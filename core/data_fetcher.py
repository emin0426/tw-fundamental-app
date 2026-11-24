"""Free official & open-source Taiwan stock data fetcher (TWSE/TPEx OpenAPI, TDCC, FinMind, Google News RSS)."""

from __future__ import annotations

import csv
import datetime as dt
import io
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, List, Optional

import requests

from .cache import TieredCache

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
TWSE_PE_URL = "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL"
TPEX_PE_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_peratio_analysis"
TWSE_STOCK_INFO_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
TPEX_STOCK_INFO_URL = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"
TWSE_MOPS_NEWS_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap04_L"
TPEX_MOPS_NEWS_URL = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap04_O"
TWSE_DAILY_CLOSE_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TPEX_DAILY_CLOSE_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"

TDCC_QRY_URL = "https://www.tdcc.com.tw/portal/zh/smWeb/qryStock"
TDCC_OPENDATA_CSV_URL = "https://smart.tdcc.com.tw/opendata/getOD.ashx?id=1-5"
TDCC_OPENAPI_URL = "https://openapi.tdcc.com.tw/v1/opendata/1-5"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, text/xml, */*",
}

# Standard 15 TDCC shareholding levels (1 to 15)
TDCC_LEVEL_LABELS: Dict[int, str] = {
    1: "1-999股 (零股)",
    2: "1,000-5,000股 (1-5張)",
    3: "5,001-10,000股 (5-10張)",
    4: "10,001-15,000股 (10-15張)",
    5: "15,001-20,000股 (15-20張)",
    6: "20,001-30,000股 (20-30張)",
    7: "30,001-40,000股 (30-40張)",
    8: "40,001-50,000股 (40-50張)",
    9: "50,001-100,000股 (50-100張)",
    10: "100,001-200,000股 (100-200張)",
    11: "200,001-400,000股 (200-400張)",
    12: "400,001-600,000股 (400-600張)",
    13: "600,001-800,000股 (600-800張)",
    14: "800,001-1,000,000股 (800-1000張)",
    15: "1,000,001股以上 (1000張以上)",
}

FINMIND_LEVEL_TO_CODE: Dict[str, int] = {
    "1-999": 1,
    "1,000-5,000": 2,
    "1000-5000": 2,
    "5,001-10,000": 3,
    "5001-10000": 3,
    "10,001-15,000": 4,
    "10001-15000": 4,
    "15,001-20,000": 5,
    "15001-20000": 5,
    "20,001-30,000": 6,
    "20001-30000": 6,
    "30,001-40,000": 7,
    "30001-40000": 7,
    "40,001-50,000": 8,
    "40001-50000": 8,
    "50,001-100,000": 9,
    "50001-100000": 9,
    "100,001-200,000": 10,
    "100001-200000": 10,
    "200,001-400,000": 11,
    "200001-400000": 11,
    "400,001-600,000": 12,
    "400001-600000": 12,
    "600,001-800,000": 13,
    "600001-800000": 13,
    "800,001-1,000,000": 14,
    "800001-1000000": 14,
    "more than 1,000,001": 15,
    "1,000,001以上": 15,
    "1000001以上": 15,
}


def _to_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace(",", "").replace("%", "")
    if not s or s in ("-", "--", "N/A", "nan", "None"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_int(val: Any) -> Optional[int]:
    f = _to_float(val)
    return int(round(f)) if f is not None else None


def _format_yyyymmdd(d: str) -> str:
    s = re.sub(r"[^0-9]", "", str(d or ""))
    if len(s) == 8:
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return str(d or "").strip()


class TaiwanStockDataFetcher:
    """Unified fetcher using free TWSE/TPEx/TDCC OpenAPI, FinMind free tier, and Google News RSS."""

    def __init__(self, finmind_token: str = "", cache: Optional[TieredCache] = None) -> None:
        self.finmind_token = (finmind_token or "").strip()
        self.cache = cache or TieredCache()
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    def _finmind_get(
        self,
        dataset: str,
        data_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeout: int = 20,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, str] = {"dataset": dataset}
        if data_id:
            params["data_id"] = data_id
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if self.finmind_token:
            params["token"] = self.finmind_token

        resp = self.session.get(FINMIND_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        if body.get("status") != 200:
            msg = body.get("msg", "FinMind API error")
            raise RuntimeError(f"FinMind 抓取 {dataset} 失敗：{msg}")
        return body.get("data", [])

    def get_stock_list(self, force_refresh: bool = False) -> List[Dict[str, str]]:
        """Load Taiwan listed/OTC stock catalog for instant code/name search."""
        cache_key = "catalog:stock_list"
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached

        stocks: Dict[str, Dict[str, str]] = {}
        try:
            rows = self._finmind_get("TaiwanStockInfo")
            for r in rows:
                sid = str(r.get("stock_id", "")).strip()
                if not sid or not sid[0].isdigit() or len(sid) > 6:
                    continue
                stocks[sid] = {
                    "stock_id": sid,
                    "stock_name": str(r.get("stock_name", "")).strip(),
                    "industry_category": str(r.get("industry_category", "其他")).strip(),
                    "market": str(r.get("type", "")).strip(),
                }
        except Exception:
            for url, mkt in ((TWSE_STOCK_INFO_URL, "twse"), (TPEX_STOCK_INFO_URL, "tpex")):
                try:
                    resp = self.session.get(url, timeout=15)
                    if resp.ok:
                        for r in resp.json():
                            sid = str(r.get("公司代號") or r.get("SecuritiesCompanyCode") or "").strip()
                            name = str(r.get("公司簡稱") or r.get("CompanyAbbreviation") or "").strip()
                            ind = str(r.get("產業別") or r.get("Industry") or "其他").strip()
                            if sid and name:
                                stocks[sid] = {
                                    "stock_id": sid,
                                    "stock_name": name,
                                    "industry_category": ind,
                                    "market": mkt,
                                }
                except Exception:
                    continue

        result = sorted(stocks.values(), key=lambda x: x["stock_id"])
        if result:
            self.cache.set(cache_key, result, TieredCache.TTL_STOCK_LIST)
        return result

    def resolve_stock(self, query: str) -> Optional[Dict[str, str]]:
        """Resolve user input like '2330' or '台積電' to stock metadata."""
        q = (query or "").strip()
        if not q:
            return None
        catalog = self.get_stock_list()
        for item in catalog:
            if item["stock_id"] == q:
                return item
        for item in catalog:
            if item["stock_name"] == q:
                return item
        q_lower = q.lower()
        for item in catalog:
            if q_lower in item["stock_name"].lower() or item["stock_id"].startswith(q_lower):
                return item
        if q.isdigit() and 4 <= len(q) <= 6:
            return {"stock_id": q, "stock_name": q, "industry_category": "未知", "market": "twse"}
        return None

    def get_market_valuation_snapshot(self, force_refresh: bool = False) -> Dict[str, Dict[str, Optional[float]]]:
        """Fetch all TWSE + TPEx PE, PB, and Dividend Yield in just 2 requests (cached 1 day)."""
        cache_key = "market:pe_pb_snapshot"
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached

        snapshot: Dict[str, Dict[str, Optional[float]]] = {}
        try:
            resp = self.session.get(TWSE_PE_URL, timeout=15)
            if resp.ok:
                for r in resp.json():
                    sid = str(r.get("Code", "")).strip()
                    if sid:
                        snapshot[sid] = {
                            "pe": _to_float(r.get("PEratio")),
                            "pb": _to_float(r.get("PBratio")),
                            "dividend_yield": _to_float(r.get("DividendYield")),
                            "name": str(r.get("Name", "")).strip(),
                        }
        except Exception:
            pass

        try:
            resp = self.session.get(TPEX_PE_URL, timeout=15)
            if resp.ok:
                for r in resp.json():
                    sid = str(r.get("SecuritiesCompanyCode", "")).strip()
                    if sid:
                        snapshot[sid] = {
                            "pe": _to_float(r.get("PriceEarningRatio")),
                            "pb": _to_float(r.get("PriceBookRatio")),
                            "dividend_yield": _to_float(r.get("YieldRatio")),
                            "name": str(r.get("CompanyName", "")).strip(),
                        }
        except Exception:
            pass

        if snapshot:
            self.cache.set(cache_key, snapshot, TieredCache.TTL_MARKET_VAL)
        return snapshot

    def get_latest_price(self, stock_id: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Get latest closing price and date for a single stock."""
        cache_key = f"quote:{stock_id}"
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached

        end_date = dt.date.today()
        start_date = end_date - dt.timedelta(days=20)
        result: Dict[str, Any] = {"stock_id": stock_id, "close": None, "date": None, "change_pct": None}
        try:
            rows = self._finmind_get(
                "TaiwanStockPrice",
                data_id=stock_id,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
            )
            if rows:
                rows_sorted = sorted(rows, key=lambda x: x.get("date", ""))
                latest = rows_sorted[-1]
                close = _to_float(latest.get("close"))
                prev_close = _to_float(rows_sorted[-2].get("close")) if len(rows_sorted) >= 2 else None
                chg_pct = ((close - prev_close) / prev_close * 100.0) if (close and prev_close) else None
                result = {
                    "stock_id": stock_id,
                    "close": close,
                    "date": latest.get("date"),
                    "change_pct": round(chg_pct, 2) if chg_pct is not None else None,
                }
        except Exception:
            for url, id_key, close_key in (
                (TWSE_DAILY_CLOSE_URL, "Code", "ClosingPrice"),
                (TPEX_DAILY_CLOSE_URL, "SecuritiesCompanyCode", "Close"),
            ):
                try:
                    resp = self.session.get(url, timeout=15)
                    if resp.ok:
                        for r in resp.json():
                            if str(r.get(id_key, "")).strip() == stock_id:
                                result["close"] = _to_float(r.get(close_key))
                                result["date"] = end_date.isoformat()
                                break
                except Exception:
                    continue
                if result["close"] is not None:
                    break

        if result["close"] is not None:
            self.cache.set(cache_key, result, TieredCache.TTL_QUOTE)
        return result

    def get_monthly_revenue(self, stock_id: str, years: int = 4, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Fetch 4 years of monthly revenue so we can compute 3 full years of YoY/MoM/Cumulative YoY."""
        cache_key = f"monthly_rev:{stock_id}:{years}"
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached

        start_year = dt.date.today().year - years
        rows = self._finmind_get(
            "TaiwanStockMonthRevenue",
            data_id=stock_id,
            start_date=f"{start_year}-01-01",
        )
        if rows:
            self.cache.set(cache_key, rows, TieredCache.TTL_MONTHLY_REV)
        return rows

    def get_financial_statements(
        self, stock_id: str, years: int = 4, force_refresh: bool = False
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch Income Statement, Balance Sheet, and Cash Flow Statement (cached 7 days)."""
        cache_key = f"financials:{stock_id}:{years}"
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached

        start_year = dt.date.today().year - years
        start_date = f"{start_year}-01-01"
        data = {
            "income_statement": self._finmind_get("TaiwanStockFinancialStatements", data_id=stock_id, start_date=start_date),
            "balance_sheet": self._finmind_get("TaiwanStockBalanceSheet", data_id=stock_id, start_date=start_date),
            "cash_flow": self._finmind_get("TaiwanStockCashFlowsStatement", data_id=stock_id, start_date=start_date),
        }
        if any(data.values()):
            self.cache.set(cache_key, data, TieredCache.TTL_FINANCIALS)
        return data

    def get_historical_pe_series(self, stock_id: str, years: int = 4, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Fetch historical daily P/E & P/B from FinMind for historical P/E river quantiles."""
        cache_key = f"hist_pe:{stock_id}:{years}"
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached

        start_year = dt.date.today().year - years
        try:
            rows = self._finmind_get(
                "TaiwanStockPER",
                data_id=stock_id,
                start_date=f"{start_year}-01-01",
            )
            sampled = rows[::5] if len(rows) > 120 else rows
            if sampled:
                self.cache.set(cache_key, sampled, TieredCache.TTL_FINANCIALS)
            return sampled
        except Exception:
            return []

    def get_peers(self, stock_id: str, industry_category: str) -> List[Dict[str, Any]]:
        """Return peer companies in the same industry with their live PE/PB from the 2-call snapshot."""
        catalog = self.get_stock_list()
        snapshot = self.get_market_valuation_snapshot()
        peers = []
        for item in catalog:
            if item["industry_category"] == industry_category and item["stock_id"] != stock_id:
                val = snapshot.get(item["stock_id"], {})
                pe = val.get("pe")
                pb = val.get("pb")
                if pe is not None or pb is not None:
                    peers.append(
                        {
                            "stock_id": item["stock_id"],
                            "stock_name": item["stock_name"],
                            "pe": pe,
                            "pb": pb,
                            "dividend_yield": val.get("dividend_yield"),
                        }
                    )
        return peers

    # =========================================================================
    # TDCC Weekly Shareholding Dispersion (集保戶股權分散表) — 3-Tier Free Fetcher
    # =========================================================================

    def _parse_tdcc_html_table(self, html: str, date_iso: str) -> List[Dict[str, Any]]:
        """Parse TDCC qryStock HTML response table into standardized rows (levels 1..15)."""
        rows_out: List[Dict[str, Any]] = []
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            for tr in soup.find_all("tr"):
                tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
                if len(tds) < 5:
                    continue
                lvl_code = _to_int(tds[0])
                if lvl_code is None or not (1 <= lvl_code <= 15):
                    continue
                people = _to_int(tds[2]) or 0
                shares = _to_int(tds[3]) or 0
                percent = _to_float(tds[4]) or 0.0
                rows_out.append(
                    {
                        "date": date_iso,
                        "level_code": lvl_code,
                        "level_label": TDCC_LEVEL_LABELS.get(lvl_code, tds[1]),
                        "people": people,
                        "shares": shares,
                        "percent": percent,
                    }
                )
        except Exception:
            pass
        return rows_out

    def _fetch_tdcc_from_official_web(
        self,
        stock_id: str,
        weeks: int = 12,
        force_refresh: bool = False,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch free weekly historical TDCC snapshots directly from tdcc.com.tw/portal/zh/smWeb/qryStock."""
        from bs4 import BeautifulSoup

        sess = requests.Session()
        sess.headers.update(
            {
                "User-Agent": DEFAULT_HEADERS["User-Agent"],
                "Referer": TDCC_QRY_URL,
                "Origin": "https://www.tdcc.com.tw",
            }
        )
        resp = sess.get(TDCC_QRY_URL, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        token_el = soup.find("input", {"name": "SYNCHRONIZER_TOKEN"})
        uri_el = soup.find("input", {"name": "SYNCHRONIZER_URI"})
        token = token_el.get("value", "") if token_el else ""
        uri = uri_el.get("value", "/portal/zh/smWeb/qryStock") if uri_el else "/portal/zh/smWeb/qryStock"

        sel = soup.find("select", {"id": "scaDate"}) or soup.find("select", {"name": "scaDate"})
        if not sel:
            return []
        avail_dates = [
            opt.get("value", "").strip()
            for opt in sel.find_all("option")
            if opt.get("value", "").strip().isdigit()
        ]
        target_dates = avail_dates[: max(1, min(weeks, len(avail_dates)))]
        if not target_dates:
            return []

        all_rows: List[Dict[str, Any]] = []
        total = len(target_dates)

        for idx, yyyymmdd in enumerate(target_dates, start=1):
            date_iso = _format_yyyymmdd(yyyymmdd)
            wk_cache_key = f"tdcc_wk:{stock_id}:{yyyymmdd}"
            cached_wk = None if force_refresh else self.cache.get(wk_cache_key)
            if cached_wk:
                all_rows.extend(cached_wk)
                if progress_cb:
                    progress_cb(idx, total)
                continue

            payload = {
                "SYNCHRONIZER_TOKEN": token,
                "SYNCHRONIZER_URI": uri,
                "method": "submit",
                "firDate": yyyymmdd,
                "scaDate": yyyymmdd,
                "sqlMethod": "StockNo",
                "stockNo": stock_id,
                "stockName": "",
            }
            try:
                post_resp = sess.post(TDCC_QRY_URL, data=payload, timeout=20)
                if post_resp.ok:
                    parsed = self._parse_tdcc_html_table(post_resp.text, date_iso)
                    if parsed:
                        self.cache.set(wk_cache_key, parsed, TieredCache.TTL_TDCC_HIST_WEEK)
                        all_rows.extend(parsed)
                    # Refresh CSRF token if TDCC rotated it in response HTML
                    next_soup = BeautifulSoup(post_resp.text, "html.parser")
                    nt = next_soup.find("input", {"name": "SYNCHRONIZER_TOKEN"})
                    if nt and nt.get("value"):
                        token = nt.get("value")
            except Exception:
                continue

            if progress_cb:
                progress_cb(idx, total)
            # Gentle polite delay so TDCC never throttles
            if idx < total:
                time.sleep(0.18)

        return all_rows

    def _fetch_tdcc_latest_opendata(self, stock_id: str, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Fallback: Official TDCC OpenData CSV/JSON (latest week for all stocks, cached 1 day)."""
        cache_key = f"tdcc_od_latest:{stock_id}"
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached

        rows_out: List[Dict[str, Any]] = []
        # Try TDCC OpenData CSV first
        try:
            resp = self.session.get(TDCC_OPENDATA_CSV_URL, timeout=25)
            if resp.ok and len(resp.text) > 100:
                text = resp.content.decode("utf-8-sig", errors="replace")
                reader = csv.reader(io.StringIO(text))
                next(reader, None)  # skip header
                for row in reader:
                    if len(row) < 6:
                        continue
                    code = row[1].strip()
                    if code != stock_id:
                        continue
                    lvl = _to_int(row[2])
                    if lvl is None or not (1 <= lvl <= 15):
                        continue
                    rows_out.append(
                        {
                            "date": _format_yyyymmdd(row[0]),
                            "level_code": lvl,
                            "level_label": TDCC_LEVEL_LABELS.get(lvl, f"級距{lvl}"),
                            "people": _to_int(row[3]) or 0,
                            "shares": _to_int(row[4]) or 0,
                            "percent": _to_float(row[5]) or 0.0,
                        }
                    )
        except Exception:
            pass

        # Secondary OpenAPI JSON fallback
        if not rows_out:
            try:
                resp = self.session.get(TDCC_OPENAPI_URL, timeout=25)
                if resp.ok:
                    for r in resp.json():
                        if str(r.get("證券代號", "")).strip() != stock_id:
                            continue
                        lvl = _to_int(r.get("持股分級"))
                        if lvl is None or not (1 <= lvl <= 15):
                            continue
                        rows_out.append(
                            {
                                "date": _format_yyyymmdd(r.get("資料日期", "")),
                                "level_code": lvl,
                                "level_label": TDCC_LEVEL_LABELS.get(lvl, f"級距{lvl}"),
                                "people": _to_int(r.get("人數")) or 0,
                                "shares": _to_int(r.get("股數")) or 0,
                                "percent": _to_float(r.get("占集保庫存數比例%")) or 0.0,
                            }
                        )
            except Exception:
                pass

        if rows_out:
            self.cache.set(cache_key, rows_out, TieredCache.TTL_TDCC_LATEST)
        return rows_out

    def get_tdcc_dispersion(
        self,
        stock_id: str,
        weeks: int = 12,
        force_refresh: bool = False,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """Fetch weekly TDCC shareholding dispersion via 3-tier free fallback.

        Priority:
        1. Official TDCC historical web query (100% free, up to 52 weeks, cached per week for 90 days)
        2. FinMind `TaiwanStockHoldingSharesPer` (if user has Backer/Sponsor token)
        3. Official TDCC OpenData CSV/JSON (latest week snapshot)
        """
        bundle_key = f"tdcc_bundle:{stock_id}:{weeks}"
        if not force_refresh:
            cached = self.cache.get(bundle_key)
            if cached:
                return cached

        rows: List[Dict[str, Any]] = []
        source_used = ""

        # Tier 1: Official TDCC historical weekly web form (100% free)
        try:
            rows = self._fetch_tdcc_from_official_web(
                stock_id, weeks=weeks, force_refresh=force_refresh, progress_cb=progress_cb
            )
            if rows:
                source_used = "集保結算所官方歷史資料（免費每週更新）"
        except Exception:
            rows = []

        # Tier 2: FinMind TaiwanStockHoldingSharesPer (works if user has Backer/Sponsor plan)
        if not rows:
            try:
                start_date = (dt.date.today() - dt.timedelta(days=weeks * 7 + 14)).isoformat()
                fm_rows = self._finmind_get(
                    "TaiwanStockHoldingSharesPer", data_id=stock_id, start_date=start_date
                )
                for r in fm_rows:
                    lvl_str = str(r.get("HoldingSharesLevel", "")).strip()
                    lvl_code = FINMIND_LEVEL_TO_CODE.get(lvl_str)
                    if lvl_code is None:
                        continue
                    rows.append(
                        {
                            "date": str(r.get("date", "")).strip(),
                            "level_code": lvl_code,
                            "level_label": TDCC_LEVEL_LABELS.get(lvl_code, lvl_str),
                            "people": _to_int(r.get("people")) or 0,
                            "shares": _to_int(r.get("unit")) or 0,
                            "percent": _to_float(r.get("percent")) or 0.0,
                        }
                    )
                if rows:
                    source_used = "FinMind 集保股權分散表"
            except Exception:
                rows = []

        # Tier 3: Official TDCC OpenData latest-week CSV/JSON
        if not rows:
            try:
                rows = self._fetch_tdcc_latest_opendata(stock_id, force_refresh=force_refresh)
                if rows:
                    source_used = "集保結算所 OpenData 最新單週資料"
            except Exception:
                rows = []

        result = {"rows": rows, "source": source_used}
        if rows:
            self.cache.set(bundle_key, result, TieredCache.TTL_TDCC_LATEST)
        return result

    def get_news_and_announcements(
        self, stock_id: str, stock_name: str, force_refresh: bool = False
    ) -> Dict[str, Any]:
        """Fetch free Google News RSS (order visibility/conference) and recent TWSE/TPEx MOPS announcements."""
        cache_key = f"news:{stock_id}"
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached

        news_items: List[Dict[str, str]] = []
        query = f'"{stock_name}" OR "{stock_id}" (法說會 OR 訂單 OR 展望 OR 產能 OR 營收 OR 目標價)'
        rss_url = (
            "https://news.google.com/rss/search?q="
            + urllib.parse.quote(query)
            + "&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        )
        try:
            resp = self.session.get(rss_url, timeout=15)
            if resp.ok:
                root = ET.fromstring(resp.content)
                for item in root.findall(".//item")[:15]:
                    title = (item.findtext("title") or "").strip()
                    link = (item.findtext("link") or "").strip()
                    pub_date = (item.findtext("pubDate") or "").strip()
                    source = ""
                    src_el = item.find("source")
                    if src_el is not None and src_el.text:
                        source = src_el.text.strip()
                    if title:
                        news_items.append(
                            {"title": title, "link": link, "pub_date": pub_date, "source": source}
                        )
        except Exception:
            pass

        announcements: List[Dict[str, str]] = []
        for url in (TWSE_MOPS_NEWS_URL, TPEX_MOPS_NEWS_URL):
            try:
                resp = self.session.get(url, timeout=15)
                if resp.ok:
                    for r in resp.json():
                        code = str(r.get("公司代號") or r.get("SecuritiesCompanyCode") or "").strip()
                        if code == stock_id:
                            announcements.append(
                                {
                                    "date": str(r.get("發言日期") or r.get("Date") or "").strip(),
                                    "subject": str(r.get("主旨 ") or r.get("主旨") or r.get("Subject") or "").strip(),
                                    "detail": str(r.get("說明") or r.get("Description") or "").strip()[:400],
                                }
                            )
            except Exception:
                continue

        official_links = {
            "集保戶股權分散表查詢": TDCC_QRY_URL,
            "產業價值鏈資訊平台": f"https://ic.tpex.org.tw/company_chain.php?stk_code={stock_id}",
            "臺灣指數公司 IR 法說會查詢": "https://irengage.taiwanindex.com.tw/conferenceList",
            "TWSE WebPro 法說會影音": "https://webpro.twse.com.tw/WebPortal/vod/101/?categoryId=148",
            "櫃買影音法說會專區": "https://www.tpex.org.tw/zh-tw/about/company/media/investor-conference.html",
        }

        result = {
            "news": news_items,
            "announcements": announcements[:10],
            "official_links": official_links,
        }
        self.cache.set(cache_key, result, TieredCache.TTL_NEWS)
        return result

    @staticmethod
    def extract_pdf_text(pdf_bytes: bytes, max_chars: int = 12000) -> str:
        """Optional helper to extract text from an uploaded or downloaded investor conference PDF."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(pdf_bytes))
            chunks: List[str] = []
            total = 0
            for page in reader.pages:
                txt = (page.extract_text() or "").strip()
                if txt:
                    chunks.append(txt)
                    total += len(txt)
                    if total >= max_chars:
                        break
            return "\n".join(chunks)[:max_chars]
        except Exception:
            return ""
