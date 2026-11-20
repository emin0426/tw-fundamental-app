"""Free official & open-source Taiwan stock data fetcher (TWSE/TPEx OpenAPI, FinMind, Google News RSS)."""

from __future__ import annotations

import datetime as dt
import io
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

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

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TW-Fundamental-Analyzer/1.0",
    "Accept": "application/json, text/xml, */*",
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


class TaiwanStockDataFetcher:
    """Unified fetcher using free TWSE/TPEx OpenAPI, FinMind free tier, and Google News RSS."""

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
            # Fallback to TWSE + TPEx OpenAPI basic info
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
        # Exact stock_id match
        for item in catalog:
            if item["stock_id"] == q:
                return item
        # Exact name match
        for item in catalog:
            if item["stock_name"] == q:
                return item
        # Partial name or id match
        q_lower = q.lower()
        for item in catalog:
            if q_lower in item["stock_name"].lower() or item["stock_id"].startswith(q_lower):
                return item
        # If catalog failed to load and user gave 4-digit code
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
        # 1. TWSE Listed
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

        # 2. TPEx OTC
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
            # Fallback to TWSE/TPEx daily close snapshot
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
            # Sample every ~5 trading days to keep cache compact while preserving distribution
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
