"""Fundamental analysis engine: 3-yr statements, Layer-1 cross-statement diagnostics, EPS projection & valuation."""

from __future__ import annotations

import datetime as dt
import statistics
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


FINANCIAL_INDUSTRIES = {
    "金融保險業",
    "金融業",
    "銀行業",
    "保險業",
    "證券期貨業",
    "Financial",
}

CYCLICAL_INDUSTRIES = {
    "航運業",
    "鋼鐵工業",
    "塑膠工業",
    "水泥工業",
    "造紙工業",
    "橡膠工業",
    "玻璃陶瓷",
    "油電燃氣業",
    "光電業",
}

# Mapping of canonical Chinese display label -> possible FinMind `type` or `origin_name` keys
INCOME_COMPACT_MAP: List[Tuple[str, List[str]]] = [
    ("營業收入", ["Revenue", "營業收入合計", "營業收入"]),
    ("營業成本", ["CostOfGoodsSold", "營業成本合計", "營業成本"]),
    ("營業毛利", ["GrossProfit", "營業毛利（毛損）淨額", "營業毛利（毛損）", "營業毛利"]),
    ("營業費用", ["OperatingExpenses", "營業費用合計", "營業費用"]),
    ("研究發展費用", ["ResearchAndDevelopmentExpenses", "研究發展費用", "研究發展費用合計"]),
    ("營業淨利", ["OperatingIncome", "營業利益（損失）", "營業淨利"]),
    ("營業外收入及支出", ["TotalNonoperatingIncomeAndExpense", "營業外收入及支出合計", "營業外收入及支出"]),
    ("稅前淨利", ["PreTaxIncome", "稅前淨利（淨損）", "繼續營業單位稅前淨利（淨損）", "稅前淨利"]),
    ("所得稅費用", ["TAX", "IncomeTaxExpense", "所得稅費用（利益）合計", "所得稅費用"]),
    ("稅後淨利", ["IncomeAfterTaxes", "本期淨利（淨損）", "繼續營業單位本期淨利（淨損）", "稅後淨利"]),
    ("其他綜合損益", ["OtherComprehensiveIncome", "其他綜合損益（淨額）", "其他綜合損益"]),
    ("綜合損益", ["TotalConsolidatedProfitForThePeriod", "本期綜合損益總額", "綜合損益"]),
    ("歸屬母公司稅後淨利", ["EquityAttributableToOwnersOfParent", "母公司業主（淨利／損）", "淨利（損）歸屬於母公司業主"]),
    ("基本每股盈餘(EPS)", ["EPS", "基本每股盈餘（元）", "基本每股盈餘"]),
    ("稀釋每股盈餘", ["DilutedEPS", "稀釋每股盈餘（元）", "稀釋每股盈餘"]),
]

FINANCIAL_INCOME_COMPACT_MAP: List[Tuple[str, List[str]]] = [
    ("利息淨收益", ["NetInterestIncome", "利息淨收益"]),
    ("利息以外淨收益", ["NetNonInterestIncome", "利息以外淨收益"]),
    ("淨收益", ["Revenue", "NetRevenue", "淨收益"]),
    ("營業費用", ["OperatingExpenses", "營業費用"]),
    ("稅前淨利", ["PreTaxIncome", "繼續營業單位稅前淨利（淨損）", "稅前淨利"]),
    ("所得稅費用", ["TAX", "所得稅費用（利益）"]),
    ("稅後淨利", ["IncomeAfterTaxes", "本期稅後淨利（淨損）", "稅後淨利"]),
    ("基本每股盈餘(EPS)", ["EPS", "基本每股盈餘（元）", "基本每股盈餘"]),
]

BALANCE_COMPACT_MAP: List[Tuple[str, List[str]]] = [
    ("現金及約當現金", ["CashAndCashEquivalents", "現金及約當現金"]),
    ("應收帳款淨額", ["AccountsReceivableNet", "應收帳款淨額", "應收票據及帳款淨額"]),
    ("存貨", ["Inventories", "存貨", "存貨合計"]),
    ("流動資產合計", ["CurrentAssets", "流動資產合計"]),
    ("不動產、廠房及設備", ["PropertyPlantAndEquipment", "不動產、廠房及設備"]),
    ("資產總計", ["TotalAssets", "資產總計", "資產總額"]),
    ("合約負債(流動)", ["CurrentContractLiabilities", "ContractLiabilitiesCurrent", "合約負債－流動", "合約負債-流動"]),
    ("短期借款", ["ShorttermBorrowings", "短期借款"]),
    ("流動負債合計", ["CurrentLiabilities", "流動負債合計"]),
    ("負債總計", ["Liabilities", "負債總計", "負債總額"]),
    ("股本", ["OrdinaryShare", "CapitalStock", "普通股股本", "股本合計"]),
    ("權益總計", ["Equity", "權益總計", "股東權益總計"]),
]

CASHFLOW_COMPACT_MAP: List[Tuple[str, List[str]]] = [
    ("營業活動之淨現金流入(出)", ["CashFlowsFromOperatingActivities", "營業活動之淨現金流入（流出）"]),
    ("取得不動產、廠房及設備(資本支出)", ["PropertyAndPlantAndEquipment", "AcquisitionOfPropertyPlantAndEquipment", "取得不動產、廠房及設備"]),
    ("折舊費用", ["Depreciation", "折舊費用"]),
    ("投資活動之淨現金流入(出)", ["CashProvidedByInvestingActivities", "投資活動之淨現金流入（流出）"]),
    ("籌資活動之淨現金流入(出)", ["CashFlowsProvidedFromFinancingActivities", "籌資活動之淨現金流入（流出）"]),
]


def _date_to_quarter_label(date_str: str) -> str:
    """Convert '2026-09-30' -> '2026Q3'."""
    try:
        y, m, _ = date_str.split("-")
        q = (int(m) - 1) // 3 + 1
        return f"{y}Q{q}"
    except Exception:
        return date_str


def _is_financial_industry(industry: str) -> bool:
    ind = (industry or "").strip()
    return any(k in ind for k in FINANCIAL_INDUSTRIES)


def _is_cyclical_industry(industry: str) -> bool:
    ind = (industry or "").strip()
    return any(k in ind for k in CYCLICAL_INDUSTRIES)


def build_monthly_revenue_table(raw_rows: List[Dict[str, Any]], max_months: int = 36) -> pd.DataFrame:
    """Build 3-year monthly revenue DataFrame matching user screenshot + MoM column."""
    if not raw_rows:
        return pd.DataFrame()

    df = pd.DataFrame(raw_rows)
    if "revenue_year" not in df.columns or "revenue_month" not in df.columns:
        return pd.DataFrame()

    df["revenue_year"] = pd.to_numeric(df["revenue_year"], errors="coerce")
    df["revenue_month"] = pd.to_numeric(df["revenue_month"], errors="coerce")
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df = df.dropna(subset=["revenue_year", "revenue_month", "revenue"]).copy()
    df["revenue_year"] = df["revenue_year"].astype(int)
    df["revenue_month"] = df["revenue_month"].astype(int)
    df = df.sort_values(["revenue_year", "revenue_month"]).drop_duplicates(
        subset=["revenue_year", "revenue_month"], keep="last"
    )

    # Convert to thousands (千元) matching MOPS / user screenshot
    df["rev_k"] = df["revenue"] / 1000.0
    df["ym"] = df.apply(lambda r: f"{int(r['revenue_year'])}/{int(r['revenue_month']):02d}", axis=1)

    # Cumulative revenue within each calendar year
    df["cum_rev_k"] = df.groupby("revenue_year")["rev_k"].cumsum()

    # Lookup map for YoY & MoM
    rev_map = {(int(r.revenue_year), int(r.revenue_month)): float(r.rev_k) for r in df.itertuples()}
    cum_map = {(int(r.revenue_year), int(r.revenue_month)): float(r.cum_rev_k) for r in df.itertuples()}

    mom_list, yoy_list, cum_yoy_list = [], [], []
    for r in df.itertuples():
        y, m = int(r.revenue_year), int(r.revenue_month)
        prev_y, prev_m = (y - 1, 12) if m == 1 else (y, m - 1)
        prev_rev = rev_map.get((prev_y, prev_m))
        last_yr_rev = rev_map.get((y - 1, m))
        last_yr_cum = cum_map.get((y - 1, m))

        mom_list.append(((r.rev_k - prev_rev) / prev_rev * 100.0) if prev_rev else np.nan)
        yoy_list.append(((r.rev_k - last_yr_rev) / last_yr_rev * 100.0) if last_yr_rev else np.nan)
        cum_yoy_list.append(((r.cum_rev_k - last_yr_cum) / last_yr_cum * 100.0) if last_yr_cum else np.nan)

    df["月增率(%)"] = mom_list
    df["年增率(%)"] = yoy_list
    df["累計年增率(%)"] = cum_yoy_list

    out = pd.DataFrame(
        {
            "月份": df["ym"],
            "營收(千元)": df["rev_k"],
            "月增率(%)": df["月增率(%)"],
            "年增率(%)": df["年增率(%)"],
            "累計營收(千元)": df["cum_rev_k"],
            "累計年增率(%)": df["累計年增率(%)"],
            "_year": df["revenue_year"],
            "_month": df["revenue_month"],
        }
    )
    # Sort newest first (like user screenshot)
    out = out.sort_values(["_year", "_month"], ascending=False).head(max_months).reset_index(drop=True)
    return out


def _pivot_statement(raw_rows: List[Dict[str, Any]], max_quarters: int = 12) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Pivot FinMind long-format statement into wide DataFrame (index=item_key, columns=newest quarters)."""
    if not raw_rows:
        return pd.DataFrame(), {}

    df = pd.DataFrame(raw_rows)
    if df.empty or "date" not in df.columns or "type" not in df.columns or "value" not in df.columns:
        return pd.DataFrame(), {}

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["quarter"] = df["date"].apply(_date_to_quarter_label)
    df["origin_name"] = df.get("origin_name", df["type"]).fillna(df["type"]).astype(str)

    # Keep mapping from type -> Chinese origin_name
    name_map = dict(zip(df["type"], df["origin_name"]))

    piv = df.pivot_table(index="type", columns="quarter", values="value", aggfunc="last")
    # Sort columns newest -> oldest
    cols = sorted(piv.columns, reverse=True)[:max_quarters]
    return piv[cols], name_map


def _extract_compact_table(
    piv: pd.DataFrame,
    name_map: Dict[str, str],
    compact_spec: List[Tuple[str, List[str]]],
    scale_thousand: bool = True,
) -> pd.DataFrame:
    """Build clean horizontal statement table (in 千元 except EPS) with requested canonical rows."""
    if piv.empty:
        return pd.DataFrame()

    # Reverse lookup from origin_name -> type key
    origin_to_type = {v: k for k, v in name_map.items()}
    rows = []
    for label, aliases in compact_spec:
        matched_series = None
        for alias in aliases:
            if alias in piv.index:
                matched_series = piv.loc[alias]
                break
            if alias in origin_to_type and origin_to_type[alias] in piv.index:
                matched_series = piv.loc[origin_to_type[alias]]
                break
        if matched_series is not None:
            is_per_share = "EPS" in label or "每股" in label
            vals = matched_series if (is_per_share or not scale_thousand) else (matched_series / 1000.0)
            row_dict = {"科目": label}
            for col in piv.columns:
                row_dict[col] = vals.get(col, np.nan)
            rows.append(row_dict)

    return pd.DataFrame(rows)


def _build_full_table(piv: pd.DataFrame, name_map: Dict[str, str], scale_thousand: bool = True) -> pd.DataFrame:
    """Build full statement table (all available rows) for the expandable detail view."""
    if piv.empty:
        return pd.DataFrame()
    rows = []
    for type_key in piv.index:
        label = name_map.get(type_key, type_key)
        is_per_share = "EPS" in str(type_key).upper() or "每股" in str(label)
        vals = piv.loc[type_key] if (is_per_share or not scale_thousand) else (piv.loc[type_key] / 1000.0)
        row_dict = {"科目": label}
        for col in piv.columns:
            row_dict[col] = vals.get(col, np.nan)
        rows.append(row_dict)
    return pd.DataFrame(rows)


def build_annual_from_quarterly(quarterly_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate quarterly income statement into annual columns (e.g., 2025, 2024, 2023)."""
    if quarterly_df.empty or "科目" not in quarterly_df.columns:
        return pd.DataFrame()

    q_cols = [c for c in quarterly_df.columns if c != "科目" and "Q" in str(c)]
    year_groups: Dict[str, List[str]] = {}
    for c in q_cols:
        yr = str(c)[:4]
        year_groups.setdefault(yr, []).append(c)

    years_sorted = sorted(year_groups.keys(), reverse=True)
    out_rows = []
    for _, row in quarterly_df.iterrows():
        rec: Dict[str, Any] = {"科目": row["科目"]}
        for yr in years_sorted:
            cols = year_groups[yr]
            vals = pd.to_numeric(row[cols], errors="coerce").dropna()
            label_col = f"{yr}年" if len(cols) == 4 else f"{yr}(前{len(cols)}季)"
            rec[label_col] = vals.sum() if not vals.empty else np.nan
        out_rows.append(rec)
    return pd.DataFrame(out_rows)


def _get_row_series(df: pd.DataFrame, label: str) -> pd.Series:
    if df.empty or "科目" not in df.columns:
        return pd.Series(dtype=float)
    sub = df[df["科目"] == label]
    if sub.empty:
        return pd.Series(dtype=float)
    cols = [c for c in df.columns if c != "科目"]
    return pd.to_numeric(sub.iloc[0][cols], errors="coerce")


def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or pd.isna(a) or pd.isna(b) or abs(float(b)) < 1e-9:
        return None
    return float(a) / float(b)


def run_cross_statement_diagnostics(
    rev_df: pd.DataFrame,
    inc_q: pd.DataFrame,
    bal_q: pd.DataFrame,
    cf_q: pd.DataFrame,
    is_financial: bool = False,
) -> List[Dict[str, str]]:
    """Layer 1: Objective Python rule-based Red/Yellow/Green cross-statement leading diagnostics."""
    flags: List[Dict[str, str]] = []

    # 1. Revenue Momentum Crossover (3M cumulative YoY vs 12M cumulative YoY)
    if not rev_df.empty and len(rev_df) >= 15:
        r3_now = rev_df.iloc[0:3]["營收(千元)"].sum()
        r3_prev = rev_df.iloc[12:15]["營收(千元)"].sum()
        yoy_3m = _safe_div(r3_now - r3_prev, r3_prev)

        r12_now = rev_df.iloc[0:12]["營收(千元)"].sum()
        r12_prev = rev_df.iloc[12:24]["營收(千元)"].sum() if len(rev_df) >= 24 else None
        yoy_12m = _safe_div(r12_now - r12_prev, r12_prev) if r12_prev else None

        if yoy_3m is not None:
            yoy_3m_pct = yoy_3m * 100
            yoy_12m_pct = yoy_12m * 100 if yoy_12m is not None else None
            if yoy_12m_pct is not None and yoy_3m_pct > yoy_12m_pct and yoy_3m_pct > 0:
                status = "green"
                msg = f"近 3 月營收年增 {yoy_3m_pct:+.1f}% 高於近 12 月年增 {yoy_12m_pct:+.1f}%，營收動能處於加速擴張期。"
            elif yoy_3m_pct > 0:
                status = "yellow"
                msg = f"近 3 月營收年增 {yoy_3m_pct:+.1f}% 仍維持成長，但較近 12 月均值略有趨緩，留意高基期效應。"
            else:
                status = "red"
                msg = f"近 3 月營收年增 {yoy_3m_pct:+.1f}% 轉為衰退，短期接單或出貨動能轉弱。"
            flags.append(
                {
                    "category": "營收加速拐點",
                    "status": status,
                    "metric": f"近3月 YoY {yoy_3m_pct:+.1f}%" + (f" / 近12月 {yoy_12m_pct:+.1f}%" if yoy_12m_pct is not None else ""),
                    "explanation": msg,
                }
            )

    # 2. Gross Margin & Operating Margin Trend (non-financial)
    if not is_financial and not inc_q.empty:
        rev_s = _get_row_series(inc_q, "營業收入")
        gp_s = _get_row_series(inc_q, "營業毛利")
        op_s = _get_row_series(inc_q, "營業淨利")
        pretax_s = _get_row_series(inc_q, "稅前淨利")

        if len(rev_s) >= 5 and not gp_s.empty and not op_s.empty:
            gm_now = _safe_div(gp_s.iloc[0], rev_s.iloc[0])
            gm_yoy = _safe_div(gp_s.iloc[4], rev_s.iloc[4])
            opm_now = _safe_div(op_s.iloc[0], rev_s.iloc[0])
            opm_yoy = _safe_div(op_s.iloc[4], rev_s.iloc[4])

            if gm_now is not None and gm_yoy is not None and opm_now is not None and opm_yoy is not None:
                gm_diff = (gm_now - gm_yoy) * 100
                opm_diff = (opm_now - opm_yoy) * 100
                if gm_diff >= 0.5 and opm_diff >= 0.5:
                    status = "green"
                    msg = f"最新季毛利率 {gm_now*100:.1f}%（年增 {gm_diff:+.1f}pp）、營益率 {opm_now*100:.1f}%（年增 {opm_diff:+.1f}pp），產品組合優化且具規模經濟。"
                elif gm_diff <= -2.0 or opm_diff <= -2.0:
                    status = "red"
                    msg = f"最新季毛利率 {gm_now*100:.1f}%（年變動 {gm_diff:+.1f}pp）、營益率 {opm_now*100:.1f}%，獲利結構承壓，需留意價格競爭或稼動率下滑。"
                else:
                    status = "yellow"
                    msg = f"最新季毛利率 {gm_now*100:.1f}%、營益率 {opm_now*100:.1f}%，較去年同期大致持平。"
                flags.append(
                    {
                        "category": "毛利與營業利益率",
                        "status": status,
                        "metric": f"毛利率 {gm_now*100:.1f}% / 營益率 {opm_now*100:.1f}%",
                        "explanation": msg,
                    }
                )

        # 3. Core Business Profit Quality (Operating Income / Pre-Tax Income)
        if not op_s.empty and not pretax_s.empty:
            core_ratio = _safe_div(op_s.iloc[0], pretax_s.iloc[0])
            if core_ratio is not None and pretax_s.iloc[0] > 0:
                if core_ratio >= 0.80:
                    status = "green"
                    msg = f"本業營業利益佔稅前淨利 {core_ratio*100:.1f}%，獲利紮實來自本業，未過度依賴業外匯兌或處分利益。"
                elif core_ratio >= 0.60:
                    status = "yellow"
                    msg = f"本業獲利佔比 {core_ratio*100:.1f}%，有一定比例來自業外收入（如利息或匯兌），推估 EPS 時應以本業為主。"
                else:
                    status = "red"
                    msg = f"本業獲利佔比僅 {core_ratio*100:.1f}%，當季獲利高度依賴業外一次性收益，持續性偏低。"
                flags.append(
                    {
                        "category": "本業獲利含金量",
                        "status": status,
                        "metric": f"營業利益／稅前淨利 = {core_ratio*100:.1f}%",
                        "explanation": msg,
                    }
                )

    # 4. Contract Liabilities (Leading Order Indicator)
    if not is_financial and not bal_q.empty:
        cl_s = _get_row_series(bal_q, "合約負債(流動)")
        rev_s = _get_row_series(inc_q, "營業收入")
        if len(cl_s) >= 5 and not pd.isna(cl_s.iloc[0]) and cl_s.iloc[0] > 0:
            cl_qoq = _safe_div(cl_s.iloc[0] - cl_s.iloc[1], cl_s.iloc[1])
            cl_yoy = _safe_div(cl_s.iloc[0] - cl_s.iloc[4], cl_s.iloc[4])
            cl_to_rev = _safe_div(cl_s.iloc[0], rev_s.iloc[0]) if not rev_s.empty else None
            if cl_yoy is not None:
                if cl_yoy >= 0.15:
                    status = "green"
                    msg = f"合約負債年增 {cl_yoy*100:+.1f}%（季增 {(cl_qoq or 0)*100:+.1f}%），客戶預收訂金明顯增加，未來 1～2 季營收認列能見度高。"
                elif cl_yoy <= -0.15:
                    status = "yellow"
                    msg = f"合約負債年減 {cl_yoy*100:+.1f}%，在手預收訂單消化速度高於新接單速度，需追蹤後續接單補庫狀況。"
                else:
                    status = "info"
                    msg = f"合約負債年變動 {cl_yoy*100:+.1f}%，在手預收款維持平穩。"
                extra = f"（佔單季營收 {cl_to_rev*100:.1f}%）" if cl_to_rev is not None else ""
                flags.append(
                    {
                        "category": "訂單前兆（合約負債）",
                        "status": status,
                        "metric": f"年增 {cl_yoy*100:+.1f}% {extra}".strip(),
                        "explanation": msg,
                    }
                )

    # 5. Inventory vs Revenue & DIO (Cross-Statement Impact on Future Income Statement!)
    if not is_financial and not bal_q.empty and not inc_q.empty:
        inv_s = _get_row_series(bal_q, "存貨")
        rev_s = _get_row_series(inc_q, "營業收入")
        cogs_s = _get_row_series(inc_q, "營業成本")
        if len(inv_s) >= 5 and len(rev_s) >= 5 and len(cogs_s) >= 5:
            inv_yoy = _safe_div(inv_s.iloc[0] - inv_s.iloc[4], inv_s.iloc[4])
            rev_yoy = _safe_div(rev_s.iloc[0] - rev_s.iloc[4], rev_s.iloc[4])
            dio_now = _safe_div(inv_s.iloc[0], cogs_s.iloc[0]) * 90 if _safe_div(inv_s.iloc[0], cogs_s.iloc[0]) else None
            dio_yoy = _safe_div(inv_s.iloc[4], cogs_s.iloc[4]) * 90 if _safe_div(inv_s.iloc[4], cogs_s.iloc[4]) else None

            if inv_yoy is not None and rev_yoy is not None and dio_now is not None and dio_yoy is not None:
                dio_diff = dio_now - dio_yoy
                if inv_yoy > rev_yoy + 0.15 and dio_diff > 10:
                    status = "red"
                    msg = (
                        f"存貨年增 {inv_yoy*100:+.1f}% 顯著高於營收年增 {rev_yoy*100:+.1f}%，且存貨週轉天數拉長至 {dio_now:.0f} 天（年增 {dio_diff:+.0f} 天）。"
                        "跨表風險：若庫存去化不及，後續可能需提列「存貨跌價損失」或降價出清，直接壓縮未來 1～2 季損益表的毛利率。"
                    )
                elif inv_yoy > rev_yoy + 0.08 and dio_diff > 5:
                    status = "yellow"
                    msg = (
                        f"存貨年增 {inv_yoy*100:+.1f}% 略高於營收年增 {rev_yoy*100:+.1f}%（週轉天數 {dio_now:.0f} 天）。"
                        "需區分為旺季主動備料（正面）或客戶拉貨放緩（警戒）。"
                    )
                else:
                    status = "green"
                    msg = (
                        f"存貨年增 {inv_yoy*100:+.1f}% 與營收年增 {rev_yoy*100:+.1f}% 匹配良好，存貨週轉天數 {dio_now:.0f} 天（年變動 {dio_diff:+.0f} 天），庫存健康。"
                    )
                flags.append(
                    {
                        "category": "存貨與損益表連動風險",
                        "status": status,
                        "metric": f"存貨 YoY {inv_yoy*100:+.1f}% / 週轉 {dio_now:.0f} 天",
                        "explanation": msg,
                    }
                )

    # 6. CapEx Expansion Confidence (CapEx / Depreciation)
    if not is_financial and not cf_q.empty:
        capex_s = _get_row_series(cf_q, "取得不動產、廠房及設備(資本支出)")
        dep_s = _get_row_series(cf_q, "折舊費用")
        if len(capex_s) >= 4 and len(dep_s) >= 4:
            capex_4q = abs(capex_s.iloc[0:4].dropna().sum())
            dep_4q = abs(dep_s.iloc[0:4].dropna().sum())
            ratio = _safe_div(capex_4q, dep_4q)
            if ratio is not None:
                if ratio >= 1.2:
                    status = "green"
                    msg = f"近四季資本支出／折舊比達 {ratio:.2f} 倍，公司處於積極擴產與資本投資階段，反映管理層對中長期需求具信心。"
                elif ratio >= 0.8:
                    status = "info"
                    msg = f"近四季資本支出／折舊比為 {ratio:.2f} 倍，屬穩健維護與溫和升級水準。"
                else:
                    status = "yellow"
                    msg = f"近四季資本支出／折舊比僅 {ratio:.2f} 倍，資本投資保守，中長期產能擴張動能有限。"
                flags.append(
                    {
                        "category": "中長期擴產信心（資本支出）",
                        "status": status,
                        "metric": f"CapEx／折舊 = {ratio:.2f}x",
                        "explanation": msg,
                    }
                )

    # 7. Cash Flow Quality (Operating Cash Flow / Net Income & FCF)
    if not cf_q.empty and not inc_q.empty:
        ocf_s = _get_row_series(cf_q, "營業活動之淨現金流入(出)")
        capex_s = _get_row_series(cf_q, "取得不動產、廠房及設備(資本支出)")
        ni_s = _get_row_series(inc_q, "稅後淨利")
        if len(ocf_s) >= 4 and len(ni_s) >= 4:
            ocf_4q = ocf_s.iloc[0:4].dropna().sum()
            ni_4q = ni_s.iloc[0:4].dropna().sum()
            capex_4q = abs(capex_s.iloc[0:4].dropna().sum()) if not capex_s.empty else 0.0
            fcf_4q = ocf_4q - capex_4q
            ocf_ratio = _safe_div(ocf_4q, ni_4q)
            if ocf_ratio is not None and ni_4q > 0:
                if ocf_ratio >= 0.85 and fcf_4q > 0:
                    status = "green"
                    msg = f"近四季營業現金流／稅後淨利達 {ocf_ratio*100:.0f}%，且自由現金流為正（{fcf_4q/1e6:,.2f} 十億元），獲利確實轉化為真現金。"
                elif ocf_ratio >= 0.60:
                    status = "yellow"
                    msg = f"近四季營業現金流／稅後淨利為 {ocf_ratio*100:.0f}%，自由現金流為 {fcf_4q/1e6:,.2f} 十億元（可能因大幅擴產資本支出或應收帳款增加所致）。"
                else:
                    status = "red"
                    msg = f"近四季營業現金流／稅後淨利僅 {ocf_ratio*100:.0f}%，帳面獲利與現金流出現背離，需提防應收帳款收現風險或庫存積壓。"
                flags.append(
                    {
                        "category": "現金流含金量與防雷",
                        "status": status,
                        "metric": f"OCF／淨利 {ocf_ratio*100:.0f}% / 4Q FCF {fcf_4q/1e6:,.2f}B",
                        "explanation": msg,
                    }
                )

    return flags


def estimate_full_year_eps_and_valuation(
    stock_meta: Dict[str, str],
    latest_price: Optional[float],
    rev_df: pd.DataFrame,
    inc_q: pd.DataFrame,
    bal_q: pd.DataFrame,
    peers: List[Dict[str, Any]],
    market_val: Optional[Dict[str, Optional[float]]],
    hist_pe_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Estimate full-year EPS via 3 models (Seasonal, Revenue Nowcasting, TTM) & compute valuation bands."""
    industry = stock_meta.get("industry_category", "其他")
    is_fin = _is_financial_industry(industry)
    is_cyc = _is_cyclical_industry(industry)

    eps_s = _get_row_series(inc_q, "基本每股盈餘(EPS)")
    rev_q_s = _get_row_series(inc_q, "營業收入")
    ni_q_s = _get_row_series(inc_q, "稅後淨利")

    # Organize EPS by year & quarter
    eps_by_year: Dict[int, Dict[int, float]] = {}
    for col, val in eps_s.items():
        if pd.isna(val) or "Q" not in str(col):
            continue
        yr = int(str(col)[:4])
        q = int(str(col)[-1])
        eps_by_year.setdefault(yr, {})[q] = float(val)

    current_year = max(eps_by_year.keys()) if eps_by_year else dt.date.today().year
    reported_qs = sorted(eps_by_year.get(current_year, {}).keys())
    k = len(reported_qs)
    cum_eps_now = sum(eps_by_year.get(current_year, {}).values()) if k > 0 else None

    # Model C: TTM EPS (sum of latest 4 quarters)
    valid_eps = eps_s.dropna()
    ttm_eps = float(valid_eps.iloc[0:4].sum()) if len(valid_eps) >= 4 else None

    # Last full year EPS for YoY growth calculation
    prev_full_year = current_year - 1
    prev_year_eps = (
        sum(eps_by_year[prev_full_year].values())
        if (prev_full_year in eps_by_year and len(eps_by_year[prev_full_year]) == 4)
        else None
    )

    # Model A: Historical Seasonal Weighting
    seasonal_eps = None
    seasonal_weight_used = None
    if k == 4:
        seasonal_eps = cum_eps_now
        seasonal_weight_used = 1.0
    elif 1 <= k <= 3 and cum_eps_now is not None:
        weights = []
        for yr, qdict in eps_by_year.items():
            if yr < current_year and len(qdict) == 4:
                full = sum(qdict.values())
                part = sum(qdict.get(q, 0.0) for q in reported_qs)
                if full > 0 and part > 0:
                    w = part / full
                    if 0.05 < w < 0.98:
                        weights.append(w)
        avg_w = statistics.mean(weights) if weights else (k / 4.0)
        seasonal_weight_used = avg_w
        seasonal_eps = cum_eps_now / avg_w

    # Model B: Latest Monthly Revenue Nowcasting (for unfinished current quarter)
    nowcast_eps = None
    nowcast_note = ""
    if 1 <= k <= 3 and not rev_df.empty and not rev_q_s.empty and not ni_q_s.empty and cum_eps_now is not None:
        next_q = max(reported_qs) + 1
        q_months = [(next_q - 1) * 3 + 1, (next_q - 1) * 3 + 2, (next_q - 1) * 3 + 3]
        sub_m = rev_df[(rev_df["_year"] == current_year) & (rev_df["_month"].isin(q_months))]
        if not sub_m.empty:
            m_count = len(sub_m)
            proj_q_rev_k = float(sub_m["營收(千元)"].sum()) * (3.0 / m_count)
            # Trailing 4Q net margin & implied shares (from latest quarter NI / EPS)
            recent_rev_4q = rev_q_s.iloc[0:4].dropna().sum()
            recent_ni_4q = ni_q_s.iloc[0:4].dropna().sum()
            net_margin = _safe_div(recent_ni_4q, recent_rev_4q)
            latest_ni = ni_q_s.iloc[0] if not pd.isna(ni_q_s.iloc[0]) else None
            latest_eps = eps_s.iloc[0] if not pd.isna(eps_s.iloc[0]) else None
            implied_shares_k = _safe_div(latest_ni, latest_eps) if (latest_ni and latest_eps and latest_eps > 0) else None

            if net_margin and implied_shares_k and implied_shares_k > 0:
                proj_next_q_eps = (proj_q_rev_k * net_margin) / implied_shares_k
                # Also compute optimistic margin using latest quarter margin
                latest_margin = _safe_div(latest_ni, rev_q_s.iloc[0]) if not pd.isna(rev_q_s.iloc[0]) else net_margin
                if next_q == 4:
                    nowcast_eps = cum_eps_now + proj_next_q_eps
                else:
                    # Chain with seasonal weight for k+1 quarters
                    w_next = (k + 1) / 4.0
                    nowcast_eps = (cum_eps_now + proj_next_q_eps) / w_next
                nowcast_note = f"已納入 {current_year}Q{next_q} 已公布 {m_count} 個月營收推估當季 EPS 約 {proj_next_q_eps:.2f} 元"

    # Combine Conservative / Base / Optimistic Full-Year EPS
    candidates = [x for x in (seasonal_eps, nowcast_eps, ttm_eps) if x is not None and x > 0]
    if candidates:
        base_eps = nowcast_eps if (nowcast_eps and nowcast_eps > 0) else (seasonal_eps or ttm_eps)
        conservative_eps = min(candidates)
        optimistic_eps = max(candidates) * (1.05 if len(candidates) == 1 else 1.0)
    else:
        base_eps = seasonal_eps or ttm_eps
        conservative_eps = base_eps
        optimistic_eps = base_eps

    eps_yoy_growth_pct = (
        ((base_eps - prev_year_eps) / abs(prev_year_eps) * 100.0)
        if (base_eps is not None and prev_year_eps is not None and abs(prev_year_eps) > 1e-6)
        else None
    )

    forward_pe = _safe_div(latest_price, base_eps) if (latest_price and base_eps and base_eps > 0) else None
    peg = (
        _safe_div(forward_pe, eps_yoy_growth_pct)
        if (forward_pe and eps_yoy_growth_pct and eps_yoy_growth_pct > 0)
        else None
    )

    # Peer Median P/E & P/B
    peer_pes = [p["pe"] for p in peers if p.get("pe") is not None and 2.0 < p["pe"] < 80.0]
    peer_pbs = [p["pb"] for p in peers if p.get("pb") is not None and 0.2 < p["pb"] < 20.0]
    peer_median_pe = statistics.median(peer_pes) if peer_pes else None
    peer_q25_pe = float(np.percentile(peer_pes, 25)) if len(peer_pes) >= 4 else (peer_median_pe * 0.85 if peer_median_pe else None)
    peer_q75_pe = float(np.percentile(peer_pes, 75)) if len(peer_pes) >= 4 else (peer_median_pe * 1.15 if peer_median_pe else None)
    peer_median_pb = statistics.median(peer_pbs) if peer_pbs else None

    # Historical P/E River quantiles (25th / 50th / 75th)
    hist_pes = [float(r["PER"]) for r in hist_pe_rows if r.get("PER") and 2.0 < float(r["PER"]) < 80.0]
    hist_pe_q25 = float(np.percentile(hist_pes, 25)) if len(hist_pes) >= 10 else None
    hist_pe_med = float(np.percentile(hist_pes, 50)) if len(hist_pes) >= 10 else None
    hist_pe_q75 = float(np.percentile(hist_pes, 75)) if len(hist_pes) >= 10 else None

    # Reasonable Price Ranges
    peer_price_band = (
        (round(base_eps * peer_q25_pe, 1), round(base_eps * peer_median_pe, 1), round(base_eps * peer_q75_pe, 1))
        if (base_eps and base_eps > 0 and peer_median_pe and peer_q25_pe and peer_q75_pe)
        else None
    )
    hist_price_band = (
        (round(base_eps * hist_pe_q25, 1), round(base_eps * hist_pe_med, 1), round(base_eps * hist_pe_q75, 1))
        if (base_eps and base_eps > 0 and hist_pe_q25 and hist_pe_med and hist_pe_q75)
        else None
    )

    # Auto Valuation Model Recommendation
    if is_fin:
        recommended_model = "股價淨值比（P/B）＋現金殖利率"
        model_reason = f"屬於「{industry}」，金融業無存貨與毛利結構且獲利受景氣／提存影響，建議以 P/B 與殖利率為主。"
    elif is_cyc:
        recommended_model = "股價淨值比（P/B）為主、預估本益比（P/E）為輔"
        model_reason = f"屬於景氣循環類股「{industry}」，景氣谷底時 P/E 容易暴衝失真，建議優先對照 P/B 位階與同業淨值比。"
    elif eps_yoy_growth_pct is not None and eps_yoy_growth_pct >= 15.0:
        recommended_model = "預估本益比（Forward P/E）＋本益成長比（PEG）"
        model_reason = f"預估全年 EPS 年增率達 {eps_yoy_growth_pct:+.1f}%（高成長股），單看 P/E 易誤判偏貴，建議搭配 PEG（小於 1 具吸引力）。"
    else:
        recommended_model = "預估本益比（Forward P/E）＋同業本益比對標"
        model_reason = f"屬於一般穩健產業「{industry}」，適合以預估全年 EPS 對照個股歷史本益比區間與同業中位數本益比。"

    return {
        "current_year": current_year,
        "reported_quarters": k,
        "cum_eps_now": round(cum_eps_now, 2) if cum_eps_now is not None else None,
        "prev_year_eps": round(prev_year_eps, 2) if prev_year_eps is not None else None,
        "ttm_eps": round(ttm_eps, 2) if ttm_eps is not None else None,
        "seasonal_eps": round(seasonal_eps, 2) if seasonal_eps is not None else None,
        "seasonal_weight": round(seasonal_weight_used * 100, 1) if seasonal_weight_used is not None else None,
        "nowcast_eps": round(nowcast_eps, 2) if nowcast_eps is not None else None,
        "nowcast_note": nowcast_note,
        "conservative_eps": round(conservative_eps, 2) if conservative_eps is not None else None,
        "base_eps": round(base_eps, 2) if base_eps is not None else None,
        "optimistic_eps": round(optimistic_eps, 2) if optimistic_eps is not None else None,
        "eps_yoy_growth_pct": round(eps_yoy_growth_pct, 1) if eps_yoy_growth_pct is not None else None,
        "latest_price": latest_price,
        "forward_pe": round(forward_pe, 2) if forward_pe is not None else None,
        "peg": round(peg, 2) if peg is not None else None,
        "current_pb": market_val.get("pb") if market_val else None,
        "current_yield": market_val.get("dividend_yield") if market_val else None,
        "peer_count": len(peer_pes),
        "peer_median_pe": round(peer_median_pe, 2) if peer_median_pe is not None else None,
        "peer_median_pb": round(peer_median_pb, 2) if peer_median_pb is not None else None,
        "peer_price_band": peer_price_band,
        "hist_pe_band": (
            (round(hist_pe_q25, 1), round(hist_pe_med, 1), round(hist_pe_q75, 1))
            if (hist_pe_q25 and hist_pe_med and hist_pe_q75)
            else None
        ),
        "hist_price_band": hist_price_band,
        "recommended_model": recommended_model,
        "model_reason": model_reason,
        "is_financial": is_fin,
        "is_cyclical": is_cyc,
    }


def analyze_stock_bundle(
    stock_meta: Dict[str, str],
    latest_price_info: Dict[str, Any],
    raw_monthly_rev: List[Dict[str, Any]],
    raw_financials: Dict[str, List[Dict[str, Any]]],
    peers: List[Dict[str, Any]],
    market_val: Optional[Dict[str, Optional[float]]],
    hist_pe_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Master analysis entrypoint shared by Windows Desktop and Mobile Streamlit."""
    is_fin = _is_financial_industry(stock_meta.get("industry_category", ""))

    rev_df = build_monthly_revenue_table(raw_monthly_rev, max_months=36)

    inc_piv, inc_names = _pivot_statement(raw_financials.get("income_statement", []), max_quarters=12)
    bal_piv, bal_names = _pivot_statement(raw_financials.get("balance_sheet", []), max_quarters=12)
    cf_piv, cf_names = _pivot_statement(raw_financials.get("cash_flow", []), max_quarters=12)

    inc_spec = FINANCIAL_INCOME_COMPACT_MAP if is_fin else INCOME_COMPACT_MAP
    inc_q_compact = _extract_compact_table(inc_piv, inc_names, inc_spec)
    inc_q_full = _build_full_table(inc_piv, inc_names)
    inc_y_compact = build_annual_from_quarterly(inc_q_compact)
    inc_y_full = build_annual_from_quarterly(inc_q_full)

    bal_q_compact = _extract_compact_table(bal_piv, bal_names, BALANCE_COMPACT_MAP)
    bal_q_full = _build_full_table(bal_piv, bal_names)

    cf_q_compact = _extract_compact_table(cf_piv, cf_names, CASHFLOW_COMPACT_MAP)
    cf_q_full = _build_full_table(cf_piv, cf_names)

    diagnostics = run_cross_statement_diagnostics(
        rev_df=rev_df,
        inc_q=inc_q_compact,
        bal_q=bal_q_compact,
        cf_q=cf_q_compact,
        is_financial=is_fin,
    )

    valuation = estimate_full_year_eps_and_valuation(
        stock_meta=stock_meta,
        latest_price=latest_price_info.get("close"),
        rev_df=rev_df,
        inc_q=inc_q_compact,
        bal_q=bal_q_compact,
        peers=peers,
        market_val=market_val,
        hist_pe_rows=hist_pe_rows,
    )

    return {
        "meta": stock_meta,
        "price": latest_price_info,
        "monthly_revenue": rev_df,
        "income_q_compact": inc_q_compact,
        "income_q_full": inc_q_full,
        "income_y_compact": inc_y_compact,
        "income_y_full": inc_y_full,
        "balance_q_compact": bal_q_compact,
        "balance_q_full": bal_q_full,
        "cashflow_q_compact": cf_q_compact,
        "cashflow_q_full": cf_q_full,
        "diagnostics": diagnostics,
        "valuation": valuation,
        "peers": peers[:15],
    }
