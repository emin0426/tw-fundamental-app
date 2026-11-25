"""TDCC weekly shareholding dispersion analyzer (v1.8: dynamic price/capital thresholds, mid-tier 100% structure, avg lots per holder)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .data_fetcher import TDCC_LEVEL_LABELS

TDCC_LEVELS = [1, 5, 10, 15, 20, 30, 40, 50, 100, 200, 400, 600, 800, 1000]
MANUAL_WHALE_OPTIONS = ["自動判定", "100 張", "200 張", "400 張", "600 張", "1000 張"]

# Maps lot threshold -> upper TDCC level_code (1..15) for retail (<= threshold).
# Whale (> threshold) strictly starts at (boundary_code + 1), guaranteeing:
# Retail (<= retail_code) + Mid-tier (retail_code < code < whale_min_code) + Whale (>= whale_min_code) = 100%
LOT_TO_UPPER_LEVEL_CODE: Dict[int, int] = {
    1: 1,     # <= 1張 (1-999股)
    5: 2,     # <= 5張 (1-5,000股)
    10: 3,    # <= 10張
    15: 4,    # <= 15張
    20: 5,    # <= 20張
    30: 6,    # <= 30張
    40: 7,    # <= 40張
    50: 8,    # <= 50張
    100: 9,   # <= 100張
    200: 10,  # <= 200張
    400: 11,  # <= 400張
    600: 12,  # <= 600張
    800: 13,  # <= 800張
    1000: 14, # <= 1000張  (so >1000張 is level_code >= 15)
}


def snap_to_level(target_lots: float, allowed_levels: List[int]) -> int:
    return min(allowed_levels, key=lambda x: abs(x - target_lots))


def get_dynamic_thresholds(price: float, total_lots: float) -> Tuple[int, int]:
    """Compute dynamic (whale_level, retail_level) based on latest stock price and total issued lots."""
    safe_price = float(price) if (price and float(price) > 0) else 100.0
    safe_lots = float(total_lots) if (total_lots and float(total_lots) > 0) else 200_000.0

    # 1. 散戶門檻：以 150 萬元資金上限反推張數
    retail_target = 1500.0 / safe_price
    retail_level = snap_to_level(retail_target, [1, 5, 10, 20, 30, 50])

    # 2. 大戶門檻：依股本（總發行張數）定基準
    if safe_lots < 100_000:
        whale_level = 200
    elif safe_lots < 500_000:
        whale_level = 400
    elif safe_lots < 1_500_000:
        whale_level = 600
    else:
        whale_level = 1000

    # 3. 非權值股 (< 100億，即 < 100萬張) 金額防呆修正
    if safe_lots < 1_000_000:
        whale_value_m = (whale_level * safe_price) / 1000.0  # 百萬元
        if whale_value_m < 15.0:
            target_lots = 20_000.0 / safe_price
            whale_level = snap_to_level(target_lots, [200, 400, 600, 800, 1000])
        elif whale_value_m > 150.0:
            target_lots = 50_000.0 / safe_price
            whale_level = snap_to_level(target_lots, [50, 100, 200, 400])

    # 4. 最後防呆確保大戶門檻嚴格大於散戶門檻
    if whale_level <= retail_level:
        whale_level = next((lv for lv in TDCC_LEVELS if lv > retail_level), 1000)

    return whale_level, retail_level


def parse_manual_whale_option(option_label: str) -> Optional[int]:
    """Convert UI dropdown string like '400 張' -> 400, or '自動判定' -> None."""
    s = (option_label or "").strip()
    if not s or "自動" in s:
        return None
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        val = int(digits)
        if val in LOT_TO_UPPER_LEVEL_CODE:
            return val
    return None


def analyze_tdcc_dispersion(
    tdcc_raw: Dict[str, Any],
    price: Optional[float] = None,
    total_lots_override: Optional[float] = None,
    manual_whale_level: Optional[int] = None,
) -> Dict[str, Any]:
    """Process weekly TDCC rows with dynamic or manual thresholds, 3-tier 100% structure, and avg lots/holder."""
    rows: List[Dict[str, Any]] = tdcc_raw.get("rows", []) if tdcc_raw else []
    source: str = tdcc_raw.get("source", "") if tdcc_raw else ""

    if not rows:
        return {
            "has_data": False,
            "source": source,
            "latest_date": None,
            "latest_table": pd.DataFrame(),
            "trend_df": pd.DataFrame(),
            "summary": {},
        }

    df = pd.DataFrame(rows)
    df["level_code"] = pd.to_numeric(df["level_code"], errors="coerce").astype("Int64")
    df["people"] = pd.to_numeric(df["people"], errors="coerce").fillna(0).astype(int)
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0).astype(int)
    df["percent"] = pd.to_numeric(df["percent"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["date", "level_code"]).copy()
    df = df.drop_duplicates(subset=["date", "level_code"], keep="last")

    dates_sorted = sorted(df["date"].unique())
    latest_date = dates_sorted[-1]
    latest_sub = df[df["date"] == latest_date].sort_values("level_code").copy()

    # Determine total issued lots from latest TDCC total shares (falls back to override if needed)
    latest_total_shares = int(latest_sub["shares"].sum())
    if latest_total_shares > 0:
        total_lots = latest_total_shares / 1000.0
    elif total_lots_override and total_lots_override > 0:
        total_lots = float(total_lots_override)
    else:
        total_lots = 200_000.0

    safe_price = float(price) if (price and float(price) > 0) else 100.0
    auto_whale_level, retail_level = get_dynamic_thresholds(safe_price, total_lots)

    is_manual = manual_whale_level is not None and manual_whale_level in LOT_TO_UPPER_LEVEL_CODE
    whale_level = int(manual_whale_level) if is_manual else auto_whale_level
    if whale_level <= retail_level:
        whale_level = next((lv for lv in TDCC_LEVELS if lv > retail_level), 1000)

    retail_max_code = LOT_TO_UPPER_LEVEL_CODE[retail_level]
    whale_min_code = LOT_TO_UPPER_LEVEL_CODE[whale_level] + 1

    # 1. Latest week 15-tier distribution table (kept internally)
    latest_records = []
    for r in latest_sub.itertuples():
        lvl = int(r.level_code)
        latest_records.append(
            {
                "級距": lvl,
                "持股分級": TDCC_LEVEL_LABELS.get(lvl, str(r.level_label)),
                "人數": int(r.people),
                "股數": int(r.shares),
                "佔集保庫存比例(%)": round(float(r.percent), 2),
            }
        )
    latest_table = pd.DataFrame(latest_records)

    # 2. Weekly historical trend DataFrame (chronological oldest -> newest)
    trend_records = []
    for d in dates_sorted:
        sub = df[df["date"] == d]
        total_people = int(sub["people"].sum())
        total_shares = int(sub["shares"].sum())
        total_pct_sum = float(sub["percent"].sum())

        whale_df = sub[sub["level_code"] >= whale_min_code]
        retail_df = sub[sub["level_code"] <= retail_max_code]
        mid_df = sub[(sub["level_code"] > retail_max_code) & (sub["level_code"] < whale_min_code)]

        raw_whale_pct = float(whale_df["percent"].sum())
        raw_retail_pct = float(retail_df["percent"].sum())

        # Normalize to exact 100.00% so Whale + Mid + Retail == 100.00% cleanly
        if total_pct_sum > 0:
            whale_pct = round(raw_whale_pct / total_pct_sum * 100.0, 2)
            retail_pct = round(raw_retail_pct / total_pct_sum * 100.0, 2)
            mid_pct = round(max(0.0, 100.0 - whale_pct - retail_pct), 2)
        else:
            whale_pct, mid_pct, retail_pct = 0.0, 0.0, 0.0

        whale_people = int(whale_df["people"].sum())
        mid_people = int(mid_df["people"].sum())
        retail_people = int(retail_df["people"].sum())
        avg_lots = round((total_shares / 1000.0) / total_people, 2) if total_people > 0 else 0.0

        trend_records.append(
            {
                "日期": d,
                "總股東人數": total_people,
                "平均每戶持股(張)": avg_lots,
                "大戶比例(%)": whale_pct,
                "中實戶比例(%)": mid_pct,
                "散戶比例(%)": retail_pct,
                "大戶人數": whale_people,
                "中實戶人數": mid_people,
                "散戶人數": retail_people,
            }
        )
    trend_df = pd.DataFrame(trend_records)

    # Week-over-week deltas
    trend_df["大戶週增減(%)"] = trend_df["大戶比例(%)"].diff().fillna(0.0).round(2)
    trend_df["中實戶週增減(%)"] = trend_df["中實戶比例(%)"].diff().fillna(0.0).round(2)
    trend_df["散戶週增減(%)"] = trend_df["散戶比例(%)"].diff().fillna(0.0).round(2)
    trend_df["大戶人數週增減"] = trend_df["大戶人數"].diff().fillna(0).astype(int)
    trend_df["散戶人數週增減"] = trend_df["散戶人數"].diff().fillna(0).astype(int)
    trend_df["總股東人數週增減"] = trend_df["總股東人數"].diff().fillna(0).astype(int)
    trend_df["平均每戶週增減(張)"] = trend_df["平均每戶持股(張)"].diff().fillna(0.0).round(2)

    # 3. Chip flow summary & anti-false-signal verification
    cur = trend_df.iloc[-1]
    prev_1w = trend_df.iloc[-2] if len(trend_df) >= 2 else None
    ref_idx = max(0, len(trend_df) - 5) if len(trend_df) >= 2 else 0
    prev_4w = trend_df.iloc[ref_idx] if len(trend_df) >= 2 else None
    lookback_weeks = (len(trend_df) - 1 - ref_idx) if len(trend_df) >= 2 else 0

    d_people_1w = int(cur["總股東人數"] - prev_1w["總股東人數"]) if prev_1w is not None else 0
    d_avg_lots_1w = round(float(cur["平均每戶持股(張)"] - prev_1w["平均每戶持股(張)"]), 2) if prev_1w is not None else 0.0
    d_whale_1w = round(float(cur["大戶比例(%)"] - prev_1w["大戶比例(%)"]), 2) if prev_1w is not None else 0.0
    d_mid_1w = round(float(cur["中實戶比例(%)"] - prev_1w["中實戶比例(%)"]), 2) if prev_1w is not None else 0.0
    d_retail_1w = round(float(cur["散戶比例(%)"] - prev_1w["散戶比例(%)"]), 2) if prev_1w is not None else 0.0
    d_whale_ppl_1w = int(cur["大戶人數"] - prev_1w["大戶人數"]) if prev_1w is not None else 0
    d_retail_ppl_1w = int(cur["散戶人數"] - prev_1w["散戶人數"]) if prev_1w is not None else 0

    d_people_4w = int(cur["總股東人數"] - prev_4w["總股東人數"]) if prev_4w is not None else 0
    d_avg_lots_4w = round(float(cur["平均每戶持股(張)"] - prev_4w["平均每戶持股(張)"]), 2) if prev_4w is not None else 0.0
    d_whale_4w = round(float(cur["大戶比例(%)"] - prev_4w["大戶比例(%)"]), 2) if prev_4w is not None else 0.0
    d_retail_4w = round(float(cur["散戶比例(%)"] - prev_4w["散戶比例(%)"]), 2) if prev_4w is not None else 0.0

    # Verify with avg_lots_per_holder & total_people to filter cross-tier boundary noise
    if lookback_weeks == 0:
        signal_status = "info"
        signal_title = "單週最新快照"
        signal_desc = "目前僅取得最新單週集保資料，後續累積歷史週期即可觀察籌碼趨勢。"
    elif d_whale_4w >= 0.4 and (d_retail_4w <= -0.1 or d_avg_lots_4w > 0):
        signal_status = "green"
        signal_title = "籌碼集中（大戶吸籌、平均持股上升）"
        signal_desc = (
            f"近 {lookback_weeks} 週大戶（>{whale_level}張）增加 {d_whale_4w:+.2f}%、散戶（≤{retail_level}張）變動 {d_retail_4w:+.2f}%；"
            f"驗證指標：總股東人數 {d_people_4w:+,} 人、平均每戶持股 {d_avg_lots_4w:+.2f} 張，確認籌碼實質集中。"
        )
    elif d_whale_4w <= -0.4 and (d_retail_4w >= 0.1 or d_avg_lots_4w < 0):
        signal_status = "red"
        signal_title = "籌碼發散（大戶調節、散戶接手）"
        signal_desc = (
            f"近 {lookback_weeks} 週大戶（>{whale_level}張）減少 {d_whale_4w:+.2f}%、散戶（≤{retail_level}張）變動 {d_retail_4w:+.2f}%；"
            f"驗證指標：總股東人數 {d_people_4w:+,} 人、平均每戶持股 {d_avg_lots_4w:+.2f} 張，留意籌碼轉趨凌亂。"
        )
    else:
        signal_status = "yellow"
        signal_title = "籌碼中性整理"
        signal_desc = (
            f"近 {lookback_weeks} 週大戶（>{whale_level}張）變動 {d_whale_4w:+.2f}%、散戶（≤{retail_level}張）變動 {d_retail_4w:+.2f}%；"
            f"總股東人數 {d_people_4w:+,} 人、平均每戶持股 {d_avg_lots_4w:+.2f} 張，尚無明顯單邊趨勢。"
        )

    whale_threshold_val_m = round((whale_level * safe_price) / 1000.0, 1)
    mode_tag = f"手動覆寫 {whale_level}張" if is_manual else "自動判定"

    summary = {
        "latest_date": latest_date,
        "weeks_count": len(trend_df),
        "mode_tag": mode_tag,
        "is_manual": is_manual,
        "price_used": round(safe_price, 2),
        "total_lots_wan": round(total_lots / 10_000.0, 1),
        "whale_level": whale_level,
        "auto_whale_level": auto_whale_level,
        "retail_level": retail_level,
        "whale_value_m": whale_threshold_val_m,
        "whale_label": f"大戶 (>{whale_level}張)",
        "mid_label": f"中實戶 ({retail_level}~{whale_level}張)",
        "retail_label": f"散戶 (≤{retail_level}張)",
        "total_people": int(cur["總股東人數"]),
        "avg_lots": float(cur["平均每戶持股(張)"]),
        "whale_pct": float(cur["大戶比例(%)"]),
        "mid_pct": float(cur["中實戶比例(%)"]),
        "retail_pct": float(cur["散戶比例(%)"]),
        "whale_people": int(cur["大戶人數"]),
        "mid_people": int(cur["中實戶人數"]),
        "retail_people": int(cur["散戶人數"]),
        "d_people_1w": d_people_1w,
        "d_avg_lots_1w": d_avg_lots_1w,
        "d_whale_1w": d_whale_1w,
        "d_mid_1w": d_mid_1w,
        "d_retail_1w": d_retail_1w,
        "d_whale_ppl_1w": d_whale_ppl_1w,
        "d_retail_ppl_1w": d_retail_ppl_1w,
        "lookback_weeks": lookback_weeks,
        "d_people_4w": d_people_4w,
        "d_avg_lots_4w": d_avg_lots_4w,
        "d_whale_4w": d_whale_4w,
        "d_retail_4w": d_retail_4w,
        "signal_status": signal_status,
        "signal_title": signal_title,
        "signal_desc": signal_desc,
    }

    return {
        "has_data": True,
        "source": source,
        "latest_date": latest_date,
        "latest_table": latest_table,
        "trend_df": trend_df,
        "summary": summary,
    }
