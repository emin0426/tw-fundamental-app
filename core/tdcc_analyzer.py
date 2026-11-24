"""TDCC weekly shareholding dispersion analyzer (major vs. retail % & headcount trends, weekly bars, chip flow)."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from .data_fetcher import TDCC_LEVEL_LABELS


def analyze_tdcc_dispersion(tdcc_raw: Dict[str, Any]) -> Dict[str, Any]:
    """Process weekly TDCC rows into:
    - trend_df: weekly time-series with major/retail percentages, weekly % deltas, and major/retail headcounts
    - latest_table: kept in data bundle for optional reference
    - summary: latest KPIs, 1-week & 4-week deltas, and automated chip flow signal (green/yellow/red)
    """
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

    # 1. Latest week 15-tier distribution table (kept internally)
    latest_sub = df[df["date"] == latest_date].sort_values("level_code").copy()
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

        m400 = sub[sub["level_code"] >= 12]   # 400張以上 (level 12-15)
        m1000 = sub[sub["level_code"] == 15]  # 1000張以上 (level 15)
        r10 = sub[sub["level_code"] <= 3]     # 10張以下散戶 (level 1-3: 1~10,000股)
        r50 = sub[sub["level_code"] <= 8]     # 50張以下 (level 1-8)

        trend_records.append(
            {
                "日期": d,
                "總股東人數": total_people,
                "400張以上大戶(%)": round(float(m400["percent"].sum()), 2),
                "400張以上大戶人數": int(m400["people"].sum()),
                "1000張以上大戶(%)": round(float(m1000["percent"].sum()), 2),
                "1000張以上大戶人數": int(m1000["people"].sum()),
                "散戶_10張以下(%)": round(float(r10["percent"].sum()), 2),
                "散戶_10張以下人數": int(r10["people"].sum()),
                "散戶_50張以下(%)": round(float(r50["percent"].sum()), 2),
            }
        )
    trend_df = pd.DataFrame(trend_records)

    # Week-over-week deltas for bar charts & tooltips
    trend_df["400張大戶週增減(%)"] = trend_df["400張以上大戶(%)"].diff().fillna(0.0).round(2)
    trend_df["1000張大戶週增減(%)"] = trend_df["1000張以上大戶(%)"].diff().fillna(0.0).round(2)
    trend_df["散戶週增減(%)"] = trend_df["散戶_10張以下(%)"].diff().fillna(0.0).round(2)
    trend_df["400張大戶人數週增減"] = trend_df["400張以上大戶人數"].diff().fillna(0).astype(int)
    trend_df["散戶人數週增減"] = trend_df["散戶_10張以下人數"].diff().fillna(0).astype(int)
    trend_df["總股東人數週增減"] = trend_df["總股東人數"].diff().fillna(0).astype(int)

    # 3. Chip flow summary & delta calculations
    cur = trend_df.iloc[-1]
    prev_1w = trend_df.iloc[-2] if len(trend_df) >= 2 else None
    ref_idx = max(0, len(trend_df) - 5) if len(trend_df) >= 2 else 0
    prev_4w = trend_df.iloc[ref_idx] if len(trend_df) >= 2 else None
    lookback_weeks = (len(trend_df) - 1 - ref_idx) if len(trend_df) >= 2 else 0

    d_people_1w = int(cur["總股東人數"] - prev_1w["總股東人數"]) if prev_1w is not None else 0
    d_m400_ppl_1w = int(cur["400張以上大戶人數"] - prev_1w["400張以上大戶人數"]) if prev_1w is not None else 0
    d_ret10_ppl_1w = int(cur["散戶_10張以下人數"] - prev_1w["散戶_10張以下人數"]) if prev_1w is not None else 0
    d_400_1w = round(float(cur["400張以上大戶(%)"] - prev_1w["400張以上大戶(%)"]), 2) if prev_1w is not None else 0.0
    d_1000_1w = round(float(cur["1000張以上大戶(%)"] - prev_1w["1000張以上大戶(%)"]), 2) if prev_1w is not None else 0.0
    d_ret10_1w = round(float(cur["散戶_10張以下(%)"] - prev_1w["散戶_10張以下(%)"]), 2) if prev_1w is not None else 0.0

    d_people_4w = int(cur["總股東人數"] - prev_4w["總股東人數"]) if prev_4w is not None else 0
    d_400_4w = round(float(cur["400張以上大戶(%)"] - prev_4w["400張以上大戶(%)"]), 2) if prev_4w is not None else 0.0
    d_1000_4w = round(float(cur["1000張以上大戶(%)"] - prev_4w["1000張以上大戶(%)"]), 2) if prev_4w is not None else 0.0
    d_ret10_4w = round(float(cur["散戶_10張以下(%)"] - prev_4w["散戶_10張以下(%)"]), 2) if prev_4w is not None else 0.0

    if lookback_weeks == 0:
        signal_status = "info"
        signal_title = "單週最新快照"
        signal_desc = "目前僅取得最新單週集保資料，後續隨每週累積或重抓歷史週期即可觀察籌碼趨勢。"
    elif d_400_4w >= 0.5 and d_ret10_4w <= -0.1:
        signal_status = "green"
        signal_title = "籌碼集中（大戶吸籌、散戶退場）"
        signal_desc = (
            f"近 {lookback_weeks} 週 400 張以上大戶持股增加 {d_400_4w:+.2f}%（千張大戶 {d_1000_4w:+.2f}%），"
            f"同時 10 張以下散戶比例減少 {d_ret10_4w:+.2f}%、總股東人數變動 {d_people_4w:+,} 人，籌碼明顯流向大戶手中。"
        )
    elif d_400_4w <= -0.5 and d_ret10_4w >= 0.1:
        signal_status = "red"
        signal_title = "籌碼發散（大戶調節、散戶接手）"
        signal_desc = (
            f"近 {lookback_weeks} 週 400 張以上大戶持股減少 {d_400_4w:+.2f}%（千張大戶 {d_1000_4w:+.2f}%），"
            f"同時 10 張以下散戶比例上升 {d_ret10_4w:+.2f}%、總股東人數變動 {d_people_4w:+,} 人，需留意籌碼趨於凌亂。"
        )
    else:
        signal_status = "yellow"
        signal_title = "籌碼中性整理"
        signal_desc = (
            f"近 {lookback_weeks} 週 400 張以上大戶變動 {d_400_4w:+.2f}%、千張大戶變動 {d_1000_4w:+.2f}%、"
            f"10 張以下散戶變動 {d_ret10_4w:+.2f}%，大戶與散戶持股比例尚無劇烈單邊移動。"
        )

    summary = {
        "latest_date": latest_date,
        "weeks_count": len(trend_df),
        "total_people": int(cur["總股東人數"]),
        "major_400_people": int(cur["400張以上大戶人數"]),
        "major_1000_people": int(cur["1000張以上大戶人數"]),
        "retail_10_people": int(cur["散戶_10張以下人數"]),
        "major_400_pct": float(cur["400張以上大戶(%)"]),
        "major_1000_pct": float(cur["1000張以上大戶(%)"]),
        "retail_10_pct": float(cur["散戶_10張以下(%)"]),
        "retail_50_pct": float(cur["散戶_50張以下(%)"]),
        "d_people_1w": d_people_1w,
        "d_m400_ppl_1w": d_m400_ppl_1w,
        "d_ret10_ppl_1w": d_ret10_ppl_1w,
        "d_400_1w": d_400_1w,
        "d_1000_1w": d_1000_1w,
        "d_ret10_1w": d_ret10_1w,
        "lookback_weeks": lookback_weeks,
        "d_people_4w": d_people_4w,
        "d_400_4w": d_400_4w,
        "d_1000_4w": d_1000_4w,
        "d_ret10_4w": d_ret10_4w,
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
