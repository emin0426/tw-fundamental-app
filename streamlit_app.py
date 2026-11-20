"""Mobile-Friendly Route-A Streamlit Web App for Taiwan Stock Fundamental Analyzer.

Deploy directly to Streamlit Community Cloud (free) so you can look up ANY Taiwan stock
from your phone with API keys safely stored in Streamlit Cloud Secrets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import altair as alt
import pandas as pd
import streamlit as st

from core.ai_analyst import DEFAULT_GEMINI_MODEL, GeminiStockAnalyst
from core.analyzer import analyze_stock_bundle
from core.cache import TieredCache
from core.data_fetcher import TaiwanStockDataFetcher

st.set_page_config(
    page_title="台股個股財報深度分析",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _load_secret_or_config(key: str, default: str = "") -> str:
    # 1. Streamlit Cloud Secrets
    try:
        if key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    # 2. Local config.json fallback
    cfg_path = Path(__file__).resolve().parent / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            lower_map = {k.lower(): v for k, v in cfg.items()}
            if key.lower() in lower_map:
                return str(lower_map[key.lower()]).strip()
        except Exception:
            pass
    return default


def _format_statement_df(df: pd.DataFrame, pct_cols: set | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    pct_cols = pct_cols or set()
    out = df.copy()
    cols = list(out.columns)
    for idx, c in enumerate(cols):
        if idx == 0:
            continue
        formatted = []
        for row_i, val in enumerate(out[c]):
            row_label = str(out.iloc[row_i, 0])
            is_eps = "EPS" in row_label or "每股" in row_label
            if pd.isna(val):
                formatted.append("-")
            elif c in pct_cols:
                formatted.append(f"{float(val):+.2f}%")
            elif is_eps:
                formatted.append(f"{float(val):,.2f}")
            else:
                formatted.append(f"{float(val):,.0f}")
        out[c] = formatted
    return out


# Sidebar Settings (optional override if not using Streamlit Cloud Secrets)
with st.sidebar:
    st.header("⚙ API 設定")
    finmind_token = st.text_input(
        "FinMind Token（選填）",
        value=_load_secret_or_config("FINMIND_TOKEN", ""),
        type="password",
    )
    gemini_api_key = st.text_input(
        "Gemini API Key（免費額度）",
        value=_load_secret_or_config("GEMINI_API_KEY", ""),
        type="password",
    )
    gemini_model = st.text_input(
        "Gemini 模型",
        value=_load_secret_or_config("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
    )
    force_refresh = st.checkbox("略過快取強制重新抓取", value=False)

st.title("📊 台股個股財報深度分析")

# Search Bar + Mobile Quick Picks
col_in, col_btn = st.columns([3, 1])
with col_in:
    default_q = st.session_state.get("stock_query", "2330")
    query = st.text_input(
        "輸入股票代號或名稱",
        value=default_q,
        placeholder="例如：2330 或 台積電",
        label_visibility="collapsed",
    )
with col_btn:
    run_search = st.button("🔍 查詢", use_container_width=True, type="primary")

q_cols = st.columns(4)
for idx, (code, label) in enumerate(
    [("2330", "2330 台積電"), ("2317", "2317 鴻海"), ("2454", "2454 聯發科"), ("2881", "2881 富邦金")]
):
    if q_cols[idx].button(label, use_container_width=True):
        st.session_state["stock_query"] = code
        query = code
        run_search = True

cache = TieredCache()
fetcher = TaiwanStockDataFetcher(finmind_token=finmind_token, cache=cache)

if run_search or ("bundle" not in st.session_state):
    st.session_state["stock_query"] = query
    with st.spinner(f"正在載入「{query}」的月營收、3 年財報與同業本益比..."):
        try:
            meta = fetcher.resolve_stock(query)
            if not meta:
                st.error(f"找不到符合「{query}」的上市櫃股票，請確認代號或名稱。")
                st.stop()
            sid = meta["stock_id"]
            price_info = fetcher.get_latest_price(sid, force_refresh=force_refresh)
            raw_rev = fetcher.get_monthly_revenue(sid, years=4, force_refresh=force_refresh)
            raw_fin = fetcher.get_financial_statements(sid, years=4, force_refresh=force_refresh)
            market_snap = fetcher.get_market_valuation_snapshot(force_refresh=force_refresh)
            peers = fetcher.get_peers(sid, meta["industry_category"])
            hist_pe = fetcher.get_historical_pe_series(sid, years=4, force_refresh=force_refresh)

            st.session_state["bundle"] = analyze_stock_bundle(
                stock_meta=meta,
                latest_price_info=price_info,
                raw_monthly_rev=raw_rev,
                raw_financials=raw_fin,
                peers=peers,
                market_val=market_snap.get(sid),
                hist_pe_rows=hist_pe,
            )
            st.session_state.pop("ai_report_text", None)
            st.session_state.pop("news_bundle", None)
        except Exception as exc:
            st.error(f"資料抓取失敗：{exc}")
            st.stop()

bundle: Dict[str, Any] = st.session_state["bundle"]
meta = bundle["meta"]
val = bundle["valuation"]
price = bundle["price"]

st.caption(
    f"**{meta['stock_id']} {meta['stock_name']}** ｜ 產業：{meta['industry_category']} ｜ 最新收盤：{price.get('close')} 元（{price.get('date')}）"
)

tabs = st.tabs(
    ["📊 總覽與估值", "📈 月營收", "📑 損益表", "🏦 資產負債表", "💰 現金流量表", "✨ AI 診斷與訂單"]
)

# --- Tab 1: Overview & Valuation ---
with tabs[0]:
    st.info(f"**系統自動推薦估值模型：{val['recommended_model']}**\n\n{val['model_reason']}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("最新股價", f"{val['latest_price']} 元" if val["latest_price"] else "-")
    m2.metric(
        f"{val['current_year']} 推估全年 EPS (基準)",
        f"{val['base_eps']} 元" if val["base_eps"] else "-",
        delta=f"{val['eps_yoy_growth_pct']:+.1f}% YoY" if val["eps_yoy_growth_pct"] is not None else None,
    )
    m3.metric(
        "預估本益比 (Forward P/E)",
        f"{val['forward_pe']}x" if val["forward_pe"] else "-",
        delta=f"同業中位 {val['peer_median_pe']}x" if val["peer_median_pe"] else None,
        delta_color="off",
    )
    m4.metric(
        "PEG / P/B",
        f"{val['peg'] or '-'} / {val['current_pb'] or '-'}x",
        delta=f"殖利率 {val['current_yield']}%" if val["current_yield"] else None,
        delta_color="off",
    )

    with st.expander("📐 查看全年 EPS 三模型推估明細與合理股價區間", expanded=True):
        st.markdown(
            f"- **全年 EPS 區間**：保守 `{val['conservative_eps']}` 元 ／ 基準 `{val['base_eps']}` 元 ／ 樂觀 `{val['optimistic_eps']}` 元"
            f"（已公布前 {val['reported_quarters']} 季累計 `{val['cum_eps_now']}` 元）\n"
            f"- **三模型對照**：歷史季節性權重法 `{val['seasonal_eps']}` 元（歷史佔比 {val['seasonal_weight']}%）｜月營收即時推估 `{val['nowcast_eps']}` 元｜近四季 TTM `{val['ttm_eps']}` 元\n"
            + (f"- **月營收即時備註**：{val['nowcast_note']}\n" if val.get("nowcast_note") else "")
            + (
                f"- **對標同業本益比合理價（{val['peer_count']} 家同業）**：`{val['peer_price_band'][0]}` ～ **`{val['peer_price_band'][1]}`（合理）** ～ `{val['peer_price_band'][2]}` 元\n"
                if val.get("peer_price_band")
                else ""
            )
            + (
                f"- **個股近 4 年歷史本益比合理價（{val['hist_pe_band']}x）**：`{val['hist_price_band'][0]}` ～ **`{val['hist_price_band'][1]}`（中位）** ～ `{val['hist_price_band'][2]}` 元"
                if val.get("hist_price_band")
                else ""
            )
        )

    st.subheader("第一層：Python 客觀跨報表領先指標診斷")
    badge_map = {"green": "🟢", "yellow": "🟡", "red": "🔴", "info": "🔵"}
    for d in bundle["diagnostics"]:
        icon = badge_map.get(d["status"], "⚪")
        st.markdown(f"**{icon} {d['category']}**　`{d['metric']}`  \n{d['explanation']}")

# --- Tab 2: Monthly Revenue ---
with tabs[1]:
    rev_df = bundle["monthly_revenue"]
    if rev_df.empty:
        st.warning("查無月營收資料。")
    else:
        chart_df = rev_df.head(24).iloc[::-1].copy()
        bar = (
            alt.Chart(chart_df)
            .mark_bar(color="#4A90E2")
            .encode(x=alt.X("月份:N", sort=None), y=alt.Y("營收(千元):Q", title="營收(千元)"), tooltip=["月份", "營收(千元)", "年增率(%)", "月增率(%)"])
        )
        st.altair_chart(bar.properties(height=260), use_container_width=True)
        disp_cols = ["月份", "營收(千元)", "月增率(%)", "年增率(%)", "累計營收(千元)", "累計年增率(%)"]
        st.dataframe(
            _format_statement_df(rev_df[disp_cols], pct_cols={"月增率(%)", "年增率(%)", "累計年增率(%)"}),
            use_container_width=True,
            hide_index=True,
        )

# --- Tab 3: Income Statement ---
with tabs[2]:
    period = st.radio("報表期間", ["季報（近 3 年）", "年報彙整"], horizontal=True)
    if period.startswith("季報"):
        st.dataframe(_format_statement_df(bundle["income_q_compact"]), use_container_width=True, hide_index=True)
        with st.expander("🔍 展開完整損益表科目"):
            st.dataframe(_format_statement_df(bundle["income_q_full"]), use_container_width=True, hide_index=True)
    else:
        st.dataframe(_format_statement_df(bundle["income_y_compact"]), use_container_width=True, hide_index=True)
        with st.expander("🔍 展開完整年報損益表科目"):
            st.dataframe(_format_statement_df(bundle["income_y_full"]), use_container_width=True, hide_index=True)

# --- Tab 4: Balance Sheet ---
with tabs[3]:
    st.dataframe(_format_statement_df(bundle["balance_q_compact"]), use_container_width=True, hide_index=True)
    with st.expander("🔍 展開完整資產負債表科目"):
        st.dataframe(_format_statement_df(bundle["balance_q_full"]), use_container_width=True, hide_index=True)

# --- Tab 5: Cash Flow ---
with tabs[4]:
    st.dataframe(_format_statement_df(bundle["cashflow_q_compact"]), use_container_width=True, hide_index=True)
    with st.expander("🔍 展開完整現金流量表科目"):
        st.dataframe(_format_statement_df(bundle["cashflow_q_full"]), use_container_width=True, hide_index=True)

# --- Tab 6: AI Layer-2 & Order Visibility ---
with tabs[5]:
    st.caption("按下按鈕才會呼叫 Gemini 免費額度，平常滑財報完全不消耗額度。")
    pdf_file = st.file_uploader("選填：上傳最新法說會簡報 PDF（強化訂單能見度分析）", type=["pdf"])
    pdf_text = ""
    if pdf_file is not None:
        pdf_text = TaiwanStockDataFetcher.extract_pdf_text(pdf_file.read())
        if pdf_text:
            st.success(f"已擷取法說會 PDF 文字（{len(pdf_text)} 字）")
        else:
            st.warning("無法從此 PDF 擷取文字（可能為純圖片掃描檔），將改以重訊與新聞分析。")

    if st.button("✨ 啟動第二層 AI 跨表解讀與訂單能見度分析", type="primary", use_container_width=True):
        with st.spinner("正在抓取最新重訊、個股新聞並呼叫 Gemini AI 撰寫深度解讀..."):
            news = fetcher.get_news_and_announcements(meta["stock_id"], meta["stock_name"], force_refresh=force_refresh)
            st.session_state["news_bundle"] = news
            analyst = GeminiStockAnalyst(api_key=gemini_api_key, model=gemini_model, cache=cache)
            st.session_state["ai_report_text"] = analyst.generate_report(
                bundle, news, extra_conference_text=pdf_text, force_refresh=force_refresh
            )

    if "ai_report_text" in st.session_state:
        st.markdown(st.session_state["ai_report_text"])

    news_bundle = st.session_state.get("news_bundle")
    if news_bundle:
        with st.expander("📰 查看近期法說／訂單相關新聞與官方連結"):
            for lbl, url in news_bundle.get("official_links", {}).items():
                st.markdown(f"- 🔗 [{lbl}]({url})")
            st.markdown("**近期新聞**")
            for n in news_bundle.get("news", [])[:10]:
                st.markdown(f"- [{n['title']}]({n['link']})（{n['source']}）")
    else:
        sid = meta["stock_id"]
        st.markdown(
            f"**官方免費查證連結**："
            f"[產業價值鏈資訊平台](https://ic.tpex.org.tw/company_chain.php?stk_code={sid}) ｜ "
            f"[IR 法說會查詢](https://irengage.taiwanindex.com.tw/conferenceList) ｜ "
            f"[TWSE WebPro 法說影音](https://webpro.twse.com.tw/WebPortal/vod/101/?categoryId=148)"
        )
