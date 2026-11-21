"""Layer-2 AI Cross-Statement Interpretation & Order Visibility Summarizer using Gemini Free Tier REST API."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import requests

from .cache import TieredCache

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
RECOMMENDED_MODELS = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
GEMINI_MODELS_LIST_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_ENDPOINT_TMPL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


def build_analysis_prompt(
    analysis_bundle: Dict[str, Any],
    news_bundle: Dict[str, Any],
    extra_conference_text: str = "",
) -> str:
    """Build a grounded prompt feeding Python-verified numbers + news/MOPS/conference text."""
    meta = analysis_bundle.get("meta", {})
    val = analysis_bundle.get("valuation", {})
    diags = analysis_bundle.get("diagnostics", [])

    diag_lines = "\n".join(
        f"- [{d['status'].upper()}] {d['category']}（{d['metric']}）：{d['explanation']}"
        for d in diags
    ) or "- 尚無足夠多期資料產生自動燈號。"

    news_lines = "\n".join(
        f"- [{n.get('pub_date', '')[:16]}] {n.get('title', '')} ({n.get('source', '')})"
        for n in news_bundle.get("news", [])[:12]
    ) or "- 近期查無相關新聞。"

    ann_lines = "\n".join(
        f"- [{a.get('date', '')}] {a.get('subject', '')}：{a.get('detail', '')}"
        for a in news_bundle.get("announcements", [])[:6]
    ) or "- 近期無重大訊息紀錄。"

    conf_section = ""
    if extra_conference_text.strip():
        conf_section = f"\n【法說會簡報擷取文字】\n{extra_conference_text.strip()[:6000]}\n"

    return f"""你是專業的台灣股市基本面研究員，請以「繁體中文（台灣慣用語）」撰寫精準、客觀、簡潔有力的個股深度解析。
嚴格遵守：所有數字必須根據下方已驗證的程式計算結果與新聞／重訊事實，不可自行編造財報數字或未出現的訂單細節；若某欄位缺乏公開資訊，請誠實註明「目前公開新聞與重訊未明確揭露」。

【標的資訊】
- 公司：{meta.get('stock_id')} {meta.get('stock_name')}（產業：{meta.get('industry_category')}）
- 最新股價：{val.get('latest_price')} 元
- 全年 EPS 推估：保守 {val.get('conservative_eps')} / 基準 {val.get('base_eps')} / 樂觀 {val.get('optimistic_eps')} 元（預估年增率：{val.get('eps_yoy_growth_pct')}%）
- 系統推薦估值模型：{val.get('recommended_model')}（預估本益比 {val.get('forward_pe')}x、同業中位數本益比 {val.get('peer_median_pe')}x、PEG {val.get('peg')}、股價淨值比 {val.get('current_pb')}x）

【第一層 Python 跨報表診斷燈號】
{diag_lines}

【近期重大訊息】
{ann_lines}

【近期法說與訂單相關新聞標題】
{news_lines}
{conf_section}

請嚴格按照以下兩個大區塊、使用條列輸出（精簡扼要，避免冗長）：

### 一、AI 跨報表連動深度解讀（承接第一層燈號）
- 針對上述亮黃燈／紅燈（或顯著綠燈）的項目，具體說明「資產負債表（如存貨、合約負債）與現金流量表（如資本支出、營業現金流）」的變化，將如何連動影響未來 1～2 季「損益表」的營收、毛利率與獲利品質。
- 總結目前估值處於偏低、合理或偏高區間，以及最該盯緊的 1～2 個下季財報驗證指標。

### 二、營運前景與訂單能見度四欄位摘要
1. **訂單能見度／接單狀況**：
2. **產能利用率與擴產計畫**：
3. **下季毛利與營收展望**：
4. **潛在風險與觀察重點**：
"""


class GeminiStockAnalyst:
    """On-demand Gemini API client with 503 auto-retry and multi-model fallback (Flash -> Flash-Lite -> Gemma)."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GEMINI_MODEL,
        cache: Optional[TieredCache] = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.model = (model or DEFAULT_GEMINI_MODEL).strip()
        self.cache = cache or TieredCache()

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

    def _call_model(
        self,
        model_name: str,
        prompt: str,
        use_thinking_budget: bool = True,
        retries: int = 2,
    ) -> requests.Response:
        clean_name = model_name.replace("models/", "").strip()
        url = GEMINI_ENDPOINT_TMPL.format(model=clean_name)
        gen_cfg: Dict[str, Any] = {
            "temperature": 0.25,
            "maxOutputTokens": 1800,
        }
        # Gemma models do not use thinkingConfig
        if use_thinking_budget and "gemma" not in clean_name.lower():
            gen_cfg["thinkingConfig"] = {"thinkingBudget": 512}

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": gen_cfg,
        }

        resp: Optional[requests.Response] = None
        for attempt in range(retries):
            resp = requests.post(
                url,
                headers=self._headers(),
                data=json.dumps(payload),
                timeout=(15, 90),
            )
            if resp.status_code == 400 and use_thinking_budget and "thinking" in resp.text.lower():
                return self._call_model(model_name, prompt, use_thinking_budget=False, retries=retries)
            if resp.status_code in (500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2.0)
                continue
            return resp
        assert resp is not None
        return resp

    def _discover_fallback_models(self) -> List[str]:
        """Discover all active free-tier text models (Flash, Flash-Lite, Gemma 4/3) to bypass 503 spikes."""
        try:
            resp = requests.get(GEMINI_MODELS_LIST_URL, headers=self._headers(), timeout=(10, 20))
            if not resp.ok:
                return []
            models = resp.json().get("models", [])
            skip_keywords = ("image", "audio", "tts", "embed", "aqa", "veo", "imagen", "robotics", "computer-use")
            flash_lite: List[str] = []
            flash_other: List[str] = []
            gemma_models: List[str] = []

            for m in models:
                methods = m.get("supportedGenerationMethods", [])
                name = str(m.get("name", "")).replace("models/", "")
                low = name.lower()
                if "generateContent" not in methods or any(k in low for k in skip_keywords):
                    continue
                if "gemini" in low and "flash-lite" in low:
                    flash_lite.append(name)
                elif "gemini" in low and "flash" in low:
                    flash_other.append(name)
                elif "gemma" in low and ("27b" in low or "31b" in low or "26b" in low or "12b" in low):
                    gemma_models.append(name)

            # Prioritize Flash-Lite (least congested) -> other Flash -> Gemma large models
            ordered = (
                sorted(flash_lite, reverse=True)
                + sorted(flash_other, reverse=True)
                + sorted(gemma_models, reverse=True)
            )
            return ordered[:6]
        except Exception:
            return []

    def generate_report(
        self,
        analysis_bundle: Dict[str, Any],
        news_bundle: Dict[str, Any],
        extra_conference_text: str = "",
        force_refresh: bool = False,
    ) -> str:
        stock_id = analysis_bundle.get("meta", {}).get("stock_id", "unknown")
        has_pdf = "pdf" if extra_conference_text.strip() else "nopdf"
        cache_key = f"ai_report:{stock_id}:{self.model}:{has_pdf}"

        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                return str(cached)

        if not self.api_key:
            return (
                "⚠️ 尚未設定 Gemini API Key，因此僅顯示下方抓取的重訊與新聞。\n"
                "請至設定填入免費的 Gemini API Key（Google AI Studio 申請），即可啟用第二層 AI 跨表解讀與訂單能見度摘要。"
            )

        prompt = build_analysis_prompt(analysis_bundle, news_bundle, extra_conference_text)
        active_model = self.model
        fallback_note = ""

        try:
            resp = self._call_model(active_model, prompt, retries=2)
        except requests.exceptions.Timeout:
            resp = None

        # Auto-fallback on timeout, 400/404 (retired model), 429 (quota), or 503 (high demand spike)
        if resp is None or resp.status_code in (400, 404, 429, 503):
            orig_code = resp.status_code if resp is not None else "逾時"
            tried = {active_model}
            retry_models = [DEFAULT_GEMINI_MODEL] + self._discover_fallback_models()

            for cand in retry_models:
                if cand in tried:
                    continue
                tried.add(cand)
                try:
                    resp_retry = self._call_model(cand, prompt, retries=1)
                except requests.exceptions.Timeout:
                    continue
                if resp_retry.ok:
                    reason = "目前尖峰塞車" if orig_code == 503 else f"暫時不可用（{orig_code}）"
                    fallback_note = (
                        f"ℹ️ 提示：主力模型 `{active_model}` {reason}，已自動切換為備援免費模型 **`{cand}`** 完成分析。\n\n"
                    )
                    resp = resp_retry
                    active_model = cand
                    break

        if resp is None:
            return "⚠️ Gemini 伺服器回應逾時，可能是免費尖峰時段較忙碌，請稍候再點一次即可。"
        if resp.status_code == 429:
            return "⚠️ 已觸及 Gemini API 免費方案每分鐘／每日請求上限，請稍候 1 分鐘後再試。"
        if resp.status_code == 503:
            return "⚠️ Google 免費伺服器目前剛好流量尖峰塞車（HTTP 503），所有備援模型亦暫時滿載，請等 30 秒後再點一次按鈕即可，不用修改任何設定。"
        if not resp.ok:
            err_msg = f"HTTP {resp.status_code}"
            try:
                err_json = resp.json()
                detail = err_json.get("error", {}).get("message", "")
                if detail:
                    err_msg = f"HTTP {resp.status_code}（{detail}）"
            except Exception:
                pass
            return f"⚠️ Gemini AI 呼叫失敗：{err_msg}"

        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return "⚠️ AI 未回傳分析結果，請稍後再試。"

        parts = candidates[0].get("content", {}).get("parts", [])
        text = "\n".join(p.get("text", "") for p in parts if p.get("text")).strip()
        final_text = (fallback_note + text) if text else "⚠️ AI 回傳內容為空。"
        if text:
            self.cache.set(cache_key, final_text, TieredCache.TTL_AI_REPORT)
        return final_text