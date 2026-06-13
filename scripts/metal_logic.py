"""
PHASE 6 (v4) — Three-Agent Investment Committee
================================================
Architecture:
  Agent 1 — Quant Analyst     (deepseek-chat, parallel)
  Agent 2 — Macro Economist   (deepseek-chat, parallel)
  Agent 3 — CIO / Risk Mgr    (deepseek-reasoner, sequential after 1+2)

Agents 1 and 2 execute concurrently via ThreadPoolExecutor to halve API
latency, then their resolved outputs are passed to Agent 3.

Deterministic brakes (hardcoded, LLM-independent):
  Oracle veto  : live_oracle_score < ORACLE_VETO_THRESHOLD (0.30)
  Regime veto  : HMM P(BEARISH) > 0.65 — loaded from current_regime.json
  Both fire AFTER the CIO returns — the LLMs are the gas pedal; math is the brakes.

Public API unchanged:
  evaluate_metal_swing(...) → {"Action": str, "Reasoning": str, "veto": bool}

New optional parameters (graceful degradation if None):
  moving_averages : dict  — {sma20, sma50, sma200}
  macro_data      : dict  — {dxy_current, dxy_1d_pct, dxy_5d_pct,
                              vix_current, vix_1d, vix_5d,
                              real_yield,
                              cot_gold_raw, cot_gold_z,
                              cot_silver_raw, cot_silver_z}
  copper_gold_z   : float — rolling-252d z-score from alt_data

Action vocabulary (strict):
    "STRONG BUY" | "SNIPER ENTRY WAIT" | "GENERATIONAL HOLD" |
    "STRATEGIC EXIT" | "CUT LOSSES" | "HOLD" | "AVOID" | "⚠️ VETO TRIGGERED"
"""

from __future__ import annotations
import os, json, re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

_DS_KEY = os.getenv("DEEPSEEK_API_KEY", "")
_client = OpenAI(api_key=_DS_KEY, base_url="https://api.deepseek.com") if _DS_KEY else None

ORACLE_VETO_THRESHOLD = 0.30
DECISION_LOG   = ROOT / "data" / "decision_log.json"
LESSONS_FILE   = ROOT / "data" / "lessons_learned.json"
_WEIGHTS_FILE  = ROOT / "data" / "committee_weights.json"
DECISION_LOG.parent.mkdir(parents=True, exist_ok=True)

_W_DEFAULT = 0.50   # equal weighting when no evaluator data exists
_W_MIN     = 0.30
_W_MAX     = 0.70


def _load_committee_weights() -> tuple[float, float]:
    """
    Load dynamic Quant / Macro weights from committee_weights.json.

    Written weekly by scripts/model_evaluator.py.  Falls back to equal
    weighting (0.50 / 0.50) if the file is absent or malformed.

    Returns (quant_weight, macro_weight) — both in [_W_MIN, _W_MAX].
    """
    try:
        if _WEIGHTS_FILE.exists():
            data = json.loads(_WEIGHTS_FILE.read_text())
            qw = float(data.get("quant_weight", _W_DEFAULT))
            mw = float(data.get("macro_weight", _W_DEFAULT))
            # Enforce bounds defensively
            qw = max(_W_MIN, min(_W_MAX, qw))
            mw = max(_W_MIN, min(_W_MAX, mw))
            return qw, mw
    except Exception:
        pass
    return _W_DEFAULT, _W_DEFAULT

VALID_ACTIONS = {
    "ACCUMULATE",       # Deploy available fiat into oz — bullish regime, no veto
    "HOLD_METAL",       # Default state — ride the trend, ignore minor noise
    "STRATEGIC_EXIT",   # Emergency fiat conversion — major regime deterioration only
    "RE_ENTER",         # Redeploy fiat after STRATEGIC_EXIT — regime stabilized
}

# Actions that deploy capital into metal (deterministic vetoes suppress these)
_DEPLOY_CLASS = {"ACCUMULATE", "RE_ENTER"}

# Actions that convert metal → fiat (vetoes force these)
_EXIT_CLASS = {"STRATEGIC_EXIT"}


# ── Data helpers ────────────────────────────────────────────────────────────────

def _load_regime_context(ticker: str) -> tuple[str, bool, str]:
    """
    Load the latest HMM regime snapshot for the given ticker.
    Returns (prompt_block: str, regime_veto: bool, state_label: str).
    Fast — reads current_regime.json, no inference.
    """
    try:
        from scripts.regime_detector import load_cached_regime
        r = load_cached_regime(ticker)
        if not r:
            return "Regime detection not yet fitted — run regime_detector.py --fit-all", False, "UNKNOWN"
        probs = r.get("probabilities", {})
        diag  = r.get("state_diagnostics", {})
        diag_lines = "\n".join(
            f"  {lbl}: drift={d.get('mean_log_ret_bps', 0):+.1f} bps/d  "
            f"sigma_ann~{d.get('ann_vol_approx', 0):.1f}%  occ={d.get('occupancy_pct', 0):.1f}%"
            for lbl, d in diag.items()
        )
        block = (
            f"Current State : {r['state_label']} ({r['confidence']:.0%} confidence)\n"
            f"P(BEARISH)    : {probs.get('BEARISH', 0):.1%}\n"
            f"P(RANGING)    : {probs.get('RANGING', 0):.1%}\n"
            f"P(BULLISH)    : {probs.get('BULLISH', 0):.1%}\n"
            f"Regime Veto   : {r['regime_veto']}\n"
            f"Model fitted  : {r.get('fitted_at', 'unknown')}\n\n"
            f"State learned parameters (from 5yr rolling HMM):\n{diag_lines}"
        )
        return block, bool(r.get("regime_veto", False)), str(r.get("state_label", "UNKNOWN"))
    except Exception as exc:
        return f"Regime detection unavailable: {exc}", False, "UNKNOWN"


def _load_master_context() -> str:
    """Load compressed macro thesis from master_context.json."""
    p = ROOT / "data" / "master_context.json"
    if not p.exists():
        return "No consolidated macro memory available."
    try:
        with open(p) as f:
            mc = json.load(f)
        thesis   = mc.get("macro_thesis", "")
        entries  = mc.get("entries", [])[:5]
        snippets = [
            f"[{e.get('date', '')} {e.get('ticker', '')}] {e.get('summary', '')}"
            for e in entries
        ]
        return thesis + "\n\nRecent signals:\n" + "\n".join(snippets)
    except Exception:
        return "Could not read master context."


def _load_lessons() -> str:
    """Load lessons learned from past trade autopsies (last 20 rules)."""
    if not LESSONS_FILE.exists():
        return "No lessons learned yet — Reflexion Engine has not run its first autopsy."
    try:
        with open(LESSONS_FILE) as f:
            data = json.load(f)
        lessons = data.get("lessons", [])
        if not lessons:
            return "Lessons file exists but is empty."
        lines = [
            f"{i}. [{l.get('date', '')} {l.get('ticker', '')}] {l.get('rule', '')}"
            for i, l in enumerate(lessons[-20:], 1)
        ]
        return "\n".join(lines)
    except Exception:
        return "Could not load lessons learned."


def _append_decision_log(
    ticker: str,
    price: float,
    action: str,
    reasoning: str,
    extra: dict | None = None,
) -> None:
    """Append one CIO decision to the rolling decision log. Never raises."""
    try:
        log = []
        if DECISION_LOG.exists():
            with open(DECISION_LOG) as f:
                log = json.load(f)
        entry = {
            "date":               datetime.utcnow().strftime("%Y-%m-%d"),
            "timestamp":          datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "ticker":             ticker,
            "price_at_time":      round(price, 4),
            "action_taken":       action,
            "original_reasoning": reasoning,
        }
        if extra:
            entry.update(extra)
        log.append(entry)
        with open(DECISION_LOG, "w") as f:
            json.dump(log, f, indent=2)
    except Exception:
        pass


# ── Low-level LLM call ──────────────────────────────────────────────────────────

def _call_ds(model: str, system: str, user: str, max_tokens: int = 3000) -> str:
    """
    Single DeepSeek API call. Returns raw content string.
    Handles deepseek-reasoner's empty-content / reasoning_content quirk.
    Fires a Telegram URGENT alert on HTTP 429 (rate/billing limit) or 5xx errors.
    """
    from openai import RateLimitError as _RateLimitError, APIStatusError as _APIStatusError

    try:
        resp = _client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
        )
    except _RateLimitError as exc:
        # HTTP 429 — covers both per-minute rate limits and billing/quota caps
        try:
            from scripts.telegram_notifier import send_urgent
            send_urgent("DeepSeek", exc, context=f"model={model}")
        except Exception:
            pass
        raise
    except _APIStatusError as exc:
        # 5xx server-side errors
        if exc.status_code >= 500:
            try:
                from scripts.telegram_notifier import send_urgent
                send_urgent("DeepSeek", exc,
                            context=f"model={model}  HTTP {exc.status_code}")
            except Exception:
                pass
        raise

    msg = resp.choices[0].message
    raw = (msg.content or "").strip()
    if not raw:
        rc = getattr(msg, "reasoning_content", "") or ""
        m  = re.search(r'\{.*\}', rc, re.DOTALL)
        if m:
            raw = m.group(0)
    return raw


def _parse_json_safe(raw: str, fallback: dict) -> dict:
    """Strip markdown fences, parse JSON. Returns fallback on any error."""
    try:
        cleaned = raw.strip().strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
        # Extract first {...} block if there's surrounding text
        m = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if m:
            cleaned = m.group(0)
        return json.loads(cleaned)
    except Exception:
        return fallback


# ── Agent system prompts ────────────────────────────────────────────────────────

_QUANT_SYSTEM = """\
You are a ruthless quantitative analyst for a commodity hedge fund. You analyze \
price model outputs, trend structure, and macro regime signals with zero emotional \
bias. You report only what the data says.

Your output MUST be a JSON object in this exact format, no extra text:
{
  "thesis": "<exactly 2 paragraphs — paragraph 1: LSTM forecast and MA trend structure; paragraph 2: HMM regime and copper/gold positioning signal>",
  "conviction": <integer from -10 to +10>
}

Conviction scale: -10 = maximum bearish certainty, 0 = neutral/conflicted, \
+10 = maximum bullish certainty.
Each paragraph is 2-3 sentences maximum. Cite specific price levels and signal \
values. No hedging language."""


_MACRO_SYSTEM = """\
You are a global macro economist specializing in fiat liquidity cycles and their \
transmission to hard assets. You track dollar strength, risk appetite, real \
interest rates, and institutional positioning. You identify the macro regime that \
makes price moves probable — you do not predict short-term prices.

Your output MUST be a JSON object in this exact format, no extra text:
{
  "thesis": "<exactly 2 paragraphs — paragraph 1: DXY and VIX dynamics; paragraph 2: real yields and CFTC COT positioning>",
  "conviction": <integer from -10 to +10>
}

Conviction scale: -10 = macro strongly headwinds gold, +10 = macro strongly \
tailwinds gold.
Each paragraph is 2-3 sentences maximum. Reference specific values. \
No hedging language."""


_CIO_SYSTEM = """\
You are the Chief Investment Officer managing a physical gold accumulation mandate \
for a high-net-worth principal in the UAE following Sharia-compliant investment \
principles.

MANDATE: The client holds PHYSICAL GOLD as the base asset. Fiat (USD/AED) is a \
TEMPORARY holding state used only during major corrections to rebuy more ounces \
at lower prices. The goal is to accumulate the maximum number of ounces over time, \
NOT to maximise short-term USD returns. Transaction costs on physical gold are \
30-80 bps bid/ask — every unnecessary trade destroys ounce accumulation.

Your output MUST be a JSON object in this exact format, no extra text:
{
  "Action": "<one of the exact action strings below>",
  "Reasoning": "<exactly 3 bullet points separated by newlines, each beginning with '• ', one sentence each, citing specific data from both specialist reports>"
}

Valid Action strings (use EXACTLY as shown):
- "ACCUMULATE"      — Deploy available fiat into gold oz. Use when regime is
                      BULLISH or RANGING, real rates not hostile, COT not crowded,
                      and fiat is available to deploy.
- "HOLD_METAL"      — The DEFAULT state. Hold existing oz, ignore minor volatility.
                      Use whenever signals are ambiguous, mixed, or not extreme.
                      This is the correct answer at least 90% of the time.
- "STRATEGIC_EXIT"  — Emergency conversion of ALL metal to fiat. EXTREMELY HIGH BAR.
                      Use ONLY when HMM veto is active OR when a regime-level
                      dislocation threatens a -15%+ drawdown. The EXPLICIT goal
                      is to sell at a local top and rebuy MORE ounces at the bottom.
                      Do NOT use for normal corrections (<10%) — that is noise.
- "RE_ENTER"        — Deploy ALL fiat back into oz after a prior STRATEGIC_EXIT.
                      Use when regime has clearly stabilized (BULLISH or RANGING),
                      price momentum is positive, and COT positioning is not crowded.

CRITICAL BIAS: When in doubt, output HOLD_METAL. A failed STRATEGIC_EXIT that \
converts metal to fiat at the wrong time will permanently destroy ounce count \
through round-trip transaction costs. The portfolio is better served by riding a \
20% drawdown than by mis-timing an exit and paying costs twice.

HARD CONSTRAINTS (applied deterministically AFTER your output — inform your reasoning):
- If HMM Veto Active: True → output will be overridden to STRATEGIC_EXIT
- If Oracle score < 0.30 → output will be overridden to STRATEGIC_EXIT

Rules:
- Never output any action not in the list above
- Each bullet point must cite at least one specific value from the specialist reports
- Flag when the setup suits direct physical gold accumulation for the UAE Sharia mandate"""


# ── Agent runners ───────────────────────────────────────────────────────────────

def _run_quant_agent(inputs: dict) -> dict:
    """
    Agent 1 — Quant Analyst.
    Inputs: LSTM predictions, moving averages, HMM regime, copper/gold z-score.
    Returns {"thesis": str, "conviction": int}
    """
    name          = inputs["name"]
    ticker        = inputs["ticker"]
    current_price = inputs["current_price"]
    pred_5d       = inputs["pred_5d"]
    pred_21d      = inputs["pred_21d"]
    pred_252d     = inputs["pred_252d"]
    regime_block  = inputs["regime_block"]
    _ma_raw       = inputs.get("moving_averages")
    ma            = _ma_raw if isinstance(_ma_raw, dict) else {}
    cu_au_z       = inputs.get("copper_gold_z")

    def _pos(price: float) -> str:
        return "ABOVE" if current_price >= price else "BELOW"

    def _cross(sma20: float, sma50: float) -> str:
        if sma20 > sma50:
            return "GOLDEN CROSS (SMA20 > SMA50)"
        if sma20 < sma50:
            return "DEATH CROSS (SMA20 < SMA50)"
        return "NO CROSS (SMA20 ~ SMA50)"

    if ma:
        sma20  = ma.get("sma20", 0.0)
        sma50  = ma.get("sma50", 0.0)
        sma200 = ma.get("sma200", 0.0)
        ma_section = (
            f"### Moving Average Structure\n"
            f"- SMA20:  ${sma20:,.2f}  — price is {_pos(sma20)}\n"
            f"- SMA50:  ${sma50:,.2f}  — price is {_pos(sma50)}\n"
            f"- SMA200: ${sma200:,.2f} — price is {_pos(sma200)}\n"
            f"- MA cross signal: {_cross(sma20, sma50)}\n"
        )
    else:
        ma_section = "### Moving Average Structure\nNot available for this session.\n"

    cu_line = (
        f"- Copper/Gold z-score: {cu_au_z:+.3f}  "
        f"(>+1.0 = industrial demand expanding vs safe-haven;  <-1.0 = risk-off flight to gold)"
        if cu_au_z is not None
        else "- Copper/Gold z-score: not available"
    )

    pg_section = ""
    try:
        _pg_path = ROOT / "data" / "proving_ground_predictions.json"
        if _pg_path.exists():
            _pg = json.loads(_pg_path.read_text())
            tag = ticker.replace("=", "").replace("^", "")
            tac = _pg.get(f"{tag}_tactical", {})
            strat = _pg.get(f"{tag}_strategic", {})
            struc = _pg.get(f"{tag}_structural", {})
            if tac or strat or struc:
                pg_section = "\n### Proving Ground Ensemble (independent tri-horizon LSTMs)\n"
                if tac:
                    pg_section += f"- Tactical (t+5d):    {tac.get('pred_pct', 0):+.2f}%\n"
                if strat:
                    pg_section += f"- Strategic (t+21d):   {strat.get('pred_pct', 0):+.2f}%\n"
                if struc:
                    pg_section += f"- Structural (t+252d): {struc.get('pred_pct', 0):+.2f}%\n"
                pg_section += "  (Compare these with the main LSTM — agreement = higher conviction)\n"
    except Exception:
        pass

    # Monte Carlo and Kelly data for quant agent
    mc_section = ""
    try:
        _mc_path = ROOT / "data" / "monte_carlo_simulation.json"
        if _mc_path.exists():
            _mc = json.loads(_mc_path.read_text())
            _probs = _mc.get("probabilities", {})
            _risk = _mc.get("risk", {})
            _term = _mc.get("terminal", {})
            mc_section = (
                f"\n### Monte Carlo Simulation ({_mc.get('horizon_days', 21)}d, "
                f"{_mc.get('n_paths', 10000):,} paths, GARCH+Student-t)\n"
                f"- P(positive return): {_probs.get('positive_return', 0):.1%}\n"
                f"- Mean return: {_term.get('mean_return_pct', 0):+.2f}%\n"
                f"- VaR-95: {_risk.get('var_95_pct', 0):+.2f}%\n"
                f"- CVaR-95: {_risk.get('cvar_95_pct', 0):+.2f}%\n"
            )
    except Exception:
        pass

    kelly_section = ""
    try:
        _ks_path = ROOT / "data" / "kelly_sizing.json"
        if _ks_path.exists():
            _ks = json.loads(_ks_path.read_text())
            _k = _ks.get("kelly", {})
            _s = _ks.get("sizing", {})
            kelly_section = (
                f"\n### Kelly Criterion\n"
                f"- Edge: {_k.get('edge', 0):+.4f}\n"
                f"- Optimal position: {_s.get('final_position_pct', 0):.1f}%\n"
                f"- Should trade: {'YES' if _k.get('should_trade') else 'NO'}\n"
            )
    except Exception:
        pass

    mtf_section = ""
    try:
        _mtf_path = ROOT / "data" / "mtf_confluence.json"
        if _mtf_path.exists():
            _mtf = json.loads(_mtf_path.read_text())
            _conf = _mtf.get("confluence", {})
            mtf_section = (
                f"\n### Multi-Timeframe Confluence\n"
                f"- Level: {_conf.get('level', 'UNKNOWN')}\n"
                f"- Score: {_conf.get('score', 0):+d}/100\n"
                f"- Alignment: {_conf.get('bullish_tfs', 0)} bullish / "
                f"{_conf.get('bearish_tfs', 0)} bearish timeframes\n"
            )
    except Exception:
        pass

    user = f"""\
## Quantitative Signal Report — {name} ({ticker}) — {datetime.utcnow().strftime('%Y-%m-%d')}

### LSTM Price Forecasts (GoldLSTM-v1, 15-feature attention model)
- Current price:   ${current_price:,.2f}
- t+5d forecast:   ${pred_5d:,.2f}  ({(pred_5d - current_price) / current_price * 100:+.1f}%)
- t+21d forecast:  ${pred_21d:,.2f}  ({(pred_21d - current_price) / current_price * 100:+.1f}%)
- t+252d forecast: ${pred_252d:,.2f}  ({(pred_252d - current_price) / current_price * 100:+.1f}%)
{pg_section}{mc_section}{kelly_section}{mtf_section}
{ma_section}
### HMM Regime (5-Year Rolling Hidden Markov Model)
{regime_block}

### Industrial Demand Signal
{cu_line}

Analyze alignment between the LSTM forecast, proving ground ensemble, Monte Carlo probabilities, \
Kelly edge, MTF confluence, MA trend structure, HMM regime, and copper/gold positioning. \
Output your JSON thesis and conviction score."""

    try:
        raw    = _call_ds("deepseek-chat", _QUANT_SYSTEM, user, max_tokens=2000)
        result = _parse_json_safe(raw, {"thesis": raw or "Quant analysis parse error.", "conviction": 0})
        result["conviction"] = max(-10, min(10, int(result.get("conviction", 0))))
        return result
    except Exception as exc:
        return {"thesis": f"Quant agent error: {exc}", "conviction": 0}


def _run_macro_agent(inputs: dict) -> dict:
    """
    Agent 2 — Macro Economist.
    Inputs: DXY, VIX, real yields (FRED), CFTC COT positioning.
    Returns {"thesis": str, "conviction": int}
    """
    md = inputs.get("macro_data") or {}

    if md:
        dxy_current  = md.get("dxy_current", 0.0)
        dxy_1d_pct   = md.get("dxy_1d_pct", 0.0)
        dxy_5d_pct   = md.get("dxy_5d_pct", 0.0)
        vix_current  = md.get("vix_current", 0.0)
        vix_1d       = md.get("vix_1d", 0.0)
        vix_5d       = md.get("vix_5d", 0.0)
        real_yield   = md.get("real_yield", 0.0)
        cot_gold_raw = md.get("cot_gold_raw", 0.0)
        cot_gold_z   = md.get("cot_gold_z", 0.0)
        cot_silv_raw = md.get("cot_silver_raw", 0.0)
        cot_silv_z   = md.get("cot_silver_z", 0.0)

        def _vix_regime(v: float) -> str:
            if v < 15:
                return "CALM (<15)"
            if v < 25:
                return "ELEVATED (15-25)"
            return "FEAR (>25)"

        macro_section = f"""\
### Dollar Strength (DXY)
- Current level:  {dxy_current:.2f}
- 1-day change:   {dxy_1d_pct:+.2f}%
- 5-day change:   {dxy_5d_pct:+.2f}%
  (Rising DXY = headwind for gold;  falling DXY = tailwind)

### Risk Appetite (VIX — CBOE Volatility Index)
- Current level:  {vix_current:.1f}
- 1-day change:   {vix_1d:+.1f} pts
- 5-day change:   {vix_5d:+.1f} pts
- Regime:         {_vix_regime(vix_current)}

### Real Interest Rates (FRED DFII10 — 10Y Treasury Inflation-Indexed)
- 10Y real yield: {real_yield:+.3f}%
  (Positive real yield = opportunity cost headwind;  negative = tailwind)

### CFTC COT — Managed Money Net Positioning
- Gold MM net:   {cot_gold_raw:>10,.0f} contracts  |  z-score: {cot_gold_z:+.3f}σ
- Silver MM net: {cot_silv_raw:>10,.0f} contracts  |  z-score: {cot_silv_z:+.3f}σ
  (>+1.5σ = crowded long, vulnerable to flush;  <-1.5σ = crowded short, coiled for squeeze)"""
    else:
        macro_section = "Macro data not available for this session — provide macro_data dict."

    user = f"""\
## Macro Environment Report — {datetime.utcnow().strftime('%Y-%m-%d')}

{macro_section}

Assess the macro tailwinds and headwinds for precious metals. What does the \
combination of dollar momentum, risk sentiment, real rates, and institutional \
positioning signal about the macro regime? Output your JSON thesis and conviction score."""

    try:
        raw    = _call_ds("deepseek-chat", _MACRO_SYSTEM, user, max_tokens=2000)
        result = _parse_json_safe(raw, {"thesis": raw or "Macro analysis parse error.", "conviction": 0})
        result["conviction"] = max(-10, min(10, int(result.get("conviction", 0))))
        return result
    except Exception as exc:
        return {"thesis": f"Macro agent error: {exc}", "conviction": 0}


def _run_cio_agent(quant: dict, macro: dict, inputs: dict) -> dict:
    """
    Agent 3 — CIO / Risk Manager (deepseek-reasoner).
    Receives resolved Quant and Macro reports.
    Returns {"Action": str, "Reasoning": str}
    """
    name          = inputs["name"]
    ticker        = inputs["ticker"]
    current_price = inputs["current_price"]
    portfolio     = inputs["portfolio"]
    regime_veto   = inputs["regime_veto"]
    oracle_str    = inputs["oracle_str"]
    lessons       = inputs["lessons"]
    master_ctx    = inputs["master_ctx"]

    shares   = float(portfolio.get("shares", 0))
    avg_cost = float(portfolio.get("avg_cost", 0.0))
    mandate  = portfolio.get("strategy_mandate", "active_swing")
    pnl_pct  = ((current_price - avg_cost) / avg_cost * 100) if avg_cost > 0 else 0.0

    q_conv = quant.get("conviction", 0)
    m_conv = macro.get("conviction", 0)

    # Dynamic weights — updated weekly by scripts/model_evaluator.py
    q_weight, m_weight = _load_committee_weights()
    w_total  = q_weight + m_weight
    avg_conv = (q_conv * q_weight + m_conv * m_weight) / w_total

    # Human-readable weight note for CIO context
    _weight_note = (
        f"Quant weight={q_weight:.0%}  Macro weight={m_weight:.0%}"
        f"  (dynamic — calibrated from recent {ticker} call accuracy)"
    )

    def _bias_label(avg: float) -> str:
        if avg >= 6:   return "STRONG BULLISH"
        if avg >= 2:   return "BULLISH"
        if avg <= -6:  return "STRONG BEARISH"
        if avg <= -2:  return "BEARISH"
        return "NEUTRAL"

    user = f"""\
## CIO Synthesis Brief — {name} ({ticker}) — {datetime.utcnow().strftime('%Y-%m-%d')}

### Specialist Reports

**[ Quant Analyst ]**  Conviction: {q_conv:+d}/10
{quant.get('thesis', 'No thesis available.')}

**[ Macro Economist ]**  Conviction: {m_conv:+d}/10
{macro.get('thesis', 'No thesis available.')}

### Combined Signal
- Weighted conviction: {avg_conv:+.1f}/10
- Signal bias: {_bias_label(avg_conv)}
- Weighting: {_weight_note}

### Portfolio Context
- Metal:          {name} ({ticker})
- Shares held:    {shares}
- Avg cost:       ${avg_cost:,.2f}
- Current price:  ${current_price:,.2f}
- Unrealized P&L: {pnl_pct:+.1f}%
- Mandate:        {mandate}

### Risk Flags
- HMM Veto Active:    {regime_veto}
- Oracle macro score: {oracle_str}  (veto threshold: {ORACLE_VETO_THRESHOLD})

### Weekly Macro Memory
{master_ctx}

### Lessons Learned (from your own past trade autopsies)
{lessons}

Synthesize both specialist reports and the portfolio context into your final \
decision. Output only the JSON object."""

    fallback = {
        "Action": "HOLD",
        "Reasoning": (
            "• CIO agent returned an unparseable response — defaulting to HOLD.\n"
            "• No capital movement recommended until system is verified.\n"
            "• Review DeepSeek API connectivity and retry."
        ),
    }

    try:
        raw    = _call_ds("deepseek-reasoner", _CIO_SYSTEM, user, max_tokens=8000)
        result = _parse_json_safe(raw, fallback)
        action = result.get("Action", "HOLD")
        if action not in VALID_ACTIONS:
            action = "HOLD"
            result["Reasoning"] = f"CIO returned unknown action. Defaulted to HOLD. Original: {result.get('Reasoning', '')}"
        result["Action"] = action
        return result
    except Exception as exc:
        fallback["Reasoning"] = (
            f"• CIO agent error: {exc}\n"
            "• Defaulting to HOLD for capital preservation.\n"
            "• Review API connectivity and retry."
        )
        return fallback


# ── Public API ──────────────────────────────────────────────────────────────────

def evaluate_metal_swing(
    ticker:            str,
    current_price:     float,
    portfolio:         dict,
    pred_5d:           float,
    pred_21d:          float,
    pred_252d:         float,
    moving_averages:   dict | None = None,
    macro_data:        dict | None = None,
    copper_gold_z:     float | None = None,
    live_oracle_score: float | None = None,
) -> dict:
    """
    Orchestrate the three-agent Investment Committee pipeline.

    Execution order:
      1. Load regime context (fast, JSON read)
      2. Agent 1 (Quant) + Agent 2 (Macro) — PARALLEL via ThreadPoolExecutor
      3. Agent 3 (CIO) — sequential, receives both reports
      4. Deterministic veto layer — oracle score + HMM regime (LLM-independent)
      5. Append to decision log

    Returns {"Action": str, "Reasoning": str, "veto": bool}
    """
    if _client is None:
        return {
            "Action":    "HOLD_METAL",
            "Reasoning": (
                "• DeepSeek Investment Committee is offline — DEEPSEEK_API_KEY not configured.\n"
                "• Defaulting to HOLD_METAL — physical gold position maintained.\n"
                "• Configure DEEPSEEK_API_KEY in .env to enable AI decisions."
            ),
            "veto": False,
        }

    name = "Gold" if ticker == "GC=F" else "Silver"
    oracle_str = f"{live_oracle_score:.2f}" if live_oracle_score is not None else "unavailable"

    # ── Step 1: regime context (fast JSON read) ─────────────────────────────────
    regime_block, regime_veto_flag, regime_state_label = _load_regime_context(ticker)
    lessons    = _load_lessons()
    master_ctx = _load_master_context()

    # ── Step 2: Quant + Macro in parallel ───────────────────────────────────────
    quant_inputs = {
        "name":           name,
        "ticker":         ticker,
        "current_price":  current_price,
        "pred_5d":        pred_5d,
        "pred_21d":       pred_21d,
        "pred_252d":      pred_252d,
        "regime_block":   regime_block,
        "moving_averages": moving_averages,
        "copper_gold_z":  copper_gold_z,
    }
    macro_inputs = {
        "macro_data": macro_data,
    }

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_quant = pool.submit(_run_quant_agent, quant_inputs)
        fut_macro = pool.submit(_run_macro_agent, macro_inputs)
        quant_result = fut_quant.result()
        macro_result = fut_macro.result()

    # ── Step 3: CIO synthesizes both reports ────────────────────────────────────
    cio_inputs = {
        "name":          name,
        "ticker":        ticker,
        "current_price": current_price,
        "portfolio":     portfolio,
        "regime_veto":   regime_veto_flag,
        "oracle_str":    oracle_str,
        "lessons":       lessons,
        "master_ctx":    master_ctx,
    }
    cio_result = _run_cio_agent(quant_result, macro_result, cio_inputs)

    action = cio_result.get("Action", "HOLD")
    reason = cio_result.get("Reasoning", "")

    # ── Step 4: Deterministic veto layer (LLM-independent brakes) ───────────────
    veto = False

    # Oracle veto — Perplexity macro score below threshold
    # Forces STRATEGIC_EXIT; suppresses any DEPLOY action.
    if (live_oracle_score is not None
            and live_oracle_score < ORACLE_VETO_THRESHOLD
            and action in _DEPLOY_CLASS):
        original = action
        action   = "STRATEGIC_EXIT"
        reason   = (
            f"• ORACLE VETO: CIO recommended {original} but Live Macro Oracle signals "
            f"severe macro risk (score {live_oracle_score:.2f} < {ORACLE_VETO_THRESHOLD} threshold).\n"
            f"• Converting to fiat — macro environment is hostile to gold deployment.\n"
            f"• CIO reasoning: {reason}"
        )
        veto = True

    # Regime veto — HMM detected high-probability BEARISH state
    # Forces STRATEGIC_EXIT on any deploy action; also fires on HOLD_METAL.
    if (not veto
            and regime_veto_flag
            and action in (_DEPLOY_CLASS | {"HOLD_METAL"})):
        original = action
        action   = "STRATEGIC_EXIT"
        reason   = (
            f"• REGIME VETO: CIO recommended {original} but HMM engine detected "
            f"high-probability BEARISH macro regime.\n"
            f"• Converting to fiat — structural regime deterioration demands capital protection.\n"
            f"• CIO reasoning: {reason}"
        )
        veto = True

    # ── Step 5: Log decision (rich entry for shadow trader & autopsy) ──────────
    _md = macro_data or {}
    _append_decision_log(ticker, current_price, action, reason, extra={
        "quant_conviction":  quant_result.get("conviction"),
        "macro_conviction":  macro_result.get("conviction"),
        "quant_thesis":      quant_result.get("thesis"),
        "macro_thesis":      macro_result.get("thesis"),
        "hmm_state":         regime_state_label,
        "hmm_veto_active":   int(regime_veto_flag),
        "oracle_score":      live_oracle_score,
        "copper_gold_z":     copper_gold_z,
        "real_yield":        _md.get("real_yield"),
        "cot_gold_z":        _md.get("cot_gold_z"),
        "dxy_current":       _md.get("dxy_current"),
        "vix_current":       _md.get("vix_current"),
    })

    return {"Action": action, "Reasoning": reason, "veto": veto}
