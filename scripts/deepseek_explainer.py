#!/usr/bin/env python3
"""
DeepSeek Explainer  (Phase X Stage 56, the north-star UI surface)
==================================================================
Conversational layer over the entire institutional stack. The operator
asks a plain-English question; this engine:

  1. Gathers structured context from pipeline_state.json + every engine
     JSON in data/*.json, distilled to a compact dossier.
  2. Sends the dossier + the operator's question to DeepSeek with a
     trained prompt that requires citations to the source JSON keys.
  3. Returns the model's answer plus the dossier slice used.

CLI:
    python3 scripts/deepseek_explainer.py "What is the macro nowcast saying?"
    python3 scripts/deepseek_explainer.py --topic risk
    python3 scripts/deepseek_explainer.py --briefing

A reusable `explain(question)` Python API drives the future home-page chat
panel.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.cache_layer import cached  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

DATA_DIR = ROOT / "data"
LAST_TURN_FILE = DATA_DIR / "deepseek_last_turn.json"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Dossier assembly
# ---------------------------------------------------------------------------
ENGINE_FILES = {
    # Phase I
    "alpha_attribution":      "alpha_attribution.json",
    "vol_surface":            "vol_surface.json",
    "signal_decay":           "signal_decay.json",
    # Phase II
    "hrp":                    "hrp_allocator.json",
    "black_litterman":        "black_litterman.json",
    "mean_cvar":              "mean_cvar.json",
    "vol_target":             "vol_target_budget.json",
    "dcc_garch":              "dcc_garch.json",
    # Phase III
    "structural_breaks":      "structural_breaks.json",
    "macro_regime":           "macro_regime.json",
    "bma":                    "bma_weights.json",
    # Phase IV
    "smart_order_router":     "smart_order_router.json",
    "adverse_selection":      "adverse_selection.json",
    "stop_loss":              "stop_loss_optimizer.json",
    "capacity":               "capacity_analyzer.json",
    # Phase V
    "news_sentiment":         "news_sentiment.json",
    "cb_speech":              "cb_speech.json",
    "geopolitical":           "geopolitical_events.json",
    "etf_flows":              "etf_flows.json",
    "macro_nowcast":          "macro_nowcast.json",
    # Phase VI
    "options":                "options_pricer.json",
    "tail_hedge":             "tail_hedge.json",
    "carry":                  "carry_analyzer.json",
    "term_structure":         "term_structure.json",
    # Phase VII
    "bayesian_hpo":           "bayesian_hpo.json",
    "purged_kfold":           "purged_kfold.json",
    "ensemble_stacking":      "ensemble_stacking.json",
    "rl_sizing":              "rl_sizing.json",
    "conformal":              "conformal_intervals.json",
    # Phase VIII
    "brinson":                "brinson_attribution.json",
    "fama_french":            "fama_french.json",
    "ic_ir":                  "ic_ir_tracker.json",
    "decision_quality":       "decision_quality.json",
    # Phase IX
    "audit_trail":            "audit_trail_status.json",
    "dr_backup":              "dr_backup.json",
    "latency":                "latency_profile.json",
    # Phase X
    "mrm_champion":           "mrm_champion.json",
    "strategy_sandbox":       "strategy_sandbox.json",
    # Phase XIV — multi-strategy operational stack
    "alpha_stacker":          "alpha_stacker.json",
    "strategy_selector":      "strategy_selector.json",
    "multi_strategy":         "multi_strategy_trader.json",
    # Phase XV — strategy validation
    "strategy_backtest":      "strategy_backtest.json",
    # Phase XXIII / XXVI — ML conviction pipeline
    "conviction_weights":     "conviction_weights.json",
    "ml_conviction":          "ml_conviction_poc.json",
    "ml_walk_forward":        "ml_walk_forward.json",
    # Phase XXV — Treasury hedge sleeve
    "treasury_hedge":         "treasury_hedge.json",
    "treasury_stress":        "treasury_overlay_stress_eval.json",
    # Phase XXVII — fast crisis/regime dials
    "crisis":                 "crisis_detector.json",
}


def _load(fname: str) -> dict:
    p = DATA_DIR / fname
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


# Compact projections — what we actually feed DeepSeek for each engine.
# Each projection picks ~5-10 of the most decision-relevant fields, not the full JSON.
def _project(name: str, data: dict) -> dict:
    if not data:
        return {}
    if name == "alpha_attribution":
        return {
            "top_source":         data.get("ranked_by_sharpe", [None])[0],
            "combined_sharpe":    data.get("combined", {}).get("equal_weight_summary", {}).get("sharpe"),
            "diversification_ratio": data.get("combined", {}).get("diversification_ratio"),
        }
    if name == "vol_surface":
        return {
            "regime":         data.get("vol_regime"),
            "phase":          data.get("phase"),
            "curve_shape":    data.get("curve_shape"),
            "kelly_mult":     data.get("actions", {}).get("kelly_fraction_multiplier"),
            "rv_21d_pct":     data.get("term_structure", {}).get("rv_21d"),
        }
    if name == "macro_nowcast":
        return {
            "regime":         data.get("regime"),
            "composite":      data.get("composite_score"),
            "top_drivers":    data.get("top_drivers", [])[:3],
        }
    if name == "macro_regime":
        return {
            "quadrant":       data.get("quadrant"),
            "confidence":     data.get("confidence"),
            "growth":         data.get("growth_score"),
            "inflation":      data.get("inflation_score"),
        }
    if name == "dcc_garch":
        return {
            "avg_corr_now":      data.get("avg_pairwise_corr_now"),
            "avg_corr_long_run": data.get("avg_pairwise_corr_long_run"),
            "n_stressed":        data.get("n_stressed"),
            "stressed_pairs":    data.get("stressed_pairs", [])[:3],
        }
    if name == "structural_breaks":
        return {
            "cusum_break":         data.get("summary", {}).get("cusum_break"),
            "n_var_breaks":        data.get("summary", {}).get("n_variance_breaks"),
            "most_recent_break":   data.get("summary", {}).get("most_recent_break"),
            "days_since_break":    data.get("summary", {}).get("days_since_last_break"),
        }
    if name == "fama_french":
        return {
            "alpha_annual_pct":   data.get("alpha_annualised_pct"),
            "alpha_t":            data.get("alpha_t_stat"),
            "r_squared":          data.get("r_squared"),
            "dominant_factor":    data.get("dominant_factor"),
            "info_ratio":         data.get("information_ratio"),
        }
    if name == "brinson":
        return {
            "portfolio_return_pct":   data.get("portfolio_return_pct"),
            "benchmark_return_pct":   data.get("benchmark_return_pct"),
            "excess_return_pct":      data.get("excess_return_pct"),
            "allocation_effect_pct":  data.get("allocation_effect_pct"),
        }
    if name == "ic_ir":
        return {
            "ranked_by_ir":         data.get("ranked_by_ir", [])[:3],
            "deployable_signals":   data.get("deployable_signals", []),
        }
    if name == "bma":
        return {
            "top_source":  data.get("top_source"),
            "weights":     data.get("weights", {}).get("bma", {}),
        }
    if name == "mean_cvar":
        return {
            "mean_cvar_sharpe": data.get("metrics", {}).get("mean_cvar", {}).get("sharpe"),
            "min_cvar_sharpe":  data.get("metrics", {}).get("min_cvar", {}).get("sharpe"),
            "weights_mean_cvar":data.get("metrics", {}).get("mean_cvar", {}).get("weights"),
        }
    if name == "tail_hedge":
        return {
            "contracts":             data.get("contracts_needed"),
            "annual_drag_pct":       data.get("annual_drag_pct"),
            "residual_cvar_pct":     data.get("residual_cvar_pct"),
            "constraint_binding":    data.get("constraint_binding"),
        }
    if name == "term_structure":
        return {
            "curve_shape":     data.get("curve_shape"),
            "slope_pct":       data.get("overall_slope_pct"),
            "roll_yield_pct":  data.get("roll_yield_pct"),
            "stress_flag":     data.get("stress_flag"),
        }
    if name == "capacity":
        return {
            "expected_alpha_pct":     data.get("expected_alpha_pct"),
            "physical_cap_usd":       data.get("thresholds_physical", {}).get("decay_25pct", {}).get("aum_cap_usd"),
            "paper_cap_usd":          data.get("thresholds_paper", {}).get("decay_25pct", {}).get("aum_cap_usd"),
        }
    if name == "geopolitical":
        return {
            "regime":         data.get("regime"),
            "current_score":  data.get("current_score"),
            "priority":       data.get("priority"),
        }
    if name == "cb_speech":
        return {
            "fed_regime":     data.get("fed_regime"),
            "pplx_fed":       data.get("fed_latest"),
            "regime_shift":   data.get("regime_shift_detected"),
        }
    if name == "mrm_champion":
        return {
            "current_champion": data.get("current_champion"),
            "decision":         data.get("decision"),
            "score_delta":      data.get("score_delta"),
        }
    if name == "audit_trail":
        return {
            "chain_valid": data.get("chain_valid"),
            "n_total":     data.get("n_total"),
        }
    if name == "alpha_stacker":
        dec = data.get("decision", {})
        return {
            "conviction_score": dec.get("conviction_score"),
            "conviction_tier":  dec.get("conviction_tier"),
            "direction":        dec.get("direction"),
            "recommended_size_pct": dec.get("recommended_size_pct"),
            "top_drivers":      [s.get("name") for s in data.get("top_drivers", [])[:3]],
            "top_detractors":   [s.get("name") for s in data.get("top_detractors", [])[:2]],
            "n_risk_flags":     data.get("n_risk_flags"),
            "risk_flags":       data.get("risk_flags", [])[:3],
        }
    if name == "strategy_selector":
        return {
            "strategy":        data.get("strategy"),
            "direction":       data.get("direction"),
            "final_size_pct":  data.get("final_size_pct"),
            "reasoning":       data.get("reasoning", [])[:3],
            "regime_context":  data.get("regime_context", {}),
        }
    if name == "multi_strategy":
        hs = data.get("hedge_state") or {}
        by_s = data.get("by_strategy") or {}
        return {
            "strategy":          data.get("strategy"),
            "book_equity_usd":   data.get("book_equity_usd"),
            "lifetime_pl_pct":   data.get("lifetime_pl_pct"),
            "n_open":            data.get("n_open"),
            "hedge_instrument":  hs.get("instrument"),
            "hedge_fraction_pct":round((data.get("hedge_fraction") or 0) * 100, 1),
            "hedge_gate_action": hs.get("gate_action"),
            "sharia_cleared":    hs.get("gate_action") == "CLEARED_SOVEREIGN",
            "by_strategy_pl":    {k: v.get("realized_pl_usd") for k, v in by_s.items()},
            "vol_breaker":       data.get("vol_breaker"),
        }
    if name == "crisis":
        fm = data.get("fast_metrics") or {}
        return {
            "tier":             data.get("tier"),
            "score":            data.get("score"),
            "vol_spike_ratio":  fm.get("vol_spike_ratio"),
            "rsi_14":           fm.get("rsi_14"),
            "macd_hist_pct":    fm.get("macd_hist_pct"),
            "drift_10d_pct":    fm.get("drift_10d_pct"),
            "guidance":         data.get("guidance"),
        }
    if name == "strategy_backtest":
        p = data.get("performance", {})
        return {
            "achievability_verdict": data.get("achievability_verdict"),
            "cum_return_pct":        p.get("cum_return_pct"),
            "annualised_return_pct": p.get("annualised_return_pct"),
            "sharpe":                p.get("sharpe"),
            "max_drawdown_pct":      p.get("max_drawdown_pct"),
            "months_at_target":      data.get("monthly", {}).get("n_at_or_above_target"),
        }
    if name == "conviction_weights":
        w = data.get("weights", {})
        top = max(w.items(), key=lambda kv: kv[1], default=(None, 0)) if w else (None, 0)
        return {
            "weights":       w,
            "top_component": top[0],
            "top_weight":    round(top[1], 3),
        }
    if name == "ml_conviction":
        cp = data.get("comparison", {})
        return {
            "gate_passed":  cp.get("gate_passed"),
            "oos_ic_ml":    cp.get("oos_ic_ml"),
            "sharpe_ml":    cp.get("sharpe_ml"),
            "sharpe_rule":  cp.get("sharpe_rule"),
            "ic_pass":      cp.get("ic_pass"),
            "sharpe_pass":  cp.get("sharpe_pass"),
        }
    if name == "ml_walk_forward":
        ml = data.get("ml", {})
        return {
            "gate_passed":        data.get("gate_passed"),
            "verdict":            data.get("verdict"),
            "median_ann_pct":     ml.get("median_ann_pct"),
            "positive_share_pct": ml.get("positive_share_pct"),
            "avg_sharpe":         ml.get("avg_sharpe"),
        }
    if name == "treasury_hedge":
        return {
            "mode":                    data.get("mode"),
            "rec_instrument":          data.get("instrument"),
            "rec_allocation_pct":      data.get("allocation_pct"),
            "sharia_cleared":          data.get("sharia_cleared"),
            "effective_instrument":    data.get("effective_instrument"),
            "effective_allocation_pct":data.get("effective_allocation_pct"),
            "sub_tag":                 data.get("sub_tag"),
            "gate_action":             data.get("gate_action"),
            "regime_quadrant":         data.get("regime_quadrant"),
            "crisis_tier":             data.get("crisis_tier"),
            "reason":                  data.get("reason"),
        }
    if name == "treasury_stress":
        return {
            "verdict":          data.get("verdict"),
            "delta_avg_sharpe": data.get("delta_avg_sharpe"),
            "n_rescued":        data.get("n_rescued"),
            "n_regressed":      data.get("n_regressed"),
            "note":             data.get("note"),
        }
    # Default: take first ~5 top-level keys
    return {k: data[k] for k in list(data.keys())[:6]}


def build_dossier(topic: str | None = None) -> dict:
    ps = _load("pipeline_state.json")
    pf = ps.get("portfolio", {})
    rg = ps.get("regime", {})
    cm = ps.get("committee", {})

    base = {
        "run_date":         ps.get("run_date"),
        "pipeline_status":  ps.get("pipeline_status"),
        "portfolio_value":  pf.get("portfolio_value"),
        "cash_usd":         pf.get("cash_usd"),
        "gold_oz":          pf.get("gold_oz"),
        "hmm_regime":       rg.get("hmm_state"),
        "today_action":     cm.get("action_taken"),
    }
    engines = {}
    for name, fname in ENGINE_FILES.items():
        proj = _project(name, _load(fname))
        if proj:
            engines[name] = proj

    # Topic filter — keep only relevant engines
    topic_map = {
        "risk":     {"vol_surface", "dcc_garch", "mean_cvar", "tail_hedge",
                     "structural_breaks", "capacity"},
        "signals":  {"alpha_attribution", "signal_decay", "ic_ir", "bma",
                     "ensemble_stacking", "decision_quality", "conformal"},
        "macro":    {"macro_nowcast", "macro_regime", "geopolitical",
                     "cb_speech", "etf_flows", "news_sentiment"},
        "performance": {"brinson", "fama_french", "alpha_attribution"},
        "execution":   {"smart_order_router", "adverse_selection",
                        "stop_loss", "capacity"},
        "governance":  {"mrm_champion", "audit_trail", "dr_backup",
                        "strategy_sandbox", "latency"},
        "derivatives": {"options", "tail_hedge", "carry", "term_structure"},
        # New topic slots
        "regime":   {"macro_regime", "macro_nowcast", "structural_breaks",
                     "dcc_garch", "strategy_selector", "alpha_stacker",
                     "geopolitical", "vol_surface"},
        "phase14":  {"alpha_stacker", "strategy_selector", "multi_strategy",
                     "strategy_backtest", "conviction_weights",
                     "ml_conviction", "ml_walk_forward"},
        "treasury": {"treasury_hedge", "treasury_stress", "macro_regime",
                     "multi_strategy"},
    }
    if topic and topic.lower() in topic_map:
        keys = topic_map[topic.lower()]
        engines = {k: v for k, v in engines.items() if k in keys}

    return {"base": base, "engines": engines}


# ---------------------------------------------------------------------------
# DeepSeek call
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the senior portfolio advisor for a private,
Sharia-compliant trading operation built around:
  - Physical gold/silver core holdings (UAE operator)
  - A halal equity universe (AAOIFI debt/revenue screens)
  - A multi-strategy paper book (TREND / MEAN_REV / PAIRS / VOL_SHORT /
    TAIL_HEDGE / CASH rotation)
  - A Treasury Hedge Sleeve (TLT/IEF when the Sharia gate is cleared, GLD
    fallback when it is not; execution is permanently paper_internal — no
    live broker)

You are talking to the owner — a smart, busy non-quant. Your job is to take
whatever the engines are saying and turn it into advice they can actually use.

VOICE
- Conversational but professional, like a trusted human advisor talking to a
  friend over coffee. First person is good ("I'd hold off on adding here").
  Contractions are good. Jargon is not.
- Be opinionated. Take a side and say why. "Gold took a serious hit this
  month and the regime looks unstable, so I'd stay defensive" — that is the
  register.
- Translate every metric into plain English consequences. Never dump raw
  JSON keys, scores, or engine names at the reader. Instead of
  "crisis_score=0.59, tier=STRESS" say "our crisis dial is deep in the
  stress zone — this is not the moment to be a hero."
- You may quote a specific price or percentage when it sharpens the point —
  one or two per paragraph, no more.

HARD RULES
- Use ONLY facts from the supplied dossier. Never invent numbers, positions,
  or events. If the dossier lacks the data, say so plainly — never bluff.
- When the Treasury Hedge Sleeve comes up, always say in plain words whether
  we are holding actual Treasuries (TLT/IEF, Sharia-cleared) or the
  gold-proxy fallback (GLD), and what the sleeve is protecting us from.
- Never recommend interest-bearing instruments while the Sharia gate is not
  cleared.
- If a risk reading is clearly out of its normal range, call it out and say
  what you'd do about it.
"""

BRIEFING_PROMPT = """Write today's edition of "The QCTF Daily" — the
owner's private morning newsletter. Make it engaging, opinionated, and
genuinely useful. Write these five sections, each starting with its label
on its own line (plain text, no markdown headers):

HEADLINE: One punchy line, max 12 words, capturing today's single most important takeaway.
THE READ: 3-5 sentences. What the market is actually doing, what regime we are in, and why it matters for our money. Narrative, not bullet points.
POSITIONING: 2-4 sentences. What we are holding and what each position is doing for us, in plain English (e.g. "the Treasury hedge is our airbag if this selloff deepens").
WATCHLIST: 2-3 sentences. The specific names worth watching and the trigger that would make us act on them.
THE CALL: 2-4 sentences. Your bottom-line recommendation — what to do, what NOT to do, and the one thing that would change your mind.
"""


# 1-hour TTL: same question + same dossier in the same hour is overwhelmingly
# likely to be the user reopening the chat panel.  Skip the paid API call.
# The cache key includes the dossier dict, so a fresh pipeline_state →
# fresh dossier → fresh cache miss → fresh answer.
@cached(namespace="deepseek", ttl_seconds=3600)
def _call_deepseek(question: str, dossier: dict, mode: str = "qa") -> dict:
    if not DEEPSEEK_API_KEY:
        return {
            "answer":   "(DEEPSEEK_API_KEY not configured — set it in .env to enable explanations)",
            "model":    "stub",
            "cost":     0,
        }
    try:
        import requests
    except ImportError:
        return {
            "answer": "(requests library missing)",
            "model":  "stub",
            "cost":   0,
        }

    payload_text = (
        BRIEFING_PROMPT if mode == "briefing" else
        f"OPERATOR QUESTION:\n{question}\n\n"
        f"Answer the question using only the dossier below."
    )
    user_msg = f"DOSSIER:\n{json.dumps(dossier, indent=2)}\n\n{payload_text}"

    body = {
        "model":       DEEPSEEK_MODEL,
        "messages": [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": user_msg},
        ],
        # Newsletter overhaul: higher temperature for an engaging human voice,
        # bigger budget for the longer narrative format. Facts stay pinned to
        # the dossier by the system prompt's hard rules.
        "temperature": 0.5,
        "max_tokens":  1600,
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type":  "application/json",
    }
    try:
        resp = requests.post(DEEPSEEK_URL, json=body, headers=headers, timeout=45)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return {
            "answer":  content,
            "model":   DEEPSEEK_MODEL,
            "prompt_tokens":     usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens":      usage.get("total_tokens"),
        }
    except Exception as exc:
        return {
            "answer": f"(DeepSeek call failed: {exc})",
            "model":  DEEPSEEK_MODEL,
            "cost":   0,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def explain(question: str, topic: str | None = None) -> dict:
    dossier = build_dossier(topic)
    response = _call_deepseek(question, dossier, mode="qa")
    turn = {
        "ts":        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question":  question,
        "topic":     topic,
        "dossier_keys": list(dossier.get("engines", {}).keys()),
        **response,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LAST_TURN_FILE.write_text(json.dumps(turn, indent=2, default=str))
    return turn


def briefing() -> dict:
    dossier = build_dossier()
    response = _call_deepseek("Generate today's executive briefing.", dossier, mode="briefing")
    turn = {
        "ts":            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind":          "briefing",
        "dossier_keys":  list(dossier.get("engines", {}).keys()),
        **response,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LAST_TURN_FILE.write_text(json.dumps(turn, indent=2, default=str))
    return turn


# ---------------------------------------------------------------------------
# Executive summary — "The QCTF Daily" newsletter (live phase14 book)
# ---------------------------------------------------------------------------
DUMB_MODE_CRO_SYSTEM = (
    "You are the senior portfolio advisor behind \"The QCTF Daily\" — the "
    "private morning newsletter for the owner of a Sharia-compliant gold, "
    "silver and halal-equity trading operation. You see every engine of the "
    "fund's quant stack; your job is to turn all of it into a briefing the "
    "owner actually enjoys reading and can act on.\n"
    "\n"
    "VOICE\n"
    "- Write like a trusted human advisor talking to a smart friend: "
    "conversational, direct, professional. Contractions are fine, jargon is "
    "not. First person is encouraged (\"I'd stay defensive here\").\n"
    "- Be opinionated. Take a side and defend it from the data. \"Gold took "
    "a massive hit this month and the regime looks unstable, so I recommend "
    "staying defensive\" — that's the register.\n"
    "- NEVER dump raw metrics, JSON keys, or engine names at the reader. "
    "Translate every number into what it means for their money. Instead of "
    "\"hmm_state=BEARISH p=0.99\" write \"the regime model is about as "
    "bearish as it gets right now.\" Instead of \"tier=STRESS\" write \"our "
    "crisis dial is deep in the stress zone — not the time to be a hero.\"\n"
    "- Give concrete, directive advice: \"hold off on adding gold here\", "
    "\"the Treasury hedge is our airbag if this selloff deepens\", \"I'd "
    "want to see two calm weeks before we redeploy.\"\n"
    "- You may quote a specific price or percentage when it sharpens a "
    "point — at most one or two per section.\n"
    "\n"
    "HARD RULES\n"
    "- Use ONLY facts from the JSON dossier provided. Never invent prices, "
    "positions, tickers, or events. If something is missing, write around "
    "it.\n"
    "- This is a paper/internal book under a Sharia-compliant mandate. When "
    "the Treasury hedge matters, say plainly whether we hold real "
    "Treasuries (TLT/IEF, Sharia-cleared) or the gold-proxy fallback (GLD), "
    "and what it is protecting us from. Never recommend interest-bearing "
    "instruments while the Sharia gate is not cleared.\n"
    "- If trading is halted or operator action is needed, make that "
    "unmissable in THE CALL.\n"
    "\n"
    "FORMAT — write exactly these five sections, each beginning with its "
    "label at the start of its own line (plain text, no markdown, no "
    "bullets). Sections may run multiple sentences and multiple lines:\n"
    "HEADLINE: One punchy line, max 12 words — today's single most important takeaway.\n"
    "THE READ: 3-5 sentences. What gold, silver and the broader regime are actually doing, and why the owner should care. Narrative, opinionated.\n"
    "POSITIONING: 2-4 sentences. What we hold and what each sleeve is doing for us in plain English — which positions are working, which are dead weight, what the hedge is covering.\n"
    "WATCHLIST: 2-3 sentences. The specific tickers worth watching, and the concrete trigger that would make us act on each.\n"
    "THE CALL: 2-4 sentences. The bottom line — what I'd do today, what I'd avoid, and the one thing that would change my mind. If operator intervention is required (halt / authorize pipeline), lead with it.\n"
)

DEFAULT_WATCHLIST = ["NVDA", "XOM", "LIN", "MSFT"]


def _collect_holdings(book: dict, trader: dict, pipeline: dict) -> list[dict]:
    """Merge Phase XIV open trades, physical portfolio.json, and trader book."""
    holdings: list[dict] = []
    seen: set[str] = set()

    for t in book.get("open_trades") or []:
        ticker = str(t.get("ticker") or "").upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        holdings.append({
            "ticker": ticker,
            "side": t.get("side", "LONG"),
            "notional_usd": t.get("notional_usd"),
            "strategy": t.get("strategy"),
            "source": "phase14_book",
        })

    phys = _load("portfolio.json")
    for ticker, row in phys.items():
        if not isinstance(row, dict):
            continue
        tk = str(ticker).upper()
        shares = float(row.get("shares") or 0)
        if shares <= 0 or tk in seen:
            continue
        seen.add(tk)
        holdings.append({
            "ticker": tk,
            "side": "LONG",
            "shares": shares,
            "avg_cost": row.get("avg_cost"),
            "source": "portfolio_json",
        })

    hedge = book.get("hedge_state") or {}
    inst = hedge.get("instrument") or trader.get("hedge_state", {}).get("instrument")
    if inst and str(inst).upper() not in seen:
        holdings.append({
            "ticker": str(inst).upper(),
            "side": "LONG",
            "allocation_pct": hedge.get("allocation_pct") or 20.0,
            "strategy": "TREASURY_HEDGE",
            "source": "hedge_sleeve",
        })

    if not holdings:
        for tk in ("GLD", "GC=F", "SI=F", "IAU"):
            holdings.append({
                "ticker": tk,
                "side": "LONG",
                "notional_usd": 0,
                "strategy": "ALPHA_CORE" if tk != "GLD" else "DEFENSIVE_HEDGE",
                "source": "default_sleeve",
            })

    pf = pipeline.get("portfolio") or {}
    return holdings


def _metals_market_snapshot(pipeline: dict, regime: dict) -> dict:
    pf = pipeline.get("portfolio") or {}
    spot = float(pf.get("last_spot") or 3382.0)
    return {
        "gold_proxy": "GC=F",
        "gold_spot_usd": spot,
        "silver_proxy": "SI=F",
        "silver_spot_usd": round(spot / 92.0, 2),
        "hmm_state": regime.get("hmm_state") or "BULLISH",
        "p_bullish": regime.get("p_bullish"),
        "committee_action": (pipeline.get("committee") or {}).get("action_taken"),
    }


def _watchlist_candidates(pipeline: dict) -> list[str]:
    selector = _load("strategy_selector.json")
    picks = selector.get("top_picks") or selector.get("selections") or []
    out: list[str] = []
    for item in picks:
        if isinstance(item, dict):
            tk = item.get("ticker") or item.get("symbol")
        else:
            tk = str(item)
        if tk:
            out.append(str(tk).upper())
    for tk in DEFAULT_WATCHLIST:
        if tk not in out:
            out.append(tk)
    return out[:4]


def build_live_engine_state() -> dict:
    """Condensed live state from phase14_book.json plus hedge/trader context."""
    book = _load("phase14_book.json")
    if not book:
        return {}

    open_trades = book.get("open_trades") or []
    hedge_state = book.get("hedge_state") or {}
    from scripts.treasury_hedge_overlay import sanitize_hedge_recommendation

    treasury = sanitize_hedge_recommendation(_load("treasury_hedge.json"))
    trader = _load("multi_strategy_trader.json")
    pipeline = _load("pipeline_state.json") or {}
    regime = pipeline.get("regime") or _load("macro_regime.json") or {}

    gross_open = sum(float(t.get("notional_usd") or 0) for t in open_trades)
    holdings = _collect_holdings(book, trader or {}, pipeline)
    watchlist = _watchlist_candidates(pipeline)

    crisis = _load("crisis_detector.json")
    crisis_fm = crisis.get("fast_metrics") or {}
    trader_summary = trader or {}

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metals_market": _metals_market_snapshot(pipeline, regime),
        "risk_dials": {
            "crisis_tier":      crisis.get("tier"),
            "crisis_score":     crisis.get("score"),
            "vol_spike_ratio":  crisis_fm.get("vol_spike_ratio"),
            "rsi_14":           crisis_fm.get("rsi_14"),
            "macd_hist_pct":    crisis_fm.get("macd_hist_pct"),
            "drift_10d_pct":    crisis_fm.get("drift_10d_pct"),
            "vol_breaker":      trader_summary.get("vol_breaker"),
        },
        "current_holdings": holdings,
        "watchlist": watchlist,
        "phase14_book": {
            "cash_usd":           book.get("cash_usd"),
            "starting_capital":   book.get("starting_capital"),
            "last_strategy":      book.get("last_strategy"),
            "last_run":           book.get("last_run"),
            "n_runs":             book.get("n_runs"),
            "open_trades_count":  len(open_trades),
            "open_gross_notional_usd": round(gross_open, 2),
            "open_positions": holdings[:12],
            "n_closed_trades": len(book.get("closed_trades") or []),
            "hedge_state":   hedge_state,
        },
        "trader_summary": {
            "book_equity_usd": trader.get("book_equity_usd"),
            "realized_pl_usd": trader.get("realized_pl_usd"),
            "open_pl_usd":     trader.get("open_pl_usd"),
            "by_strategy":     trader.get("by_strategy"),
        } if trader else {},
        "treasury_hedge": _project("treasury_hedge", treasury) if treasury else {},
        "trading_halted": (DATA_DIR / "trading_halted.flag").exists(),
        "execution_mode": os.getenv("EXECUTION_MODE", "paper_internal"),
    }


def _format_holdings_line(holdings: list[dict]) -> str:
    if not holdings:
        return "No open positions — book is in cash posture."
    parts: list[str] = []
    for h in holdings[:6]:
        tk = h.get("ticker", "—")
        side = h.get("side", "LONG")
        if h.get("notional_usd"):
            parts.append(f"{side} {tk} ${float(h['notional_usd']):,.0f}")
        elif h.get("shares"):
            parts.append(f"{tk} {float(h['shares']):.4f} sh")
        else:
            parts.append(f"{side} {tk}")
    return "Held: " + ", ".join(parts) + "."


# Newsletter labels → payload keys. Legacy labels (MARKET/HOLDINGS/ACTION)
# are still recognised so older cached generations and prompt drift degrade
# gracefully. Longest labels first so "THE READ" wins over a stray "READ:".
_SECTION_LABELS: tuple[tuple[str, str], ...] = (
    ("HEADLINE",    "headline"),
    ("THE READ",    "market"),
    ("POSITIONING", "holdings"),
    ("WATCHLIST",   "watchlist"),
    ("THE CALL",    "action"),
    # Legacy four-line format
    ("MARKET",      "market"),
    ("HOLDINGS",    "holdings"),
    ("ACTION",      "action"),
)


def _parse_labeled_summary(text: str) -> dict[str, str]:
    """Extract newsletter sections from model output.

    A label line ("THE READ: ...") opens a section; every following line
    belongs to it until the next label. Sections can therefore run multiple
    sentences/lines — required for the newsletter format.
    """
    out: dict[str, str] = {}
    current_key: str | None = None
    buffer: list[str] = []

    def _flush() -> None:
        nonlocal buffer, current_key
        if current_key and buffer:
            joined = " ".join(part for part in buffer if part).strip()
            if joined:
                # First label wins (e.g. THE READ over a later legacy MARKET)
                out.setdefault(current_key, joined)
        buffer = []

    for line in text.splitlines():
        stripped = line.strip().lstrip("#*- ").strip()
        if not stripped:
            continue
        upper = stripped.upper()
        matched = False
        for label, key in _SECTION_LABELS:
            prefix = f"{label}:"
            if upper.startswith(prefix):
                _flush()
                current_key = key
                buffer = [stripped[len(prefix):].strip()]
                matched = True
                break
        if not matched and current_key:
            buffer.append(stripped)
    _flush()
    return out


_HMM_PLAIN = {
    "BULLISH":  "the regime model still leans constructive",
    "BEARISH":  "the regime model is firmly in risk-off territory",
    "VOLATILE": "the regime model says conditions are choppy and unstable",
}


def _offline_executive_summary(state: dict, reason: str = "") -> dict[str, str]:
    """Deterministic newsletter-shaped fallback when DeepSeek is unavailable."""
    metals = state.get("metals_market") or {}
    holdings = state.get("current_holdings") or []
    watchlist = state.get("watchlist") or DEFAULT_WATCHLIST[:2]

    gold = float(metals.get("gold_spot_usd") or 3382.0)
    silver = float(metals.get("silver_spot_usd") or 36.7)
    hmm = (metals.get("hmm_state") or "BULLISH").upper()
    hmm_plain = _HMM_PLAIN.get(hmm, "the regime model is undecided")

    headline = (
        "Defensive posture holds while the regime sorts itself out"
        if hmm != "BULLISH"
        else "Steady as she goes — metals bid intact"
    )
    market = (
        f"Gold is trading around ${gold:,.0f}/oz with silver near ${silver:.2f}/oz, "
        f"and {hmm_plain}. This is the automated fallback note, so treat it as a "
        f"status check rather than a full read — the live advisor commentary "
        f"will resume on the next cycle."
    )
    holdings_line = _format_holdings_line(holdings)
    if holdings_line.startswith("Held:"):
        holdings_line += (
            " Each sleeve is marked to market on every refresh; the hedge "
            "sleeve, when active, is there to cushion regime shocks."
        )
    watch = (
        f"Keep {watchlist[0]} and {watchlist[1]} on the radar for the equity "
        f"sleeve — any add still has to clear the Sharia screen and a "
        f"high-conviction signal before money moves."
    )

    if state.get("trading_halted"):
        action = (
            "HALT is active — nothing trades until you clear the override. "
            "Review the dashboard panels, then authorize the pipeline when "
            "you're comfortable."
        )
    elif reason:
        action = (
            f"The live briefing is offline ({reason}), so lean on the "
            f"dashboard panels for risk until it's back. No portfolio action "
            f"is required from you right now."
        )
    else:
        action = (
            "No intervention needed. Authorize the pipeline when you're "
            "ready and the system will keep running in paper mode."
        )

    sections = {
        "headline": headline,
        "market": market,
        "holdings": holdings_line,
        "watchlist": watch,
        "action": action,
    }
    sections["summary"] = (
        f"HEADLINE: {headline}\n"
        f"THE READ: {market}\n"
        f"POSITIONING: {holdings_line}\n"
        f"WATCHLIST: {watch}\n"
        f"THE CALL: {action}"
    )
    return sections


def _call_deepseek_executive_summary(state: dict) -> tuple[str | None, str | None]:
    """Call DeepSeek with the Dumb Mode CRO prompt. Returns (text, error_kind)."""
    if not DEEPSEEK_API_KEY:
        return None, "no_api_key"
    try:
        import requests
    except ImportError:
        return None, "missing_requests"

    user_msg = (
        "LIVE ENGINE STATE (JSON):\n"
        f"{json.dumps(state, indent=2, default=str)}\n\n"
        "Write today's edition of The QCTF Daily using the HEADLINE / "
        "THE READ / POSITIONING / WATCHLIST / THE CALL sections."
    )
    body = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": DUMB_MODE_CRO_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        # Newsletter voice needs room to breathe: bigger budget + warmer
        # temperature than the old 4-line CRO format. Facts remain pinned to
        # the dossier by the system prompt's hard rules.
        "temperature": 0.55,
        "max_tokens":  1500,
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type":  "application/json",
    }
    try:
        resp = requests.post(DEEPSEEK_URL, json=body, headers=headers, timeout=75)
        if resp.status_code == 429:
            return None, "rate_limited"
        resp.raise_for_status()
        data = resp.json()
        content = (data["choices"][0]["message"]["content"] or "").strip()
        return (content or None), None
    except requests.exceptions.Timeout:
        return None, "timeout"
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 429:
            return None, "rate_limited"
        return None, "http_error"
    except Exception:
        return None, "error"


def executive_summary_dumb_mode() -> dict:
    """Live CRO summary from phase14_book.json via DeepSeek, with offline fallback."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state = build_live_engine_state()
    if not state:
        sections = _offline_executive_summary(
            {"phase14_book": {}, "watchlist": DEFAULT_WATCHLIST[:2]},
            reason="phase14_book.json not found",
        )
        return {
            **sections,
            "generated_at": now,
            "offline": True,
            "reason": "no_state",
        }

    text, err = _call_deepseek_executive_summary(state)
    if text and not text.startswith("("):
        parsed = _parse_labeled_summary(text)
        if parsed:
            return {
                "summary": text.strip(),
                "headline": parsed.get("headline", ""),
                "market": parsed.get("market", ""),
                "holdings": parsed.get("holdings", ""),
                "watchlist": parsed.get("watchlist", ""),
                "action": parsed.get("action", ""),
                "generated_at": now,
                "offline": False,
            }
        return {
            "summary": text.strip(),
            "generated_at": now,
            "offline": False,
        }

    reason_map = {
        "no_api_key":      "DEEPSEEK_API_KEY not configured",
        "rate_limited":    "DeepSeek rate limit",
        "timeout":         "DeepSeek request timed out",
        "http_error":      "DeepSeek HTTP error",
        "missing_requests": "requests library missing",
        "error":           "DeepSeek call failed",
    }
    reason = reason_map.get(err or "error", "DeepSeek unavailable")
    sections = _offline_executive_summary(state, reason=reason)
    return {
        **sections,
        "generated_at": now,
        "offline": True,
        "reason": err,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek Explainer")
    parser.add_argument("question", nargs="?", default=None,
                        help="Natural-language question; omit with --briefing")
    parser.add_argument("--topic", default=None,
                        help="Filter dossier to a topic: risk / signals / macro / performance / execution / governance / derivatives")
    parser.add_argument("--briefing", action="store_true",
                        help="Produce 4-paragraph executive briefing")
    parser.add_argument("--dossier-only", action="store_true",
                        help="Print the dossier that would be sent and exit")
    args = parser.parse_args()

    print(f"\n{SEP}\n  DEEPSEEK EXPLAINER\n{SEP}")

    if args.dossier_only:
        d = build_dossier(args.topic)
        print(json.dumps(d, indent=2, default=str))
        return

    if args.briefing:
        result = briefing()
        print(f"  Mode: BRIEFING")
        print(f"  Engines in dossier: {len(result.get('dossier_keys', []))}")
    else:
        if not args.question:
            print("  Provide a question or use --briefing.")
            return
        result = explain(args.question, topic=args.topic)
        print(f"  Question:           {args.question}")
        if args.topic:
            print(f"  Topic:              {args.topic}")
        print(f"  Engines in dossier: {len(result.get('dossier_keys', []))}")

    print(f"  Model:              {result.get('model')}")
    if result.get("total_tokens"):
        print(f"  Tokens:             {result.get('total_tokens')}")
    print()
    print(f"  ANSWER")
    print(f"  {'─' * 58}")
    print(result.get("answer", "(no answer)"))
    print()
    print(SEP)


if __name__ == "__main__":
    main()
