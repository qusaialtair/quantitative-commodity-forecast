#!/usr/bin/env python3
"""
agents/equity_swarm/cio.py
==========================
CIO Synthesizer — combines swarm output (ranker + macro + insider + sentiment)
into a final portfolio using Kelly Criterion sizing.

Two-stage decision:

  (1) Gemini-2.5-Pro reviews each top-50 candidate with all swarm context
      and returns {conviction ∈ [-1,+1], thesis, include}. This is
      qualitative reasoning only — the LLM never touches position sizing.

  (2) Fractional Kelly (deterministic code) turns each candidate's
      (expected_return, volatility, conviction) into a weight:

          μ_i  = E[r_10d]_i
                 · (1 + 0.30·conviction)
                 · (1 + 0.10·sentiment_score)
                 · (1 + 0.05·insider_signal)
          σ_i  = vol_ann · √(10/252)
          f*_i = max(0, μ_i) / σ_i²
          f_i  = min(0.08, 0.25 · f*_i)          # ¼-Kelly, 8% position cap

  (3) Macro regime caps gross exposure:
          RISK_ON  → 1.00   NEUTRAL → 0.65   RISK_OFF → 0.30

  (4) Residual (1 − gross_equity_weight) → SPSK sukuk / cash.
"""
from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent.parent
load_dotenv(ROOT / ".env")
logger = logging.getLogger("CIO")

POSITION_CAP      = 0.08
KELLY_FRACTION    = 0.25
HORIZON_DAYS      = 10
TRADING_DAYS_Y    = 252

# Gemini cascade — cheapest/free-tier-friendly first, escalate on empty response.
# Pro is paywalled on this project's key; flash-lite is the reliable free-tier hop.
GEMINI_CASCADE = ("gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro")

_CIO_SYSTEM = """\
You are the Chief Investment Officer of a Sharia-compliant institutional fund.
Your north-star mandate is CAPITAL PRESERVATION; outperformance is earned only
after the downside is neutered. You will receive:

  · macro.regime  ∈ {RISK_ON, NEUTRAL, RISK_OFF}  ← hard gate on gross exposure
  · macro.hmm_regime + hmm_state_probs            ← quantitative HMM (Crash Shield)
  · per-ticker: ML score, 10d expected return, vol, beta, Sharia flag,
                sentiment (Perplexity), insider signal (Form 4),
                INSIDER ALT-DATA: insider_buy_count_90d, insider_net_usd_90d,
                                  insider_cluster_flag
                CONGRESS ALT-DATA: congress_buy_count_180d,
                                  congress_net_direction_180d,
                                  congress_committee_flag

For EACH ticker return an object:
  {
    "ticker":    str,
    "include":   bool,     # false = exclude from final book entirely
    "conviction": float,   # -1.0 (strong avoid) to +1.0 (strong buy)
    "thesis":    str       # ≤50 words — name the DOMINANT driver explicitly
  }

HARD rules (violations invalidate your output):
  • Reject any ticker with sharia_pass=false. Non-negotiable.
  • If macro.regime == "RISK_OFF": RUTHLESSLY cut the book. Exclude
    (include=false) any ticker with beta_252d > 1.10 OR vol_63d_ann > 0.35
    OR sent_signal in ("SELL","STRONG_SELL"). Only HIGH-QUALITY DEFENSIVES
    survive: healthcare (ex-biotech), consumer staples, utilities,
    dividend-paying mega-caps with low beta. Max conviction in RISK_OFF is +0.60.
  • If macro.regime == "NEUTRAL": prefer quality & low-beta; max conviction +0.75
    unless ALL four corroborators align (ML ≥ 0.70, sentiment BUY, insider>0,
    insider_cluster_flag=1 OR congress_committee_flag=1).
  • Never assign conviction > 0.80 without ≥2 independent positive signals
    (ML, sentiment, insider-classic, insider-alt, congress-alt count as
    independent — macro does NOT).

ALT-DATA CONVICTION BOOST (mandatory weighting):
  • insider_cluster_flag == 1 → +0.25 to base conviction (multiple officers
    buying within 30 days is the strongest single Form-4 signal).
  • congress_committee_flag == 1 AND congress_net_direction_180d > 0 → +0.15
    (committee-jurisdiction politicians accumulating a covered-sector name).
  • congress_net_direction_180d ≥ 3 (≥3 more buys than sells across 180d)
    → +0.10 even without committee flag.
  • insider_net_usd_90d < -50 ($M) (large net selling) → −0.20.
  • Stack these additively after your base reasoning, then clamp to [-1, +1].

Return ONLY a valid JSON object with key "candidates" mapping to a list of
the above. No prose outside the JSON.
"""


def _maybe_call_gemini(context: dict[str, Any]) -> dict[str, dict] | None:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set — CIO falls back to heuristic conviction.")
        return None

    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        logger.warning("google-genai import failed (%s) — using heuristic.", exc)
        return None

    client = genai.Client(api_key=api_key)
    payload = json.dumps(context, default=str, indent=2)
    prompt = (
        _CIO_SYSTEM
        + "\n\n=== SWARM CONTEXT (JSON) ===\n"
        + payload
        + "\n\nReturn only the JSON object now."
    )

    for model in GEMINI_CASCADE:
        try:
            logger.info("Calling Gemini %s  (context %d chars)", model, len(payload))
            resp = client.models.generate_content(
                model=model, contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3, max_output_tokens=6000,
                ),
            )
            text = (resp.text or "").strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.lower().startswith("json"):
                    text = text[4:]
            obj = json.loads(text)
            cands = obj.get("candidates") or obj.get("results") or []
            if not isinstance(cands, list) or not cands:
                raise ValueError("empty candidate list")
            return {c["ticker"]: c for c in cands if "ticker" in c}
        except Exception as exc:
            logger.warning("Gemini %s failed: %s", model, exc)
            continue
    return None


def _alt_data_boost(row: dict) -> float:
    """Mirror of the alt-data conviction boost rules in _CIO_SYSTEM.

    Kept as a deterministic fallback so the heuristic path + the Kelly
    μ-boost stay consistent with what the LLM is instructed to do.
    """
    boost = 0.0
    if int(row.get("insider_cluster_flag", 0) or 0) == 1:
        boost += 0.25
    cong_net = float(row.get("congress_net_direction_180d", 0.0) or 0.0)
    if int(row.get("congress_committee_flag", 0) or 0) == 1 and cong_net > 0:
        boost += 0.15
    elif cong_net >= 3:
        boost += 0.10
    if float(row.get("insider_net_usd_90d", 0.0) or 0.0) < -50.0:
        boost -= 0.20
    return boost


def _heuristic_conviction(row: dict, regime: str | None = None) -> dict:
    """Fallback when no LLM is available — transparent deterministic formula.

    Mirrors the hard rules from _CIO_SYSTEM so the two decision paths stay
    coherent: Sharia gate, RISK_OFF exclusions, alt-data boost, conviction cap.
    """
    ai     = float(row.get("ai_score", 0.0))
    sent   = float(row.get("sent_score", 0.0) or 0.0)
    ins    = float(row.get("insider_signal", 0.0) or 0.0)
    beta   = float(row.get("beta_252d", 1.0) or 1.0)
    vol    = float(row.get("vol_63d_ann", 0.3) or 0.3)
    sent_sig  = str(row.get("sent_signal", "HOLD") or "HOLD").upper()
    sharia_ok = bool(row.get("sharia_pass", True))

    conv = 0.60 * (2*ai - 1) + 0.25 * sent + 0.15 * ins
    conv += _alt_data_boost(row)
    conv = max(-1.0, min(1.0, conv))

    include = bool(sharia_ok and ai >= 0.4)
    # RISK_OFF hard exclusions — mirror the LLM prompt verbatim.
    if regime == "RISK_OFF":
        if beta > 1.10 or vol > 0.35 or sent_sig in ("SELL", "STRONG_SELL"):
            include = False
        conv = min(conv, 0.60)
    elif regime == "NEUTRAL":
        conv = min(conv, 0.75)

    return {
        "ticker": row["ticker"],
        "include": include,
        "conviction": round(conv, 3),
        "thesis": (f"heuristic[{regime or 'n/a'}]: ai={ai:.2f}, sent={sent:+.2f}, "
                   f"insider={ins:+.2f}, alt_boost={_alt_data_boost(row):+.2f}"),
    }


def _kelly_weight(mu_base: float, vol_ann: float, conviction: float,
                  sent: float, ins: float,
                  alt_multiplier: float = 1.0) -> float:
    """Fractional-Kelly μ with conviction + sentiment + classic-insider boost,
    then scaled by `alt_multiplier` for high-signal alt-data alignment.

    alt_multiplier is expected to be ≥ 1.0 (1.25× when insider-cluster or a
    positive committee-flagged congressional buy is present — see synthesize).
    """
    if vol_ann <= 0 or mu_base is None or math.isnan(mu_base) or math.isnan(vol_ann):
        return 0.0
    mu = (mu_base
          * (1 + 0.30 * conviction)
          * (1 + 0.10 * sent)
          * (1 + 0.05 * ins)
          * alt_multiplier)
    sigma_h = vol_ann * math.sqrt(HORIZON_DAYS / TRADING_DAYS_Y)
    var_h = sigma_h ** 2
    if var_h <= 0:
        return 0.0
    f_star = max(0.0, mu) / var_h
    return min(POSITION_CAP, KELLY_FRACTION * f_star)


ALT_COLS = (
    "insider_buy_count_90d", "insider_net_usd_90d", "insider_cluster_flag",
    "congress_buy_count_180d", "congress_net_direction_180d", "congress_committee_flag",
)


def _alt_multiplier(row: dict) -> float:
    """1.25× if either insider-cluster OR positive committee-flagged congress buy;
    1.0× otherwise. This is the Kelly μ-boost gate for validated alt-data alignment.
    """
    insider_cluster = int(row.get("insider_cluster_flag", 0) or 0) == 1
    cong_positive   = (int(row.get("congress_committee_flag", 0) or 0) == 1
                       and float(row.get("congress_net_direction_180d", 0.0) or 0.0) > 0)
    return 1.25 if (insider_cluster or cong_positive) else 1.0


def synthesize(ranking: pd.DataFrame, macro: dict, insider: dict,
               sentiment: dict, top_n: int = 50) -> dict[str, Any]:
    top = ranking.head(top_n).copy()

    # ── Attach swarm fields ──────────────────────────────────────────────────
    top["sent_score"]     = top.index.map(lambda t: (sentiment.get(t) or {}).get("score", 0.0))
    top["sent_signal"]    = top.index.map(lambda t: (sentiment.get(t) or {}).get("signal", "HOLD"))
    top["insider_signal"] = top.index.map(lambda t: (insider.get(t) or {}).get("signal", 0.0))
    top["insider_net_$"]  = top.index.map(lambda t: (insider.get(t) or {}).get("net_dollars_90d", 0.0))

    # Alt-data columns come from equity_features.py (Phase 5 merge). If the
    # ranker frame was built before the merge landed, default to 0 so the LLM
    # and Kelly paths stay robust.
    for col in ALT_COLS:
        if col not in top.columns:
            top[col] = 0

    regime = macro.get("regime")

    # ── Build compact LLM context ────────────────────────────────────────────
    context_rows = []
    for ticker, row in top.iterrows():
        context_rows.append({
            "ticker":     ticker,
            "name":       row.get("name", ticker),
            "sector":     row.get("sector", "Unknown"),
            "ai_score":           round(float(row.get("ai_score", 0.0)), 3),
            "prob_beats_median":  round(float(row.get("prob_beats_median", 0.5)), 3),
            "expected_return_10d": round(float(row.get("expected_return_10d", 0.0)), 4),
            "vol_63d_ann":        round(float(row.get("vol_63d_ann", 0.3)), 3),
            "beta_252d":          round(float(row.get("beta_252d", 1.0) or 1.0), 3),
            "sharia_pass":        bool(row.get("sharia_pass", True)),
            "sent_score":         round(float(row["sent_score"] or 0.0), 3),
            "sent_signal":        row["sent_signal"],
            "insider_signal":     round(float(row["insider_signal"] or 0.0), 3),
            # Phase 5 alt-data — explicit columns so the LLM can cite them.
            "insider_buy_count_90d":        int(row.get("insider_buy_count_90d", 0) or 0),
            "insider_net_usd_90d":          round(float(row.get("insider_net_usd_90d", 0.0) or 0.0), 2),
            "insider_cluster_flag":         int(row.get("insider_cluster_flag", 0) or 0),
            "congress_buy_count_180d":      int(row.get("congress_buy_count_180d", 0) or 0),
            "congress_net_direction_180d":  int(row.get("congress_net_direction_180d", 0) or 0),
            "congress_committee_flag":      int(row.get("congress_committee_flag", 0) or 0),
            "drivers":            row.get("ai_drivers_top3", []),
        })
    llm_ctx = {"macro": macro, "candidates": context_rows}

    cio_out = _maybe_call_gemini(llm_ctx)

    # Merge LLM conviction back onto the frame
    convictions: dict[str, dict] = {}
    for _, row in top.iterrows():
        row_d = {**row.to_dict(), "ticker": row.name}
        if cio_out and row.name in cio_out:
            convictions[row.name] = cio_out[row.name]
        else:
            convictions[row.name] = _heuristic_conviction(row_d, regime=regime)

    # ── Kelly sizing (alt-data gated μ-boost) ────────────────────────────────
    raw: list[dict] = []
    for ticker, row in top.iterrows():
        row_d = row.to_dict()
        conv = convictions[ticker]
        if not conv.get("include", True):
            raw.append({"ticker": ticker, "weight_raw": 0.0,
                        "alt_multiplier": 1.0, **conv, "row": row})
            continue
        alt_mult = _alt_multiplier(row_d)
        w = _kelly_weight(
            mu_base        = float(row.get("expected_return_10d", 0.0) or 0.0),
            vol_ann        = float(row.get("vol_63d_ann", 0.3) or 0.3),
            conviction     = float(conv.get("conviction", 0.0)),
            sent           = float(row.get("sent_score", 0.0) or 0.0),
            ins            = float(row.get("insider_signal", 0.0) or 0.0),
            alt_multiplier = alt_mult,
        )
        raw.append({"ticker": ticker, "weight_raw": w,
                    "alt_multiplier": alt_mult, **conv, "row": row})

    # ── Regime gross-exposure cap + normalisation ────────────────────────────
    gross_cap = float(macro.get("target_gross_exposure", 0.65))
    total_raw = sum(r["weight_raw"] for r in raw)
    if total_raw > gross_cap:
        scale = gross_cap / total_raw
    elif total_raw == 0:
        scale = 0.0
    else:
        scale = 1.0  # already within the cap — no upscaling

    positions: list[dict] = []
    for r in raw:
        w = r["weight_raw"] * scale
        if w < 0.002:
            continue
        row = r["row"]
        positions.append({
            "ticker":     r["ticker"],
            "name":       row.get("name", r["ticker"]),
            "sector":     row.get("sector", "Unknown"),
            "weight":     round(w, 4),
            "conviction": r.get("conviction", 0.0),
            "alt_multiplier": round(float(r.get("alt_multiplier", 1.0)), 3),
            "thesis":     r.get("thesis", "")[:300],
            "ai_score":            round(float(row.get("ai_score", 0.0)), 3),
            "prob_beats_median":   round(float(row.get("prob_beats_median", 0.5)), 3),
            "expected_return_10d": round(float(row.get("expected_return_10d", 0.0)), 4),
            "vol_63d_ann":         round(float(row.get("vol_63d_ann", 0.3)), 3),
            "sent_signal":         row.get("sent_signal", "HOLD"),
            "insider_net_$":       round(float(row.get("insider_net_$", 0.0)), 0),
            # Phase 5 alt-data surfaced for UI + audit.
            "insider_buy_count_90d":       int(row.get("insider_buy_count_90d", 0) or 0),
            "insider_net_usd_90d":         round(float(row.get("insider_net_usd_90d", 0.0) or 0.0), 2),
            "insider_cluster_flag":        int(row.get("insider_cluster_flag", 0) or 0),
            "congress_buy_count_180d":     int(row.get("congress_buy_count_180d", 0) or 0),
            "congress_net_direction_180d": int(row.get("congress_net_direction_180d", 0) or 0),
            "congress_committee_flag":     int(row.get("congress_committee_flag", 0) or 0),
        })

    positions.sort(key=lambda p: -p["weight"])
    equity_gross = round(sum(p["weight"] for p in positions), 4)
    cash_sukuk   = round(max(0.0, 1.0 - equity_gross), 4)

    decision = {
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "horizon_days":   HORIZON_DAYS,
        "macro_regime":   macro.get("regime"),
        "macro_gross_cap": gross_cap,
        "equity_gross":   equity_gross,
        "cash_sukuk":     cash_sukuk,
        "n_positions":    len(positions),
        "used_llm":       bool(cio_out),
        "positions":      positions,
        "macro_snapshot": macro,
    }
    logger.info("CIO decision: %d positions  equity=%.1f%%  sukuk=%.1f%%  regime=%s",
                len(positions), equity_gross * 100, cash_sukuk * 100, macro.get("regime"))
    return decision
