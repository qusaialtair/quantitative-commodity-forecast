#!/usr/bin/env python3
"""
Alpha Stacker  (Phase XIV Stage 72)
====================================
Meta-engine that consumes the JSON outputs of every signal-producing engine
in the system and fuses them into one institutional-grade decision:

    direction         : BUY / SELL / HOLD
    conviction_score  : continuous in [-1, +1]
    conviction_tier   : VERY_LOW / LOW / MEDIUM / HIGH / VERY_HIGH
    recommended_size  : 0 - 100 % of risk budget
    top_drivers       : signals reinforcing the direction
    top_detractors    : signals fighting the direction

The math is deliberately transparent:

    For each signal source k we extract a triple
        (d_k, q_k, w_k)
    where
        d_k  ∈ [-1, +1]   directional contribution
        q_k  ≥ 0           quality (Sharpe / IR / hit-rate proxy)
        w_k  ≥ 0           posterior weight (BMA / IR / 1/N fallback)

    Combined conviction:
        C = Σ_k w_k · d_k · tanh(q_k)
            ──────────────────────────
                       Σ_k w_k

    Tier thresholds:
        |C| < 0.10  →  VERY_LOW   (CASH)
        |C| < 0.25  →  LOW
        |C| < 0.45  →  MEDIUM
        |C| < 0.65  →  HIGH
        |C| ≥ 0.65  →  VERY_HIGH

This stacker DOES NOT veto trades — it just produces conviction.  Veto logic
lives in the strategy selector + risk_manager.

Inputs (every read is best-effort; missing engines silently drop out):
    bma_weights.json, alpha_attribution.json, ic_ir_tracker.json,
    macro_nowcast.json, macro_regime.json, mtf_confluence.json,
    conformal_intervals.json, decision_quality.json,
    ensemble_stacking.json, rl_sizing.json, vol_surface.json,
    signal_decay.json, cointegration_engine.json, pairs_trader.json,
    structural_breaks.json, geopolitical_detector.json, cb_speech.json,
    etf_flow_tracker.json, news_sentiment.json, tail_risk_engine.json,
    drawdown_controller.json, carry_analyzer.json, term_structure.json,
    current_regime.json, decision_log.json

Output: data/alpha_stacker.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "alpha_stacker.json"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

# In-process memoization keyed on (path, mtime).  Engine JSON files change
# only when their producer engine re-runs (once per pipeline run, typically
# once a day).  Re-parsing them on every stacker call wastes wall-clock and
# disk I/O.  The cache invalidates automatically when the file's mtime
# changes — no manual invalidation needed.
_JSON_CACHE: dict[str, tuple[float, dict]] = {}


def _load(name: str) -> dict:
    p = DATA_DIR / name
    if not p.exists():
        return {}
    try:
        mtime = p.stat().st_mtime
        cached = _JSON_CACHE.get(name)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        data = json.loads(p.read_text())
        _JSON_CACHE[name] = (mtime, data)
        return data
    except Exception:
        return {}


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except Exception:
        return default


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _sigmoid_q(q: float) -> float:
    """Quality squashed to (0, 1) via tanh so big Sharpes don't dominate."""
    return float(0.5 + 0.5 * math.tanh(_safe_float(q)))


# ──────────────────────────────────────────────────────────────────────────────
# Signal extractors
# ──────────────────────────────────────────────────────────────────────────────
def _extract_bma() -> list[dict]:
    """
    Bayesian posterior weights over alpha sources.  Each source becomes one
    signal with d = sign(2*hit-1), q = (hit-0.5)*5 (so 0.6 hit → q=0.5).
    """
    bma = _load("bma_weights.json")
    if not bma:
        return []

    per = bma.get("per_source", [])
    sigs = []
    for s in per:
        name = s.get("source")
        hit = _safe_float(s.get("hit_rate"), 0.5)
        w = _safe_float(s.get("weight"), 0.0)
        if not name or w <= 0:
            continue
        d = _clamp((hit - 0.5) * 2.0)
        q = abs(hit - 0.5) * 5.0  # 0.6 → 0.5, 0.7 → 1.0
        sigs.append({
            "name":   f"bma:{name}",
            "family": "ensemble",
            "d":      d,
            "q":      q,
            "w":      w,
            "weight_basis": "posterior",
            "note":   f"hit={hit:.3f}",
        })
    return sigs


def _extract_alpha_attribution() -> list[dict]:
    """Per-source full-history Sharpe; IR vs equal-weight."""
    aa = _load("alpha_attribution.json")
    if not aa:
        return []

    full = aa.get("full_history", {}) or {}
    irs = aa.get("information_ratios", {}) or {}
    sigs = []
    for name, m in full.items():
        sharpe = _safe_float(m.get("sharpe"))
        ann_ret = _safe_float(m.get("ann_return_pct"))
        ir = _safe_float((irs.get(name) or {}).get("information_ratio"))
        if sharpe == 0 and ir == 0:
            continue
        d = _clamp(0.6 * math.tanh(sharpe) + 0.4 * math.tanh(ir))
        q = max(abs(sharpe), abs(ir))
        sigs.append({
            "name":   f"alpha:{name}",
            "family": "directional",
            "d":      d,
            "q":      q,
            "w":      max(abs(ir), 0.1),
            "weight_basis": "ir",
            "note":   f"Sharpe={sharpe:+.2f} IR={ir:+.2f}",
        })
    return sigs


def _extract_ic_ir() -> list[dict]:
    """IC/IR tracker — deployable signals only."""
    ii = _load("ic_ir_tracker.json")
    if not ii:
        return []

    per = ii.get("per_signal", {}) or {}
    sigs = []
    for name, m in per.items():
        ic63 = _safe_float(m.get("ic_63d"))
        ir63 = _safe_float(m.get("ir_63d"))
        deployable = bool(m.get("deployable", False))
        d = _clamp(math.tanh(ic63 * 8.0))  # IC tiny so scale up
        q = abs(ir63)
        w = 0.8 if deployable else 0.2
        if abs(ic63) < 1e-4 and abs(ir63) < 1e-4:
            continue
        sigs.append({
            "name":   f"icir:{name}",
            "family": "directional",
            "d":      d,
            "q":      q,
            "w":      w,
            "weight_basis": "deployable",
            "note":   f"IC63={ic63:+.4f} IR63={ir63:+.2f} deploy={deployable}",
        })
    return sigs


def _extract_mtf() -> list[dict]:
    mtf = _load("metals_pipeline.json") or _load("mtf_confluence.json")
    # Pull from current_regime.json fallback when canonical MTF doesn't exist
    if not mtf:
        return []
    conf = mtf.get("confluence") or mtf
    score = _safe_float(conf.get("score"))
    if score == 0:
        return []
    d = _clamp(score / 100.0)
    q = abs(d) * 1.5
    return [{
        "name":   "mtf:confluence",
        "family": "directional",
        "d":      d,
        "q":      q,
        "w":      0.7,
        "weight_basis": "static",
        "note":   f"score={score:+.0f}/100  level={conf.get('level')}",
    }]


def _extract_conformal() -> list[dict]:
    co = _load("conformal_intervals.json")
    if not co:
        return []
    fc = _safe_float(co.get("latest_forecast_pct"))
    a05 = co.get("intervals", {}).get("alpha_05", {}) or {}
    width = _safe_float(a05.get("interval_width_pct"))
    if fc == 0 and width == 0:
        return []
    # Narrow intervals → high quality.  Use 1/width as quality proxy.
    q = 1.0 / max(width, 0.5)
    d = _clamp(fc / 2.0)  # 2% forecast → d=1
    return [{
        "name":   "conformal:forecast",
        "family": "directional",
        "d":      d,
        "q":      q,
        "w":      0.6,
        "weight_basis": "interval_tightness",
        "note":   f"forecast={fc:+.2f}%  95%-width={width:.2f}%",
    }]


def _extract_macro_nowcast() -> list[dict]:
    mn = _load("macro_nowcast.json")
    if not mn:
        return []
    score = _safe_float(mn.get("composite_score"))
    regime = (mn.get("regime") or "").upper()
    d = _clamp(score)
    q = abs(score) * 1.5
    return [{
        "name":   "macro:nowcast",
        "family": "macro",
        "d":      d,
        "q":      q,
        "w":      0.8,
        "weight_basis": "macro_high",
        "note":   f"regime={regime} score={score:+.3f}",
    }]


def _extract_macro_regime() -> list[dict]:
    mr = _load("macro_regime.json")
    if not mr:
        return []
    tilts = mr.get("asset_tilts") or {}
    gold_tilt = _safe_float(tilts.get("gold"))
    conf = _safe_float(mr.get("confidence"), 0.5)
    quadrant = mr.get("quadrant")
    if gold_tilt == 0:
        return []
    d = _clamp(gold_tilt / 1.5)
    q = conf
    return [{
        "name":   "macro:quadrant",
        "family": "macro",
        "d":      d,
        "q":      q,
        "w":      0.7,
        "weight_basis": "regime_confidence",
        "note":   f"{quadrant} gold_tilt={gold_tilt:+.1f}",
    }]


def _extract_cb_speech() -> list[dict]:
    cb = _load("cb_speech.json")
    if not cb:
        return []
    fed = _safe_float(cb.get("fed_latest"))
    regime = (cb.get("fed_regime") or "").upper()
    if fed == 0:
        return []
    # Hawkish (positive fed score) is bearish gold → invert
    d = _clamp(-fed)
    q = abs(fed)
    return [{
        "name":   "macro:cb_speech",
        "family": "macro",
        "d":      d,
        "q":      q,
        "w":      0.5,
        "weight_basis": "static",
        "note":   f"fed={fed:+.2f} regime={regime}",
    }]


def _extract_geopolitical() -> list[dict]:
    ge = _load("geopolitical_detector.json")
    if not ge:
        return []
    score = _safe_float(ge.get("current_score"))
    regime = (ge.get("regime") or "").upper()
    if score == 0:
        return []
    # High geo risk is bullish gold
    d = _clamp(score / 5.0)
    q = abs(d)
    return [{
        "name":   "macro:geo",
        "family": "macro",
        "d":      d,
        "q":      q,
        "w":      0.5,
        "weight_basis": "static",
        "note":   f"{regime} score={score:.2f}",
    }]


def _extract_etf_flows() -> list[dict]:
    ef = _load("etf_flow_tracker.json")
    if not ef:
        return []
    gold = ef.get("gold_bucket", {}) or {}
    flow7d = _safe_float(gold.get("flow_7d_usd"))
    if flow7d == 0:
        return []
    # >$1B = strong inflow.  Normalize.
    d = _clamp(flow7d / 2e9)
    q = abs(d)
    return [{
        "name":   "flow:gold_etf",
        "family": "positioning",
        "d":      d,
        "q":      q,
        "w":      0.4,
        "weight_basis": "static",
        "note":   f"7d=${flow7d/1e9:+.2f}B",
    }]


def _extract_news_sentiment() -> list[dict]:
    ns = _load("news_sentiment.json")
    if not ns:
        return []
    agg = ns.get("aggregate") or {}
    avg = _safe_float(agg.get("avg_sentiment"))
    disp = _safe_float(agg.get("dispersion"))
    if avg == 0:
        return []
    d = _clamp(avg * 2)
    q = max(0.1, 1.0 - disp)  # high dispersion → low quality
    return [{
        "name":   "sentiment:news",
        "family": "sentiment",
        "d":      d,
        "q":      q,
        "w":      0.4,
        "weight_basis": "static",
        "note":   f"avg={avg:+.3f} dispersion={disp:.3f}",
    }]


def _extract_cointegration() -> list[dict]:
    co = _load("cointegration_engine.json")
    if not co:
        return []
    sigs = []
    for s in co.get("actionable_signals", []) or []:
        z = _safe_float(s.get("z_score"))
        halflife = _safe_float(s.get("half_life_days"), 99)
        sig_name = s.get("signal", "")
        if z == 0:
            continue
        d = _clamp(-z / 3.0)  # high positive z → short
        # Map name e.g. "LONG_SPREAD" → keep sign
        if "LONG" in sig_name and d < 0:
            d = abs(d)
        elif "SHORT" in sig_name and d > 0:
            d = -abs(d)
        q = max(0.1, 1.0 - halflife / 60.0)
        sigs.append({
            "name":   f"pair:{s.get('name', '?')}",
            "family": "stat_arb",
            "d":      d,
            "q":      q,
            "w":      0.5,
            "weight_basis": "actionable",
            "note":   f"z={z:+.2f} ½={halflife:.0f}d {sig_name}",
        })
    return sigs


def _extract_carry() -> list[dict]:
    ca = _load("carry_analyzer.json")
    if not ca:
        return []
    carry = ca.get("carry", {}) or {}
    fair = _safe_float(carry.get("fair_pct"))
    burden = (carry.get("burden") or "").upper()
    excess = (ca.get("excess_vs_carry") or {}).get("21d_pct")
    excess = _safe_float(excess)
    if excess == 0:
        return []
    d = _clamp(excess / 5.0)
    q = 0.5
    return [{
        "name":   "carry:excess",
        "family": "carry",
        "d":      d,
        "q":      q,
        "w":      0.3,
        "weight_basis": "static",
        "note":   f"fair={fair:+.2f}% 21d-excess={excess:+.2f}% {burden}",
    }]


def _extract_term_structure() -> list[dict]:
    ts = _load("term_structure.json")
    if not ts:
        return []
    slope = _safe_float(ts.get("overall_slope_pct"))
    roll = _safe_float(ts.get("roll_yield_pct"))
    shape = (ts.get("curve_shape") or "").upper()
    if roll == 0 and slope == 0:
        return []
    d = _clamp(roll / 3.0)
    q = abs(d)
    return [{
        "name":   "carry:term_structure",
        "family": "carry",
        "d":      d,
        "q":      q,
        "w":      0.3,
        "weight_basis": "static",
        "note":   f"{shape} slope={slope:+.2f}% roll={roll:+.2f}%",
    }]


def _extract_ensemble_stacking() -> list[dict]:
    es = _load("ensemble_stacking.json")
    if not es:
        return []
    meta = es.get("meta_metrics") or {}
    auc = _safe_float(meta.get("auc"), 0.5)
    prob_up = _safe_float((es.get("latest_prediction") or {}).get("prob_up"))
    if prob_up == 0:
        # fallback: signal from auc only
        return []
    d = _clamp((prob_up - 0.5) * 2.0)
    q = max(0.0, (auc - 0.5) * 4.0)
    return [{
        "name":   "ensemble:stack",
        "family": "ensemble",
        "d":      d,
        "q":      q,
        "w":      0.6,
        "weight_basis": "auc",
        "note":   f"prob_up={prob_up:.3f} AUC={auc:.3f}",
    }]


def _extract_rl_sizing() -> list[dict]:
    rl = _load("rl_sizing.json")
    if not rl:
        return []
    action = _safe_float(rl.get("latest_action"))
    lift = _safe_float(rl.get("test_lift_sharpe"))
    if action == 0:
        return []
    d = _clamp(action)
    q = max(0.1, math.tanh(lift))
    return [{
        "name":   "rl:sizing",
        "family": "ensemble",
        "d":      d,
        "q":      q,
        "w":      0.4,
        "weight_basis": "lift",
        "note":   f"action={action:+.2f} lift={lift:+.3f}",
    }]


def _extract_committee() -> list[dict]:
    """The existing committee (LSTM + macro oracle + CIO) from decision_log."""
    dl = _load("decision_log.json")
    if not dl:
        return []
    qc = _safe_float(dl.get("quant_conviction"))
    mc = _safe_float(dl.get("macro_conviction"))
    sigs = []
    if qc != 0:
        sigs.append({
            "name":   "committee:quant",
            "family": "directional",
            "d":      _clamp(qc / 5.0),
            "q":      0.7,
            "w":      0.9,
            "weight_basis": "committee",
            "note":   f"conviction={qc:+.0f}/10",
        })
    if mc != 0:
        sigs.append({
            "name":   "committee:macro",
            "family": "macro",
            "d":      _clamp(mc / 5.0),
            "q":      0.7,
            "w":      0.9,
            "weight_basis": "committee",
            "note":   f"conviction={mc:+.0f}/10",
        })
    return sigs


# ──────────────────────────────────────────────────────────────────────────────
# Combining
# ──────────────────────────────────────────────────────────────────────────────
def _conviction_tier(c: float) -> str:
    a = abs(c)
    if a < 0.10:
        return "VERY_LOW"
    if a < 0.25:
        return "LOW"
    if a < 0.45:
        return "MEDIUM"
    if a < 0.65:
        return "HIGH"
    return "VERY_HIGH"


def _direction(c: float, tier: str) -> str:
    if tier == "VERY_LOW":
        return "HOLD"
    return "BUY" if c > 0 else "SELL"


def _stack(signals: list[dict]) -> dict:
    """
    Pure-weight stacked conviction:
        C = Σ w_k · d_k · sigmoid(q_k)  /  Σ w_k
    """
    if not signals:
        return {
            "conviction_score": 0.0,
            "weighted_sum":     0.0,
            "total_weight":     0.0,
            "n_signals":        0,
        }

    weighted = 0.0
    wsum = 0.0
    for s in signals:
        d = _clamp(_safe_float(s.get("d")))
        q = _safe_float(s.get("q"))
        w = max(_safe_float(s.get("w")), 0.0)
        contribution = w * d * _sigmoid_q(q)
        s["contribution"] = round(contribution, 6)
        weighted += contribution
        wsum += w
    score = weighted / wsum if wsum > 1e-12 else 0.0
    return {
        "conviction_score": round(_clamp(score), 6),
        "weighted_sum":     round(weighted, 6),
        "total_weight":     round(wsum, 6),
        "n_signals":        len(signals),
    }


def _family_view(signals: list[dict]) -> dict:
    """Per-family aggregate conviction so the UI can show one bar per family."""
    out = {}
    for s in signals:
        fam = s.get("family", "other")
        d = _clamp(_safe_float(s.get("d")))
        q = _safe_float(s.get("q"))
        w = max(_safe_float(s.get("w")), 0.0)
        e = out.setdefault(fam, {"weighted": 0.0, "wsum": 0.0, "n": 0})
        e["weighted"] += w * d * _sigmoid_q(q)
        e["wsum"]    += w
        e["n"]       += 1
    return {
        fam: {
            "conviction": round(v["weighted"] / v["wsum"], 4) if v["wsum"] > 1e-12 else 0.0,
            "n_signals":  v["n"],
            "weight_share": round(v["wsum"], 3),
        }
        for fam, v in out.items()
    }


def _risk_flags() -> list[str]:
    flags = []
    dd = _load("drawdown_controller.json")
    tier = (dd.get("tier_name") or dd.get("tier") or "").upper()
    if tier in ("ELEVATED", "SEVERE", "CRITICAL"):
        flags.append(f"DRAWDOWN_{tier}")

    sb = _load("structural_breaks.json")
    if (sb.get("summary") or {}).get("cusum_break"):
        flags.append("CUSUM_BREAK")

    eg = _load("economic_calendar.json")
    if eg.get("blocked_today"):
        flags.append("ECON_BLACKOUT")

    er = _load("earnings_calendar.json")
    if er.get("n_blocked"):
        flags.append("EARNINGS_BLACKOUT")

    dq = _load("data_quality.json")
    if (dq.get("overall_status") or "").upper() == "FAIL":
        flags.append("DATA_QUALITY_FAIL")

    tr = _load("tail_risk_engine.json")
    premium = _safe_float((tr.get("tail_risk") or {}).get("tail_fatness_premium_pct"))
    if premium > 100:
        flags.append(f"FAT_TAIL_{premium:.0f}%")
    return flags


def _recommended_size_pct(score: float, tier: str) -> float:
    """
    Maps conviction → recommended size of risk-budget (0 - 100 %).
        VERY_LOW  → 0
        LOW       → 15 - 25
        MEDIUM    → 25 - 50
        HIGH      → 50 - 80
        VERY_HIGH → 80 - 100
    """
    a = abs(score)
    if tier == "VERY_LOW":
        return 0.0
    if tier == "LOW":
        return 15 + 40 * (a - 0.10) / 0.15
    if tier == "MEDIUM":
        return 25 + 50 * (a - 0.25) / 0.20
    if tier == "HIGH":
        return 50 + 30 * (a - 0.45) / 0.20
    return min(100.0, 80 + 50 * (a - 0.65))


# ──────────────────────────────────────────────────────────────────────────────
# Public entrypoint
# ──────────────────────────────────────────────────────────────────────────────
def run_alpha_stacker() -> dict:
    """Fuse all signal JSONs into conviction and write ``data/alpha_stacker.json``."""
    extractors = [
        _extract_bma,
        _extract_alpha_attribution,
        _extract_ic_ir,
        _extract_mtf,
        _extract_conformal,
        _extract_macro_nowcast,
        _extract_macro_regime,
        _extract_cb_speech,
        _extract_geopolitical,
        _extract_etf_flows,
        _extract_news_sentiment,
        _extract_cointegration,
        _extract_carry,
        _extract_term_structure,
        _extract_ensemble_stacking,
        _extract_rl_sizing,
        _extract_committee,
    ]

    signals: list[dict] = []
    extractor_failures: list[str] = []
    for fn in extractors:
        try:
            sigs = fn() or []
            signals.extend(sigs)
        except Exception as exc:
            extractor_failures.append(f"{fn.__name__}: {exc}")

    stack = _stack(signals)
    score = stack["conviction_score"]
    tier = _conviction_tier(score)
    direction = _direction(score, tier)
    family_view = _family_view(signals)
    flags = _risk_flags()

    # Top drivers / detractors (sorted by signed contribution)
    # Definition: drivers = signals pushing toward current direction (or strongest if HOLD)
    sorted_signals = sorted(signals, key=lambda s: s.get("contribution", 0.0))
    if direction == "BUY" or (direction == "HOLD" and score >= 0):
        drivers = [s for s in reversed(sorted_signals) if s.get("contribution", 0) > 0][:5]
        detractors = [s for s in sorted_signals if s.get("contribution", 0) < 0][:5]
    else:
        drivers = [s for s in sorted_signals if s.get("contribution", 0) < 0][:5]
        detractors = [s for s in reversed(sorted_signals) if s.get("contribution", 0) > 0][:5]

    rec_size_pct = _recommended_size_pct(score, tier)

    out = {
        "schema_version": "1.0",
        "engine":         "alpha_stacker",
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decision": {
            "direction":          direction,
            "conviction_score":   round(score, 4),
            "conviction_tier":    tier,
            "recommended_size_pct": round(rec_size_pct, 2),
        },
        "stack": stack,
        "by_family": family_view,
        "top_drivers": [
            {
                "name":         s["name"],
                "family":       s.get("family"),
                "d":            round(s["d"], 3),
                "q":            round(s["q"], 3),
                "w":            round(s["w"], 3),
                "contribution": round(s["contribution"], 4),
                "note":         s.get("note", ""),
            }
            for s in drivers
        ],
        "top_detractors": [
            {
                "name":         s["name"],
                "family":       s.get("family"),
                "d":            round(s["d"], 3),
                "q":            round(s["q"], 3),
                "w":            round(s["w"], 3),
                "contribution": round(s["contribution"], 4),
                "note":         s.get("note", ""),
            }
            for s in detractors
        ],
        "signals": [
            {
                "name":         s["name"],
                "family":       s.get("family"),
                "d":            round(s["d"], 3),
                "q":            round(s["q"], 3),
                "w":            round(s["w"], 3),
                "contribution": round(s["contribution"], 4),
                "note":         s.get("note", ""),
                "weight_basis": s.get("weight_basis"),
            }
            for s in signals
        ],
        "risk_flags":     flags,
        "n_risk_flags":   len(flags),
        "n_extractor_failures": len(extractor_failures),
        "extractor_failures":   extractor_failures,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(out, indent=2))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    """CLI entrypoint for the alpha stacker."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    out = run_alpha_stacker()
    if args.quiet:
        return 0
    d = out["decision"]
    print("=" * 64)
    print(f"ALPHA STACKER  ({out['generated_at']})")
    print("=" * 64)
    print(f"  Direction       : {d['direction']}")
    print(f"  Conviction      : {d['conviction_score']:+.4f}  [{d['conviction_tier']}]")
    print(f"  Recommended size: {d['recommended_size_pct']:.1f}% of risk budget")
    print(f"  Signals stacked : {out['stack']['n_signals']}")
    print(f"  Risk flags      : {out['n_risk_flags']}  ({', '.join(out['risk_flags']) or 'none'})")
    print()
    print("Top drivers:")
    for s in out["top_drivers"]:
        print(f"  + {s['name']:<28s}  {s['contribution']:+.4f}  ({s['note']})")
    print("Top detractors:")
    for s in out["top_detractors"]:
        print(f"  - {s['name']:<28s}  {s['contribution']:+.4f}  ({s['note']})")
    print()
    print("By family:")
    for fam, fv in out["by_family"].items():
        print(f"  {fam:<14s} conviction={fv['conviction']:+.3f}  "
              f"n={fv['n_signals']}  weight={fv['weight_share']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
