#!/usr/bin/env python3
"""
Crisis Detector  (Phase XVIII Stage 80)
========================================
Pure-price crisis-regime detector.  Emits a continuous score in [0, 1]
plus a discrete tier label that downstream sizing / strategy logic can
gate on.

Motivation
----------
The Phase XVII stress backtest revealed the rule cascade is
REGIME_FRAGILE: it fails the 2008 GFC, 2015 China rout, and 2022
inflation rout windows.  Those three windows share three observable
features in the *price series itself*:

  1. Realised vol sitting at the top of its 5-year distribution
  2. Price deep in drawdown from its trailing-252d high
  3. Trend persistence broken (the sign of the 21-day drift is flipping)

This module fuses those features into a single score so the live
selector and the backtests can use identical detection logic without
needing macro feeds (VIX, real yields, etc.) which aren't available for
historical reconstructions.

Score formula  (Phase XXVII reactivity upgrade — 2026-06)
----------------------------------------------------------
The original five-component blend measured vol acceleration as
vol_21d / vol_63d, which is backward-looking by ~a month: during the
2026-06 gold break it read 0.00 while 5-day realised vol sat at its
98th percentile.  Two faster components were added and the weights
rebalanced so the score ignites within days of a shock instead of
weeks:

    crisis_score =
          0.22 * vol_percentile_rank      # rank of fast vol vs 5y of 21d vols
        + 0.20 * drawdown_intensity       # |dd_from_252d_high| / 25%
        + 0.18 * crash_5d                 # max(0, -5d_return / 10%)
        + 0.18 * fast_vol_spike           # EWMA(λ=0.87, ≈5d half-life) / vol_21d − 1
        + 0.12 * crash_21d                # max(0, -21d_return / 15%)  (month-long routs)
        + 0.10 * trend_reversal_burst     # |Δ sign(drift_21d)|

vol_percentile_rank now ranks max(vol_10d, vol_21d) — asymmetric on
purpose: a fresh spike registers immediately, while a calm-down still
has to earn its way back through the slow window.

Damage gate: the two pure-volatility components (vol_percentile,
fast_vol_spike) are scaled 0.6→1.0 by realised damage
max(drawdown, crash_5d, crash_21d)/0.25.  Volatility alone cannot push
the book past ELEVATED — gold frequently RALLIES on crisis vol
(2011, 2020) and capping longs there gives away the safe-haven upside.
STRESS and CRISIS must be earned by actual price damage.

Each component is clipped to [0, 1]; the weighted sum is also clipped.

Tier mapping
------------

    NORMAL      score < 0.30
    ELEVATED    0.30 <= score < 0.50
    STRESS      0.50 <= score < 0.70
    CRISIS      score >= 0.70

Two public APIs:

    classify_from_prices(closes)   — for live selector / backtest use
    classify_from_engines()        — reads pipeline_state-style JSONs
                                     and fuses with the price detector;
                                     used by `scripts/strategy_selector.py`

Outputs `data/crisis_detector.json` so the UI can render it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "crisis_detector.json"


# ──────────────────────────────────────────────────────────────────────────────
# Tier thresholds
# ──────────────────────────────────────────────────────────────────────────────
TIER_THRESHOLDS = {
    "ELEVATED": 0.30,
    "STRESS":   0.50,
    "CRISIS":   0.70,
}


def _tier_for_score(score: float) -> str:
    if score >= TIER_THRESHOLDS["CRISIS"]:
        return "CRISIS"
    if score >= TIER_THRESHOLDS["STRESS"]:
        return "STRESS"
    if score >= TIER_THRESHOLDS["ELEVATED"]:
        return "ELEVATED"
    return "NORMAL"


def _clamp01(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return max(0.0, min(1.0, x))


# ──────────────────────────────────────────────────────────────────────────────
# Component computations (price-only)
# ──────────────────────────────────────────────────────────────────────────────
def _vol_percentile_rank(returns: np.ndarray, current_idx: int,
                         current_window: int = 21,
                         history_window: int = 1260) -> float:
    """
    Rank current realised vol against the prior 5 years of rolling 21d vols.

    Reactivity upgrade: "current vol" is max(vol_10d, vol_21d).  A fresh
    spike shows up in the 10d window within days (the 21d window dilutes it
    for weeks), while in calm markets the two converge and behaviour matches
    the original definition.  Both estimators are unbiased daily-σ measures,
    so ranking the max against the 21d history keeps the scale comparable.

    Returns the percentile in [0, 1] — 1.0 means current vol is at all-time
    high vs the rolling-5y sample, 0.0 means rock-bottom.
    """
    start = max(0, current_idx - history_window)
    seg = returns[start:current_idx]
    if len(seg) < current_window * 4:
        return 0.0
    # Build rolling 21d vols
    vols = []
    for i in range(current_window, len(seg)):
        vols.append(float(np.std(seg[i - current_window:i], ddof=0)))
    if not vols:
        return 0.0
    vol_21 = float(np.std(seg[-current_window:], ddof=0))
    vol_10 = float(np.std(seg[-10:], ddof=0)) if len(seg) >= 10 else vol_21
    current_vol = max(vol_10, vol_21)
    rank = float(np.searchsorted(np.sort(vols), current_vol)) / max(len(vols), 1)
    return _clamp01(rank)


def _drawdown_intensity(closes: np.ndarray, current_idx: int,
                       lookback: int = 252,
                       severity_pct: float = 25.0) -> float:
    """
    Magnitude of current drawdown from trailing-252d high, normalised so
    25% drawdown == 1.0.
    """
    start = max(0, current_idx - lookback)
    seg = closes[start:current_idx + 1]
    if len(seg) < 5:
        return 0.0
    peak = float(seg.max())
    cur  = float(seg[-1])
    if peak <= 0:
        return 0.0
    dd_pct = (cur / peak - 1.0) * 100  # negative
    return _clamp01(abs(dd_pct) / severity_pct)


def _crash_5d(closes: np.ndarray, current_idx: int,
              severity_pct: float = 10.0) -> float:
    """5d return crash magnitude; -10% over 5d == 1.0."""
    if current_idx < 5:
        return 0.0
    p_now = float(closes[current_idx])
    p_5d  = float(closes[current_idx - 5])
    if p_5d <= 0:
        return 0.0
    ret_5d_pct = (p_now / p_5d - 1.0) * 100
    return _clamp01(max(0.0, -ret_5d_pct) / severity_pct)


def _crash_21d(closes: np.ndarray, current_idx: int,
               severity_pct: float = 15.0) -> float:
    """21d return crash magnitude; -15% over a month == 1.0.

    Catches grinding month-long routs (e.g. a 'massive hit this month')
    that bleed too slowly to trip the 5d crash component on any single day.
    """
    if current_idx < 21:
        return 0.0
    p_now  = float(closes[current_idx])
    p_21d  = float(closes[current_idx - 21])
    if p_21d <= 0:
        return 0.0
    ret_21d_pct = (p_now / p_21d - 1.0) * 100
    return _clamp01(max(0.0, -ret_21d_pct) / severity_pct)


# RiskMetrics-style decay. λ=0.87 → half-life ≈ 5 trading days, so the
# estimator reprices a vol shock within a week instead of a month.
EWMA_LAMBDA = 0.87


def _ewma_vol(returns: np.ndarray, current_idx: int,
              span: int = 63) -> float:
    """Exponentially-weighted daily vol over the trailing `span` returns."""
    start = max(0, current_idx - span)
    seg = returns[start:current_idx]
    if len(seg) < 10:
        return 0.0
    weights = EWMA_LAMBDA ** np.arange(len(seg) - 1, -1, -1)
    weights /= weights.sum()
    mean = float(np.dot(weights, seg))
    var = float(np.dot(weights, (seg - mean) ** 2))
    return math.sqrt(max(var, 0.0))


def _fast_vol_spike(returns: np.ndarray, current_idx: int) -> float:
    """
    EWMA(≈5d half-life) vol vs trailing 21d vol.  Replaces the old
    vol_acceleration (21d vs 63d), which read 0.00 during the 2026-06 break
    because both slow windows had already absorbed the prior regime.
    1.0 means fast vol is running at ≥2× the 21d baseline.
    """
    if current_idx < 30:
        return 0.0
    fast = _ewma_vol(returns, current_idx)
    vol_21 = float(np.std(returns[max(0, current_idx - 21):current_idx], ddof=0))
    if vol_21 < 1e-9 or fast <= 0.0:
        return 0.0
    ratio_excess = fast / vol_21 - 1.0   # 0 == no spike, 1.0 == 2x baseline
    return _clamp01(ratio_excess)


def fast_vol_ratio(returns: np.ndarray, current_idx: int | None = None) -> float:
    """Raw EWMA-fast / 21d vol ratio (1.0 = calm). Exposed for selector + UI."""
    returns = np.asarray(returns, dtype=float)
    if current_idx is None:
        current_idx = len(returns)
    if current_idx < 30:
        return 1.0
    fast = _ewma_vol(returns, current_idx)
    vol_21 = float(np.std(returns[max(0, current_idx - 21):current_idx], ddof=0))
    if vol_21 < 1e-9 or fast <= 0.0:
        return 1.0
    return float(fast / vol_21)


# ──────────────────────────────────────────────────────────────────────────────
# Fast momentum diagnostics (RSI-14 / MACD histogram) — price-only
# ──────────────────────────────────────────────────────────────────────────────
def rsi_14(closes: np.ndarray, period: int = 14) -> float | None:
    """Wilder RSI over the trailing window. None when history is too short."""
    closes = np.asarray(closes, dtype=float)
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes[-(period * 4):])  # bounded lookback for speed
    if len(deltas) < period:
        return None
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for g, l in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
    if avg_loss < 1e-12:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - 100.0 / (1.0 + rs), 2)


def macd_hist_pct(closes: np.ndarray,
                  fast: int = 12, slow: int = 26, signal: int = 9) -> float | None:
    """MACD(12,26,9) histogram as % of price — sign/magnitude of fast momentum.

    Negative = downside momentum building faster than the signal line.
    None when history is too short.
    """
    closes = np.asarray(closes, dtype=float)
    if len(closes) < slow + signal + 5:
        return None

    def _ema(arr: np.ndarray, span: int) -> np.ndarray:
        alpha = 2.0 / (span + 1.0)
        out = np.empty_like(arr)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
        return out

    seg = closes[-(slow * 4):]
    macd_line = _ema(seg, fast) - _ema(seg, slow)
    signal_line = _ema(macd_line, signal)
    hist = float(macd_line[-1] - signal_line[-1])
    price = float(seg[-1])
    if price <= 0:
        return None
    return round(hist / price * 100.0, 4)


def _trend_reversal_burst(returns: np.ndarray, current_idx: int) -> float:
    """
    Did the sign of 21d drift just flip vs the prior 21d drift?
    1.0 when the system just executed a hard reversal.
    """
    if current_idx < 50:
        return 0.0
    drift_recent = float(np.mean(returns[current_idx - 21:current_idx]))
    drift_prior  = float(np.mean(returns[current_idx - 42:current_idx - 21]))
    if drift_prior * drift_recent >= 0:
        return 0.0
    magnitude = (abs(drift_recent) + abs(drift_prior)) / max(np.std(returns[current_idx - 42:current_idx], ddof=0), 1e-9)
    return _clamp01(magnitude / 1.5)


# ──────────────────────────────────────────────────────────────────────────────
# Public API — price-based
# ──────────────────────────────────────────────────────────────────────────────
# Phase XXVII reactivity weights — see module docstring for rationale.
WEIGHTS = {
    "vol_percentile":    0.22,
    "drawdown_intensity":0.20,
    "crash_5d":          0.18,
    "fast_vol_spike":    0.18,
    "crash_21d":         0.12,
    "trend_reversal":    0.10,
}


def classify_from_prices(closes: np.ndarray, current_idx: int | None = None) -> dict:
    """
    Compute the crisis score + tier at ``closes[current_idx]``.

    Pass ``current_idx=None`` to evaluate at the latest close.
    Returns a dict with all components for transparency.
    """
    closes = np.asarray(closes, dtype=float)
    if current_idx is None:
        current_idx = len(closes) - 1
    if current_idx < 60 or current_idx >= len(closes):
        return {
            "score":  0.0,
            "tier":   "NORMAL",
            "components": {k: 0.0 for k in WEIGHTS},
            "note":   "insufficient history",
        }

    returns = np.diff(np.log(closes))
    # current_idx applies to closes; for returns use current_idx - 1
    r_idx = max(0, current_idx - 1)

    components = {
        "vol_percentile":    _vol_percentile_rank(returns, r_idx),
        "drawdown_intensity":_drawdown_intensity(closes, current_idx),
        "crash_5d":          _crash_5d(closes, current_idx),
        "fast_vol_spike":    _fast_vol_spike(returns, r_idx),
        "crash_21d":         _crash_21d(closes, current_idx),
        "trend_reversal":    _trend_reversal_burst(returns, r_idx),
    }

    # Damage gate: volatility alone must not push the score past ELEVATED.
    # Gold's crisis behaviour is often a safe-haven RALLY on elevated vol
    # (2011 euro crisis, 2020 COVID) — throttling longs there gives away the
    # exact upside the book exists to capture. The pure-vol components
    # (vol_percentile, fast_vol_spike) carry full weight only when there is
    # realised damage (drawdown / crash evidence); with zero damage they are
    # scaled to 60%. Smooth ramp so the gate cannot flap.
    damage = max(
        components["drawdown_intensity"],
        components["crash_5d"],
        components["crash_21d"],
    )
    vol_scale = 0.6 + 0.4 * _clamp01(damage / 0.25)

    weighted = {
        k: WEIGHTS[k] * components[k] * (
            vol_scale if k in ("vol_percentile", "fast_vol_spike") else 1.0
        )
        for k in WEIGHTS
    }
    score = _clamp01(sum(weighted.values()))
    tier = _tier_for_score(score)
    return {
        "score":      round(score, 4),
        "tier":       tier,
        "components": {k: round(v, 4) for k, v in components.items()},
        "weights":    WEIGHTS,
        "damage_gate": {
            "damage":    round(damage, 4),
            "vol_scale": round(vol_scale, 4),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Engine-aware classifier (live use only)
# ──────────────────────────────────────────────────────────────────────────────
def _load(name: str) -> dict:
    p = DATA_DIR / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _safe(v, d=0.0):
    try:
        if v is None:
            return d
        f = float(v)
        return d if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return d


def classify_from_engines() -> dict:
    """
    Live classifier: fuses the price-based score with engine-derived
    signals (DCC-GARCH stress, tail-risk premium, geopolitical regime).

    Falls back gracefully when an engine JSON is missing — the price
    portion always works as long as `metals_pipeline` data is available.
    """
    closes = _fetch_recent_closes()
    if closes is None or len(closes) < 60:
        return {
            "schema_version": "1.0",
            "engine":         "crisis_detector",
            "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "score":          0.0,
            "tier":           "NORMAL",
            "note":           "insufficient price history",
        }

    price_result = classify_from_prices(closes)
    price_score = price_result["score"]

    # ── Fast regime diagnostics (Phase XXVII) ─────────────────────────────
    # Price-only momentum/vol dials consumed by strategy_selector's
    # falling-knife veto and rendered on the dashboard Regime Pulse panel.
    returns = np.diff(np.log(np.asarray(closes, dtype=float)))
    ewma_fast = _ewma_vol(returns, len(returns))
    vol_21 = float(np.std(returns[-21:], ddof=0)) if len(returns) >= 21 else 0.0
    drift_10d_pct = (
        float(closes[-1] / closes[-11] - 1.0) * 100.0 if len(closes) >= 11 else 0.0
    )
    fast_metrics = {
        "vol_spike_ratio":   round(fast_vol_ratio(returns), 3),
        "ewma_vol_ann_pct":  round(ewma_fast * math.sqrt(252) * 100, 2),
        "vol_21d_ann_pct":   round(vol_21 * math.sqrt(252) * 100, 2),
        "rsi_14":            rsi_14(closes),
        "macd_hist_pct":     macd_hist_pct(closes),
        "drift_10d_pct":     round(drift_10d_pct, 2),
    }

    # ── Engine bumps ──────────────────────────────────────────────────────
    bump = 0.0
    bump_notes: list[str] = []

    # DCC-GARCH correlation stress
    dcc = _load("dcc_garch.json")
    if dcc:
        n_stressed = int(_safe(dcc.get("n_stressed"), 0))
        if n_stressed >= 3:
            bump += 0.10
            bump_notes.append(f"DCC stressed pairs={n_stressed}")

    # Tail-risk EVT premium
    tr = _load("tail_risk_engine.json")
    premium = _safe((tr.get("tail_risk") or {}).get("tail_fatness_premium_pct"))
    if premium > 200:
        bump += 0.10
        bump_notes.append(f"tail premium {premium:.0f}% > 200%")
    elif premium > 100:
        bump += 0.05
        bump_notes.append(f"tail premium {premium:.0f}% > 100%")

    # Geopolitical regime
    geo = _load("geopolitical_detector.json")
    geo_regime = (geo.get("regime") or "").upper()
    if geo_regime == "EXTREME":
        bump += 0.15
        bump_notes.append("geopolitical EXTREME")
    elif geo_regime == "ELEVATED":
        bump += 0.05
        bump_notes.append("geopolitical ELEVATED")

    # Vol surface — only adds if it's worse than what we see in returns
    vs = _load("vol_surface.json")
    vol_regime = (vs.get("vol_regime") or "").upper()
    if vol_regime == "EXTREME":
        bump += 0.05
        bump_notes.append("vol_surface EXTREME")

    final_score = _clamp01(price_score + bump)
    final_tier = _tier_for_score(final_score)

    out = {
        "schema_version": "1.1",
        "engine":         "crisis_detector",
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "score":          round(final_score, 4),
        "tier":           final_tier,
        "price_score":    round(price_score, 4),
        "engine_bump":    round(bump, 4),
        "components":     price_result["components"],
        "weights":        price_result["weights"],
        "fast_metrics":   fast_metrics,
        "engine_bumps_applied": bump_notes,
        "size_caps": {
            "NORMAL":   None,
            "ELEVATED": 50.0,
            "STRESS":   20.0,
            "CRISIS":   0.0,
        },
        "guidance": _guidance_for_tier(final_tier),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(out, indent=2))
    return out


def _fetch_recent_closes() -> np.ndarray | None:
    """Pull ~600 days of GC=F for the live classifier."""
    try:
        from scripts.cache_layer import cached
        import yfinance as yf

        @cached(namespace="yfinance", ttl_seconds=6 * 3600)
        def _load_history():
            df = yf.Ticker("GC=F").history(
                period="3y", interval="1d", auto_adjust=True,
            )
            return df["Close"].astype(float).values

        return _load_history()
    except Exception:
        return None


def _guidance_for_tier(tier: str) -> str:
    return {
        "NORMAL":   "No special guard — use stacker conviction as usual.",
        "ELEVATED": "Cap final_size_pct at 50%. Prefer TREND only on tier HIGH+.",
        "STRESS":   "Cap final_size_pct at 20%. Prefer VOL_SHORT or TAIL_HEDGE; mean-rev only on extreme z.",
        "CRISIS":   "Force CASH or TAIL_HEDGE only. All other strategies suspended until score < 0.7.",
    }[tier]


# ──────────────────────────────────────────────────────────────────────────────
# Guard helpers — called by selectors / backtesters
# ──────────────────────────────────────────────────────────────────────────────
SIZE_CAP_BY_TIER: dict[str, float | None] = {
    "NORMAL":   None,
    "ELEVATED": 50.0,
    "STRESS":   20.0,
    "CRISIS":   0.0,
}


def apply_guard(
    strategy: str, final_size_pct: float, crisis_tier: str,
) -> tuple[str, float, str | None]:
    """
    Apply the crisis guard rule to a (strategy, size) decision.

    Returns (possibly-overridden strategy, possibly-capped size_pct, reason).
    ``reason`` is None when the guard didn't fire.
    """
    if crisis_tier == "CRISIS":
        # Only TAIL_HEDGE survives a CRISIS regime.
        if strategy != "TAIL_HEDGE":
            return "CASH", 0.0, "CRISIS tier — strategy forced to CASH"
        return strategy, min(final_size_pct, 30.0), "CRISIS tier — TAIL_HEDGE allowed at <=30%"

    cap = SIZE_CAP_BY_TIER.get(crisis_tier)
    if cap is not None and final_size_pct > cap:
        return strategy, cap, f"{crisis_tier} tier size cap {cap:.0f}%"

    return strategy, final_size_pct, None


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    out = classify_from_engines()
    if args.quiet:
        return 0
    print("=" * 64)
    print(f"CRISIS DETECTOR  ({out['generated_at']})")
    print("=" * 64)
    print(f"  Score        : {out['score']:.4f}")
    print(f"  Tier         : {out['tier']}")
    print(f"  Price score  : {out.get('price_score')}")
    print(f"  Engine bump  : {out.get('engine_bump')}")
    print()
    print("  Components:")
    for k, v in (out.get("components") or {}).items():
        w = out["weights"].get(k, 0)
        print(f"    {k:<22s} = {v:.3f}  (weight {w:.2f}  contribution {v*w:.3f})")
    if out.get("engine_bumps_applied"):
        print()
        print("  Engine bumps applied:")
        for b in out["engine_bumps_applied"]:
            print(f"    + {b}")
    print()
    print(f"  Guidance     : {out.get('guidance')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
