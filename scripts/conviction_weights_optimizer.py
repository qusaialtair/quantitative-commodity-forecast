#!/usr/bin/env python3
"""
Conviction Weights Optimizer  (Phase XXIII Stage 84)
======================================================
Re-weights the five conviction components in `strategy_backtester.
_technical_conviction` based on their *actual* historical signal
quality, rather than the hard-coded judgement-call weights.

Motivation
----------
Phase XXII's walk-forward validator showed median annual +3.96 %/y at
Sharpe 0.49 — meaningful edge but regime-dependent.  The conviction
blend is the most direct lever on signal quality.  Today it uses:

    0.32 * trend_short + 0.22 * trend_long + 0.28 * mom_combined
  + 0.08 * mean_rev_fade + 0.10 * pivot

These weights were chosen by intuition.  This module:

  1. For each component, runs a single-component backtest across the
     full history — going long when the component value is positive,
     short when negative, with no other rules.
  2. Computes that component's standalone Sharpe + information-
     coefficient (sign-correlation with next-day return).
  3. Combines IC and Sharpe into a posterior weight via Bayesian
     softmax with a temperature parameter.
  4. Writes the recommended weights to `data/conviction_weights.json`.

`strategy_backtester._technical_conviction` reads that file at runtime
(falls back to defaults if missing or stale).

Output schema
-------------

    {
      "component_metrics": {
        "trend_short": {"sharpe": +1.20, "ic": +0.08, "hit_rate": 0.54},
        ...
      },
      "weights": {"trend_short": 0.31, ...},   # sum to 1.0
      "default_weights": {...},                 # original hard-coded for comparison
      "delta_vs_default": {...},
      "generated_at": ISO-8601 UTC
    }
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.cache_layer import cached  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Independent fetcher — avoid circular import with strategy_backtester
# ──────────────────────────────────────────────────────────────────────────────
@cached(namespace="yfinance", ttl_seconds=12 * 3600)
def _fetch_full_history(ticker: str = "GC=F") -> tuple[np.ndarray, list[str]]:
    """Local copy; matches stress_backtester._fetch_full_history signature."""
    import yfinance as yf
    hist = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=True)
    if hist is None or hist.empty:
        raise SystemExit(f"no history for {ticker}")
    closes = hist["Close"].astype(float).values
    dates = [d.strftime("%Y-%m-%d") for d in hist.index]
    return closes, dates

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "conviction_weights.json"

SQ252 = float(np.sqrt(252))

# Original hard-coded weights from `strategy_backtester._technical_conviction`
DEFAULT_WEIGHTS = {
    "trend_short":    0.32,
    "trend_long":     0.22,
    "mom_combined":   0.28,
    "mean_rev_fade":  0.08,
    "pivot":          0.10,
}

# Softmax temperature: lower = more concentrated on best component;
# higher = more uniform.  0.5 is a sensible default — see ablation below.
SOFTMAX_TEMPERATURE = 0.6
MIN_WEIGHT = 0.05   # floor so no component disappears entirely


# ──────────────────────────────────────────────────────────────────────────────
# Component value computations  (mirror strategy_backtester._technical_conviction)
# ──────────────────────────────────────────────────────────────────────────────
def _trend_short(closes: np.ndarray, i: int) -> float:
    if i < 50:
        return 0.0
    sma20 = float(np.mean(closes[i - 20:i]))
    sma50 = float(np.mean(closes[i - 50:i]))
    return math.tanh((sma20 - sma50) / max(abs(sma50), 1.0) * 80)


def _trend_long(closes: np.ndarray, i: int) -> float:
    if i < 200:
        return 0.0
    sma50  = float(np.mean(closes[i - 50:i]))
    sma200 = float(np.mean(closes[i - 200:i]))
    return math.tanh((sma50 - sma200) / max(abs(sma200), 1.0) * 80)


def _mom_combined(closes: np.ndarray, i: int) -> float:
    if i < 64:
        return 0.0
    mom_5d  = (closes[i - 1] - closes[i - 6])  / max(closes[i - 6], 1e-9)
    mom_21d = (closes[i - 1] - closes[i - 22]) / max(closes[i - 22], 1e-9)
    mom_63d = (closes[i - 1] - closes[i - 64]) / max(closes[i - 64], 1e-9)
    return math.tanh((mom_5d * 30) + (mom_21d * 12) + (mom_63d * 4))


def _mean_rev_fade(closes: np.ndarray, i: int) -> float:
    if i < 20:
        return 0.0
    sma20 = float(np.mean(closes[i - 20:i]))
    std20 = float(np.std(closes[i - 20:i]))
    if std20 < 1e-9:
        return 0.0
    bb = (closes[i - 1] - sma20) / (2.0 * std20)
    bb = max(-1.0, min(1.0, bb))
    return -bb * 0.4


def _pivot_value(closes: np.ndarray, i: int) -> float:
    """Direct port of strategy_backtester._pivot_score."""
    if i < 60:
        return 0.0
    if i >= 200:
        sma50  = float(np.mean(closes[i - 50:i]))
        sma200 = float(np.mean(closes[i - 200:i]))
        prevailing = 1.0 if sma50 > sma200 else -1.0
    else:
        sma50  = float(np.mean(closes[i - 50:i]))
        sma100 = float(np.mean(closes[max(0, i - 100):i]))
        prevailing = 1.0 if sma50 > sma100 else -1.0

    mom_5d  = (closes[i - 1] - closes[i - 6])  / max(closes[i - 6], 1e-9) if i >= 6 else 0.0
    mom_21d = (closes[i - 1] - closes[i - 22]) / max(closes[i - 22], 1e-9) if i >= 22 else 0.0

    short_disagrees = math.copysign(1.0, mom_5d)  != math.copysign(1.0, prevailing)
    mid_disagrees   = math.copysign(1.0, mom_21d) != math.copysign(1.0, prevailing)
    if not (short_disagrees and mid_disagrees):
        return 0.0
    if abs(mom_5d) < 0.012 or abs(mom_21d) < 0.008:
        return 0.0

    accel = abs(mom_5d) + 0.5 * abs(mom_21d)
    intensity = math.tanh(accel * 22.0)
    new_direction = -math.copysign(1.0, prevailing)
    return float(new_direction * intensity)


COMPONENT_FUNCTIONS = {
    "trend_short":   _trend_short,
    "trend_long":    _trend_long,
    "mom_combined":  _mom_combined,
    "mean_rev_fade": _mean_rev_fade,
    "pivot":         _pivot_value,
}


# ──────────────────────────────────────────────────────────────────────────────
# Per-component evaluation
# ──────────────────────────────────────────────────────────────────────────────
def _evaluate_component(
    closes: np.ndarray, fn, start_idx: int = 200,
) -> dict:
    """
    Single-component standalone backtest.  Long when component > 0, short when < 0.
    No size scaling, no other rules — pure signal evaluation.
    """
    n = len(closes)
    pct_returns = np.diff(closes) / closes[:-1] * 100  # daily %
    signals: list[float] = []
    next_day_returns: list[float] = []
    daily_pnl: list[float] = []

    for i in range(start_idx, n - 1):
        component_value = fn(closes, i + 1)
        # Trade direction: sign of component, only when above magnitude threshold
        if abs(component_value) < 0.05:
            position = 0.0
        else:
            position = math.copysign(1.0, component_value)
        next_ret_pct = float(pct_returns[i])
        signals.append(component_value)
        next_day_returns.append(next_ret_pct)
        daily_pnl.append(position * next_ret_pct - 0.005)  # tiny cost

    if not daily_pnl:
        return {"n_days": 0, "sharpe": 0.0, "ic": 0.0, "hit_rate": 0.5}

    pnl = np.asarray(daily_pnl)
    mu = float(pnl.mean())
    sd = float(pnl.std(ddof=0))
    sharpe = (mu / sd * SQ252) if sd > 1e-12 else 0.0

    # Information coefficient: Spearman-style sign-correlation
    sigs = np.asarray(signals)
    rets = np.asarray(next_day_returns)
    if len(sigs) >= 30 and sigs.std() > 1e-9 and rets.std() > 1e-9:
        ic_pearson = float(np.corrcoef(sigs, rets)[0, 1])
    else:
        ic_pearson = 0.0
    # Sign-correlation (more robust for non-linear signals)
    sign_match = np.sign(sigs) == np.sign(rets)
    # Ignore zero-signal days
    mask = sigs != 0
    if mask.sum() > 0:
        sign_hit_rate = float(sign_match[mask].mean())
    else:
        sign_hit_rate = 0.5

    return {
        "n_days":     int(len(pnl)),
        "n_signaled":int(int((sigs != 0).sum())),
        "sharpe":    round(float(sharpe), 3),
        "ic":        round(float(ic_pearson), 4),
        "hit_rate":  round(float(sign_hit_rate), 4),
        "ann_return_pct": round(float(((1 + mu / 100) ** 252 - 1) * 100), 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Weight derivation
# ──────────────────────────────────────────────────────────────────────────────
def _softmax_weights(scores: dict[str, float], temperature: float = SOFTMAX_TEMPERATURE) -> dict[str, float]:
    """
    Softmax over per-component scores (Sharpe-based) with a temperature
    parameter.  Clamps each weight at MIN_WEIGHT so no component
    disappears entirely.
    """
    keys = list(scores.keys())
    raw = np.array([scores[k] for k in keys], dtype=float)
    # Numerical stability
    raw -= raw.max()
    exp_vals = np.exp(raw / max(temperature, 1e-6))
    w = exp_vals / exp_vals.sum()
    # Apply floor + re-normalise
    w = np.maximum(w, MIN_WEIGHT)
    w /= w.sum()
    return {k: float(round(w[i], 4)) for i, k in enumerate(keys)}


def _combine_score(metrics: dict) -> float:
    """
    Combine Sharpe + IC into one quality score.
    Sharpe is the main signal (it's already risk-adjusted return).
    IC adds a secondary 'predictive consistency' boost.
    """
    sharpe = float(metrics.get("sharpe", 0.0))
    ic     = float(metrics.get("ic", 0.0))
    return sharpe + 8.0 * ic   # IC is small (~0.05); scale to match Sharpe magnitude


# ──────────────────────────────────────────────────────────────────────────────
# Public entry
# ──────────────────────────────────────────────────────────────────────────────
def run_optimizer(temperature: float = SOFTMAX_TEMPERATURE) -> dict:
    closes, dates = _fetch_full_history("GC=F")
    if len(closes) < 250:
        raise SystemExit("insufficient history")

    component_metrics: dict[str, dict] = {}
    component_scores:  dict[str, float] = {}
    for name, fn in COMPONENT_FUNCTIONS.items():
        m = _evaluate_component(closes, fn)
        component_metrics[name] = m
        component_scores[name]  = _combine_score(m)

    new_weights = _softmax_weights(component_scores, temperature=temperature)

    delta = {
        k: round(new_weights.get(k, 0) - DEFAULT_WEIGHTS.get(k, 0), 4)
        for k in DEFAULT_WEIGHTS
    }

    # Build a ranking by score
    ranked = sorted(component_metrics.items(),
                    key=lambda kv: -_combine_score(kv[1]))

    out = {
        "schema_version":   "1.0",
        "engine":           "conviction_weights_optimizer",
        "generated_at":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "history_range":    {"start": dates[0], "end": dates[-1]},
        "n_history_days":   len(closes),
        "temperature":      temperature,
        "min_weight_floor": MIN_WEIGHT,
        "component_metrics": component_metrics,
        "weights":          new_weights,
        "default_weights":  DEFAULT_WEIGHTS,
        "delta_vs_default": delta,
        "ranked": [{"component": k, "score": _combine_score(m), **m} for k, m in ranked],
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(out, indent=2, default=str))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Loader used by strategy_backtester
# ──────────────────────────────────────────────────────────────────────────────
def load_weights() -> dict[str, float]:
    """
    Load optimised weights from disk.  Returns DEFAULT_WEIGHTS if the
    optimiser hasn't been run or the file is corrupt.
    """
    if not OUTPUT_FILE.exists():
        return DEFAULT_WEIGHTS.copy()
    try:
        data = json.loads(OUTPUT_FILE.read_text())
        weights = data.get("weights") or {}
        # Validate — must have all 5 keys + sum within tolerance
        if set(weights.keys()) != set(DEFAULT_WEIGHTS.keys()):
            return DEFAULT_WEIGHTS.copy()
        total = sum(float(v) for v in weights.values())
        if not (0.95 <= total <= 1.05):
            return DEFAULT_WEIGHTS.copy()
        return {k: float(v) for k, v in weights.items()}
    except Exception:
        return DEFAULT_WEIGHTS.copy()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--temperature", type=float, default=SOFTMAX_TEMPERATURE)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    out = run_optimizer(temperature=args.temperature)
    if args.quiet:
        return 0
    print("=" * 76)
    print(f"CONVICTION WEIGHTS OPTIMIZER  ({out['generated_at']})")
    print("=" * 76)
    print(f"  History  : {out['history_range']['start']} → {out['history_range']['end']}  ({out['n_history_days']} bars)")
    print(f"  Temperature : {out['temperature']}  (lower = more concentrated)")
    print()
    print(f"  {'Component':<16s} {'Sharpe':>7s} {'IC':>7s} {'Hit':>6s} {'Default':>8s} {'New':>7s} {'Δ':>7s}")
    print("  " + "-" * 65)
    for r in out["ranked"]:
        name = r["component"]
        print(
            f"  {name:<16s} "
            f"{r['sharpe']:>+7.2f} "
            f"{r['ic']:>+7.3f} "
            f"{r['hit_rate']:>6.3f} "
            f"{DEFAULT_WEIGHTS[name]:>8.3f} "
            f"{out['weights'][name]:>7.3f} "
            f"{out['delta_vs_default'][name]:>+7.3f}"
        )
    print()
    print(f"  weights.json → {OUTPUT_FILE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
