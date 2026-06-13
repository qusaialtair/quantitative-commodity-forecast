#!/usr/bin/env python3
"""
Bayesian Model Averaging (BMA)
================================
Mathematically coherent ensemble weighting that replaces ad-hoc averaging:

    p(M_k | data) ∝ p(data | M_k) · p(M_k)

For each alpha source (treated as a probabilistic model of next-day gold
returns), we:

  1. Build a per-source point forecast on each historical day t:
        ŷ_k(t) = signal_k(t-1) · E[|r| | signal active]
     This is the simplest non-trivial mapping — direction from the signal,
     magnitude from the historical average absolute return.

  2. Compute Gaussian residual log-likelihoods over a rolling 252d window:
        LL_k = Σ_t [ −½ log(2π σ_k²) − (r_t − ŷ_k(t))² / (2σ_k²) ]
     where σ_k is estimated from the same window's residuals.

  3. Convert LLs to posterior model weights via stable softmax:
        w_k = exp(LL_k − max_LL) / Σ exp(LL_j − max_LL)

The output is then compared head-to-head to:
  - Equal weight  (1/N baseline)
  - IR weight     (from alpha_attribution.json, edge-tilt baseline)

For each weighting scheme the engine backtests the blended signal in-sample
and reports Sharpe / vol / max DD.

Output: data/bma_weights.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import yfinance as yf
except ImportError:
    yf = None

from scripts.alpha_attribution import _fetch_panel, _generate_signals

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "bma_weights.json"
ALPHA_FILE = DATA_DIR / "alpha_attribution.json"

DEFAULT_TICKER = "GC=F"
DEFAULT_LOOKBACK = "5y"
BMA_WINDOW = 252
SQ252 = float(np.sqrt(252))

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# BMA core
# ---------------------------------------------------------------------------
def compute_log_likelihoods(
    returns: pd.Series,
    signals: pd.DataFrame,
    window: int = BMA_WINDOW,
) -> tuple[dict, dict, dict]:
    """
    Directional log-score relative to a 50/50 null.

    For each model k, on days where signal[t-1] is non-zero:
        hit_t   = 1 if sign(signal[t-1]) == sign(r[t]) else 0
        p_k     = empirical hit rate over the window
        LL_k    = Σ_t [ hit_t · log(p_k) + (1 - hit_t) · log(1 - p_k) ]
        Bayes   = LL_k − n_k · log(½)   (information gain over a coin flip)

    Returns (bayes_factors, hit_rates, n_active_obs). Bayes factors are what
    feed into the softmax; hit rates are reported for diagnostics.

    This intentionally replaces a Gaussian-residual likelihood, which would
    perversely reward signals that almost never trade (small residual σ ≠
    predictive skill).
    """
    recent_r = returns.tail(window)
    recent_signals = signals.shift(1).tail(window)

    lls = {}
    hit_rates = {}
    n_obs = {}
    for col in signals.columns:
        sig = recent_signals[col]
        active = sig != 0
        n = int(active.sum())
        n_obs[col] = n
        if n < 30:
            lls[col] = -np.inf
            hit_rates[col] = float("nan")
            continue
        s = sig[active].values
        r = recent_r[active].values
        hits = (np.sign(s) == np.sign(r)).astype(int)
        n_hits = int(hits.sum())
        hit_rate = float(n_hits / n)
        # Clip to avoid log(0)
        p = max(0.01, min(0.99, hit_rate))
        ll = n_hits * np.log(p) + (n - n_hits) * np.log(1 - p)
        null_ll = n * np.log(0.5)
        lls[col] = float(ll - null_ll)
        hit_rates[col] = round(hit_rate, 4)
    return lls, hit_rates, n_obs


def softmax_weights(log_likelihoods: dict) -> dict:
    """Numerically stable softmax over LLs; falls back to equal-weight if no
    finite entries."""
    keys = list(log_likelihoods.keys())
    lls = np.array([log_likelihoods[k] for k in keys], dtype=float)
    finite = np.isfinite(lls)
    if not finite.any():
        return {k: 1.0 / len(keys) for k in keys}
    max_ll = lls[finite].max()
    exp_lls = np.where(finite, np.exp(np.clip(lls - max_ll, -50, 50)), 0.0)
    total = exp_lls.sum()
    if total <= 0:
        return {k: 1.0 / len(keys) for k in keys}
    return {k: float(exp_lls[i] / total) for i, k in enumerate(keys)}


def equal_weights(signals: pd.DataFrame) -> dict:
    n = signals.shape[1]
    return {c: 1.0 / n for c in signals.columns}


def ir_weights() -> dict:
    """Load Information Ratios from alpha_attribution.json; +IR → weight ∝ IR."""
    if not ALPHA_FILE.exists():
        return {}
    try:
        aa = json.loads(ALPHA_FILE.read_text())
    except Exception:
        return {}
    irs = aa.get("information_ratios", {})
    raw = {s: max(0.0, irs.get(s, {}).get("information_ratio", 0))
           for s in aa.get("sources", [])}
    total = sum(raw.values())
    if total <= 0:
        return {s: 1.0 / max(1, len(raw)) for s in raw}
    return {s: v / total for s, v in raw.items()}


# ---------------------------------------------------------------------------
# Backtest each weighting scheme
# ---------------------------------------------------------------------------
def backtest(
    returns: pd.Series, signals: pd.DataFrame, weights: dict,
) -> dict:
    if not weights:
        return {"sharpe": 0.0, "ann_return_pct": 0.0, "ann_vol_pct": 0.0,
                "max_drawdown_pct": 0.0, "n_obs": 0}
    w = pd.Series(weights)
    lagged = signals.shift(1).fillna(0)
    blended_signal = (lagged * w).sum(axis=1)
    # Clip to a sensible position range; signal values are typically [-1, +1] per source
    blended_signal = blended_signal.clip(-1.0, 1.0)
    strat_returns = (blended_signal * returns).dropna()
    if len(strat_returns) < 30:
        return {"sharpe": 0.0, "ann_return_pct": 0.0, "ann_vol_pct": 0.0,
                "max_drawdown_pct": 0.0, "n_obs": int(len(strat_returns))}
    ann_ret = float(strat_returns.mean() * 252)
    ann_vol = float(strat_returns.std() * SQ252)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
    cum = (1 + strat_returns).cumprod()
    max_dd = float((cum / cum.cummax() - 1).min())
    return {
        "sharpe":           round(sharpe, 3),
        "ann_return_pct":   round(ann_ret * 100, 3),
        "ann_vol_pct":      round(ann_vol * 100, 3),
        "max_drawdown_pct": round(max_dd * 100, 3),
        "n_obs":            int(len(strat_returns)),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_bma(
    ticker: str = DEFAULT_TICKER,
    lookback: str = DEFAULT_LOOKBACK,
    window: int = BMA_WINDOW,
) -> dict:
    panel = _fetch_panel(ticker, lookback)
    signals = _generate_signals(panel)
    returns = panel["gold"].pct_change().dropna()

    # Align indices
    signals = signals.reindex(returns.index).fillna(0)

    # Compute LLs in the last `window` days
    lls, hit_rates, n_obs = compute_log_likelihoods(returns, signals, window)
    w_bma = softmax_weights(lls)
    w_eq = equal_weights(signals)
    w_ir = ir_weights()
    if not w_ir:
        w_ir = w_eq

    bt_bma = backtest(returns, signals, w_bma)
    bt_eq = backtest(returns, signals, w_eq)
    bt_ir = backtest(returns, signals, w_ir)

    # Per-source diagnostics
    per_source = []
    for s in signals.columns:
        per_source.append({
            "source":            s,
            "log_bayes_factor":  round(lls.get(s, 0), 3),
            "hit_rate":          hit_rates.get(s, 0),
            "n_active_obs":      int(n_obs.get(s, 0)),
            "bma_weight":        round(w_bma.get(s, 0), 4),
            "eq_weight":         round(w_eq.get(s, 0), 4),
            "ir_weight":         round(w_ir.get(s, 0), 4),
        })

    ranked = sorted(per_source, key=lambda x: x["bma_weight"], reverse=True)
    top_source = ranked[0]["source"] if ranked else None

    result = {
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker":         ticker,
        "lookback":       lookback,
        "bma_window_days":window,
        "n_obs_total":    int(len(returns)),
        "top_source":     top_source,
        "per_source":     per_source,
        "weights": {
            "bma":         {s: round(v, 4) for s, v in w_bma.items()},
            "equal":       {s: round(v, 4) for s, v in w_eq.items()},
            "ir":          {s: round(v, 4) for s, v in w_ir.items()},
        },
        "backtest": {
            "bma":         bt_bma,
            "equal":       bt_eq,
            "ir":          bt_ir,
        },
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2))
    _print_report(result)
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_report(r: dict) -> None:
    print(f"\n{SEP}")
    print(f"  BAYESIAN MODEL AVERAGING -- {r['ticker']}")
    print(SEP)
    print(f"  BMA window:   {r['bma_window_days']}d")
    print(f"  Top source:   {r['top_source']}")
    print()

    print(f"  PER-SOURCE DIAGNOSTICS")
    print(f"  {'─' * 64}")
    print(
        f"  {'source':<16s}  {'logBF':>8s}  {'hit':>6s}  {'n':>5s}  "
        f"{'BMA':>7s}  {'IR':>7s}  {'EqW':>7s}"
    )
    for d in r["per_source"]:
        print(
            f"  {d['source']:<16s}  "
            f"{d['log_bayes_factor']:>+8.2f}  "
            f"{d['hit_rate']:>6.1%}  "
            f"{d['n_active_obs']:>5d}  "
            f"{d['bma_weight']:>7.2%}  "
            f"{d['ir_weight']:>7.2%}  "
            f"{d['eq_weight']:>7.2%}"
        )
    print()

    print(f"  BACKTEST COMPARISON (blended signal × forward return)")
    print(f"  {'─' * 58}")
    cols = [("bma", "BMA"), ("ir", "IR-weight"), ("equal", "EqualW")]
    print(f"  {'metric':<22s}  " + "  ".join(f"{n:>9s}" for _, n in cols))
    for fld, label in [
        ("ann_return_pct",   "Ann Return (%)"),
        ("ann_vol_pct",      "Ann Vol (%)"),
        ("sharpe",           "Sharpe"),
        ("max_drawdown_pct", "Max DD (%)"),
    ]:
        print(
            f"  {label:<22s}  " + "  ".join(
                f"{r['backtest'][k][fld]:>9.3f}" for k, _ in cols
            )
        )
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Model Averaging")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--lookback", default=DEFAULT_LOOKBACK)
    parser.add_argument("--window", type=int, default=BMA_WINDOW)
    args = parser.parse_args()
    run_bma(ticker=args.ticker, lookback=args.lookback, window=args.window)
