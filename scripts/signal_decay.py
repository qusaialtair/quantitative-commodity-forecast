#!/usr/bin/env python3
"""
Signal Decay Half-Life Analyzer
================================
Measures the predictive power of every alpha-source signal across multiple
forward horizons (1 / 3 / 5 / 10 / 21 days), fits an exponential decay
curve to the per-horizon Information Coefficient (IC), and reports:

  - IC at each horizon
  - IC half-life (days until predictive power halves)
  - t-stat of the strongest-horizon IC
  - decay status: STABLE / DECAYING / STRENGTHENING
        (compared to the prior 252d window)
  - suggested rebalance frequency (days)

Signals examined (re-using the alpha_attribution signal generators):
  lstm_momentum, macro_overlay, regime_filter, technical, mean_reversion

Output: data/signal_decay.json
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

from scripts.alpha_attribution import _generate_signals, _fetch_panel

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "signal_decay.json"

DEFAULT_TICKER = "GC=F"
DEFAULT_LOOKBACK = "5y"
HORIZONS = [1, 3, 5, 10, 21]

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# IC and decay computation
# ---------------------------------------------------------------------------
def _spearman_ic(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    """
    Spearman rank correlation, returning (rho, n_obs_used).
    Pure pandas implementation, no scipy dependency.
    """
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 30:
        return 0.0, len(df)
    rx = df.iloc[:, 0].rank()
    ry = df.iloc[:, 1].rank()
    # Pearson on ranks = Spearman
    rho = float(rx.corr(ry))
    if not np.isfinite(rho):
        rho = 0.0
    return rho, len(df)


def _compute_ic_curve(signal: pd.Series, returns: pd.Series, horizons: list[int]) -> dict:
    """For each horizon, return spearman(signal[t], R[t+1..t+h])."""
    out = {}
    for h in horizons:
        fwd = returns.rolling(h).sum().shift(-h)  # forward h-day return
        ic, n = _spearman_ic(signal, fwd)
        # Two-sided t-stat under H0: rho = 0
        tstat = ic * np.sqrt(max(n - 2, 1) / max(1 - ic ** 2, 1e-9))
        out[f"h_{h}"] = {
            "horizon_days": h,
            "ic": round(float(ic), 4),
            "t_stat": round(float(tstat), 3),
            "n_obs": int(n),
        }
    return out


def _fit_half_life(ic_curve: dict) -> tuple[float, float, str]:
    """
    Fit an exponential decay model to |IC|(h) and return:
        (half_life_days, ic_0, fit_status)

    Model: |IC|(h) = IC_0 * exp(-h / tau)
    Half-life = tau * ln(2)

    fit_status:
      OK              — clean fit
      UNRELIABLE      — small |IC| or non-monotonic
      INSUFFICIENT    — fewer than 3 horizons usable
    """
    horizons = []
    ics = []
    for k, v in ic_curve.items():
        ic = abs(v["ic"])
        if ic > 1e-4:
            horizons.append(v["horizon_days"])
            ics.append(ic)

    if len(horizons) < 3:
        return 999.0, 0.0, "INSUFFICIENT"

    h_arr = np.array(horizons, dtype=float)
    ic_arr = np.array(ics, dtype=float)

    # log(|IC|) = log(IC_0) - h / tau
    try:
        slope, intercept = np.polyfit(h_arr, np.log(ic_arr), 1)
    except Exception:
        return 999.0, 0.0, "UNRELIABLE"

    if slope >= 0:
        # IC strengthening with horizon → no traditional half-life
        return 999.0, float(np.exp(intercept)), "STRENGTHENING"

    tau = -1.0 / slope
    half_life = tau * np.log(2.0)
    if half_life > 365 or half_life < 0.5:
        return min(max(half_life, 0.5), 365.0), float(np.exp(intercept)), "UNRELIABLE"

    return float(half_life), float(np.exp(intercept)), "OK"


def _classify_decay(current_ic: float, prior_ic: float) -> str:
    """Compare current vs prior best |IC| to classify alpha decay."""
    if abs(prior_ic) < 1e-4:
        return "STABLE"
    delta = (abs(current_ic) - abs(prior_ic)) / abs(prior_ic)
    if delta < -0.25:
        return "DECAYING"
    if delta > 0.25:
        return "STRENGTHENING"
    return "STABLE"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_signal_decay(
    ticker: str = DEFAULT_TICKER,
    lookback: str = DEFAULT_LOOKBACK,
) -> dict:
    df = _fetch_panel(ticker, lookback)
    signals = _generate_signals(df)
    returns = df["gold"].pct_change()

    # Split: most recent 252d vs prior 252d for decay comparison
    n = len(df)
    split_idx = max(int(n - 252), int(n * 0.6))
    current_signals = signals.iloc[split_idx:]
    current_returns = returns.iloc[split_idx:]
    prior_signals = signals.iloc[max(split_idx - 252, 0):split_idx]
    prior_returns = returns.iloc[max(split_idx - 252, 0):split_idx]

    out = {}
    for col in signals.columns:
        cur_curve = _compute_ic_curve(current_signals[col], current_returns, HORIZONS)
        prior_curve = _compute_ic_curve(prior_signals[col], prior_returns, HORIZONS)

        half_life, ic_0, fit_status = _fit_half_life(cur_curve)

        # Best-horizon IC (highest absolute) for headline metric
        best_h = max(cur_curve.values(), key=lambda v: abs(v["ic"]))
        prior_best = max(prior_curve.values(), key=lambda v: abs(v["ic"]))
        decay = _classify_decay(best_h["ic"], prior_best["ic"])

        # Rebalance frequency: half of the half-life, floored at 1d, capped at 21d
        rebalance = int(np.clip(round(half_life * 0.5), 1, 21))
        if fit_status == "INSUFFICIENT" or fit_status == "STRENGTHENING":
            rebalance = HORIZONS[-1]  # 21d if no clear half-life

        out[col] = {
            "ic_curve":        cur_curve,
            "prior_ic_curve":  prior_curve,
            "best_horizon_ic": round(best_h["ic"], 4),
            "best_horizon":    best_h["horizon_days"],
            "best_horizon_t":  round(best_h["t_stat"], 3),
            "ic_0_estimate":   round(ic_0, 4),
            "half_life_days":  round(half_life, 2),
            "fit_status":      fit_status,
            "decay_status":    decay,
            "prior_best_ic":   round(prior_best["ic"], 4),
            "rebalance_days":  rebalance,
        }

    # Rank
    ranked_ic = sorted(out.items(), key=lambda kv: abs(kv[1]["best_horizon_ic"]), reverse=True)
    ranked_persistent = sorted(
        [(k, v) for k, v in out.items() if v["fit_status"] == "OK"],
        key=lambda kv: kv[1]["half_life_days"],
        reverse=True,
    )

    decaying = [k for k, v in out.items() if v["decay_status"] == "DECAYING"]
    strengthening = [k for k, v in out.items() if v["decay_status"] == "STRENGTHENING"]

    result = {
        "generated_at":    datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker":          ticker,
        "lookback":        lookback,
        "horizons_days":   HORIZONS,
        "signals":         out,
        "ranked_by_ic":    [k for k, _ in ranked_ic],
        "ranked_by_half_life": [k for k, _ in ranked_persistent],
        "decaying_signals":      decaying,
        "strengthening_signals": strengthening,
        "n_signals":       len(out),
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
    print(f"  SIGNAL DECAY ANALYZER -- {r['ticker']}")
    print(SEP)
    print(f"  Signals: {r['n_signals']}  Lookback: {r['lookback']}")
    print()

    print(f"  PER-SIGNAL DIAGNOSTICS")
    print(f"  {'─' * 58}")
    print(
        f"  {'signal':<16s}  {'best_h':>6s}  "
        f"{'IC':>7s}  {'t':>6s}  {'½-life':>7s}  "
        f"{'rebal':>6s}  {'status':>10s}"
    )
    for sig, v in r["signals"].items():
        decay_marker = {
            "DECAYING":      "↓",
            "STRENGTHENING": "↑",
            "STABLE":        "─",
        }.get(v["decay_status"], "?")
        hl_str = f"{v['half_life_days']:.1f}d" if v['fit_status'] == "OK" else "n/a"
        print(
            f"  {sig:<16s}  {v['best_horizon']:>5d}d  "
            f"{v['best_horizon_ic']:+7.4f}  {v['best_horizon_t']:+6.2f}  "
            f"{hl_str:>7s}  {v['rebalance_days']:>5d}d  "
            f"{decay_marker} {v['decay_status']:>8s}"
        )
    print()

    print(f"  ALPHA DECAY DASHBOARD")
    print(f"  {'─' * 58}")
    if r["decaying_signals"]:
        print(f"  ⚠ Decaying:       {', '.join(r['decaying_signals'])}")
    if r["strengthening_signals"]:
        print(f"  ✓ Strengthening:  {', '.join(r['strengthening_signals'])}")
    if not r["decaying_signals"] and not r["strengthening_signals"]:
        print(f"    All signals stable vs prior 252d window")
    print()
    print(f"  RANKED BY |IC|:        {', '.join(r['ranked_by_ic'])}")
    if r["ranked_by_half_life"]:
        print(f"  RANKED BY HALF-LIFE:   {', '.join(r['ranked_by_half_life'])}")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signal Decay Half-Life Analyzer")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--lookback", default=DEFAULT_LOOKBACK)
    args = parser.parse_args()
    run_signal_decay(ticker=args.ticker, lookback=args.lookback)
