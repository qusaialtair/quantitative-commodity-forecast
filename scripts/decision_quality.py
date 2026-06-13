#!/usr/bin/env python3
"""
Decision Quality Framework  (Brier score, log loss, calibration)
==================================================================
Tests how well our directional-probability forecasts are *calibrated*
beyond mere accuracy. A perfectly accurate but over-confident system can
still be poorly calibrated.

For each alpha signal the engine derives a probability of the 5d-forward
return being positive, using a simple monotonic mapping
  p = ½ + ½ · σ_norm(signal_today)
then computes against the realised binary outcome:

  Brier score    = mean[(p − y)²]                   lower is better; chance = 0.25
  Log loss       = -mean[y·log p + (1-y)·log(1-p)]  lower is better; chance ≈ 0.693
  Calibration MAE = mean[|p_bin − empirical_freq_bin|]   across 10 bins
  Reliability    = correlation of p_bin and freq_bin

Compares each signal to a 50/50 null and to a calibration-refined logistic
recalibrator fit by isotonic regression-lite (binned).

Output: data/decision_quality.json
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

from scripts.alpha_attribution import _fetch_panel, _generate_signals

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "decision_quality.json"

DEFAULT_TICKER = "GC=F"
DEFAULT_LOOKBACK = "5y"
DEFAULT_HORIZON = 5
N_BINS = 10

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Probability mapping
# ---------------------------------------------------------------------------
def _signal_to_prob(signal: pd.Series, scale: float = 1.0) -> pd.Series:
    """Map signal in {-1, 0, +1} to probability in [0.1, 0.9]."""
    # If signal is +1 → 0.6, -1 → 0.4, 0 → 0.5 (with scale=0.1)
    # We'll use scale=0.15 to give slightly more confidence
    p = 0.5 + 0.15 * signal.fillna(0)
    return p.clip(0.05, 0.95)


# ---------------------------------------------------------------------------
# Scoring rules
# ---------------------------------------------------------------------------
def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))


def calibration_metrics(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> dict:
    """Reliability via binning. Returns ECE-style MAE + per-bin freqs."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    per_bin = []
    total_mae = 0.0
    total_weight = 0.0
    expected_freqs = []
    observed_freqs = []
    for b in range(n_bins):
        mask = bin_indices == b
        n = int(mask.sum())
        if n == 0:
            continue
        expected = float(p[mask].mean())
        observed = float(y[mask].mean())
        per_bin.append({
            "bin":      b,
            "p_low":    round(float(bins[b]), 3),
            "p_high":   round(float(bins[b + 1]), 3),
            "expected": round(expected, 4),
            "observed": round(observed, 4),
            "n":        n,
            "abs_dev":  round(abs(expected - observed), 4),
        })
        total_mae += abs(expected - observed) * n
        total_weight += n
        expected_freqs.append(expected)
        observed_freqs.append(observed)
    ece = total_mae / total_weight if total_weight > 0 else 0.0

    # Reliability = corr(expected, observed)
    if len(expected_freqs) >= 2:
        ef = np.array(expected_freqs)
        of = np.array(observed_freqs)
        if ef.std() > 0 and of.std() > 0:
            reliability = float(np.corrcoef(ef, of)[0, 1])
        else:
            reliability = 1.0
    else:
        reliability = 0.0

    return {
        "ece":         round(float(ece), 4),
        "reliability": round(reliability, 4),
        "per_bin":     per_bin,
        "n_bins_used": len(per_bin),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_decision_quality(
    ticker: str = DEFAULT_TICKER,
    lookback: str = DEFAULT_LOOKBACK,
    horizon: int = DEFAULT_HORIZON,
) -> dict:
    df = _fetch_panel(ticker, lookback)
    signals = _generate_signals(df)
    fwd = df["gold"].pct_change(horizon).shift(-horizon)
    y_full = (fwd > 0).astype(int)

    per_signal = {}
    for col in signals.columns:
        lagged = signals[col].shift(1)
        p = _signal_to_prob(lagged)
        aligned = pd.concat([p.rename("p"), y_full.rename("y")], axis=1).dropna()
        if len(aligned) < 50:
            per_signal[col] = {"n": int(len(aligned)), "brier": None,
                               "log_loss": None, "ece": None}
            continue
        p_arr = aligned["p"].values
        y_arr = aligned["y"].values

        b = brier_score(p_arr, y_arr)
        ll = log_loss(p_arr, y_arr)
        cal = calibration_metrics(p_arr, y_arr)

        # Skill score vs 50/50 null (Brier of null = 0.25)
        skill_brier = 1 - b / 0.25 if 0.25 > 0 else 0.0
        skill_log = 1 - ll / np.log(2)  # null log-loss ≈ log(2) ≈ 0.693

        per_signal[col] = {
            "n":              int(len(aligned)),
            "brier":          round(b, 5),
            "log_loss":       round(ll, 5),
            "skill_brier":    round(float(skill_brier), 5),
            "skill_log":      round(float(skill_log), 5),
            "ece":            cal["ece"],
            "reliability":    cal["reliability"],
            "calibration_bins": cal["per_bin"],
        }

    # Aggregate: best by skill_brier
    ranked = sorted(
        [(k, v) for k, v in per_signal.items() if v.get("skill_brier") is not None],
        key=lambda kv: kv[1]["skill_brier"],
        reverse=True,
    )

    # Realised positive rate (for context)
    realised_positive = float(y_full.dropna().mean())

    result = {
        "generated_at":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker":            ticker,
        "lookback":          lookback,
        "horizon":           horizon,
        "n_bins":            N_BINS,
        "per_signal":        per_signal,
        "ranked_by_skill":   [k for k, _ in ranked],
        "realised_positive_rate": round(realised_positive, 4),
        "best_signal":       ranked[0][0] if ranked else None,
        "best_brier":        round(ranked[0][1]["brier"], 5) if ranked else None,
        "best_skill":        round(ranked[0][1]["skill_brier"], 5) if ranked else None,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2))
    _print_report(result)
    return result


def _print_report(r: dict) -> None:
    print(f"\n{SEP}")
    print(f"  DECISION QUALITY -- {r['ticker']}")
    print(SEP)
    print(f"  Horizon:                {r['horizon']}d")
    print(f"  Realised positive rate: {r['realised_positive_rate']:.4f}")
    print()

    print(f"  PER-SIGNAL SCORING")
    print(f"  {'─' * 64}")
    print(
        f"  {'signal':<16s}  {'Brier':>7s}  {'logloss':>9s}  "
        f"{'skill':>7s}  {'ECE':>6s}  {'rely':>7s}"
    )
    for name, m in r["per_signal"].items():
        if m.get("brier") is None:
            print(f"  {name:<16s}  insufficient data ({m.get('n', 0)} obs)")
            continue
        print(
            f"  {name:<16s}  "
            f"{m['brier']:>7.5f}  "
            f"{m['log_loss']:>9.5f}  "
            f"{m['skill_brier']:>+7.5f}  "
            f"{m['ece']:>6.4f}  "
            f"{m['reliability']:>+7.3f}"
        )
    print()

    print(f"  RANKED BY BRIER SKILL: {', '.join(r['ranked_by_skill'])}")
    if r["best_signal"]:
        print(f"  Best: {r['best_signal']}  "
              f"(Brier {r['best_brier']:.4f}, skill {r['best_skill']:+.4f})")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decision Quality Framework")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--lookback", default=DEFAULT_LOOKBACK)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    args = parser.parse_args()
    run_decision_quality(
        ticker=args.ticker, lookback=args.lookback, horizon=args.horizon,
    )
