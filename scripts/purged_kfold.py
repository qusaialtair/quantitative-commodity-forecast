#!/usr/bin/env python3
"""
Purged K-Fold Cross-Validation  (López de Prado, AFML 2018)
=============================================================
Eliminates two forms of leakage that break standard K-fold on financial
time series:

  1. Label overlap: a training sample at time t whose target spans into
     the test fold gives the model future information.
  2. Serial correlation: even after purging, samples adjacent to the test
     fold's edges contain near-duplicate information. An "embargo" period
     after each test fold removes them.

For each fold k:
  Test  = fold k                                  (contiguous block)
  Train = all samples EXCEPT
          • the test fold
          • samples whose [t, t+H] horizon overlaps the test fold (purged)
          • samples in [test_end, test_end + embargo)  (embargoed)

We evaluate the equal-weighted signal blend (from alpha_attribution) on
each test fold and report:
  - per-fold Sharpe / DD / return
  - mean ± std across folds
  - stability ratio = mean(Sharpe) / std(Sharpe)  (higher = more stable)

Output: data/purged_kfold.json
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
OUTPUT_FILE = DATA_DIR / "purged_kfold.json"

DEFAULT_TICKER = "GC=F"
DEFAULT_LOOKBACK = "5y"
DEFAULT_N_SPLITS = 5
DEFAULT_LABEL_HORIZON = 5
DEFAULT_EMBARGO_PCT = 0.02
SQ252 = float(np.sqrt(252))

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Purged K-Fold splitter
# ---------------------------------------------------------------------------
class PurgedKFold:
    """
    Generates (train_idx, test_idx) pairs that respect:
      - chronological ordering (test fold is contiguous, never randomised)
      - label-horizon purging (no train sample's label leaks into test)
      - embargo (no train sample from immediately after test)
    """
    def __init__(
        self,
        n_splits: int = DEFAULT_N_SPLITS,
        label_horizon_days: int = DEFAULT_LABEL_HORIZON,
        embargo_pct: float = DEFAULT_EMBARGO_PCT,
    ):
        self.n_splits = n_splits
        self.label_horizon_days = label_horizon_days
        self.embargo_pct = embargo_pct

    def split(self, n_samples: int):
        embargo = int(np.ceil(n_samples * self.embargo_pct))
        fold_size = n_samples // self.n_splits
        for k in range(self.n_splits):
            test_start = k * fold_size
            test_end = test_start + fold_size if k < self.n_splits - 1 else n_samples
            test_idx = np.arange(test_start, test_end)

            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_idx] = False

            # Purge: train samples whose label horizon enters the test fold
            # Sample at t has label spanning [t, t + horizon)
            for i in range(max(0, test_start - self.label_horizon_days), test_start):
                if i + self.label_horizon_days >= test_start:
                    train_mask[i] = False

            # Embargo: skip samples in [test_end, test_end + embargo)
            embargo_end = min(test_end + embargo, n_samples)
            train_mask[test_end:embargo_end] = False

            train_idx = np.where(train_mask)[0]
            yield train_idx, test_idx


# ---------------------------------------------------------------------------
# Fold-level backtest
# ---------------------------------------------------------------------------
def _fold_backtest(
    returns: pd.Series, signals: pd.DataFrame, test_idx: np.ndarray,
) -> dict:
    """Evaluate equal-weighted blended-signal strategy on a test fold."""
    if len(test_idx) < 30:
        return {"sharpe": 0.0, "ann_return_pct": 0.0,
                "ann_vol_pct": 0.0, "max_drawdown_pct": 0.0, "n_obs": int(len(test_idx))}

    sigs_test = signals.iloc[test_idx]
    rets_test = returns.iloc[test_idx]
    lagged = sigs_test.shift(1).fillna(0)
    blended = lagged.mean(axis=1)
    strat_rets = (blended * rets_test).dropna()

    if strat_rets.std() <= 1e-9:
        return {"sharpe": 0.0, "ann_return_pct": 0.0,
                "ann_vol_pct": 0.0, "max_drawdown_pct": 0.0, "n_obs": int(len(strat_rets))}

    ann_ret = float(strat_rets.mean() * 252)
    ann_vol = float(strat_rets.std() * SQ252)
    sharpe = ann_ret / ann_vol
    cum = (1 + strat_rets).cumprod()
    max_dd = float((cum / cum.cummax() - 1).min())

    return {
        "n_obs":            int(len(strat_rets)),
        "sharpe":           round(sharpe, 3),
        "ann_return_pct":   round(ann_ret * 100, 3),
        "ann_vol_pct":      round(ann_vol * 100, 3),
        "max_drawdown_pct": round(max_dd * 100, 3),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_purged_kfold(
    ticker: str = DEFAULT_TICKER,
    lookback: str = DEFAULT_LOOKBACK,
    n_splits: int = DEFAULT_N_SPLITS,
    label_horizon: int = DEFAULT_LABEL_HORIZON,
    embargo_pct: float = DEFAULT_EMBARGO_PCT,
) -> dict:
    df = _fetch_panel(ticker, lookback)
    signals = _generate_signals(df)
    returns = df["gold"].pct_change().fillna(0)
    n = len(df)

    splitter = PurgedKFold(
        n_splits=n_splits,
        label_horizon_days=label_horizon,
        embargo_pct=embargo_pct,
    )

    fold_results = []
    for k, (train_idx, test_idx) in enumerate(splitter.split(n)):
        m = _fold_backtest(returns, signals, test_idx)
        fold_results.append({
            "fold":           k + 1,
            "train_size":     int(len(train_idx)),
            "test_size":      int(len(test_idx)),
            "test_range":     [int(test_idx[0]), int(test_idx[-1])],
            **m,
        })

    sharpes = np.array([f["sharpe"] for f in fold_results])
    mean_sharpe = float(sharpes.mean())
    std_sharpe = float(sharpes.std(ddof=1)) if len(sharpes) > 1 else 0.0
    stability = mean_sharpe / std_sharpe if std_sharpe > 1e-9 else float("inf")

    dds = np.array([f["max_drawdown_pct"] for f in fold_results])
    avg_dd = float(dds.mean())
    worst_dd = float(dds.min())

    result = {
        "generated_at":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker":           ticker,
        "lookback":         lookback,
        "n_obs":            int(n),
        "n_splits":         n_splits,
        "label_horizon":    label_horizon,
        "embargo_pct":      embargo_pct,
        "embargo_days":     int(np.ceil(n * embargo_pct)),
        "fold_results":     fold_results,
        "summary": {
            "mean_sharpe":           round(mean_sharpe, 3),
            "std_sharpe":            round(std_sharpe, 3),
            "stability_ratio":       round(stability, 3) if std_sharpe > 0 else None,
            "min_sharpe":            round(float(sharpes.min()), 3),
            "max_sharpe":            round(float(sharpes.max()), 3),
            "avg_max_drawdown_pct":  round(avg_dd, 3),
            "worst_max_drawdown_pct":round(worst_dd, 3),
            "n_positive_folds":      int((sharpes > 0).sum()),
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
    print(f"  PURGED K-FOLD CV -- {r['ticker']}")
    print(SEP)
    print(f"  Observations:   {r['n_obs']}")
    print(f"  Folds:          {r['n_splits']}")
    print(f"  Label horizon:  {r['label_horizon']}d  (purge window)")
    print(f"  Embargo:        {r['embargo_days']}d  ({r['embargo_pct']:.1%})")
    print()

    print(f"  PER-FOLD METRICS")
    print(f"  {'─' * 58}")
    print(
        f"  {'fold':>4s}  {'train':>6s}  {'test':>6s}  "
        f"{'Sharpe':>7s}  {'Ret %':>6s}  {'DD %':>6s}"
    )
    for f in r["fold_results"]:
        print(
            f"  {f['fold']:>4d}  "
            f"{f['train_size']:>6d}  "
            f"{f['test_size']:>6d}  "
            f"{f['sharpe']:>+7.3f}  "
            f"{f['ann_return_pct']:>+6.2f}  "
            f"{f['max_drawdown_pct']:>+6.2f}"
        )
    print()

    s = r["summary"]
    print(f"  AGGREGATE STABILITY")
    print(f"  {'─' * 50}")
    print(f"  Mean Sharpe:        {s['mean_sharpe']:+.3f}")
    print(f"  Std Sharpe:         {s['std_sharpe']:.3f}")
    stab = s["stability_ratio"]
    print(f"  Stability ratio:    {stab if stab is not None else 'inf'}")
    print(f"  Range:              {s['min_sharpe']:+.3f}  →  {s['max_sharpe']:+.3f}")
    print(f"  Positive folds:     {s['n_positive_folds']} / {r['n_splits']}")
    print(f"  Avg max DD:         {s['avg_max_drawdown_pct']:.2f}%")
    print(f"  Worst fold DD:      {s['worst_max_drawdown_pct']:.2f}%")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Purged K-Fold CV")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--lookback", default=DEFAULT_LOOKBACK)
    parser.add_argument("--splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--horizon", type=int, default=DEFAULT_LABEL_HORIZON)
    parser.add_argument("--embargo", type=float, default=DEFAULT_EMBARGO_PCT)
    args = parser.parse_args()
    run_purged_kfold(
        ticker=args.ticker,
        lookback=args.lookback,
        n_splits=args.splits,
        label_horizon=args.horizon,
        embargo_pct=args.embargo,
    )
