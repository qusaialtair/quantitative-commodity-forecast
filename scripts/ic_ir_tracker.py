#!/usr/bin/env python3
"""
Information Coefficient / Information Ratio Tracker
=====================================================
Rolling Spearman rank-correlation (Information Coefficient) of each alpha
signal vs realised 5-day forward gold return, plus the derived Information
Ratio:

    IC_t  = spearman(signal[t-N..t], forward_return[t-N..t])
    IR    = mean(IC) / std(IC)            (Grinold-Kahn)

Windows: 21d, 63d, 252d.

For each signal also tracks:
  - current IC values across the three windows
  - IR
  - decay slope (linear fit of IC against window length)
  - IC half-life (days for IC to halve, computed in signal_decay)

This is the realised-performance counterpart to the predictive signal_decay
engine. Use IR > 0.5 as the institutional rule-of-thumb for a deployable signal.

Output: data/ic_ir_tracker.json
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
OUTPUT_FILE = DATA_DIR / "ic_ir_tracker.json"

DEFAULT_TICKER = "GC=F"
DEFAULT_LOOKBACK = "5y"
DEFAULT_HORIZON = 5
WINDOWS = [21, 63, 252]

LINE_W = 62
SEP = "━" * LINE_W


def _rolling_spearman_ic(signal: pd.Series, forward_ret: pd.Series, window: int) -> pd.Series:
    """Rolling Spearman correlation over a moving window."""
    df = pd.concat([signal.rename("s"), forward_ret.rename("y")], axis=1).dropna()
    ic_list = []
    idx_list = []
    for i in range(window, len(df) + 1):
        sub = df.iloc[i - window:i]
        if sub["s"].std() <= 0 or sub["y"].std() <= 0:
            ic_list.append(0.0)
        else:
            rs = sub["s"].rank()
            ry = sub["y"].rank()
            ic = float(rs.corr(ry))
            ic_list.append(0.0 if not np.isfinite(ic) else ic)
        idx_list.append(df.index[i - 1])
    return pd.Series(ic_list, index=idx_list)


def _summarise_window(ic_series: pd.Series) -> dict:
    s = ic_series.dropna()
    if len(s) == 0:
        return {"n": 0, "mean_ic": 0.0, "std_ic": 0.0, "ir": 0.0, "latest_ic": 0.0}
    mean_ic = float(s.mean())
    std_ic = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    ir = mean_ic / std_ic if std_ic > 1e-9 else 0.0
    return {
        "n":         int(len(s)),
        "mean_ic":   round(mean_ic, 4),
        "std_ic":    round(std_ic, 4),
        "ir":        round(ir, 3),
        "latest_ic": round(float(s.iloc[-1]), 4),
    }


def run_ic_ir_tracker(
    ticker: str = DEFAULT_TICKER,
    lookback: str = DEFAULT_LOOKBACK,
    horizon: int = DEFAULT_HORIZON,
) -> dict:
    df = _fetch_panel(ticker, lookback)
    signals = _generate_signals(df)
    returns = df["gold"].pct_change()
    forward = returns.rolling(horizon).sum().shift(-horizon)

    out = {}
    for col in signals.columns:
        per_window = {}
        for w in WINDOWS:
            ic_series = _rolling_spearman_ic(signals[col].shift(1), forward, w)
            per_window[f"w_{w}"] = _summarise_window(ic_series)

        # Decay slope: linear fit of mean_ic across windows
        windows_arr = np.array(WINDOWS, dtype=float)
        means_arr = np.array([per_window[f"w_{w}"]["mean_ic"] for w in WINDOWS])
        if means_arr.std() > 1e-9:
            slope, intercept = np.polyfit(windows_arr, means_arr, 1)
        else:
            slope, intercept = 0.0, float(means_arr.mean())

        # Headline (use 63d as canonical)
        canonical = per_window["w_63"]
        out[col] = {
            "per_window":     per_window,
            "ir_63d":         canonical["ir"],
            "ic_63d":         canonical["mean_ic"],
            "latest_ic":      canonical["latest_ic"],
            "decay_slope":    round(float(slope), 7),
            "decay_intercept":round(float(intercept), 4),
            "deployable":     bool(abs(canonical["ir"]) > 0.5),
        }

    # Rank signals
    ranked_ir = sorted(out.items(), key=lambda kv: kv[1]["ir_63d"], reverse=True)
    ranked_ic = sorted(out.items(), key=lambda kv: kv[1]["ic_63d"], reverse=True)

    deployable_signals = [name for name, m in out.items() if m["deployable"]]

    result = {
        "generated_at":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker":           ticker,
        "lookback":         lookback,
        "horizon":          horizon,
        "windows_days":     WINDOWS,
        "per_signal":       out,
        "ranked_by_ir":     [k for k, _ in ranked_ir],
        "ranked_by_ic":     [k for k, _ in ranked_ic],
        "deployable_signals": deployable_signals,
        "n_deployable":     len(deployable_signals),
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2))
    _print_report(result)
    return result


def _print_report(r: dict) -> None:
    print(f"\n{SEP}")
    print(f"  IC / IR TRACKER -- {r['ticker']}")
    print(SEP)
    print(f"  Horizon:    {r['horizon']}d forward return")
    print(f"  Windows:    {', '.join(f'{w}d' for w in r['windows_days'])}")
    print()

    print(f"  PER-SIGNAL ROLLING IC (Spearman vs forward return)")
    print(f"  {'─' * 64}")
    hdr = f"  {'signal':<16s}  "
    for w in r["windows_days"]:
        hdr += f"{'IC ' + str(w) + 'd':>9s}  "
    hdr += f"{'IR(63d)':>8s}  {'depl':>6s}"
    print(hdr)
    for name, m in r["per_signal"].items():
        pw = m["per_window"]
        row = f"  {name:<16s}  "
        for w in r["windows_days"]:
            row += f"{pw[f'w_{w}']['mean_ic']:>+9.4f}  "
        row += f"{m['ir_63d']:>+8.3f}  "
        row += f"{'YES' if m['deployable'] else 'no':>6s}"
        print(row)
    print()

    print(f"  RANKED BY IR (63d):    {', '.join(r['ranked_by_ir'])}")
    print(f"  RANKED BY IC (63d):    {', '.join(r['ranked_by_ic'])}")
    print(f"  Deployable signals:    {len(r['deployable_signals'])}  ({', '.join(r['deployable_signals']) or 'none'})")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IC/IR Tracker")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--lookback", default=DEFAULT_LOOKBACK)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    args = parser.parse_args()
    run_ic_ir_tracker(
        ticker=args.ticker, lookback=args.lookback, horizon=args.horizon,
    )
