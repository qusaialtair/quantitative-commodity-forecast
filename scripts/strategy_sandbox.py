#!/usr/bin/env python3
"""
Strategy Sandbox  (Phase X Stage 53)
======================================
Paper-trading harness that runs alternative strategies side-by-side with
the live champion so new ideas can soak for the institutional minimum of
6 months before touching production capital.

Each strategy is a dict spec with a signal generator function reference
and a position-sizing config. The sandbox compounds a virtual book daily
and reports rolling metrics.

Built-in strategies:
  - champion         current champion alpha source × full position
  - half_kelly       champion signal × half-Kelly sizing
  - mean_reversion   pure Bollinger %B fade (counter to champion)
  - long_only        passive 100% long (benchmark)

Records per-strategy daily P&L in data/sandbox_history.csv and reports
30d / 90d / 252d Sharpe + DD.

Output: data/strategy_sandbox.json
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
OUTPUT_FILE = DATA_DIR / "strategy_sandbox.json"
HISTORY_CSV = DATA_DIR / "sandbox_history.csv"

DEFAULT_TICKER = "GC=F"
DEFAULT_LOOKBACK = "2y"
SQ252 = float(np.sqrt(252))

LINE_W = 62
SEP = "━" * LINE_W


def _summarise(returns: pd.Series, label: str) -> dict:
    r = returns.dropna()
    if len(r) < 30:
        return {"strategy": label, "n": int(len(r)), "sharpe": 0.0,
                "ann_return_pct": 0.0, "ann_vol_pct": 0.0, "max_dd_pct": 0.0}
    ann_r = float(r.mean() * 252)
    ann_v = float(r.std() * SQ252)
    sharpe = ann_r / ann_v if ann_v > 1e-9 else 0.0
    cum = (1 + r).cumprod()
    dd = float((cum / cum.cummax() - 1).min())
    return {
        "strategy":        label,
        "n":               int(len(r)),
        "sharpe":          round(sharpe, 3),
        "ann_return_pct":  round(ann_r * 100, 3),
        "ann_vol_pct":     round(ann_v * 100, 3),
        "max_dd_pct":      round(dd * 100, 3),
    }


def _window_summary(strategy_returns: dict) -> dict:
    out = {}
    for label, r in strategy_returns.items():
        out[label] = {
            "30d":   _summarise(r.tail(30), label),
            "90d":   _summarise(r.tail(90), label),
            "252d":  _summarise(r.tail(252), label),
            "total": _summarise(r, label),
        }
    return out


def run_sandbox(
    ticker: str = DEFAULT_TICKER,
    lookback: str = DEFAULT_LOOKBACK,
) -> dict:
    df = _fetch_panel(ticker, lookback)
    signals = _generate_signals(df)
    r = df["gold"].pct_change().fillna(0)

    # Strategy spec
    strategies = {
        "champion":        signals["technical"].shift(1).clip(0, 1),   # long-only filter
        "half_kelly":      0.5 * signals["technical"].shift(1).clip(-1, 1),
        "mean_reversion":  signals["mean_reversion"].shift(1).clip(-1, 1),
        "long_only":       pd.Series(1.0, index=df.index),
        "regime_gated":    (signals["technical"] * signals["regime_filter"]).shift(1).clip(-1, 1),
    }

    strategy_returns = {}
    for label, position in strategies.items():
        strategy_returns[label] = (position * r).fillna(0)

    windows = _window_summary(strategy_returns)

    # Best by full-period Sharpe
    ranked = sorted(
        [(label, m["total"]["sharpe"]) for label, m in windows.items()],
        key=lambda kv: kv[1], reverse=True,
    )

    # Persist daily series for soak-test history
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    hist = pd.DataFrame({label: ret for label, ret in strategy_returns.items()})
    hist.to_csv(HISTORY_CSV)

    result = {
        "generated_at":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker":        ticker,
        "lookback":      lookback,
        "n_obs":         int(len(df)),
        "strategies":    list(strategies.keys()),
        "per_strategy":  windows,
        "ranked_by_total_sharpe": [k for k, _ in ranked],
        "best_strategy": ranked[0][0] if ranked else None,
        "history_csv":   str(HISTORY_CSV),
    }
    OUTPUT_FILE.write_text(json.dumps(result, indent=2))
    _print_report(result)
    return result


def _print_report(r: dict) -> None:
    print(f"\n{SEP}\n  STRATEGY SANDBOX -- {r['ticker']}\n{SEP}")
    print(f"  Obs: {r['n_obs']}  strategies: {len(r['strategies'])}")
    print()
    print(f"  TOTAL-PERIOD METRICS")
    print(f"  {'─' * 58}")
    print(
        f"  {'strategy':<16s}  {'Sharpe':>7s}  {'Ret %':>7s}  "
        f"{'Vol %':>6s}  {'DD %':>6s}"
    )
    for label in r["ranked_by_total_sharpe"]:
        m = r["per_strategy"][label]["total"]
        marker = " *" if label == r["best_strategy"] else "  "
        print(
            f"  {marker} {label:<14s}  "
            f"{m['sharpe']:>+7.3f}  "
            f"{m['ann_return_pct']:>+7.2f}  "
            f"{m['ann_vol_pct']:>6.2f}  "
            f"{m['max_dd_pct']:>+6.2f}"
        )
    print()
    print(f"  90-DAY SOAK-TEST WINDOW")
    print(f"  {'─' * 58}")
    for label in r["ranked_by_total_sharpe"]:
        m = r["per_strategy"][label]["90d"]
        print(
            f"  {label:<16s}  Sharpe {m['sharpe']:>+.3f}   "
            f"Ret {m['ann_return_pct']:>+.2f}%   DD {m['max_dd_pct']:>+.2f}%"
        )
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy Sandbox")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--lookback", default=DEFAULT_LOOKBACK)
    args = parser.parse_args()
    run_sandbox(ticker=args.ticker, lookback=args.lookback)
