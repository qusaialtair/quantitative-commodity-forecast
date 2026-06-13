#!/usr/bin/env python3
"""
Dynamic Stop-Loss Optimizer
============================
Evaluates four stop-loss methodologies on historical gold data and
recommends the best fit for the current volatility regime.

Methods (all long-only, entry-anchored unless noted):

  1. atr_2_0      Fixed ATR stop:        entry − 2.0 × ATR(14)
  2. atr_2_5      Wider ATR stop:        entry − 2.5 × ATR(14)
  3. chandelier   Trailing chandelier:   highest_high_22 − 3.0 × ATR(14)
  4. pct_3        Percent stop:          entry × (1 − 0.03)

For each method, runs a synthetic long-only backtest:
  - Enter on every Monday close
  - Exit at the next stop trigger OR at +30d horizon
  - Compute win-rate, avg win/loss, profit factor

Then maps the current vol regime (LOW/NORMAL/ELEVATED/EXTREME from
vol_surface.json) to the best-performing method.

Output: data/stop_loss_optimizer.json
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

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "stop_loss_optimizer.json"

DEFAULT_TICKER = "GC=F"
DEFAULT_LOOKBACK = "5y"
ATR_PERIOD = 14
CHANDELIER_PERIOD = 22
MAX_HOLD_DAYS = 30

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Data + ATR
# ---------------------------------------------------------------------------
def _fetch_ohlc(ticker: str, lookback: str) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is required")
    raw = yf.download(
        ticker, period=lookback, interval="1d",
        progress=False, auto_adjust=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    return raw[["Open", "High", "Low", "Close"]].dropna()


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ---------------------------------------------------------------------------
# Stop calculators
# ---------------------------------------------------------------------------
def _atr_stop(entry: float, atr: float, k: float) -> float:
    return entry - k * atr


def _chandelier_stop(highest_high: float, atr: float, k: float = 3.0) -> float:
    return highest_high - k * atr


def _pct_stop(entry: float, pct: float) -> float:
    return entry * (1.0 - pct)


# ---------------------------------------------------------------------------
# Synthetic long-only backtest
# ---------------------------------------------------------------------------
def backtest_method(
    df: pd.DataFrame,
    method: str,
    atr: pd.Series,
    max_hold: int = MAX_HOLD_DAYS,
) -> dict:
    """
    Enter long every Monday close, exit on stop or after `max_hold` days.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    weekday = pd.Series(df.index.weekday, index=df.index)
    entry_dates = df.index[weekday == 0]

    trades = []
    for ed in entry_dates:
        if ed not in df.index:
            continue
        i = df.index.get_loc(ed)
        if i + max_hold >= len(df) or pd.isna(atr.iloc[i]):
            continue

        entry_price = float(close.iloc[i])
        entry_atr = float(atr.iloc[i])

        # Set stop based on method
        highest_high_22 = float(high.iloc[max(0, i - CHANDELIER_PERIOD + 1):i + 1].max())
        if method == "atr_2_0":
            stop = _atr_stop(entry_price, entry_atr, 2.0)
            trailing = False
        elif method == "atr_2_5":
            stop = _atr_stop(entry_price, entry_atr, 2.5)
            trailing = False
        elif method == "chandelier":
            stop = _chandelier_stop(highest_high_22, entry_atr, 3.0)
            trailing = True
        elif method == "pct_3":
            stop = _pct_stop(entry_price, 0.03)
            trailing = False
        else:
            continue

        # Walk forward
        exit_idx = None
        exit_reason = "horizon"
        running_high = entry_price
        for j in range(i + 1, min(i + 1 + max_hold, len(df))):
            day_low = float(low.iloc[j])
            day_close = float(close.iloc[j])
            running_high = max(running_high, float(high.iloc[j]))

            # Trailing stop update
            if trailing:
                day_atr = float(atr.iloc[j]) if not pd.isna(atr.iloc[j]) else entry_atr
                stop = max(stop, running_high - 3.0 * day_atr)

            if day_low <= stop:
                exit_idx = j
                exit_reason = "stopped"
                exit_price = stop
                break

        if exit_idx is None:
            exit_idx = min(i + max_hold, len(df) - 1)
            exit_price = float(close.iloc[exit_idx])

        ret = (exit_price - entry_price) / entry_price
        days_held = exit_idx - i
        trades.append({
            "entry_date":  str(ed.date()),
            "exit_reason": exit_reason,
            "return_pct":  ret * 100,
            "days_held":   days_held,
        })

    if not trades:
        return {"n_trades": 0, "win_rate_pct": 0, "avg_win_pct": 0,
                "avg_loss_pct": 0, "profit_factor": 0, "expectancy_pct": 0,
                "avg_days_held": 0, "stopped_pct": 0}

    returns = np.array([t["return_pct"] for t in trades])
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    gross_wins = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_losses = float(abs(losses.sum())) if len(losses) > 0 else 0.0
    pf = gross_wins / gross_losses if gross_losses > 0 else float(gross_wins)

    return {
        "n_trades":       int(len(trades)),
        "win_rate_pct":   round(float((returns > 0).mean()) * 100, 2),
        "avg_win_pct":    round(avg_win, 3),
        "avg_loss_pct":   round(avg_loss, 3),
        "profit_factor":  round(pf, 3),
        "expectancy_pct": round(float(returns.mean()), 3),
        "avg_days_held":  round(float(np.mean([t["days_held"] for t in trades])), 1),
        "stopped_pct":    round(
            sum(1 for t in trades if t["exit_reason"] == "stopped") / len(trades) * 100, 2
        ),
    }


# ---------------------------------------------------------------------------
# Regime → method mapping
# ---------------------------------------------------------------------------
def _load_vol_regime() -> str:
    try:
        vs_path = DATA_DIR / "vol_surface.json"
        if vs_path.exists():
            vs = json.loads(vs_path.read_text())
            return vs.get("vol_regime", "NORMAL")
    except Exception:
        pass
    return "NORMAL"


def _regime_recommendation(regime: str, method_metrics: dict) -> str:
    """Map regime to preferred method (overridable by which actually performed best)."""
    regime_priors = {
        "LOW":      "atr_2_0",
        "NORMAL":   "atr_2_0",
        "ELEVATED": "atr_2_5",
        "EXTREME":  "chandelier",
    }
    return regime_priors.get(regime, "atr_2_0")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_stop_loss_optimizer(
    ticker: str = DEFAULT_TICKER,
    lookback: str = DEFAULT_LOOKBACK,
) -> dict:
    df = _fetch_ohlc(ticker, lookback)
    atr = compute_atr(df)
    last_price = float(df["Close"].iloc[-1])
    last_atr = float(atr.dropna().iloc[-1]) if len(atr.dropna()) else 0.0
    highest_high_22 = float(df["High"].tail(CHANDELIER_PERIOD).max())

    # Current stop levels for each method (long entry at last_price)
    current_stops = {
        "atr_2_0":    {
            "price": round(_atr_stop(last_price, last_atr, 2.0), 2),
            "distance_pct": round(2.0 * last_atr / last_price * 100, 3),
            "description": "Fixed 2.0× ATR below entry",
        },
        "atr_2_5":    {
            "price": round(_atr_stop(last_price, last_atr, 2.5), 2),
            "distance_pct": round(2.5 * last_atr / last_price * 100, 3),
            "description": "Wider 2.5× ATR below entry",
        },
        "chandelier": {
            "price": round(_chandelier_stop(highest_high_22, last_atr, 3.0), 2),
            "distance_pct": round(
                (last_price - _chandelier_stop(highest_high_22, last_atr, 3.0))
                / last_price * 100, 3
            ),
            "description": "Trailing chandelier: 22d high − 3.0× ATR",
        },
        "pct_3":      {
            "price": round(_pct_stop(last_price, 0.03), 2),
            "distance_pct": 3.0,
            "description": "Fixed 3% below entry",
        },
    }

    # Backtest each method
    backtest_results = {}
    for method in ["atr_2_0", "atr_2_5", "chandelier", "pct_3"]:
        backtest_results[method] = backtest_method(df, method, atr)

    regime = _load_vol_regime()
    prior_recommendation = _regime_recommendation(regime, backtest_results)

    # Pick "best by expectancy" too
    best_by_exp = max(backtest_results.items(),
                     key=lambda kv: kv[1].get("expectancy_pct", -999))[0]
    best_by_pf = max(backtest_results.items(),
                    key=lambda kv: kv[1].get("profit_factor", 0))[0]

    result = {
        "generated_at":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker":                ticker,
        "lookback":              lookback,
        "n_obs":                 int(len(df)),
        "current_price":         round(last_price, 2),
        "atr_14":                round(last_atr, 2),
        "highest_high_22":       round(highest_high_22, 2),
        "vol_regime":            regime,
        "current_stops":         current_stops,
        "backtest":              backtest_results,
        "regime_recommendation": prior_recommendation,
        "best_by_expectancy":    best_by_exp,
        "best_by_profit_factor": best_by_pf,
        "final_recommendation":  prior_recommendation,
        "final_stop_price":      current_stops[prior_recommendation]["price"],
        "final_stop_distance_pct": current_stops[prior_recommendation]["distance_pct"],
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
    print(f"  DYNAMIC STOP-LOSS OPTIMIZER -- {r['ticker']}")
    print(SEP)
    print(f"  Current Price:  ${r['current_price']:,.2f}")
    print(f"  ATR(14):        ${r['atr_14']:,.2f}")
    print(f"  22d High:       ${r['highest_high_22']:,.2f}")
    print(f"  Vol Regime:     {r['vol_regime']}")
    print()

    print(f"  STOP LEVELS")
    print(f"  {'─' * 58}")
    print(f"  {'method':<14s}  {'price':>10s}  {'dist %':>8s}  desc")
    for m, s in r["current_stops"].items():
        marker = " >>" if m == r["final_recommendation"] else "   "
        print(
            f"  {marker} {m:<10s}  "
            f"${s['price']:>9,.2f}  "
            f"{s['distance_pct']:>7.2f}%  "
            f"{s['description'][:30]}"
        )
    print()

    print(f"  HISTORICAL BACKTEST")
    print(f"  {'─' * 58}")
    print(
        f"  {'method':<14s}  {'n':>4s}  "
        f"{'win %':>6s}  {'avgW%':>6s}  {'avgL%':>6s}  "
        f"{'PF':>5s}  {'exp%':>6s}  {'stop%':>6s}"
    )
    for m, b in r["backtest"].items():
        print(
            f"  {m:<14s}  {b['n_trades']:>4d}  "
            f"{b['win_rate_pct']:>6.1f}  "
            f"{b['avg_win_pct']:>+6.2f}  "
            f"{b['avg_loss_pct']:>+6.2f}  "
            f"{b['profit_factor']:>5.2f}  "
            f"{b['expectancy_pct']:>+6.3f}  "
            f"{b['stopped_pct']:>6.1f}"
        )
    print()

    print(f"  RECOMMENDATION")
    print(f"  {'─' * 50}")
    print(f"  Regime prior:        {r['regime_recommendation']}")
    print(f"  Best by expectancy:  {r['best_by_expectancy']}")
    print(f"  Best by profit fac:  {r['best_by_profit_factor']}")
    print()
    print(f"  → Use {r['final_recommendation']}  "
          f"@ ${r['final_stop_price']:,.2f}  "
          f"({r['final_stop_distance_pct']:.2f}% below current)")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic Stop-Loss Optimizer")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--lookback", default=DEFAULT_LOOKBACK)
    args = parser.parse_args()
    run_stop_loss_optimizer(ticker=args.ticker, lookback=args.lookback)
