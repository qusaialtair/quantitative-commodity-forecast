#!/usr/bin/env python3
"""
Adverse Selection Detector
============================
Quantifies the post-execution markout risk by hour-of-day. Adverse selection
is the systematic loss a liquidity demander suffers when prices move against
their fills in the seconds and minutes after the trade.

Lacking real fill data, this engine uses 1-hour bar history as a proxy:

  For each hour h ∈ [0..23] (UTC):
      mean_fwd_return_pct   average 1-hour-forward return at this hour
      mean_abs_return_pct   average absolute 1-hour return (microstructure noise)
      realized_vol_pct      std of 1-hour returns at this hour
      n_obs                 number of historical hours

  adverse_selection_score = mean_abs_fwd_return × realized_vol (annualised)

The intuition is that hours combining persistent post-trade drift with high
volatility are the worst times to cross the spread.

Output: data/adverse_selection.json
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
OUTPUT_FILE = DATA_DIR / "adverse_selection.json"

DEFAULT_TICKER = "GC=F"
DEFAULT_LOOKBACK = "60d"
SQ_HOURS_YEAR = float(np.sqrt(252 * 24))  # ~24h gold market

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _fetch_hourly(ticker: str, lookback: str) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is required")
    raw = yf.download(
        ticker, period=lookback, interval="1h",
        progress=False, auto_adjust=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    return raw.dropna()


# ---------------------------------------------------------------------------
# Per-hour metrics
# ---------------------------------------------------------------------------
def compute_hour_metrics(hourly: pd.DataFrame) -> pd.DataFrame:
    close = hourly["Close"].copy()
    fwd_returns = close.pct_change().shift(-1)
    df = pd.DataFrame({
        "fwd_return": fwd_returns,
        "hour":       hourly.index.hour,
    }).dropna()

    grouped = df.groupby("hour")["fwd_return"].agg(
        mean_fwd="mean",
        std_fwd="std",
        mean_abs_fwd=lambda x: float(np.abs(x).mean()),
        n="count",
    ).reset_index()

    # Adverse selection score: persistent drift × volatility (annualised pct units)
    grouped["adverse_score"] = (
        np.abs(grouped["mean_fwd"]) * grouped["std_fwd"] * SQ_HOURS_YEAR * 10_000
    )
    return grouped


def rank_hours(hour_df: pd.DataFrame) -> tuple[list, list]:
    sorted_df = hour_df.sort_values("adverse_score", ascending=False)
    worst = sorted_df.head(5).to_dict(orient="records")
    best = sorted_df.tail(5).to_dict(orient="records")
    return worst, best


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_adverse_selection(
    ticker: str = DEFAULT_TICKER,
    lookback: str = DEFAULT_LOOKBACK,
) -> dict:
    hourly = _fetch_hourly(ticker, lookback)
    if len(hourly) < 50:
        raise RuntimeError(
            f"Only {len(hourly)} hourly bars — yfinance limits intraday "
            f"to recent history. Try lookback='30d'."
        )

    hour_df = compute_hour_metrics(hourly)
    worst, best = rank_hours(hour_df)

    # Cleaner outputs
    hour_records = []
    for _, row in hour_df.iterrows():
        hour_records.append({
            "hour_utc":            int(row["hour"]),
            "mean_fwd_return_bps": round(float(row["mean_fwd"]) * 10_000, 3),
            "std_fwd_pct":         round(float(row["std_fwd"]) * 100, 3),
            "mean_abs_fwd_bps":    round(float(row["mean_abs_fwd"]) * 10_000, 3),
            "adverse_score":       round(float(row["adverse_score"]), 3),
            "n_obs":               int(row["n"]),
        })

    # Sessions  (rough — exchange clocks vary; this is just UTC bucketing)
    sessions = {
        "ASIA":   list(range(0, 8)),
        "LONDON": list(range(8, 13)),
        "OVERLAP":list(range(13, 17)),
        "NY":     list(range(17, 22)),
        "LATE":   list(range(22, 24)),
    }
    session_summary = {}
    for sess_name, hrs in sessions.items():
        subset = hour_df[hour_df["hour"].isin(hrs)]
        if len(subset) == 0:
            continue
        # Weighted by n_obs
        n_total = subset["n"].sum()
        weighted_adverse = float(
            (subset["adverse_score"] * subset["n"]).sum() / max(n_total, 1)
        )
        weighted_vol = float(
            (subset["std_fwd"] * subset["n"]).sum() / max(n_total, 1) * 100
        )
        session_summary[sess_name] = {
            "avg_adverse_score":   round(weighted_adverse, 3),
            "avg_vol_pct":         round(weighted_vol, 3),
            "n_obs":               int(n_total),
        }

    result = {
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker":         ticker,
        "lookback":       lookback,
        "n_hourly_bars":  int(len(hourly)),
        "hour_metrics":   hour_records,
        "worst_hours":    [
            {"hour_utc": int(w["hour"]),
             "adverse_score": round(float(w["adverse_score"]), 3),
             "n_obs": int(w["n"])}
            for w in worst
        ],
        "best_hours": [
            {"hour_utc": int(w["hour"]),
             "adverse_score": round(float(w["adverse_score"]), 3),
             "n_obs": int(w["n"])}
            for w in best
        ],
        "session_summary": session_summary,
        "recommendation": _recommendation(worst, best, session_summary),
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2))
    _print_report(result)
    return result


def _recommendation(worst: list, best: list, sessions: dict) -> str:
    if not sessions:
        return "Not enough data for recommendation."
    worst_sess = max(sessions.items(), key=lambda kv: kv[1]["avg_adverse_score"])
    best_sess = min(sessions.items(), key=lambda kv: kv[1]["avg_adverse_score"])
    worst_hours = [str(int(w["hour"])) for w in worst[:3]]
    return (
        f"Avoid {worst_sess[0]} session (score {worst_sess[1]['avg_adverse_score']:.2f}); "
        f"prefer {best_sess[0]} (score {best_sess[1]['avg_adverse_score']:.2f}). "
        f"Highest-risk hours (UTC): {', '.join(worst_hours)}."
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_report(r: dict) -> None:
    print(f"\n{SEP}")
    print(f"  ADVERSE SELECTION DETECTOR -- {r['ticker']}")
    print(SEP)
    print(f"  Hourly bars:    {r['n_hourly_bars']}")
    print(f"  Lookback:       {r['lookback']}")
    print()

    print(f"  HOURLY METRICS (UTC)")
    print(f"  {'─' * 58}")
    print(
        f"  {'hr':>3s}  {'fwd_mean':>10s}  {'fwd_abs':>9s}  "
        f"{'vol %':>7s}  {'score':>7s}  {'n':>4s}"
    )
    for h in r["hour_metrics"]:
        print(
            f"  {h['hour_utc']:>3d}  "
            f"{h['mean_fwd_return_bps']:>+10.2f}  "
            f"{h['mean_abs_fwd_bps']:>9.2f}  "
            f"{h['std_fwd_pct']:>7.3f}  "
            f"{h['adverse_score']:>7.2f}  "
            f"{h['n_obs']:>4d}"
        )
    print()

    print(f"  SESSION SUMMARY")
    print(f"  {'─' * 48}")
    print(f"  {'session':<10s}  {'score':>8s}  {'vol %':>8s}  {'n':>5s}")
    for sess_name, s in r["session_summary"].items():
        print(
            f"  {sess_name:<10s}  "
            f"{s['avg_adverse_score']:>8.2f}  "
            f"{s['avg_vol_pct']:>8.3f}  "
            f"{s['n_obs']:>5d}"
        )
    print()

    print(f"  TOP-3 WORST HOURS (highest adverse selection)")
    for w in r["worst_hours"][:3]:
        print(f"    {w['hour_utc']:>2d} UTC  score={w['adverse_score']:.2f}  n={w['n_obs']}")
    print()
    print(f"  TOP-3 BEST HOURS  (lowest adverse selection)")
    for w in r["best_hours"][:3]:
        print(f"    {w['hour_utc']:>2d} UTC  score={w['adverse_score']:.2f}  n={w['n_obs']}")
    print()
    print(f"  → {r['recommendation']}")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adverse Selection Detector")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--lookback", default=DEFAULT_LOOKBACK)
    args = parser.parse_args()
    run_adverse_selection(ticker=args.ticker, lookback=args.lookback)
