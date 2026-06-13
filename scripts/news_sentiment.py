#!/usr/bin/env python3
"""
News Sentiment NLP Aggregator
==============================
Builds per-asset and aggregate sentiment time-series from the Perplexity
oracle scores already logged in data/oracle_history.csv. Each oracle score
is a 0-1 narrative-sentiment value pulled from Sonar Pro responses.

For each ticker the engine reports:
  - latest score
  - 7d EWMA  (short-term)
  - 21d EWMA (medium-term)
  - momentum (latest − 21d EWMA)
  - z-score vs full history
  - trend regime: BULLISH / NEUTRAL / BEARISH / SHIFTING

Plus aggregate cross-asset sentiment dispersion (max − min) and divergence
flags (e.g., gold bullish while silver bearish).

Output: data/news_sentiment.json
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

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "news_sentiment.json"
ORACLE_CSV = DATA_DIR / "oracle_history.csv"

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _load_oracle() -> pd.DataFrame:
    if not ORACLE_CSV.exists():
        raise FileNotFoundError(f"oracle_history.csv not found at {ORACLE_CSV}")
    df = pd.read_csv(ORACLE_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])
    return df


# ---------------------------------------------------------------------------
# Per-ticker stats
# ---------------------------------------------------------------------------
def _classify_regime(score: float, momentum: float, z: float) -> str:
    if abs(momentum) > 0.15 and abs(z) > 1.5:
        return "SHIFTING"
    if score > 0.60:
        return "BULLISH"
    if score < 0.40:
        return "BEARISH"
    return "NEUTRAL"


def per_ticker_stats(df: pd.DataFrame) -> dict:
    out = {}
    for ticker, group in df.groupby("ticker"):
        scores = group["score"].astype(float).reset_index(drop=True)
        if len(scores) < 2:
            out[ticker] = {
                "n_obs":       int(len(scores)),
                "latest":      float(scores.iloc[-1]) if len(scores) else None,
                "regime":      "NEUTRAL",
            }
            continue

        latest = float(scores.iloc[-1])
        # EWMAs (span = N observations, not days)
        ewma_short = float(scores.ewm(span=min(7, len(scores)), adjust=False).mean().iloc[-1])
        ewma_med = float(scores.ewm(span=min(21, len(scores)), adjust=False).mean().iloc[-1])
        momentum = latest - ewma_med
        mean = float(scores.mean())
        std = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0
        z = (latest - mean) / std if std > 1e-6 else 0.0
        regime = _classify_regime(latest, momentum, z)

        out[ticker] = {
            "n_obs":         int(len(scores)),
            "latest_date":   str(group["date"].iloc[-1].date()),
            "latest":        round(latest, 4),
            "ewma_7":        round(ewma_short, 4),
            "ewma_21":       round(ewma_med, 4),
            "momentum":      round(momentum, 4),
            "mean":          round(mean, 4),
            "std":           round(std, 4),
            "z_score":       round(z, 3),
            "regime":        regime,
        }
    return out


# ---------------------------------------------------------------------------
# Aggregate / cross-asset
# ---------------------------------------------------------------------------
def cross_asset(per_ticker: dict) -> dict:
    if not per_ticker:
        return {}
    scores = [v.get("latest") for v in per_ticker.values() if v.get("latest") is not None]
    if not scores:
        return {}
    avg = float(np.mean(scores))
    dispersion = float(max(scores) - min(scores))
    consensus = (
        "BULLISH" if avg > 0.6
        else "BEARISH" if avg < 0.4
        else "NEUTRAL"
    )

    # Divergence detection: do tickers disagree?
    regimes = [v.get("regime", "NEUTRAL") for v in per_ticker.values()]
    divergent = ("BULLISH" in regimes) and ("BEARISH" in regimes)

    # Top movers by absolute momentum
    movers = sorted(
        [(t, v.get("momentum", 0)) for t, v in per_ticker.items() if v.get("momentum")],
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )[:3]

    return {
        "avg_sentiment":       round(avg, 4),
        "dispersion":          round(dispersion, 4),
        "consensus_regime":    consensus,
        "divergent":           bool(divergent),
        "n_tickers":           len(per_ticker),
        "top_movers":          [
            {"ticker": t, "momentum": round(m, 4)} for t, m in movers
        ],
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_news_sentiment() -> dict:
    df = _load_oracle()
    per_ticker = per_ticker_stats(df)
    aggregate = cross_asset(per_ticker)

    result = {
        "generated_at":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_total_obs":   int(len(df)),
        "n_tickers":     int(df["ticker"].nunique()),
        "date_range":    {
            "from": str(df["date"].min().date()),
            "to":   str(df["date"].max().date()),
        },
        "per_ticker":    per_ticker,
        "aggregate":     aggregate,
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
    print(f"  NEWS SENTIMENT NLP")
    print(SEP)
    print(f"  Observations: {r['n_total_obs']}  tickers: {r['n_tickers']}")
    print(f"  Range:        {r['date_range']['from']} → {r['date_range']['to']}")
    print()

    print(f"  PER-TICKER SENTIMENT")
    print(f"  {'─' * 58}")
    print(
        f"  {'ticker':<10s}  {'latest':>7s}  {'7d':>6s}  {'21d':>6s}  "
        f"{'mom':>7s}  {'z':>6s}  {'regime':>10s}"
    )
    for t, v in r["per_ticker"].items():
        print(
            f"  {t:<10s}  "
            f"{v.get('latest', 0):>7.3f}  "
            f"{v.get('ewma_7', 0):>6.3f}  "
            f"{v.get('ewma_21', 0):>6.3f}  "
            f"{v.get('momentum', 0):>+7.3f}  "
            f"{v.get('z_score', 0):>+6.2f}  "
            f"{v.get('regime', ''):>10s}"
        )
    print()

    a = r["aggregate"]
    print(f"  CROSS-ASSET")
    print(f"  {'─' * 40}")
    print(f"  Avg sentiment:     {a.get('avg_sentiment', 0):.3f}")
    print(f"  Dispersion:        {a.get('dispersion', 0):.3f}")
    print(f"  Consensus:         {a.get('consensus_regime', 'n/a')}")
    print(f"  Divergent:         {a.get('divergent', False)}")
    if a.get("top_movers"):
        print(f"  Top movers:")
        for m in a["top_movers"]:
            print(f"    {m['ticker']:<10s}  Δ={m['momentum']:+.3f}")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="News Sentiment NLP")
    args = parser.parse_args()
    run_news_sentiment()
