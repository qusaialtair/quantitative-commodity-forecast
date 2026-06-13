#!/usr/bin/env python3
"""
ETF Flow Tracker
==================
Approximates daily flows into the major precious-metals ETFs using a
volume-weighted price-change proxy (signed dollar volume).

For each ETF in (GLD, SLV, IAU, GLDM, SIVR, BAR):

    daily_dollar_volume = close × volume
    flow_proxy_usd      = daily_dollar_volume × sign(close_change)
    cumulative          = running sum over the lookback window
    7d / 21d / 63d EWMAs

The flow_proxy is not exact AUM accounting — it's a sentiment-grade proxy
that goes up on net buying pressure and down on net selling. The relative
shape across ETFs is what matters.

Reports:
  - Latest daily flow per ETF
  - 7d / 21d cumulative net flow
  - Trend regime: INFLOW / NEUTRAL / OUTFLOW
  - Aggregate gold-bucket vs silver-bucket flows
  - Divergence flag (gold inflow + silver outflow, or vice versa)

Output: data/etf_flows.json
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
OUTPUT_FILE = DATA_DIR / "etf_flows.json"

DEFAULT_LOOKBACK = "3mo"
GOLD_ETFS = ["GLD", "IAU", "GLDM", "BAR"]
SILVER_ETFS = ["SLV", "SIVR"]

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _fetch_etf(ticker: str, lookback: str) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is required")
    raw = yf.download(
        ticker, period=lookback, interval="1d",
        progress=False, auto_adjust=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    return raw[["Close", "Volume"]].dropna()


# ---------------------------------------------------------------------------
# Flow proxy
# ---------------------------------------------------------------------------
def compute_flows(df: pd.DataFrame) -> pd.Series:
    """Signed dollar volume per day."""
    close = df["Close"]
    vol = df["Volume"]
    change = close.diff()
    sign = np.sign(change)
    dollar_volume = close * vol
    flow = dollar_volume * sign
    return flow.fillna(0)


def per_etf_summary(ticker: str, df: pd.DataFrame) -> dict:
    flow = compute_flows(df)
    latest = float(flow.iloc[-1]) if len(flow) else 0.0
    flow_7d = float(flow.tail(7).sum()) if len(flow) >= 7 else float(flow.sum())
    flow_21d = float(flow.tail(21).sum()) if len(flow) >= 21 else float(flow.sum())
    flow_63d = float(flow.sum())

    avg_daily_volume_usd = float((df["Close"] * df["Volume"]).mean())

    # Z-score of latest flow vs window
    if len(flow) > 5 and flow.std() > 0:
        z = float((flow.iloc[-1] - flow.mean()) / flow.std())
    else:
        z = 0.0

    # Regime
    threshold = avg_daily_volume_usd * 5  # 5 days worth of average
    if flow_21d > threshold:
        regime = "INFLOW"
    elif flow_21d < -threshold:
        regime = "OUTFLOW"
    else:
        regime = "NEUTRAL"

    return {
        "ticker":              ticker,
        "n_obs":               int(len(df)),
        "latest_close":        round(float(df["Close"].iloc[-1]), 2),
        "latest_flow_usd":     round(latest, 2),
        "flow_7d_usd":         round(flow_7d, 2),
        "flow_21d_usd":        round(flow_21d, 2),
        "flow_63d_usd":        round(flow_63d, 2),
        "avg_daily_volume_usd":round(avg_daily_volume_usd, 2),
        "z_score":             round(z, 3),
        "regime":              regime,
    }


def aggregate_bucket(per_etf: dict, bucket_tickers: list) -> dict:
    members = [per_etf[t] for t in bucket_tickers if t in per_etf]
    if not members:
        return {"n_etfs": 0}
    flow_7d_sum = float(sum(m["flow_7d_usd"] for m in members))
    flow_21d_sum = float(sum(m["flow_21d_usd"] for m in members))
    flow_63d_sum = float(sum(m["flow_63d_usd"] for m in members))
    n_inflow = sum(1 for m in members if m["regime"] == "INFLOW")
    n_outflow = sum(1 for m in members if m["regime"] == "OUTFLOW")
    if n_inflow > n_outflow:
        bucket_regime = "INFLOW"
    elif n_outflow > n_inflow:
        bucket_regime = "OUTFLOW"
    else:
        bucket_regime = "MIXED"
    return {
        "n_etfs":           len(members),
        "flow_7d_usd":      round(flow_7d_sum, 2),
        "flow_21d_usd":     round(flow_21d_sum, 2),
        "flow_63d_usd":     round(flow_63d_sum, 2),
        "n_inflow":         n_inflow,
        "n_outflow":        n_outflow,
        "bucket_regime":    bucket_regime,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_etf_flow_tracker(lookback: str = DEFAULT_LOOKBACK) -> dict:
    all_etfs = GOLD_ETFS + SILVER_ETFS
    per_etf = {}
    for tick in all_etfs:
        try:
            df = _fetch_etf(tick, lookback)
            if len(df) < 5:
                continue
            per_etf[tick] = per_etf_summary(tick, df)
        except Exception as exc:
            per_etf[tick] = {"ticker": tick, "error": str(exc)}

    gold_bucket = aggregate_bucket(per_etf, GOLD_ETFS)
    silver_bucket = aggregate_bucket(per_etf, SILVER_ETFS)

    # Divergence flag
    divergent = (
        gold_bucket.get("bucket_regime") == "INFLOW"
        and silver_bucket.get("bucket_regime") == "OUTFLOW"
    ) or (
        gold_bucket.get("bucket_regime") == "OUTFLOW"
        and silver_bucket.get("bucket_regime") == "INFLOW"
    )

    # Headline read: dominant regime across all ETFs
    inflows = sum(1 for v in per_etf.values() if v.get("regime") == "INFLOW")
    outflows = sum(1 for v in per_etf.values() if v.get("regime") == "OUTFLOW")
    if inflows > outflows:
        headline = "NET_INFLOW"
    elif outflows > inflows:
        headline = "NET_OUTFLOW"
    else:
        headline = "MIXED"

    result = {
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lookback":       lookback,
        "per_etf":        per_etf,
        "gold_bucket":    gold_bucket,
        "silver_bucket":  silver_bucket,
        "divergent":      bool(divergent),
        "headline":       headline,
        "n_inflows":      int(inflows),
        "n_outflows":     int(outflows),
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
    print(f"  ETF FLOW TRACKER")
    print(SEP)
    print(f"  Lookback:       {r['lookback']}")
    print()

    print(f"  PER-ETF SUMMARY")
    print(f"  {'─' * 64}")
    print(
        f"  {'ticker':<8s}  {'last':>8s}  "
        f"{'flow_7d':>14s}  {'flow_21d':>14s}  {'regime':>10s}"
    )
    for t, v in r["per_etf"].items():
        if "error" in v:
            print(f"  {t:<8s}  ERROR: {v['error'][:40]}")
            continue
        print(
            f"  {t:<8s}  ${v['latest_close']:>7,.2f}  "
            f"${v['flow_7d_usd']:>13,.0f}  "
            f"${v['flow_21d_usd']:>13,.0f}  "
            f"{v['regime']:>10s}"
        )
    print()

    g = r["gold_bucket"]
    s = r["silver_bucket"]
    print(f"  BUCKET SUMMARY")
    print(f"  {'─' * 50}")
    print(
        f"  GOLD ({g.get('n_etfs', 0)} ETFs):   "
        f"7d ${g.get('flow_7d_usd', 0):>12,.0f}  "
        f"21d ${g.get('flow_21d_usd', 0):>12,.0f}  "
        f"{g.get('bucket_regime', 'n/a')}"
    )
    print(
        f"  SILVER ({s.get('n_etfs', 0)} ETFs): "
        f"7d ${s.get('flow_7d_usd', 0):>12,.0f}  "
        f"21d ${s.get('flow_21d_usd', 0):>12,.0f}  "
        f"{s.get('bucket_regime', 'n/a')}"
    )
    print()
    print(f"  HEADLINE:    {r['headline']}  ({r['n_inflows']}↑ / {r['n_outflows']}↓)")
    if r["divergent"]:
        print(f"  ⚠ DIVERGENT — gold and silver flows are opposite")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETF Flow Tracker")
    parser.add_argument("--lookback", default=DEFAULT_LOOKBACK)
    args = parser.parse_args()
    run_etf_flow_tracker(lookback=args.lookback)
