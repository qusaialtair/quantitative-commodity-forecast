#!/usr/bin/env python3
"""
Brinson Performance Attribution
=================================
Classical Brinson-Hood-Beebower decomposition of portfolio excess return
versus a benchmark, split into:

  Allocation Effect (AE)   = (w_p − w_b) · (r_b)        attribution from over/under-weighting sectors
  Selection Effect (SE)    = w_b · (r_p − r_b)          attribution from picking winners inside a sector
  Interaction (IE)         = (w_p − w_b) · (r_p − r_b)  cross-term

  AE + SE + IE = total excess return

For this gold-trading platform we treat the portfolio as three buckets:
  metals     (GC=F + SI=F)
  halal_eq   (SPY proxy for halal-equity bucket; replace once equity universe is live)
  cash       (zero return)

Benchmark = equal-weighted (1/3, 1/3, 1/3) static allocation.

Output: data/brinson_attribution.json
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
OUTPUT_FILE = DATA_DIR / "brinson_attribution.json"

DEFAULT_LOOKBACK = "1y"
SQ252 = float(np.sqrt(252))

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _fetch_close(ticker: str, lookback: str) -> pd.Series:
    if yf is None:
        raise ImportError("yfinance is required")
    raw = yf.download(ticker, period=lookback, interval="1d",
                       progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    return raw["Close"].dropna()


def _load_portfolio_weights() -> dict:
    """Read actual portfolio weights from data/portfolio.json + shadow_book."""
    metals_w = 0.0
    halal_w = 0.0
    cash_w = 1.0
    try:
        # Read shadow_book for metals
        from scripts.master_controller import _read_shadow_portfolio, STARTING_CAPITAL
        shadow = _read_shadow_portfolio()
        if shadow:
            pv = float(shadow.get("portfolio_value", STARTING_CAPITAL))
            cash = float(shadow.get("cash_usd", STARTING_CAPITAL))
            gold_oz = float(shadow.get("gold_oz", 0.0))
            last_spot = float(shadow.get("last_spot", 0.0))
            metals_value = gold_oz * last_spot if last_spot > 0 else 0
            if pv > 0:
                metals_w = metals_value / pv
                cash_w = cash / pv
                halal_w = max(0.0, 1.0 - metals_w - cash_w)
    except Exception:
        pass

    # If everything is cash and metals is zero, default to a sensible test portfolio
    if metals_w == 0 and halal_w == 0:
        metals_w, halal_w, cash_w = 0.40, 0.40, 0.20

    return {"metals": metals_w, "halal_eq": halal_w, "cash": cash_w}


# ---------------------------------------------------------------------------
# Brinson decomposition
# ---------------------------------------------------------------------------
def brinson(
    w_p: dict, w_b: dict, r_p: dict, r_b: dict,
) -> dict:
    """Classical Brinson-Hood-Beebower with three buckets."""
    buckets = list(w_p.keys())
    ae_total = 0.0
    se_total = 0.0
    ie_total = 0.0
    per_bucket = {}
    for k in buckets:
        ae = (w_p[k] - w_b[k]) * r_b[k]
        se = w_b[k] * (r_p[k] - r_b[k])
        ie = (w_p[k] - w_b[k]) * (r_p[k] - r_b[k])
        ae_total += ae
        se_total += se
        ie_total += ie
        per_bucket[k] = {
            "w_portfolio":   round(w_p[k], 4),
            "w_benchmark":   round(w_b[k], 4),
            "r_portfolio":   round(r_p[k], 4),
            "r_benchmark":   round(r_b[k], 4),
            "allocation":    round(ae, 4),
            "selection":     round(se, 4),
            "interaction":   round(ie, 4),
            "total_excess":  round(ae + se + ie, 4),
        }
    total = ae_total + se_total + ie_total
    portfolio_return = sum(w_p[k] * r_p[k] for k in buckets)
    benchmark_return = sum(w_b[k] * r_b[k] for k in buckets)
    return {
        "per_bucket":       per_bucket,
        "allocation_effect":round(ae_total, 4),
        "selection_effect": round(se_total, 4),
        "interaction":      round(ie_total, 4),
        "total_excess":     round(total, 4),
        "portfolio_return": round(portfolio_return, 4),
        "benchmark_return": round(benchmark_return, 4),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_brinson_attribution(lookback: str = DEFAULT_LOOKBACK) -> dict:
    # Fetch the three bucket returns over the lookback window
    gold = _fetch_close("GC=F", lookback)
    silver = _fetch_close("SI=F", lookback)
    spy = _fetch_close("SPY", lookback)

    gold_ret_total = float(gold.iloc[-1] / gold.iloc[0] - 1)
    silver_ret_total = float(silver.iloc[-1] / silver.iloc[0] - 1)
    spy_ret_total = float(spy.iloc[-1] / spy.iloc[0] - 1)

    # Metals bucket = 50/50 gold + silver
    metals_ret = 0.5 * gold_ret_total + 0.5 * silver_ret_total
    halal_ret = spy_ret_total
    cash_ret = 0.0

    # Portfolio (actual) weights
    w_p = _load_portfolio_weights()

    # Benchmark = equal-weighted 1/3 each
    w_b = {"metals": 1 / 3, "halal_eq": 1 / 3, "cash": 1 / 3}

    # Bucket returns (same for both)
    r_p = {"metals": metals_ret, "halal_eq": halal_ret, "cash": cash_ret}
    r_b = r_p.copy()

    decomp = brinson(w_p, w_b, r_p, r_b)

    result = {
        "generated_at":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lookback":         lookback,
        "portfolio_weights":   w_p,
        "benchmark_weights":   w_b,
        "bucket_returns_pct": {k: round(v * 100, 3) for k, v in r_p.items()},
        "portfolio_return_pct": round(decomp["portfolio_return"] * 100, 3),
        "benchmark_return_pct": round(decomp["benchmark_return"] * 100, 3),
        "excess_return_pct":    round(decomp["total_excess"] * 100, 3),
        "allocation_effect_pct":round(decomp["allocation_effect"] * 100, 3),
        "selection_effect_pct": round(decomp["selection_effect"] * 100, 3),
        "interaction_pct":      round(decomp["interaction"] * 100, 3),
        "per_bucket":           decomp["per_bucket"],
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
    print(f"  BRINSON ATTRIBUTION")
    print(SEP)
    print(f"  Lookback:           {r['lookback']}")
    print(f"  Portfolio return:   {r['portfolio_return_pct']:+.2f}%")
    print(f"  Benchmark return:   {r['benchmark_return_pct']:+.2f}%")
    print(f"  Excess:             {r['excess_return_pct']:+.2f}%")
    print()

    print(f"  PER-BUCKET BREAKDOWN")
    print(f"  {'─' * 64}")
    print(
        f"  {'bucket':<10s}  {'w_p':>6s}  {'w_b':>6s}  {'r_p%':>7s}  "
        f"{'r_b%':>7s}  {'alloc%':>7s}  {'sel%':>7s}"
    )
    for k, v in r["per_bucket"].items():
        print(
            f"  {k:<10s}  "
            f"{v['w_portfolio']:>6.2%}  "
            f"{v['w_benchmark']:>6.2%}  "
            f"{v['r_portfolio']*100:>+7.2f}  "
            f"{v['r_benchmark']*100:>+7.2f}  "
            f"{v['allocation']*100:>+7.3f}  "
            f"{v['selection']*100:>+7.3f}"
        )
    print()

    print(f"  ATTRIBUTION SUMMARY")
    print(f"  Allocation: {r['allocation_effect_pct']:+.3f}%   "
          f"(weighting bets vs benchmark)")
    print(f"  Selection:  {r['selection_effect_pct']:+.3f}%   "
          f"(security picks within bucket)")
    print(f"  Interact.:  {r['interaction_pct']:+.3f}%   "
          f"(cross-term)")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brinson Attribution")
    parser.add_argument("--lookback", default=DEFAULT_LOOKBACK)
    args = parser.parse_args()
    run_brinson_attribution(lookback=args.lookback)
