#!/usr/bin/env python3
"""
Strategy Capacity Analyzer
============================
Estimates the maximum AUM at which the trading strategy still produces a
positive net alpha after accounting for impact and spread costs.

Model:
  - Take per-source expected alpha from alpha_attribution.json
  - Per-asset ADV from yfinance Volume × Price
  - Almgren-Chriss square-root impact:  impact_bps = K·√(participation%)
  - Annualised cost = roundtrip_cost × turnover_per_year
  - Net alpha = expected_alpha − annualised_cost

Reports the AUM ceilings at three alpha-decay thresholds:
  - Conservative   alpha eaten by 25%
  - Moderate       alpha eaten by 50%
  - Aggressive     alpha eaten by 100% (breakeven)

Output: data/capacity_analyzer.json
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
OUTPUT_FILE = DATA_DIR / "capacity_analyzer.json"
ALPHA_FILE = DATA_DIR / "alpha_attribution.json"

DEFAULT_TICKER = "GC=F"

# Cost parameters
SPREAD_BPS = {"GC=F": 3.0, "SI=F": 5.0, "SPY": 1.0, "TLT": 2.0}
PHYSICAL_PREMIUM_BPS = {"GC=F": 75.0, "SI=F": 75.0}
DEFAULT_SPREAD_BPS = 5.0
AC_IMPACT_K = 0.20  # bps per √(participation%)

# Strategy parameters
DEPLOY_PCT_OF_CAPITAL = 25.0   # avg position size as % of AUM
TURNOVER_PER_YEAR = 8           # round trips per year
DECAY_THRESHOLDS_PCT = [25.0, 50.0, 100.0]

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _fetch_panel(ticker: str) -> tuple[float, float]:
    if yf is None:
        raise ImportError("yfinance is required")
    raw = yf.download(
        ticker, period="3mo", interval="1d",
        progress=False, auto_adjust=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    close = raw["Close"].dropna()
    last_price = float(close.iloc[-1])
    if "Volume" in raw.columns:
        adv_shares = float(raw["Volume"].dropna().tail(21).mean())
        adv_usd = adv_shares * last_price
    else:
        adv_usd = 0.0
    return last_price, adv_usd


def _load_expected_alpha() -> tuple[float, str]:
    """Pull top-source annual return from alpha_attribution.json."""
    if not ALPHA_FILE.exists():
        return 6.0, "fallback"  # 6% annualised default
    try:
        aa = json.loads(ALPHA_FILE.read_text())
        top = aa.get("ranked_by_sharpe", [None])[0]
        if top:
            ret = aa.get("full_history", {}).get(top, {}).get("ann_return_pct", 0)
            if ret > 0:
                return float(ret), top
    except Exception:
        pass
    return 6.0, "fallback"


# ---------------------------------------------------------------------------
# Cost curve
# ---------------------------------------------------------------------------
def _impact_bps(participation_pct: float) -> float:
    return AC_IMPACT_K * float(np.sqrt(max(participation_pct, 0.0)))


def _spread_bps(ticker: str, physical: bool) -> float:
    base = SPREAD_BPS.get(ticker, DEFAULT_SPREAD_BPS)
    phys = PHYSICAL_PREMIUM_BPS.get(ticker, 0.0) if physical else 0.0
    return base + phys


def _annual_cost_pct(
    aum_usd: float, adv_usd: float, ticker: str, physical: bool,
    deploy_pct: float = DEPLOY_PCT_OF_CAPITAL,
    turnover: int = TURNOVER_PER_YEAR,
) -> float:
    """Annual cost as a % of AUM."""
    position_usd = aum_usd * deploy_pct / 100.0
    if adv_usd <= 0:
        return 0.0
    participation = position_usd / adv_usd * 100.0  # %ADV
    impact = _impact_bps(participation)
    spread = _spread_bps(ticker, physical)
    oneway_bps = impact + spread
    roundtrip_bps = 2 * oneway_bps
    annual_bps = roundtrip_bps * turnover
    return annual_bps / 100.0  # convert bps → %


# ---------------------------------------------------------------------------
# Capacity curve
# ---------------------------------------------------------------------------
def build_capacity_curve(
    ticker: str, adv_usd: float, expected_alpha_pct: float,
    physical: bool,
) -> list:
    """Sweep AUM from $100K to $10B in log steps; compute net alpha."""
    aums = np.logspace(5, 10, 25)  # $100K → $10B
    out = []
    for aum in aums:
        cost_pct = _annual_cost_pct(aum, adv_usd, ticker, physical)
        net_alpha = expected_alpha_pct - cost_pct
        decay_pct = (cost_pct / expected_alpha_pct * 100) if expected_alpha_pct > 0 else 0.0
        out.append({
            "aum_usd":            int(aum),
            "cost_pct_per_year":  round(cost_pct, 3),
            "net_alpha_pct":      round(net_alpha, 3),
            "alpha_decay_pct":    round(decay_pct, 2),
            "position_usd":       round(aum * DEPLOY_PCT_OF_CAPITAL / 100, 2),
            "participation_pct":  round(
                (aum * DEPLOY_PCT_OF_CAPITAL / 100) / max(adv_usd, 1) * 100, 4
            ),
        })
    return out


def find_capacity_thresholds(curve: list, expected_alpha_pct: float) -> dict:
    """Find AUM at each decay threshold (25%, 50%, 100%)."""
    thresholds = {}
    for thr_pct in DECAY_THRESHOLDS_PCT:
        target_cost = expected_alpha_pct * thr_pct / 100.0
        # Find first AUM where cost >= target_cost
        cap = None
        for row in curve:
            if row["cost_pct_per_year"] >= target_cost:
                cap = row["aum_usd"]
                break
        if cap is None:
            cap = curve[-1]["aum_usd"]  # never reached; AUM > $10B is fine
        thresholds[f"decay_{int(thr_pct)}pct"] = {
            "aum_cap_usd": cap,
            "description": (
                "Conservative" if thr_pct == 25 else
                "Moderate"     if thr_pct == 50 else
                "Breakeven"
            ),
        }
    return thresholds


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_capacity_analyzer(
    ticker: str = DEFAULT_TICKER,
    physical: bool = True,
) -> dict:
    price, adv_usd = _fetch_panel(ticker)
    expected_alpha, alpha_source = _load_expected_alpha()

    curve = build_capacity_curve(ticker, adv_usd, expected_alpha, physical)
    thresholds = find_capacity_thresholds(curve, expected_alpha)

    # Compare paper vs physical
    if PHYSICAL_PREMIUM_BPS.get(ticker, 0) > 0:
        paper_curve = build_capacity_curve(ticker, adv_usd, expected_alpha, False)
        paper_thresholds = find_capacity_thresholds(paper_curve, expected_alpha)
    else:
        paper_curve = curve
        paper_thresholds = thresholds

    result = {
        "generated_at":         datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker":               ticker,
        "last_price":           round(price, 2),
        "adv_usd":              round(adv_usd, 2),
        "expected_alpha_pct":   round(expected_alpha, 3),
        "alpha_source":         alpha_source,
        "deploy_pct_capital":   DEPLOY_PCT_OF_CAPITAL,
        "turnover_per_year":    TURNOVER_PER_YEAR,
        "physical_execution":   physical,
        "thresholds_physical":  thresholds,
        "thresholds_paper":     paper_thresholds,
        "capacity_curve":       curve,
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
    print(f"  STRATEGY CAPACITY ANALYZER -- {r['ticker']}")
    print(SEP)
    print(f"  Last Price:        ${r['last_price']:,.2f}")
    print(f"  ADV:               ${r['adv_usd']:,.0f}")
    print(f"  Expected Alpha:    {r['expected_alpha_pct']:+.2f}% (from {r['alpha_source']})")
    print(f"  Deploy Fraction:   {r['deploy_pct_capital']:.1f}% of AUM")
    print(f"  Turnover:          {r['turnover_per_year']} round-trips / year")
    print(f"  Physical exec.:    {r['physical_execution']}")
    print()

    print(f"  CAPACITY THRESHOLDS  (physical execution)")
    print(f"  {'─' * 50}")
    for k, v in r["thresholds_physical"].items():
        print(f"  {k:<16s}  ${v['aum_cap_usd']:>15,.0f}  ({v['description']})")
    print()

    if r["physical_execution"]:
        print(f"  vs PAPER EXECUTION (no UAE physical premium)")
        print(f"  {'─' * 50}")
        for k, v in r["thresholds_paper"].items():
            print(f"  {k:<16s}  ${v['aum_cap_usd']:>15,.0f}")
        print()

    print(f"  CAPACITY CURVE (selected points)")
    print(f"  {'─' * 58}")
    print(
        f"  {'AUM':>12s}  {'pos USD':>12s}  {'%ADV':>7s}  "
        f"{'cost %':>7s}  {'net α%':>7s}  {'decay %':>7s}"
    )
    sample_idx = [0, 4, 8, 12, 16, 20, 24]
    for i in sample_idx:
        if i >= len(r["capacity_curve"]):
            continue
        row = r["capacity_curve"][i]
        decay_str = f"{row['alpha_decay_pct']:.1f}" if row['alpha_decay_pct'] < 999 else "∞"
        print(
            f"  ${row['aum_usd']:>10,}  "
            f"${row['position_usd']:>10,.0f}  "
            f"{row['participation_pct']:>7.3f}  "
            f"{row['cost_pct_per_year']:>7.3f}  "
            f"{row['net_alpha_pct']:>+7.3f}  "
            f"{decay_str:>7s}"
        )
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy Capacity Analyzer")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--no-physical", action="store_true")
    args = parser.parse_args()
    run_capacity_analyzer(
        ticker=args.ticker,
        physical=not args.no_physical,
    )
