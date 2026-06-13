#!/usr/bin/env python3
"""
Smart Order Router (SOR)
=========================
Decides how to execute a parent order across child orders. Compares four
slicing schedules using the Almgren-Chriss square-root impact law:

  - SINGLE   immediate single execution         (best at tiny size)
  - TWAP     evenly-spaced slices               (linear time impact)
  - VWAP     volume-weighted intraday schedule  (~30% impact reduction)
  - POV      participation-of-volume cap        (slower, lowest impact)

Algo selection rules of thumb (driven by participation rate %ADV):
  participation < 1%     → SINGLE
  1-5%                   → TWAP
  5-15%                  → VWAP
  > 15%                  → POV

Inputs:
  - ticker
  - notional in USD
  - urgency  (low/medium/high)
  - max_participation_pct (default 10%)

Outputs:
  data/smart_order_router.json
  - recommended algo, slice schedule
  - per-algo expected cost in bps (impact + spread + physical premium)
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
OUTPUT_FILE = DATA_DIR / "smart_order_router.json"

DEFAULT_TICKER = "GC=F"
DEFAULT_NOTIONAL = 50_000.0
DEFAULT_URGENCY = "medium"
DEFAULT_MAX_POV = 10.0

# Cost parameters (bps)
SPREAD_BPS = {"GC=F": 3.0, "SI=F": 5.0, "SPY": 1.0, "TLT": 2.0}
PHYSICAL_PREMIUM_BPS = {"GC=F": 75.0, "SI=F": 75.0}  # UAE physical premium for metals
DEFAULT_SPREAD_BPS = 5.0

# Almgren-Chriss impact constant (calibrated to roughly match SI's 10bp at 5% ADV)
AC_IMPACT_K = 0.20  # in bps per √(participation_pct)

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _fetch_panel(ticker: str) -> tuple[float, float]:
    """Return (last_price, average_daily_volume_usd)."""
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

    # ADV in USD = avg(volume) × avg(close); use 21d
    if "Volume" in raw.columns:
        vol = raw["Volume"].dropna().tail(21)
        adv_shares = float(vol.mean()) if len(vol) else 0.0
        adv_usd = adv_shares * last_price
    else:
        adv_usd = 0.0
    return last_price, adv_usd


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
def _spread_bps(ticker: str) -> float:
    return SPREAD_BPS.get(ticker, DEFAULT_SPREAD_BPS)


def _physical_premium_bps(ticker: str, physical: bool) -> float:
    if not physical:
        return 0.0
    return PHYSICAL_PREMIUM_BPS.get(ticker, 0.0)


def _ac_impact_bps(participation_pct: float, algo: str) -> float:
    """
    Almgren-Chriss square-root law:
        impact_bps = K · √(participation_pct)
    with algo-specific multipliers reflecting how well each algo blends with
    natural volume.
    """
    base = AC_IMPACT_K * np.sqrt(max(participation_pct, 0.0))
    multipliers = {
        "SINGLE": 1.5,   # aggressive — eats top of book
        "TWAP":   1.0,
        "VWAP":   0.7,   # ~30% impact reduction by tracking natural volume
        "POV":    0.5,   # slowest, lowest impact
    }
    return float(base * multipliers.get(algo, 1.0))


def cost_for_algo(
    algo: str,
    notional: float,
    adv_usd: float,
    ticker: str,
    physical: bool,
    horizon_minutes: int,
) -> dict:
    """Estimate total one-way cost in bps for a given algo."""
    # Participation rate: child-order size as % of expected volume in horizon
    # For SINGLE / TWAP / VWAP we assume the order completes within `horizon_minutes`
    # POV by definition caps participation
    if adv_usd <= 0:
        participation = 0.0
    else:
        # 6.5 trading hours/day = 390 minutes
        horizon_volume = adv_usd * (horizon_minutes / 390.0)
        participation = (notional / max(horizon_volume, 1.0)) * 100.0

    impact = _ac_impact_bps(participation, algo)
    spread = _spread_bps(ticker) * (0.5 if algo != "SINGLE" else 1.0)
    physical_bps = _physical_premium_bps(ticker, physical)

    total = impact + spread + physical_bps
    return {
        "algo":               algo,
        "horizon_minutes":    horizon_minutes,
        "participation_pct":  round(participation, 3),
        "impact_bps":         round(impact, 3),
        "spread_bps":         round(spread, 3),
        "physical_bps":       round(physical_bps, 3),
        "total_oneway_bps":   round(total, 3),
        "total_cost_usd":     round(total / 10000.0 * notional, 2),
    }


# ---------------------------------------------------------------------------
# Algo selection
# ---------------------------------------------------------------------------
def select_algo(participation_pct: float, urgency: str) -> str:
    """Rule-based algo choice from participation and urgency."""
    if urgency == "high":
        # Bias toward faster execution
        if participation_pct < 5:
            return "SINGLE"
        if participation_pct < 15:
            return "TWAP"
        return "VWAP"

    # Normal / low urgency
    if participation_pct < 1.0:
        return "SINGLE"
    if participation_pct < 5.0:
        return "TWAP"
    if participation_pct < 15.0:
        return "VWAP"
    return "POV"


def _horizon_for_algo(algo: str, participation_pct: float, max_pov: float) -> int:
    """How many minutes to spread the order over."""
    if algo == "SINGLE":
        return 1
    if algo == "TWAP":
        return 30
    if algo == "VWAP":
        return 60
    # POV — extend until participation falls below max_pov
    if max_pov <= 0:
        return 390
    factor = max(1.0, participation_pct / max_pov)
    return int(min(390, round(60 * factor)))


def _slice_schedule(algo: str, notional: float, horizon_minutes: int) -> list:
    """Generate per-slice timing and size (USD)."""
    if algo == "SINGLE":
        return [{"t_minutes": 0, "size_usd": round(notional, 2), "frac": 1.0}]
    # Generate 6-20 slices depending on horizon
    n_slices = max(2, min(20, horizon_minutes // 5))
    out = []
    if algo == "TWAP":
        size = notional / n_slices
        for i in range(n_slices):
            out.append({
                "t_minutes": round(i * horizon_minutes / n_slices, 1),
                "size_usd":  round(size, 2),
                "frac":      round(1.0 / n_slices, 4),
            })
    elif algo == "VWAP":
        # U-shape intraday volume profile: heavy at open and close
        x = np.linspace(0, 1, n_slices)
        weights = 1.5 - 4.0 * (x - 0.5) ** 2  # parabolic up at edges
        weights = np.clip(weights, 0.3, None)
        weights = weights / weights.sum()
        for i in range(n_slices):
            out.append({
                "t_minutes": round(i * horizon_minutes / n_slices, 1),
                "size_usd":  round(notional * weights[i], 2),
                "frac":      round(float(weights[i]), 4),
            })
    else:  # POV
        size = notional / n_slices
        for i in range(n_slices):
            out.append({
                "t_minutes": round(i * horizon_minutes / n_slices, 1),
                "size_usd":  round(size, 2),
                "frac":      round(1.0 / n_slices, 4),
            })
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_smart_order_router(
    ticker: str = DEFAULT_TICKER,
    notional: float = DEFAULT_NOTIONAL,
    urgency: str = DEFAULT_URGENCY,
    max_pov: float = DEFAULT_MAX_POV,
    physical: bool = True,
) -> dict:
    price, adv_usd = _fetch_panel(ticker)

    # First-pass participation against a "natural" 60min horizon for algo choice
    if adv_usd <= 0:
        participation_60m = 0.0
    else:
        participation_60m = (notional / (adv_usd * 60.0 / 390.0)) * 100.0

    recommended = select_algo(participation_60m, urgency)
    horizon = _horizon_for_algo(recommended, participation_60m, max_pov)
    schedule = _slice_schedule(recommended, notional, horizon)

    # Cost comparison across all algos
    comparison = {}
    for algo in ["SINGLE", "TWAP", "VWAP", "POV"]:
        h = _horizon_for_algo(algo, participation_60m, max_pov)
        comparison[algo] = cost_for_algo(algo, notional, adv_usd, ticker, physical, h)

    rec_cost = comparison[recommended]
    cheapest = min(comparison.items(), key=lambda kv: kv[1]["total_oneway_bps"])

    result = {
        "generated_at":        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker":              ticker,
        "last_price":          round(price, 2),
        "notional_usd":        notional,
        "adv_usd":             round(adv_usd, 2),
        "participation_60min": round(participation_60m, 3),
        "urgency":             urgency,
        "max_pov_pct":         max_pov,
        "physical":            physical,
        "recommended_algo":    recommended,
        "horizon_minutes":     horizon,
        "n_slices":            len(schedule),
        "slice_schedule":      schedule,
        "recommended_cost":    rec_cost,
        "comparison":          comparison,
        "cheapest_algo":       cheapest[0],
        "savings_vs_cheapest_bps": round(
            rec_cost["total_oneway_bps"] - cheapest[1]["total_oneway_bps"], 3
        ),
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
    print(f"  SMART ORDER ROUTER -- {r['ticker']}")
    print(SEP)
    print(f"  Notional:       ${r['notional_usd']:,.2f}")
    print(f"  Last Price:     ${r['last_price']:,.2f}")
    print(f"  ADV:            ${r['adv_usd']:,.0f}")
    print(f"  60m Particip.:  {r['participation_60min']:.3f}%")
    print(f"  Urgency:        {r['urgency']}")
    print()

    print(f"  ALGO COMPARISON (one-way cost in bps)")
    print(f"  {'─' * 58}")
    print(f"  {'algo':<8s}  {'horiz':>6s}  {'partic%':>8s}  "
          f"{'impact':>7s}  {'spread':>7s}  {'phys':>6s}  {'total':>7s}")
    for algo, c in r["comparison"].items():
        marker = ">>" if algo == r["recommended_algo"] else "  "
        print(
            f"  {marker}{algo:<6s}  {c['horizon_minutes']:>5d}m  "
            f"{c['participation_pct']:>8.3f}  "
            f"{c['impact_bps']:>7.2f}  "
            f"{c['spread_bps']:>7.2f}  "
            f"{c['physical_bps']:>6.1f}  "
            f"{c['total_oneway_bps']:>7.2f}"
        )
    print()

    rc = r["recommended_cost"]
    print(f"  RECOMMENDATION: {r['recommended_algo']}")
    print(f"  {'─' * 40}")
    print(f"  Horizon:        {r['horizon_minutes']} minutes")
    print(f"  Slices:         {r['n_slices']}")
    print(f"  Expected cost:  {rc['total_oneway_bps']:.2f} bps  "
          f"(${rc['total_cost_usd']:,.2f})")
    print(f"  Cheapest algo:  {r['cheapest_algo']}  "
          f"(savings if chosen: {r['savings_vs_cheapest_bps']:+.2f}bp)")
    print()

    if r["n_slices"] > 1 and r["n_slices"] <= 12:
        print(f"  SLICE SCHEDULE (first {min(8, r['n_slices'])})")
        print(f"  {'─' * 40}")
        for sl in r["slice_schedule"][:8]:
            print(f"  t+{sl['t_minutes']:>5.1f}m  "
                  f"${sl['size_usd']:>10,.0f}  "
                  f"({sl['frac']:.1%})")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Order Router")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--notional", type=float, default=DEFAULT_NOTIONAL)
    parser.add_argument("--urgency", choices=["low", "medium", "high"], default=DEFAULT_URGENCY)
    parser.add_argument("--max-pov", type=float, default=DEFAULT_MAX_POV)
    parser.add_argument("--no-physical", action="store_true",
                        help="Exclude UAE physical premium (paper / ETF execution)")
    args = parser.parse_args()
    run_smart_order_router(
        ticker=args.ticker,
        notional=args.notional,
        urgency=args.urgency,
        max_pov=args.max_pov,
        physical=not args.no_physical,
    )
