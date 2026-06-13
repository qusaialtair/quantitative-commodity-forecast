#!/usr/bin/env python3
"""
Gold Futures Term Structure Analyzer
======================================
Pulls live prices for several COMEX gold futures contracts (front-month
through ~1y out), computes the curve slope, classifies the curve shape,
and reports implied roll yields and stress flags.

CME gold futures month codes:
    F=Jan G=Feb H=Mar J=Apr K=May M=Jun N=Jul Q=Aug U=Sep V=Oct X=Nov Z=Dec
Contract expires ~3rd-to-last business day of the prior month.

Curve classifications (slope = annualised premium of far vs front, %):
    BACKWARDATION    slope < -1%        physical stress / supply tightness
    FLAT             −1% ≤ slope < +1%  rare; often pre-crisis
    NORMAL_CONTANGO  +1% ≤ slope < +5%  typical for gold
    STEEP_CONTANGO   slope ≥ +5%        carry exceeds fair value

Output: data/term_structure.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, date
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
OUTPUT_FILE = DATA_DIR / "term_structure.json"

# CME gold futures month-code letters (in order)
MONTH_CODES = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}

# Contracts to fetch (assumes we have ~6 active going forward)
CONTRACT_SYMBOLS = [
    "GCM26.CMX", "GCQ26.CMX", "GCV26.CMX",
    "GCZ26.CMX", "GCG27.CMX", "GCM27.CMX",
]

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _contract_expiry(symbol: str) -> date | None:
    """
    Parse 'GCM26.CMX' → June 2026 → approximate expiry as the last day of May.
    (Real CME expiry is 3rd-to-last business day of prior month; this is close
    enough for term-structure slope estimation.)
    """
    try:
        if not symbol.startswith("GC") or "." not in symbol:
            return None
        body = symbol.split(".")[0]
        if len(body) < 5:
            return None
        month_letter = body[2]
        year_2d = int(body[3:5])
        if month_letter not in MONTH_CODES:
            return None
        month = MONTH_CODES[month_letter]
        year_full = 2000 + year_2d
        prev_month = month - 1 if month > 1 else 12
        prev_year = year_full if month > 1 else year_full - 1
        # End-of-prior-month
        if prev_month == 12:
            next_first = date(prev_year + 1, 1, 1)
        else:
            next_first = date(prev_year, prev_month + 1, 1)
        return next_first - pd.Timedelta(days=1)
    except Exception:
        return None


def _fetch_last_close(symbol: str) -> float | None:
    try:
        raw = yf.download(
            symbol, period="5d", interval="1d",
            progress=False, auto_adjust=True,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        s = raw["Close"].dropna()
        if len(s) == 0:
            return None
        return float(s.iloc[-1])
    except Exception:
        return None


def _classify_curve(slope_pct: float) -> str:
    if slope_pct < -1.0:
        return "BACKWARDATION"
    if slope_pct < 1.0:
        return "FLAT"
    if slope_pct < 5.0:
        return "NORMAL_CONTANGO"
    return "STEEP_CONTANGO"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_term_structure(symbols: list[str] = None) -> dict:
    if yf is None:
        raise ImportError("yfinance is required")
    symbols = symbols or CONTRACT_SYMBOLS
    today = date.today()

    contracts = []
    for sym in symbols:
        price = _fetch_last_close(sym)
        if price is None:
            continue
        exp = _contract_expiry(sym)
        if exp is None:
            continue
        dte = (exp - today).days
        if dte <= 0:
            continue
        contracts.append({
            "symbol":         sym,
            "price":          round(price, 2),
            "expiry_date":    str(exp),
            "days_to_expiry": int(dte),
        })

    if len(contracts) < 2:
        result = {
            "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_contracts":    len(contracts),
            "contracts":      contracts,
            "warning":        "Need at least 2 active contracts for curve analysis.",
        }
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(json.dumps(result, indent=2))
        _print_report(result)
        return result

    # Sort by expiry
    contracts.sort(key=lambda c: c["days_to_expiry"])

    front = contracts[0]
    back = contracts[-1]

    # Basis vs front month
    for c in contracts:
        basis_pct = (c["price"] / front["price"] - 1) * 100
        dte_delta = c["days_to_expiry"] - front["days_to_expiry"]
        if dte_delta > 0:
            ann_slope = basis_pct * (365.0 / dte_delta)
        else:
            ann_slope = 0.0
        c["basis_vs_front_pct"] = round(basis_pct, 4)
        c["dte_from_front"] = int(dte_delta)
        c["annualised_slope_pct"] = round(float(ann_slope), 3)

    # Overall curve slope = (back - front) / front, annualised
    dte_span = back["days_to_expiry"] - front["days_to_expiry"]
    overall_basis_pct = (back["price"] / front["price"] - 1) * 100
    overall_slope_pct = overall_basis_pct * (365.0 / max(dte_span, 1))
    curve_shape = _classify_curve(overall_slope_pct)

    # Roll yield: if curve is contango, holders of paper lose this each year
    roll_yield_pct = -overall_slope_pct

    # Curve linearity: is it smooth or kinked?
    # Compute R² of price vs DTE — perfect contango is linear in log-time
    if len(contracts) >= 3:
        dtes = np.array([c["days_to_expiry"] for c in contracts], dtype=float)
        prices = np.array([c["price"] for c in contracts], dtype=float)
        # Linear fit on log price vs DTE
        log_p = np.log(prices)
        if dtes.std() > 0:
            slope, intercept = np.polyfit(dtes, log_p, 1)
            preds = slope * dtes + intercept
            ss_res = ((log_p - preds) ** 2).sum()
            ss_tot = ((log_p - log_p.mean()) ** 2).sum()
            r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 1.0
        else:
            r_squared = 1.0
    else:
        r_squared = None

    # Stress flag: flat or backwardated curve
    stress_flag = curve_shape in ("FLAT", "BACKWARDATION")

    result = {
        "generated_at":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_contracts":           len(contracts),
        "contracts":             contracts,
        "front_symbol":          front["symbol"],
        "front_price":           front["price"],
        "back_symbol":           back["symbol"],
        "back_price":            back["price"],
        "front_to_back_basis_pct": round(overall_basis_pct, 4),
        "overall_slope_pct":     round(float(overall_slope_pct), 4),
        "curve_shape":           curve_shape,
        "roll_yield_pct":        round(float(roll_yield_pct), 4),
        "curve_r_squared":       round(r_squared, 4) if r_squared is not None else None,
        "stress_flag":           bool(stress_flag),
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
    print(f"  GOLD FUTURES TERM STRUCTURE")
    print(SEP)

    if r.get("warning"):
        print(f"  ⚠ {r['warning']}")
        print(SEP)
        return

    print(f"  Front:    {r['front_symbol']}  @ ${r['front_price']:,.2f}")
    print(f"  Back:     {r['back_symbol']}  @ ${r['back_price']:,.2f}")
    print(f"  Spread:   {r['front_to_back_basis_pct']:+.3f}%  "
          f"over {r['contracts'][-1]['dte_from_front']}d")
    print()

    print(f"  PER-CONTRACT")
    print(f"  {'─' * 64}")
    print(f"  {'symbol':<12s}  {'price':>10s}  {'DTE':>6s}  "
          f"{'basis':>8s}  {'ann slope':>10s}")
    for c in r["contracts"]:
        print(
            f"  {c['symbol']:<12s}  "
            f"${c['price']:>9,.2f}  "
            f"{c['days_to_expiry']:>5d}d  "
            f"{c['basis_vs_front_pct']:>+7.3f}%  "
            f"{c['annualised_slope_pct']:>+9.3f}%"
        )
    print()

    shape_color = {
        "BACKWARDATION":   "\033[31;1m",
        "FLAT":            "\033[33m",
        "NORMAL_CONTANGO": "\033[36m",
        "STEEP_CONTANGO":  "\033[31m",
    }.get(r["curve_shape"], "\033[0m")

    print(f"  Curve shape:     {shape_color}{r['curve_shape']}\033[0m")
    print(f"  Annualised slope: {r['overall_slope_pct']:+.3f}%")
    print(f"  Roll yield (paper holder):  {r['roll_yield_pct']:+.3f}% / yr")
    print(f"  Curve R²:        {r['curve_r_squared']}")
    if r["stress_flag"]:
        print(f"  ⚠ STRESS — flat/backwardated curve suggests physical tightness")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gold Futures Term Structure")
    parser.add_argument("--symbols", default=",".join(CONTRACT_SYMBOLS),
                        help="Comma-separated contract symbols")
    args = parser.parse_args()
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    run_term_structure(symbols=syms)
