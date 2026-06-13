#!/usr/bin/env python3
"""
Black-Scholes Options Pricer + Greeks
=======================================
Closed-form Black-Scholes pricing and the full Greek suite for European
calls and puts on GLD (the most liquid gold ETF), with strikes laid out as
a ladder around the current spot.

Inputs:
  - Spot S        last GLD close
  - σ (vol)       21d realised vol from vol_surface.json, fallback 16%
  - r (rate)      4.5% (configurable)
  - T (tenor)     30 days (configurable)

Greeks (per option; standard market conventions):
  Delta            ∂price/∂S
  Gamma            ∂²price/∂S²
  Vega             ∂price/∂σ          (per 1.0 σ, divide by 100 for "per 1%")
  Theta            ∂price/∂t          (per year; divide by 365 for "per day")
  Rho              ∂price/∂r          (per 1.0 r, divide by 100 for "per 1%")

Reports an ATM table plus strike ladder (-10% to +10%, 1% steps), and
identifies the call & put with the highest gamma (most leverage on small
moves).

Output: data/options_pricer.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import yfinance as yf
except ImportError:
    yf = None

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "options_pricer.json"
VOL_SURFACE_FILE = DATA_DIR / "vol_surface.json"

DEFAULT_TICKER = "GLD"
DEFAULT_RATE = 0.045
DEFAULT_TENOR_DAYS = 30
DEFAULT_VOL = 0.16  # fallback if vol_surface unavailable

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def _fetch_spot(ticker: str) -> float:
    if yf is None:
        raise ImportError("yfinance is required")
    raw = yf.download(
        ticker, period="5d", interval="1d",
        progress=False, auto_adjust=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    return float(raw["Close"].dropna().iloc[-1])


def _load_iv_from_surface() -> float:
    if VOL_SURFACE_FILE.exists():
        try:
            vs = json.loads(VOL_SURFACE_FILE.read_text())
            v_pct = vs.get("term_structure", {}).get("rv_21d")
            if v_pct:
                return float(v_pct) / 100.0
        except Exception:
            pass
    return DEFAULT_VOL


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------
def black_scholes(
    S: float, K: float, T: float, sigma: float, r: float, kind: str = "call",
) -> dict:
    """Price and Greeks for a single European option."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if kind == "call" else max(K - S, 0)
        return {
            "price":  round(intrinsic, 4),
            "delta":  1.0 if kind == "call" and S > K else (-1.0 if kind == "put" and S < K else 0.0),
            "gamma":  0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0,
        }

    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    nd1 = norm.pdf(d1)  # density

    if kind == "call":
        price = S * Nd1 - K * np.exp(-r * T) * Nd2
        delta = Nd1
        theta = (
            -S * nd1 * sigma / (2 * sqrtT)
            - r * K * np.exp(-r * T) * Nd2
        )
        rho = K * T * np.exp(-r * T) * Nd2
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = Nd1 - 1.0
        theta = (
            -S * nd1 * sigma / (2 * sqrtT)
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        )
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    gamma = nd1 / (S * sigma * sqrtT)
    vega = S * nd1 * sqrtT

    return {
        "price":  round(float(price), 4),
        "delta":  round(float(delta), 5),
        "gamma":  round(float(gamma), 5),
        "vega":   round(float(vega), 4),
        "theta":  round(float(theta), 4),
        "rho":    round(float(rho), 4),
    }


# ---------------------------------------------------------------------------
# Strike ladder
# ---------------------------------------------------------------------------
def strike_ladder(spot: float, pct_range: float = 0.10, step_pct: float = 0.01) -> list[float]:
    steps = int(round(2 * pct_range / step_pct)) + 1
    moneynesses = np.linspace(-pct_range, pct_range, steps)
    strikes = [round(spot * (1 + m), 2) for m in moneynesses]
    return strikes


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_options_pricer(
    ticker: str = DEFAULT_TICKER,
    rate: float = DEFAULT_RATE,
    tenor_days: int = DEFAULT_TENOR_DAYS,
    sigma_override: float | None = None,
) -> dict:
    spot = _fetch_spot(ticker)
    sigma = sigma_override if sigma_override is not None else _load_iv_from_surface()
    T = tenor_days / 365.0

    strikes = strike_ladder(spot)
    calls = []
    puts = []
    for K in strikes:
        c = black_scholes(spot, K, T, sigma, rate, "call")
        p = black_scholes(spot, K, T, sigma, rate, "put")
        moneyness_pct = (K / spot - 1) * 100
        calls.append({"strike": K, "moneyness_pct": round(moneyness_pct, 2), **c})
        puts.append({"strike":  K, "moneyness_pct": round(moneyness_pct, 2), **p})

    atm_call = black_scholes(spot, spot, T, sigma, rate, "call")
    atm_put = black_scholes(spot, spot, T, sigma, rate, "put")

    # Highest gamma
    best_call = max(calls, key=lambda c: c["gamma"])
    best_put = max(puts, key=lambda p: p["gamma"])

    # Put-call parity check (should be near zero)
    parity = atm_call["price"] - atm_put["price"] - (spot - spot * np.exp(-rate * T))

    result = {
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker":         ticker,
        "spot":           round(spot, 2),
        "sigma":          round(sigma, 4),
        "rate":           rate,
        "tenor_days":     tenor_days,
        "tenor_years":    round(T, 5),
        "atm_call":       atm_call,
        "atm_put":        atm_put,
        "highest_gamma_call": best_call,
        "highest_gamma_put":  best_put,
        "parity_residual":round(float(parity), 5),
        "strikes":        strikes,
        "calls_ladder":   calls,
        "puts_ladder":    puts,
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
    print(f"  BLACK-SCHOLES OPTIONS PRICER -- {r['ticker']}")
    print(SEP)
    print(f"  Spot:     ${r['spot']:,.2f}")
    print(f"  σ:        {r['sigma']:.2%}")
    print(f"  r:        {r['rate']:.2%}")
    print(f"  T:        {r['tenor_days']} days ({r['tenor_years']:.4f}y)")
    print()

    print(f"  ATM (K = ${r['spot']:,.2f})")
    print(f"  {'─' * 50}")
    print(f"  {'':<10s}  {'price':>8s}  {'delta':>8s}  {'gamma':>8s}  {'vega':>8s}  {'theta':>8s}")
    ac = r["atm_call"]
    ap = r["atm_put"]
    print(f"  {'CALL':<10s}  {ac['price']:>8.4f}  {ac['delta']:>8.4f}  "
          f"{ac['gamma']:>8.4f}  {ac['vega']:>8.4f}  {ac['theta']:>8.4f}")
    print(f"  {'PUT':<10s}  {ap['price']:>8.4f}  {ap['delta']:>8.4f}  "
          f"{ap['gamma']:>8.4f}  {ap['vega']:>8.4f}  {ap['theta']:>8.4f}")
    print(f"  Put-call parity residual: {r['parity_residual']:+.5f}")
    print()

    print(f"  STRIKE LADDER (calls)")
    print(f"  {'─' * 58}")
    print(f"  {'K':>8s}  {'%S':>7s}  {'price':>8s}  {'delta':>7s}  {'gamma':>8s}  {'vega':>8s}")
    for c in r["calls_ladder"]:
        print(
            f"  {c['strike']:>8.2f}  "
            f"{c['moneyness_pct']:>+7.2f}  "
            f"{c['price']:>8.4f}  "
            f"{c['delta']:>7.4f}  "
            f"{c['gamma']:>8.4f}  "
            f"{c['vega']:>8.4f}"
        )
    print()

    bc = r["highest_gamma_call"]
    bp = r["highest_gamma_put"]
    print(f"  HIGHEST-GAMMA OPTIONS  (best for small-move leverage)")
    print(f"  CALL  K=${bc['strike']:.2f}  γ={bc['gamma']:.4f}  price=${bc['price']:.4f}")
    print(f"  PUT   K=${bp['strike']:.2f}  γ={bp['gamma']:.4f}  price=${bp['price']:.4f}")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Black-Scholes Options Pricer")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE)
    parser.add_argument("--tenor", type=int, default=DEFAULT_TENOR_DAYS)
    parser.add_argument("--sigma", type=float, default=None,
                        help="Override σ. Defaults to vol_surface rv_21d.")
    args = parser.parse_args()
    run_options_pricer(
        ticker=args.ticker,
        rate=args.rate,
        tenor_days=args.tenor,
        sigma_override=args.sigma,
    )
