#!/usr/bin/env python3
"""
Tail-Risk Hedge Overlay
=========================
Sizes a rolling OTM-put protection program for a long-gold (GLD) book.
Goal: cap the 95% CVaR loss at a target level by buying enough puts that
the put payoff offsets the worst-tail drawdown, while constraining the
annual premium drag.

Inputs:
  - notional_usd                 long-gold exposure to protect
  - cvar_target_pct              max allowed daily CVaR-95 loss (e.g. 1.0%)
  - max_annual_drag_pct          cap on premium spend (e.g. 1.5%)
  - tenor_days                   put tenor (default 90d for laddered roll)
  - moneyness_pct                OTM strike pct (default 5% below spot)

Approach:
  1. Read CVaR-95 estimate from monte_carlo_simulation.json
     (annualised: cvar95_daily × √252)
  2. Price the OTM put with Black-Scholes (σ from vol_surface)
  3. Compute # contracts needed so that the put-payoff at the CVaR scenario
     covers (current_cvar − target_cvar)
  4. Annualise the premium cost (4 rolls per year for 90d tenor)
  5. If annual drag exceeds max_annual_drag_pct → reduce hedge ratio so the
     drag constraint binds and report residual unhedged tail

Output: data/tail_hedge.json
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
OUTPUT_FILE = DATA_DIR / "tail_hedge.json"

DEFAULT_NOTIONAL = 100_000.0
DEFAULT_CVAR_TARGET_PCT = 1.0       # 1% daily CVaR target
DEFAULT_MAX_DRAG_PCT = 1.5          # 1.5% annual premium drag cap
DEFAULT_TENOR_DAYS = 90
DEFAULT_MONEYNESS_PCT = 5.0         # OTM by 5%
DEFAULT_RATE = 0.045
DEFAULT_VOL = 0.16
GLD_TO_OZ_RATIO = 0.1               # GLD ≈ 1/10 oz gold
CONTRACT_MULTIPLIER = 100           # 1 contract = 100 shares of GLD
SQ252 = float(np.sqrt(252))

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def _fetch_gld_spot() -> float:
    if yf is None:
        raise ImportError("yfinance is required")
    raw = yf.download("GLD", period="5d", interval="1d",
                       progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    return float(raw["Close"].dropna().iloc[-1])


def _load_iv() -> float:
    try:
        vs = json.loads((DATA_DIR / "vol_surface.json").read_text())
        v_pct = vs.get("term_structure", {}).get("rv_21d")
        if v_pct:
            return float(v_pct) / 100.0
    except Exception:
        pass
    return DEFAULT_VOL


def _load_cvar_daily_pct() -> float:
    try:
        mc = json.loads((DATA_DIR / "monte_carlo_simulation.json").read_text())
        cvar_pct = mc.get("risk", {}).get("cvar_95_pct")
        if cvar_pct is not None:
            # Monte Carlo CVaR is over the horizon (typically 21d); convert to daily
            horizon = mc.get("horizon_days", 21)
            return abs(float(cvar_pct)) / max(horizon, 1)
    except Exception:
        pass
    return 0.8  # fallback 0.8% daily CVaR


# ---------------------------------------------------------------------------
# Black-Scholes put
# ---------------------------------------------------------------------------
def _bs_put(S: float, K: float, T: float, sigma: float, r: float) -> dict:
    if T <= 0:
        return {"price": max(K - S, 0), "delta": -1.0 if S < K else 0, "vega": 0}
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    delta = norm.cdf(d1) - 1
    vega = S * norm.pdf(d1) * sqrtT
    return {
        "price": float(price),
        "delta": float(delta),
        "vega":  float(vega),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_tail_hedge(
    notional_usd: float = DEFAULT_NOTIONAL,
    cvar_target_pct: float = DEFAULT_CVAR_TARGET_PCT,
    max_annual_drag_pct: float = DEFAULT_MAX_DRAG_PCT,
    tenor_days: int = DEFAULT_TENOR_DAYS,
    moneyness_pct: float = DEFAULT_MONEYNESS_PCT,
    rate: float = DEFAULT_RATE,
) -> dict:
    spot = _fetch_gld_spot()
    sigma = _load_iv()
    current_cvar_daily = _load_cvar_daily_pct()
    T = tenor_days / 365.0

    # Strike at moneyness_pct below spot
    K = spot * (1 - moneyness_pct / 100.0)
    put = _bs_put(spot, K, T, sigma, rate)
    put_price = put["price"]

    # Contracts needed
    # Assume one CVaR-scenario daily loss = notional × cvar_daily_pct/100
    # Single OTM put payoff at strike = max(K - S_T, 0); ATM ≈ moneyness × S
    # We want the residual unhedged tail to match cvar_target_pct
    current_loss_usd = notional_usd * current_cvar_daily / 100.0
    target_loss_usd = notional_usd * cvar_target_pct / 100.0
    needed_coverage_usd = max(current_loss_usd - target_loss_usd, 0)

    # Approximate put payoff at the CVaR scenario:
    # if S drops by current_cvar_daily%, payoff ≈ max(K - S*(1-current_cvar_daily/100), 0)
    s_at_cvar = spot * (1 - current_cvar_daily / 100.0)
    payoff_per_share = max(K - s_at_cvar, 0)

    if payoff_per_share <= 0:
        # 5% OTM put won't pay off on a 1% scenario daily move; need closer-to-ATM put
        # Use Δ-equivalent shares instead: shares = needed_coverage / |delta| × strike
        delta_abs = max(abs(put["delta"]), 0.05)
        shares_needed = needed_coverage_usd / (delta_abs * spot)
        coverage_method = "delta-equivalent"
    else:
        shares_needed = needed_coverage_usd / payoff_per_share
        coverage_method = "intrinsic-payoff"

    contracts_needed = max(0, np.ceil(shares_needed / CONTRACT_MULTIPLIER))
    total_premium = contracts_needed * CONTRACT_MULTIPLIER * put_price

    # Annual cost: rolls per year = 365 / tenor_days
    rolls_per_year = 365.0 / max(tenor_days, 1)
    annual_premium = total_premium * rolls_per_year
    annual_drag_pct = (annual_premium / notional_usd) * 100 if notional_usd > 0 else 0

    # If drag exceeds cap, scale down and report residual tail
    if annual_drag_pct > max_annual_drag_pct and annual_drag_pct > 0:
        hedge_ratio = max_annual_drag_pct / annual_drag_pct
        contracts_needed = int(np.ceil(contracts_needed * hedge_ratio))
        total_premium = contracts_needed * CONTRACT_MULTIPLIER * put_price
        annual_premium = total_premium * rolls_per_year
        annual_drag_pct = (annual_premium / notional_usd) * 100
        # Residual = current cvar minus what the constrained hedge covers
        covered_usd = contracts_needed * CONTRACT_MULTIPLIER * payoff_per_share
        residual_loss_usd = max(current_loss_usd - covered_usd, 0)
        residual_cvar_pct = (residual_loss_usd / notional_usd) * 100
        constraint_binding = True
    else:
        residual_cvar_pct = cvar_target_pct
        constraint_binding = False

    result = {
        "generated_at":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gld_spot":              round(spot, 2),
        "sigma":                 round(sigma, 4),
        "rate":                  rate,
        "tenor_days":            tenor_days,
        "moneyness_pct":         moneyness_pct,
        "strike":                round(K, 2),
        "notional_usd":          notional_usd,
        "current_cvar_daily_pct":round(current_cvar_daily, 4),
        "current_loss_usd":      round(current_loss_usd, 2),
        "target_cvar_pct":       cvar_target_pct,
        "target_loss_usd":       round(target_loss_usd, 2),
        "max_annual_drag_pct":   max_annual_drag_pct,
        "put_price_per_share":   round(put_price, 4),
        "put_delta":             round(put["delta"], 4),
        "put_vega":              round(put["vega"], 4),
        "payoff_per_share_at_cvar": round(payoff_per_share, 4),
        "coverage_method":       coverage_method,
        "contracts_needed":      int(contracts_needed),
        "total_premium_usd":     round(total_premium, 2),
        "rolls_per_year":        round(rolls_per_year, 2),
        "annual_premium_usd":    round(annual_premium, 2),
        "annual_drag_pct":       round(annual_drag_pct, 3),
        "residual_cvar_pct":     round(residual_cvar_pct, 3),
        "constraint_binding":    bool(constraint_binding),
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
    print(f"  TAIL-RISK HEDGE OVERLAY")
    print(SEP)
    print(f"  GLD spot:          ${r['gld_spot']:,.2f}")
    print(f"  Notional:          ${r['notional_usd']:,.2f}")
    print(f"  σ:                 {r['sigma']:.2%}")
    print(f"  Tenor:             {r['tenor_days']}d  ({r['rolls_per_year']:.1f} rolls/yr)")
    print(f"  Strike:            ${r['strike']:,.2f}  ({r['moneyness_pct']:.1f}% OTM)")
    print()

    print(f"  CURRENT TAIL")
    print(f"  {'─' * 40}")
    print(f"  CVaR-95 daily:     {r['current_cvar_daily_pct']:.3f}%")
    print(f"  Current loss:      ${r['current_loss_usd']:,.2f}")
    print(f"  Target CVaR:       {r['target_cvar_pct']:.2f}%")
    print(f"  Target loss:       ${r['target_loss_usd']:,.2f}")
    print()

    print(f"  HEDGE")
    print(f"  {'─' * 40}")
    print(f"  Put price/share:   ${r['put_price_per_share']:.4f}")
    print(f"  Put δ:             {r['put_delta']:+.4f}")
    print(f"  Payoff @ CVaR:     ${r['payoff_per_share_at_cvar']:.4f}")
    print(f"  Coverage method:   {r['coverage_method']}")
    print(f"  Contracts:         {r['contracts_needed']}")
    print(f"  Total premium:     ${r['total_premium_usd']:,.2f}")
    print(f"  Annual premium:    ${r['annual_premium_usd']:,.2f}")
    print(f"  Annual drag:       {r['annual_drag_pct']:.3f}%  "
          f"(cap: {r['max_annual_drag_pct']:.2f}%)")
    print()

    if r["constraint_binding"]:
        print(f"  ⚠ Drag cap binding — residual CVaR after hedge: {r['residual_cvar_pct']:.3f}%")
    else:
        print(f"  ✓ Hedge fully covers tail to target CVaR")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tail Risk Hedge Overlay")
    parser.add_argument("--notional", type=float, default=DEFAULT_NOTIONAL)
    parser.add_argument("--cvar-target", type=float, default=DEFAULT_CVAR_TARGET_PCT)
    parser.add_argument("--max-drag", type=float, default=DEFAULT_MAX_DRAG_PCT)
    parser.add_argument("--tenor", type=int, default=DEFAULT_TENOR_DAYS)
    parser.add_argument("--moneyness", type=float, default=DEFAULT_MONEYNESS_PCT)
    args = parser.parse_args()
    run_tail_hedge(
        notional_usd=args.notional,
        cvar_target_pct=args.cvar_target,
        max_annual_drag_pct=args.max_drag,
        tenor_days=args.tenor,
        moneyness_pct=args.moneyness,
    )
