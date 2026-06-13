#!/usr/bin/env python3
"""
scripts/transaction_cost_model.py
==================================
Realistic Transaction Cost Analysis (TCA).

Models the gap between paper-Sharpe and deliverable-Sharpe — the difference
between "what the backtest claims" and "what real money earns."

Cost components (each reported in basis points and USD):

  1. Half-spread cost
        spread_bps = bid-ask / mid * 1e4 / 2
     Pulled from yfinance bid/ask when available; falls back to a Roll (1984)
     estimator from negative serial covariance of mid-price changes.

  2. Square-root market impact (Almgren-Chriss / Kissell)
        impact_bps = η * σ_daily * sqrt(Q / ADV) * 1e4
     η = 0.142 (institutional calibration). Q = trade shares.
     Decomposes into permanent (1/3) + temporary (2/3) per Almgren-Chriss.

  3. Vol-regime multiplier
        × 1.0 normal, × 1.5 elevated, × 2.0 panic    (VIX-aware)

  4. UAE physical gold premium (metals only, when physical=True)
        institutional dealer: 75 bps over spot
        retail (Dubai souk):  150 bps over spot

  5. Optimal slicing schedule (Almgren-Chriss limit)
        If Q > 5% of ADV, split across N days where
            N = ceil(Q / (ADV * max_participation_rate))

Output:
    data/transaction_cost_model.json

Single-trade CLI:
    python3 scripts/transaction_cost_model.py --ticker GC=F --notional 50000 --side BUY
    python3 scripts/transaction_cost_model.py --ticker SNOW --notional 25000 --side BUY

Bulk CLI (portfolio TCA from data/equity_ranker/ranking_latest.parquet):
    python3 scripts/transaction_cost_model.py --portfolio-tca --top-n 10 --portfolio-value 100000
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import yfinance as yf
except ImportError:
    yf = None

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "transaction_cost_model.json"
RANKING_PARQUET = DATA_DIR / "equity_ranker" / "ranking_latest.parquet"

LINE_W = 80
SEP = "━" * LINE_W

# Almgren-Chriss / Kissell impact coefficient
IMPACT_ETA = 0.142
PERMANENT_FRAC = 1.0 / 3.0
TEMPORARY_FRAC = 2.0 / 3.0

# Slicing
MAX_PARTICIPATION = 0.05      # 5% of ADV
MAX_DAYS_PER_SLICE = 10        # never schedule beyond 10 days

# Physical premium (UAE) in basis points over spot
PHYSICAL_PREMIUM_INSTITUTIONAL_BPS = 75    # 0.75%
PHYSICAL_PREMIUM_RETAIL_BPS        = 150   # 1.50%

METAL_TICKERS = {"GC=F", "SI=F", "PL=F", "PA=F", "HG=F"}


@dataclass
class TradeTCA:
    ticker: str
    side: str
    notional_usd: float
    last_price: float
    shares: float
    adv_shares: float
    participation_pct: float
    spread_bps: float
    spread_source: str
    impact_bps_total: float
    impact_bps_permanent: float
    impact_bps_temporary: float
    vol_regime: str
    vol_multiplier: float
    physical_premium_bps: float
    total_oneway_bps: float
    total_oneway_usd: float
    roundtrip_bps: float
    optimal_slices: int
    days_to_execute: int
    annualised_vol_pct: float
    cost_alpha_breakeven_bps: float


# ── Data layer ────────────────────────────────────────────────────────────────

def _fetch_quote(ticker: str, lookback: int = 90) -> dict:
    """Pull recent OHLCV + bid/ask + volume for a ticker."""
    if yf is None:
        raise ImportError("yfinance is required")

    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=lookback * 2)

    try:
        raw = yf.download(
            ticker, start=start, end=end + pd.Timedelta(days=1),
            progress=False, auto_adjust=True, threads=False,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
    except Exception as exc:
        raise RuntimeError(f"yfinance failed for {ticker}: {exc}")

    if raw is None or raw.empty:
        raise RuntimeError(f"No data for {ticker}")

    df = raw.tail(lookback).dropna()
    last_close = float(df["Close"].iloc[-1])
    high_low = float((df["High"].iloc[-1] - df["Low"].iloc[-1]) / df["Close"].iloc[-1])
    adv_shares = float(df["Volume"].rolling(21).mean().iloc[-1]) if "Volume" in df.columns else 0.0
    log_rets = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    daily_vol = float(log_rets.std(ddof=1))
    ann_vol = daily_vol * math.sqrt(252)

    # Try to pull live bid/ask via Ticker.info
    bid = ask = None
    try:
        tinfo = yf.Ticker(ticker).fast_info
        bid = float(tinfo.get("bid")) if tinfo.get("bid") is not None else None
        ask = float(tinfo.get("ask")) if tinfo.get("ask") is not None else None
        if bid is not None and ask is not None and bid > 0 and ask > bid:
            pass
        else:
            bid = ask = None
    except Exception:
        bid = ask = None

    return {
        "ticker": ticker,
        "last_close": last_close,
        "bid": bid, "ask": ask,
        "intraday_range_pct": high_low * 100,
        "adv_shares": adv_shares,
        "daily_vol": daily_vol,
        "ann_vol": ann_vol,
        "log_returns": log_rets.values,
    }


# ── Spread estimation ─────────────────────────────────────────────────────────

def _estimate_spread_bps(quote: dict) -> tuple[float, str]:
    """
    Returns (half_spread_bps, source).
    1. If yfinance bid/ask exists, use the quoted spread.
    2. Else use Roll (1984) estimator, bounded by realistic liquidity priors.
    3. Else fall back to a vol-scaled heuristic.

    Liquidity priors (half-spread bps, based on real institutional execution):
        Liquid futures (=F):       0.5  – 3
        Large-cap equities:        1    – 8
        Mid-cap equities:          3    – 15
        Generic fallback ceiling:  50
    """
    ticker = quote.get("ticker", "")
    is_future = ticker.endswith("=F")
    is_dxy = ticker == "DX-Y.NYB"

    if is_future or is_dxy:
        floor_bps, ceil_bps = 0.5, 3.0
    else:
        # Estimate liquidity tier from price × ADV (dollar-volume)
        adv_dollar = quote["adv_shares"] * quote["last_close"]
        if adv_dollar > 100_000_000:        # > $100M/day = mega-cap
            floor_bps, ceil_bps = 1.0, 5.0
        elif adv_dollar > 10_000_000:       # > $10M/day = mid-cap
            floor_bps, ceil_bps = 2.0, 12.0
        else:
            floor_bps, ceil_bps = 5.0, 50.0

    bid, ask = quote["bid"], quote["ask"]
    if bid is not None and ask is not None and bid > 0 and ask > bid:
        mid = (bid + ask) / 2.0
        full_bps = (ask - bid) / mid * 1e4
        half_bps = max(floor_bps, min(ceil_bps, full_bps / 2.0))
        return half_bps, "quote"

    # Roll estimator from log returns, bounded by liquidity tier
    rets = quote["log_returns"]
    if len(rets) >= 30:
        cov = float(np.cov(rets[1:], rets[:-1])[0, 1])
        if cov < 0:
            roll_full_bps = 2.0 * math.sqrt(-cov) * 1e4
            half_bps = max(floor_bps, min(ceil_bps, roll_full_bps / 2.0))
            return half_bps, "roll"

    # Heuristic: midpoint of liquidity tier
    return (floor_bps + ceil_bps) / 2.0, "heuristic"


# ── Vol regime classification ─────────────────────────────────────────────────

def _classify_vol_regime() -> tuple[str, float]:
    """Pull VIX and bucket: <16 normal, 16-25 elevated, >25 panic."""
    if yf is None:
        return "unknown", 1.0
    try:
        vix = yf.download("^VIX", period="5d", interval="1d",
                          progress=False, auto_adjust=True, threads=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.droplevel(1)
        last = float(vix["Close"].iloc[-1])
    except Exception:
        return "unknown", 1.0

    if last < 16:
        return f"normal (VIX={last:.1f})", 1.0
    if last < 25:
        return f"elevated (VIX={last:.1f})", 1.5
    return f"panic (VIX={last:.1f})", 2.0


# ── Square-root market impact (Almgren-Chriss / Kissell) ──────────────────────

def _market_impact_bps(quote: dict, shares: float) -> dict:
    """
    Square-root market impact model.
        impact_bps = IMPACT_ETA * σ_daily * sqrt(Q / ADV) * 1e4
    Decomposes 1/3 permanent + 2/3 temporary per Almgren-Chriss.
    """
    daily_vol = quote.get("daily_vol", 0.01)
    adv = quote.get("adv_shares", 0)
    if adv <= 0:
        # Liquid futures with unreported volume — assume tight impact
        total_bps = max(0.5, daily_vol * 100 * 1.0)
    else:
        total_bps = IMPACT_ETA * daily_vol * math.sqrt(shares / adv) * 1e4
    return {
        "total":     round(total_bps, 3),
        "permanent": round(total_bps * PERMANENT_FRAC, 3),
        "temporary": round(total_bps * TEMPORARY_FRAC, 3),
    }


# ── Optimal slicing ───────────────────────────────────────────────────────────

def _optimal_slicing(quote: dict, shares: float) -> tuple[int, int]:
    """
    Almgren-Chriss-style schedule capped at MAX_PARTICIPATION of ADV per day.
    Returns (n_slices, days_to_execute).
    """
    adv = quote.get("adv_shares", 0)
    if adv <= 0:
        return 1, 1
    participation = shares / adv
    if participation <= MAX_PARTICIPATION:
        return 1, 1
    days = min(int(math.ceil(shares / (adv * MAX_PARTICIPATION))), MAX_DAYS_PER_SLICE)
    return max(days, 1), days


# ── Single-trade TCA ──────────────────────────────────────────────────────────

def estimate_trade_cost(
    ticker: str,
    side: str = "BUY",
    notional_usd: float = 10_000.0,
    physical: bool = False,
    physical_retail: bool = False,
    vol_regime_cache: dict | None = None,
    quote_cache: dict | None = None,
) -> TradeTCA:
    """Run end-to-end TCA on a single trade."""
    side = side.upper()
    if side not in ("BUY", "SELL"):
        raise ValueError("side must be BUY or SELL")

    quote = quote_cache or _fetch_quote(ticker)
    last = quote["last_close"]
    if last <= 0:
        raise RuntimeError(f"Bad price for {ticker}: {last}")

    shares = notional_usd / last
    adv = quote["adv_shares"]
    participation = (shares / adv * 100) if adv > 0 else 0.0

    # Spread
    spread_bps, spread_source = _estimate_spread_bps(quote)

    # Impact
    impact = _market_impact_bps(quote, shares)

    # Vol regime multiplier (apply to impact + spread, not physical premium)
    if vol_regime_cache is None:
        vol_regime, vol_mult = _classify_vol_regime()
    else:
        vol_regime, vol_mult = vol_regime_cache["regime"], vol_regime_cache["mult"]

    # Physical premium (only for metals + buy side; sell side gets a haircut, ~ -50 bps)
    physical_bps = 0.0
    if physical and ticker in METAL_TICKERS:
        if side == "BUY":
            physical_bps = (PHYSICAL_PREMIUM_RETAIL_BPS
                            if physical_retail
                            else PHYSICAL_PREMIUM_INSTITUTIONAL_BPS)
        else:
            # Selling physical: dealer pays you ~50 bps below spot
            physical_bps = (PHYSICAL_PREMIUM_INSTITUTIONAL_BPS / 1.5
                            if not physical_retail
                            else PHYSICAL_PREMIUM_RETAIL_BPS / 2.0)

    # Total one-way cost in bps
    total_oneway_bps = (
        spread_bps * vol_mult
        + impact["total"] * vol_mult
        + physical_bps
    )
    total_oneway_usd = total_oneway_bps / 1e4 * notional_usd

    # Roundtrip = 2 × one-way (buy then sell at same notional in expectation)
    # Physical premium asymmetric: buy cost + sell discount
    roundtrip_bps = 2.0 * (spread_bps * vol_mult + impact["total"] * vol_mult) + 2.0 * physical_bps

    # Slicing
    n_slices, days = _optimal_slicing(quote, shares)

    # Alpha breakeven: at a 60% hit rate, what edge does an alpha need to beat costs?
    # Required alpha-bps = roundtrip_bps / hit_rate_advantage
    cost_alpha_breakeven_bps = roundtrip_bps  # simplest interpretation

    return TradeTCA(
        ticker=ticker,
        side=side,
        notional_usd=round(notional_usd, 2),
        last_price=round(last, 4),
        shares=round(shares, 4),
        adv_shares=round(adv, 0),
        participation_pct=round(participation, 3),
        spread_bps=round(spread_bps, 3),
        spread_source=spread_source,
        impact_bps_total=impact["total"],
        impact_bps_permanent=impact["permanent"],
        impact_bps_temporary=impact["temporary"],
        vol_regime=vol_regime,
        vol_multiplier=vol_mult,
        physical_premium_bps=round(physical_bps, 1),
        total_oneway_bps=round(total_oneway_bps, 2),
        total_oneway_usd=round(total_oneway_usd, 2),
        roundtrip_bps=round(roundtrip_bps, 2),
        optimal_slices=n_slices,
        days_to_execute=days,
        annualised_vol_pct=round(quote["ann_vol"] * 100, 2),
        cost_alpha_breakeven_bps=round(cost_alpha_breakeven_bps, 2),
    )


# ── Portfolio-level TCA (batch over halal universe top-N) ─────────────────────

def run_portfolio_tca(
    top_n: int = 10,
    portfolio_value: float = 100_000.0,
    weight_per_position: float | None = None,
    metals_physical: bool = True,
) -> dict:
    """
    Bulk TCA across the top-N halal-equity ranker plus the metals book.

    For halal equities, weight defaults to 1/top_n if not specified.
    For metals, applies a fixed 5% gold + 5% silver illustrative allocation.
    """
    if weight_per_position is None:
        weight_per_position = 1.0 / max(top_n, 1)

    # Cache vol regime once
    vol_regime, vol_mult = _classify_vol_regime()
    vrc = {"regime": vol_regime, "mult": vol_mult}

    equity_tca: list[dict] = []
    universe_loaded = False
    if RANKING_PARQUET.exists():
        try:
            df = pd.read_parquet(RANKING_PARQUET)
            df = df.head(top_n)
            universe_loaded = True
            for ticker in df.index.tolist():
                try:
                    notional = portfolio_value * weight_per_position
                    tca = estimate_trade_cost(
                        ticker=ticker, side="BUY", notional_usd=notional,
                        physical=False, vol_regime_cache=vrc,
                    )
                    equity_tca.append(asdict(tca))
                except Exception as exc:
                    equity_tca.append({"ticker": ticker, "error": str(exc)})
        except Exception as exc:
            equity_tca.append({"error": f"Could not load ranker: {exc}"})

    # Metals TCA
    metals_tca: list[dict] = []
    for metal in ("GC=F", "SI=F"):
        try:
            notional = portfolio_value * 0.05
            tca = estimate_trade_cost(
                ticker=metal, side="BUY", notional_usd=notional,
                physical=metals_physical, vol_regime_cache=vrc,
            )
            metals_tca.append(asdict(tca))
        except Exception as exc:
            metals_tca.append({"ticker": metal, "error": str(exc)})

    # Aggregate stats
    all_costs_bps = [t.get("total_oneway_bps", 0) for t in equity_tca + metals_tca
                     if "error" not in t]
    aggregate = {
        "n_trades":            len(all_costs_bps),
        "avg_oneway_cost_bps": round(float(np.mean(all_costs_bps)), 2) if all_costs_bps else 0,
        "max_oneway_cost_bps": round(float(np.max(all_costs_bps)), 2) if all_costs_bps else 0,
        "min_oneway_cost_bps": round(float(np.min(all_costs_bps)), 2) if all_costs_bps else 0,
        "vol_regime":          vol_regime,
    }

    output = {
        "generated_at":    datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "portfolio_value": portfolio_value,
        "top_n_equities":  top_n,
        "weight_per_pos":  weight_per_position,
        "vol_regime":      vol_regime,
        "vol_multiplier":  vol_mult,
        "metals":          metals_tca,
        "equities":        equity_tca,
        "aggregate":       aggregate,
        "universe_loaded": universe_loaded,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    _print_portfolio_report(output)
    return output


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_single_report(tca: TradeTCA) -> None:
    print(f"\n{SEP}")
    print(f"  TRANSACTION COST ANALYSIS  — {tca.ticker}  {tca.side}")
    print(SEP)
    print(f"  Notional:           ${tca.notional_usd:>14,.2f}")
    print(f"  Last Price:         ${tca.last_price:>14,.4f}")
    print(f"  Shares:              {tca.shares:>14,.4f}")
    print(f"  ADV (21d):           {tca.adv_shares:>14,.0f}")
    print(f"  Participation:       {tca.participation_pct:>13.3f}% of ADV")
    print(f"  Annualised Vol:      {tca.annualised_vol_pct:>13.2f}%")
    print(f"  Vol Regime:          {tca.vol_regime}  (×{tca.vol_multiplier})")
    print(f"  {'─' * 50}")
    print(f"  Half-Spread:         {tca.spread_bps:>9.3f} bps   ({tca.spread_source})")
    print(f"  Impact (Permanent):  {tca.impact_bps_permanent:>9.3f} bps")
    print(f"  Impact (Temporary):  {tca.impact_bps_temporary:>9.3f} bps")
    if tca.physical_premium_bps > 0:
        print(f"  Physical Premium:    {tca.physical_premium_bps:>9.1f} bps")
    print(f"  {'─' * 50}")
    print(f"  TOTAL ONE-WAY:       {tca.total_oneway_bps:>9.2f} bps  "
          f"(${tca.total_oneway_usd:,.2f})")
    print(f"  TOTAL ROUND-TRIP:    {tca.roundtrip_bps:>9.2f} bps")
    print(f"  Alpha breakeven:     {tca.cost_alpha_breakeven_bps:>9.2f} bps")
    print()
    if tca.optimal_slices > 1:
        print(f"  OPTIMAL SCHEDULE:    {tca.optimal_slices} slices over {tca.days_to_execute} days")
    else:
        print(f"  OPTIMAL SCHEDULE:    Single execution OK ({tca.participation_pct:.2f}% of ADV)")
    print(SEP)


def _print_portfolio_report(out: dict) -> None:
    print(f"\n{SEP}")
    print(f"  PORTFOLIO TCA  —  ${out['portfolio_value']:,.0f}  |  "
          f"Vol regime: {out['vol_regime']}")
    print(SEP)

    print(f"  METALS")
    print(f"  {'Ticker':<10s} {'Notional':>12s} {'Spread':>9s} {'Impact':>9s} "
          f"{'Phys':>7s} {'1-way':>9s} {'Schedule':>14s}")
    for t in out["metals"]:
        if "error" in t:
            print(f"  {t.get('ticker', '?'):<10s}  {t['error'][:60]}")
            continue
        sched = (f"{t['optimal_slices']}x{t['days_to_execute']}d"
                 if t["optimal_slices"] > 1 else "single")
        print(f"  {t['ticker']:<10s} ${t['notional_usd']:>10,.0f} "
              f"{t['spread_bps']:>7.2f}bp {t['impact_bps_total']:>7.2f}bp "
              f"{t['physical_premium_bps']:>5.0f}bp {t['total_oneway_bps']:>7.2f}bp "
              f"{sched:>14s}")

    print()
    print(f"  HALAL EQUITIES (top-{out['top_n_equities']})")
    print(f"  {'Ticker':<10s} {'Notional':>12s} {'Spread':>9s} {'Impact':>9s} "
          f"{'Part%':>8s} {'1-way':>9s} {'Schedule':>14s}")
    for t in out["equities"]:
        if "error" in t:
            print(f"  {t.get('ticker', '?'):<10s}  {t.get('error', '')[:60]}")
            continue
        sched = (f"{t['optimal_slices']}x{t['days_to_execute']}d"
                 if t["optimal_slices"] > 1 else "single")
        print(f"  {t['ticker']:<10s} ${t['notional_usd']:>10,.0f} "
              f"{t['spread_bps']:>7.2f}bp {t['impact_bps_total']:>7.2f}bp "
              f"{t['participation_pct']:>7.3f}% {t['total_oneway_bps']:>7.2f}bp "
              f"{sched:>14s}")

    agg = out["aggregate"]
    print()
    print(f"  AGGREGATE")
    print(f"  {'─' * 50}")
    print(f"  Trades analysed:      {agg['n_trades']}")
    print(f"  Avg one-way cost:     {agg['avg_oneway_cost_bps']:.2f} bps")
    print(f"  Range:                {agg['min_oneway_cost_bps']:.2f} – "
          f"{agg['max_oneway_cost_bps']:.2f} bps")
    print(f"  Saved: {OUTPUT_FILE}")
    print(SEP)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transaction Cost Analysis")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--side", default="BUY", choices=["BUY", "SELL"])
    parser.add_argument("--notional", type=float, default=10_000.0)
    parser.add_argument("--physical", action="store_true",
                        help="Apply UAE physical premium (metals only)")
    parser.add_argument("--retail", action="store_true",
                        help="Use retail premium instead of institutional")
    parser.add_argument("--portfolio-tca", action="store_true",
                        help="Run portfolio-level TCA over halal universe + metals")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--portfolio-value", type=float, default=100_000.0)
    args = parser.parse_args()

    if args.portfolio_tca:
        run_portfolio_tca(
            top_n=args.top_n,
            portfolio_value=args.portfolio_value,
            metals_physical=args.physical,
        )
    elif args.ticker:
        tca = estimate_trade_cost(
            ticker=args.ticker, side=args.side, notional_usd=args.notional,
            physical=args.physical, physical_retail=args.retail,
        )
        _print_single_report(tca)
        OUTPUT_FILE.write_text(json.dumps({"single_trade": asdict(tca)}, indent=2))
    else:
        # Default: portfolio TCA
        run_portfolio_tca(top_n=10, portfolio_value=100_000.0, metals_physical=True)
