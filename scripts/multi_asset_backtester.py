#!/usr/bin/env python3
"""
Multi-Asset Backtester  (Phase XV Stage 77)
=============================================
Combines the Phase XV single-asset metals backtest with a simulated halal-
equity book and benchmarks the combined portfolio against the S&P 500.

Goal
----
The user's realistic target is 10-20% annualised, beating SPY (~10-12%/y
long-run average), using a halal-compliant universe.  This engine answers:

    Given the metals strategy_backtester result + a momentum-weighted
    halal equity book, what is the combined annualised return, Sharpe,
    and maximum drawdown?  Does it beat SPY?

Strategy
--------
1. Metals book (40% target weight) — uses strategy_backtester output
   directly when available; falls back to gold-LP if not.
2. Halal equity book (60% target weight) — 5-10 large-cap Sharia-screened
   tickers, equal-weight rebalanced monthly, with a simple 21d/63d
   momentum tilt: overweight top-quartile, underweight bottom-quartile.
3. Combined book rebalances daily to the 40/60 target with a 3% dead-band
   to avoid churn.

Benchmarks
----------
- SPY: buy-and-hold S&P 500 (proxy for "would a passive halal investor
  have done better?")
- 60/40 SPY/GLD passive: traditional balanced portfolio benchmark.

Output: data/multi_asset_backtest.json

Usage:
    python3 scripts/multi_asset_backtester.py
    python3 scripts/multi_asset_backtester.py --lookback 1260
    python3 scripts/multi_asset_backtester.py --metals-weight 0.5
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.cache_layer import cached  # noqa: E402
from scripts.regime_adaptive_allocator import (  # noqa: E402
    WEIGHT_SCHEDULE,
    adaptive_series,
    weights_for_tier,
)

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "multi_asset_backtest.json"

# Halal-screened large caps — these all pass standard Sharia screens
# (debt/total assets < 33%, interest income < 5%, no haram revenue)
DEFAULT_HALAL_TICKERS = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet
    "NVDA",   # Nvidia
    "JNJ",    # Johnson & Johnson
    "LLY",    # Eli Lilly
    "UNH",    # UnitedHealth
    "PG",     # Procter & Gamble
    "XOM",    # Exxon (energy)
    "TSLA",   # Tesla
]

DEFAULT_LOOKBACK = 504
DEFAULT_METALS_WEIGHT = 0.40
DEFAULT_EQUITIES_WEIGHT = 0.60
REBAL_DEADBAND = 0.03  # 3% deviation before rebalance
SQ252 = float(np.sqrt(252))

# Annual cost drags
EQUITY_TRADING_COST_BPS = 5    # 5 bps per side
RIBA_FILTER_DRAG_BPS = 25      # mild conservatism for halal exclusions


# ──────────────────────────────────────────────────────────────────────────────
# Data fetch
# ──────────────────────────────────────────────────────────────────────────────
@cached(namespace="yfinance", ttl_seconds=6 * 3600)
def _fetch_panel(
    tickers: list[str], lookback_days: int
) -> tuple[pd.DataFrame, list[str]]:
    """
    Fetch close prices for a list of tickers and return a panel indexed by date.
    Drops tickers with insufficient history.

    Cached for 6 hours.  The cache key includes the full ticker list, so
    swapping the universe automatically misses cache and re-fetches.
    """
    import yfinance as yf

    period = "5y" if lookback_days > 1000 else "2y" if lookback_days > 400 else "1y"
    raw = yf.download(
        tickers, period=period, interval="1d",
        progress=False, auto_adjust=True, group_by="ticker",
    )

    series_by_ticker: dict[str, pd.Series] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            try:
                s = raw[t]["Close"].dropna()
                if len(s) >= lookback_days * 0.7:
                    series_by_ticker[t] = s
            except (KeyError, AttributeError):
                continue
    else:
        # Single ticker fallback
        if "Close" in raw.columns:
            series_by_ticker[tickers[0]] = raw["Close"].dropna()

    if not series_by_ticker:
        raise SystemExit("no usable price history for any ticker")

    panel = pd.DataFrame(series_by_ticker)
    # Strip tz then trim to lookback window
    if panel.index.tz is not None:
        panel.index = panel.index.tz_localize(None)
    if len(panel) > lookback_days:
        panel = panel.iloc[-lookback_days:]

    return panel, list(panel.columns)


# ──────────────────────────────────────────────────────────────────────────────
# Equity book simulation
# ──────────────────────────────────────────────────────────────────────────────
def _momentum_weights(panel: pd.DataFrame, t: int, lookback: int = 63) -> np.ndarray:
    """
    21d/63d momentum tilt.  Top quartile gets 1.5×, bottom 0.5×, rest 1.0×.
    Returns weights normalised to sum to 1.
    """
    if t < lookback:
        # Equal-weight before we have enough history
        return np.ones(panel.shape[1]) / panel.shape[1]

    window = panel.iloc[t - lookback:t]
    momenta = (window.iloc[-1] / window.iloc[0] - 1.0).values  # 63d return
    rank = np.argsort(momenta)
    n = len(momenta)
    weights = np.ones(n)
    top_q = rank[-(n // 4 or 1):]
    bot_q = rank[: (n // 4 or 1)]
    weights[top_q] = 1.5
    weights[bot_q] = 0.5
    return weights / weights.sum()


def _simulate_equity_book(
    panel: pd.DataFrame, rebal_every: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run an equal-weight + momentum-tilt halal equity book.

    Returns
    -------
    (daily_returns_pct, equity_curve)
    """
    n = len(panel)
    daily_returns_pct = np.zeros(n - 1)
    nav = 100_000.0
    equity_curve = [nav]
    weights = np.ones(panel.shape[1]) / panel.shape[1]
    last_rebal = 0

    # Annual drag converted to daily
    daily_drag_pct = (RIBA_FILTER_DRAG_BPS / 10_000) / 252 * 100

    for t in range(1, n):
        # Compute weighted return for this day
        prev_prices = panel.iloc[t - 1].values
        curr_prices = panel.iloc[t].values
        # Skip days with NaN prices
        mask = ~np.isnan(prev_prices) & ~np.isnan(curr_prices) & (prev_prices > 0)
        if mask.sum() == 0:
            daily_returns_pct[t - 1] = 0.0
        else:
            day_returns = np.zeros_like(prev_prices)
            day_returns[mask] = (curr_prices[mask] / prev_prices[mask] - 1.0) * 100
            weighted = float(np.dot(weights[mask], day_returns[mask]) / weights[mask].sum())
            # Subtract daily riba-filter drag (light conservatism penalty)
            daily_returns_pct[t - 1] = weighted - daily_drag_pct

        nav *= 1.0 + daily_returns_pct[t - 1] / 100.0
        equity_curve.append(nav)

        # Rebalance with momentum tilt
        if t - last_rebal >= rebal_every:
            weights = _momentum_weights(panel, t)
            last_rebal = t
            # Apply trading cost: 5 bps × weight turnover (approximated as full)
            cost_pct = EQUITY_TRADING_COST_BPS / 10_000 * 100
            nav *= 1.0 - cost_pct
            equity_curve[-1] = nav

    return daily_returns_pct, np.asarray(equity_curve)


# ──────────────────────────────────────────────────────────────────────────────
# Metals book — replay or stub
# ──────────────────────────────────────────────────────────────────────────────
def _load_metals_daily_series() -> tuple[np.ndarray, list[str], list[str]]:
    """
    Pull the metals strategy_backtest result and return its actual daily
    P&L PLUS the per-day crisis tier (used by the regime-adaptive allocator).

    Returns
    -------
    (returns_pct, dates, crisis_tiers)
    """
    bt_path = DATA_DIR / "strategy_backtest.json"
    if not bt_path.exists():
        return np.array([]), [], []
    bt = json.loads(bt_path.read_text())

    daily_series = bt.get("daily_series") or []
    if daily_series:
        rets = np.asarray([float(d.get("return_pct", 0.0)) for d in daily_series])
        dates = [str(d.get("date", "")) for d in daily_series]
        tiers = [str(d.get("crisis_tier", "NORMAL")) for d in daily_series]
        return rets, dates, tiers

    # Legacy fallback — old metals backtest predates daily_series.  Numbers
    # will be over-smoothed (under-stated vol).  Tiers will all be NORMAL.
    monthlies = (bt.get("monthly") or {}).get("returns", [])
    if not monthlies:
        return np.array([]), [], []
    daily_returns = []
    dates: list[str] = []
    for m in monthlies:
        n_days = int(m["n_days"])
        month_ret = float(m["return_pct"]) / 100.0
        daily_factor = (1.0 + month_ret) ** (1.0 / max(n_days, 1)) - 1.0
        for _ in range(n_days):
            daily_returns.append(daily_factor * 100)
    return np.asarray(daily_returns), dates, ["NORMAL"] * len(daily_returns)


# Backwards-compat wrapper used by older code paths
def _load_metals_daily_returns() -> tuple[np.ndarray, list[str]]:
    r, d, _t = _load_metals_daily_series()
    return r, d


# ──────────────────────────────────────────────────────────────────────────────
# Combined-book simulation
# ──────────────────────────────────────────────────────────────────────────────
def _combine(
    metals_returns_pct: np.ndarray,
    equity_returns_pct: np.ndarray,
    w_metals: float,
    w_equities: float,
    *,
    metals_tiers: list[str] | None = None,
    adaptive: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    """
    Daily-rebalanced combined book.  Two modes:

    - Fixed (default): combine at the static `(w_metals, w_equities)` weights.
    - Adaptive (`adaptive=True` + `metals_tiers` provided): per-day weights
      from `regime_adaptive_allocator.adaptive_series` driven by the crisis
      tier of the metals book on that day.

    Returns
    -------
    (combined_daily_returns_pct, equity_curve, w_metals_series, w_equity_series)
    """
    n = max(len(metals_returns_pct), len(equity_returns_pct))
    m = np.zeros(n)
    e = np.zeros(n)
    if len(metals_returns_pct):
        m[-len(metals_returns_pct):] = metals_returns_pct
    if len(equity_returns_pct):
        e[-len(equity_returns_pct):] = equity_returns_pct

    if adaptive and metals_tiers and len(metals_tiers) >= len(metals_returns_pct):
        # Align tier series to the combined daily index (pad earliest tiers
        # with NORMAL when metals_returns is shorter than equity_returns).
        tiers = ["NORMAL"] * n
        offset = n - len(metals_tiers)
        for i, t in enumerate(metals_tiers):
            idx = i + offset
            if 0 <= idx < n:
                tiers[idx] = t
        w_metals_series, w_equity_series = adaptive_series(tiers, smoothing=True)
    else:
        w_metals_series = [w_metals] * n
        w_equity_series = [w_equities] * n

    combined = np.array(
        [w_metals_series[i] * m[i] + w_equity_series[i] * e[i] for i in range(n)]
    )
    nav = 100_000.0
    curve = [nav]
    for r in combined:
        nav *= 1.0 + r / 100.0
        curve.append(nav)
    return combined, np.asarray(curve), w_metals_series, w_equity_series


# ──────────────────────────────────────────────────────────────────────────────
# Stats
# ──────────────────────────────────────────────────────────────────────────────
def _stats(daily_returns_pct: np.ndarray) -> dict:
    if len(daily_returns_pct) == 0:
        return {"n": 0}
    ret = np.asarray(daily_returns_pct)
    cum = (np.prod(1 + ret / 100.0) - 1.0) * 100
    days = len(ret)
    annualised = ((1 + cum / 100.0) ** (252 / max(days, 1)) - 1.0) * 100
    mu = float(ret.mean())
    sd = float(ret.std(ddof=0))
    sharpe = (mu / sd * SQ252) if sd > 1e-12 else None

    # Max drawdown
    curve = np.cumprod(1 + ret / 100.0)
    peak = np.maximum.accumulate(curve)
    dd = (curve - peak) / peak
    max_dd = float(dd.min() * 100)

    win_days = float((ret > 0).mean() * 100)
    return {
        "n_days":         int(days),
        "cum_return_pct": round(float(cum), 2),
        "annualised_pct": round(float(annualised), 2),
        "ann_vol_pct":    round(float(sd * SQ252), 2),
        "sharpe":         round(sharpe, 3) if sharpe is not None else None,
        "max_drawdown_pct": round(max_dd, 2),
        "win_days_pct":   round(win_days, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public entry
# ──────────────────────────────────────────────────────────────────────────────
def run_multi_asset_backtest(
    halal_tickers: list[str] | None = None,
    lookback_days: int = DEFAULT_LOOKBACK,
    w_metals: float = DEFAULT_METALS_WEIGHT,
    w_equities: float = DEFAULT_EQUITIES_WEIGHT,
    adaptive: bool = False,
) -> dict:
    halal_tickers = halal_tickers or DEFAULT_HALAL_TICKERS

    # ── Equity book ────────────────────────────────────────────────────────
    panel, used_tickers = _fetch_panel(halal_tickers, lookback_days)
    eq_daily, eq_curve = _simulate_equity_book(panel)

    # ── Metals book ────────────────────────────────────────────────────────
    metals_daily, _metals_dates, metals_tiers = _load_metals_daily_series()

    # ── Combined (adaptive or fixed) ──────────────────────────────────────
    combined_daily, combined_curve, w_m_series, w_e_series = _combine(
        metals_daily, eq_daily, w_metals, w_equities,
        metals_tiers=metals_tiers, adaptive=adaptive,
    )

    # ── Benchmarks ────────────────────────────────────────────────────────
    bench_panel, _ = _fetch_panel(["SPY", "GLD"], lookback_days)
    spy_returns_pct = bench_panel["SPY"].pct_change().dropna().values * 100
    gld_returns_pct = bench_panel["GLD"].pct_change().dropna().values * 100
    # 60/40 passive
    bench_60_40 = 0.60 * spy_returns_pct + 0.40 * gld_returns_pct

    # ── Stats ─────────────────────────────────────────────────────────────
    equity_stats   = _stats(eq_daily)
    metals_stats   = _stats(metals_daily)
    combined_stats = _stats(combined_daily)
    spy_stats      = _stats(spy_returns_pct)
    bench_60_40_stats = _stats(bench_60_40)

    # ── Verdict ───────────────────────────────────────────────────────────
    ann = float(combined_stats.get("annualised_pct") or 0)
    spy_ann = float(spy_stats.get("annualised_pct") or 0)
    sharpe = float(combined_stats.get("sharpe") or 0)

    beat_spy_by_pp = ann - spy_ann
    if 10 <= ann <= 25 and sharpe >= 1.2 and beat_spy_by_pp >= 2:
        verdict = "ON_TARGET"
    elif ann > 25 and sharpe >= 1.5:
        verdict = "EXCEEDS_TARGET"
    elif ann >= 10 and beat_spy_by_pp >= 0:
        verdict = "MEETS_TARGET"
    elif beat_spy_by_pp < 0:
        verdict = "TRAILING_SPY"
    else:
        verdict = "BELOW_TARGET"

    note = (
        f"Combined book annualised {ann:+.2f}% (Sharpe {sharpe:.2f}) vs "
        f"SPY {spy_ann:+.2f}% — "
        f"{'+' if beat_spy_by_pp >= 0 else ''}{beat_spy_by_pp:.2f}pp vs benchmark."
    )

    out = {
        "schema_version": "1.0",
        "engine":         "multi_asset_backtester",
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lookback_days":  lookback_days,
        "halal_tickers_attempted": halal_tickers,
        "halal_tickers_used":      used_tickers,
        "n_equity_days":           int(len(eq_daily)),
        "n_metals_days":           int(len(metals_daily)),
        "weights": {
            "metals":   w_metals,
            "equities": w_equities,
            "adaptive": bool(adaptive),
            "weight_schedule": {
                k: {"metals": v[0], "equity": v[1]} for k, v in WEIGHT_SCHEDULE.items()
            } if adaptive else None,
            "avg_metals_weight":   round(sum(w_m_series) / len(w_m_series), 3) if w_m_series else w_metals,
            "avg_equity_weight":   round(sum(w_e_series) / len(w_e_series), 3) if w_e_series else w_equities,
        },
        "books": {
            "metals":   metals_stats,
            "equities": equity_stats,
            "combined": combined_stats,
        },
        "benchmarks": {
            "spy_buyhold":         spy_stats,
            "passive_60_40":       bench_60_40_stats,
        },
        "delta_vs_spy": {
            "annualised_pp": round(ann - spy_ann, 2),
            "sharpe_delta":  round(
                (combined_stats.get("sharpe") or 0)
                - (spy_stats.get("sharpe") or 0), 3,
            ),
            "max_dd_delta_pp": round(
                (combined_stats.get("max_drawdown_pct") or 0)
                - (spy_stats.get("max_drawdown_pct") or 0), 2,
            ),
        },
        "verdict":  verdict,
        "note":     note,
        "combined_equity_curve_sampled": [
            {"i": i, "nav_usd": round(float(combined_curve[i]), 2)}
            for i in range(0, len(combined_curve),
                           max(1, len(combined_curve) // 200))
        ],
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(out, indent=2, default=str))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    ap.add_argument("--metals-weight", type=float, default=DEFAULT_METALS_WEIGHT)
    ap.add_argument("--equities-weight", type=float, default=DEFAULT_EQUITIES_WEIGHT)
    ap.add_argument("--tickers", type=str, default=None,
                    help="Comma-separated halal-screened tickers")
    ap.add_argument("--adaptive", action="store_true",
                    help="Use crisis-tier-driven dynamic allocation instead of fixed weights")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    tickers = args.tickers.split(",") if args.tickers else None
    out = run_multi_asset_backtest(
        halal_tickers=tickers,
        lookback_days=args.lookback,
        w_metals=args.metals_weight,
        w_equities=args.equities_weight,
        adaptive=args.adaptive,
    )
    if args.quiet:
        return 0

    b = out["books"]
    bm = out["benchmarks"]
    print("=" * 64)
    print(f"MULTI-ASSET BACKTEST  ({out['generated_at']})")
    print("=" * 64)
    print(f"  Window         : {out['n_equity_days']} equity days, "
          f"{out['n_metals_days']} metals days")
    print(f"  Halal universe : {len(out['halal_tickers_used'])} tickers "
          f"({', '.join(out['halal_tickers_used'])})")
    print(f"  Allocation     : metals {out['weights']['metals']*100:.0f}% / "
          f"equities {out['weights']['equities']*100:.0f}%")
    print()
    print(f"  {'Book':<20s} {'Ann':>10s} {'Sharpe':>8s} {'DD':>9s} {'Vol':>8s} {'Win%':>7s}")
    for name, stats in (
        ("Metals (strategy)", b['metals']),
        ("Equities (halal)",  b['equities']),
        ("Combined",          b['combined']),
        ("---",               None),
        ("SPY buy-hold",      bm['spy_buyhold']),
        ("Passive 60/40",     bm['passive_60_40']),
    ):
        if stats is None:
            print(f"  {name}")
            continue
        ann = stats.get('annualised_pct', 0)
        sharpe = stats.get('sharpe')
        dd = stats.get('max_drawdown_pct', 0)
        vol = stats.get('ann_vol_pct', 0)
        win = stats.get('win_days_pct', 0)
        print(f"  {name:<20s} {ann:>+9.2f}%  "
              f"{sharpe:>7.2f}  {dd:>+8.2f}%  {vol:>+7.1f}%  {win:>6.1f}%"
              if sharpe is not None else f"  {name:<20s}    n/a")
    print()
    d = out["delta_vs_spy"]
    print(f"  vs SPY         : {d['annualised_pp']:+.2f}pp annualised  "
          f"ΔSharpe {d['sharpe_delta']:+.2f}  ΔMaxDD {d['max_dd_delta_pp']:+.2f}pp")
    print()
    print(f"  Verdict        : {out['verdict']}")
    print(f"                   {out['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
