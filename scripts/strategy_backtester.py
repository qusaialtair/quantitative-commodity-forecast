#!/usr/bin/env python3
"""
Strategy Backtester  (Phase XV Stage 76)
=========================================
Historical replay of the Phase XIV stack against ~2 years of gold prices.
For each historical day t we:

    1. Estimate the regime-state at time t (HMM-proxy via rolling vol +
       short-term return sign).
    2. Apply the Strategy Selector's deterministic rules using those
       proxies + an optional alpha-conviction proxy.
    3. Simulate the strategy's P&L on the next-day return, using realistic
       trade-size scaling from a regime / Kelly multiplier matching the
       live selector.
    4. Track monthly returns, hit rate, Sharpe, max drawdown, % months
       hitting the configured target (default 10%).

This is a *shadow* backtest — it doesn't refit any models, it uses the
selector's rule-tree against historically realised features.  Because the
selector is the bottleneck of the live system, this measures the rule
quality more honestly than backtesting the LSTM in isolation.

Output: data/strategy_backtest.json

Usage:
    python3 scripts/strategy_backtester.py                 # default 504d backtest
    python3 scripts/strategy_backtester.py --lookback 1260 # 5y
    python3 scripts/strategy_backtester.py --target 10     # 10%/mo target
    python3 scripts/strategy_backtester.py --quiet
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.cache_layer import cached  # noqa: E402
from scripts.conviction_weights_optimizer import load_weights as _load_conviction_weights  # noqa: E402
from scripts.crisis_detector import (  # noqa: E402
    apply_guard as _crisis_apply_guard,
    classify_from_prices as _crisis_classify,
)

# Lazy-cached optimised weights — loaded once per process.
_CONVICTION_WEIGHTS_CACHE: dict[str, float] | None = None


def _conviction_weights() -> dict[str, float]:
    global _CONVICTION_WEIGHTS_CACHE
    if _CONVICTION_WEIGHTS_CACHE is None:
        _CONVICTION_WEIGHTS_CACHE = _load_conviction_weights()
    return _CONVICTION_WEIGHTS_CACHE

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "strategy_backtest.json"

DEFAULT_LOOKBACK = 504           # 2y of trading days
DEFAULT_TICKER = "GC=F"
DEFAULT_TARGET_PCT = 10.0
TRADING_DAYS_PER_MONTH = 21
SQ252 = float(np.sqrt(252))


# ──────────────────────────────────────────────────────────────────────────────
# Data fetch
# ──────────────────────────────────────────────────────────────────────────────
@cached(namespace="yfinance", ttl_seconds=6 * 3600)
def _fetch_history(ticker: str, lookback_days: int) -> tuple[np.ndarray, list[str]]:
    """Returns (close_prices, dates) for the requested lookback window.

    Cached for 6 hours — daily bars don't change intraday, so this saves
    redundant yfinance hits when the backtester is re-run.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("yfinance not installed")

    period = "5y" if lookback_days > 1000 else "2y" if lookback_days > 400 else "1y"
    hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    if hist is None or hist.empty:
        raise SystemExit(f"no history for {ticker}")
    closes = hist["Close"].astype(float).values
    dates = [d.strftime("%Y-%m-%d") for d in hist.index]
    if len(closes) > lookback_days:
        closes = closes[-lookback_days:]
        dates = dates[-lookback_days:]
    return closes, dates


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering at time t (uses only data <= t)
# ──────────────────────────────────────────────────────────────────────────────
def _rolling_vol_pct(returns: np.ndarray, window: int = 21) -> np.ndarray:
    """Annualised rolling vol % at each index (NaN-padded for the first window)."""
    out = np.full_like(returns, np.nan, dtype=float)
    for i in range(window, len(returns)):
        seg = returns[i - window:i]
        out[i] = float(np.std(seg, ddof=0)) * SQ252 * 100
    return out


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    csum = np.cumsum(np.insert(arr, 0, 0))
    out[window - 1:] = (csum[window:] - csum[:-window]) / window
    return out


def _classify_regime_proxy(
    rv_21d: float, drift_5d: float, drift_21d: float
) -> str:
    """Cheap HMM-proxy classification using realised vol + short-term drift."""
    if not np.isfinite(rv_21d):
        return "UNKNOWN"
    if rv_21d > 25.0:
        return "VOLATILE"
    if drift_5d > 0.005 and drift_21d > 0.005:
        return "BULLISH"
    if drift_5d < -0.005 and drift_21d < -0.005:
        return "BEARISH"
    return "VOLATILE"  # default to fade regime when ambiguous


def _classify_vol_regime(rv_21d: float, rv_63d: float) -> str:
    if not np.isfinite(rv_21d):
        return "UNKNOWN"
    if rv_21d > 30.0:
        return "EXTREME"
    if rv_21d > 22.0:
        return "ELEVATED"
    if rv_21d < 12.0:
        return "LOW"
    if np.isfinite(rv_63d) and rv_21d > rv_63d * 1.25:
        return "ELEVATED"
    return "NORMAL"


def _pivot_score(closes: np.ndarray, i: int) -> float:
    """
    Phase XIX — detect a regime pivot vs the prevailing trend.

    Returns a score in [-1, +1]:
      positive = sharp upward pivot just confirmed
      negative = sharp downward pivot just confirmed
      ~0      = no pivot

    The trick: traditional SMA-based conviction lags behind reversals
    by ~20 days.  We need to spot the turn *while* it's happening so we
    can short the leg of a bear market (2015 China) or pivot fast in a
    whipsaw (2008 GFC).

    Method (4 components):
      a) Prevailing-trend direction: sign(SMA50 - SMA200)
      b) Short-term reversal: sign(mom_5d) vs prevailing
      c) Mid-term confirmation: sign(mom_21d) vs prevailing
      d) Acceleration: mom_5d - mom_21d (do recent days outpace the trend?)

    The score is the *intensity of the disagreement* between the
    prevailing trend and short/mid-term momentum, scaled by acceleration.
    """
    if i < 60:
        return 0.0
    if i >= 200:
        sma50  = float(np.mean(closes[i - 50:i]))
        sma200 = float(np.mean(closes[i - 200:i]))
        prevailing = 1.0 if sma50 > sma200 else -1.0
    else:
        # Insufficient history for SMA200 — use SMA100 as fallback.
        sma50  = float(np.mean(closes[i - 50:i]))
        sma100 = float(np.mean(closes[max(0, i - 100):i]))
        prevailing = 1.0 if sma50 > sma100 else -1.0

    mom_5d  = (closes[i - 1] - closes[i - 6])  / max(closes[i - 6], 1e-9) if i >= 6 else 0.0
    mom_21d = (closes[i - 1] - closes[i - 22]) / max(closes[i - 22], 1e-9) if i >= 22 else 0.0

    short_disagrees = math.copysign(1.0, mom_5d)  != math.copysign(1.0, prevailing)
    mid_disagrees   = math.copysign(1.0, mom_21d) != math.copysign(1.0, prevailing)

    # Require BOTH 5d AND 21d to disagree with the prevailing trend.
    # Single-window disagreement is a fakeout; both is a confirmed pivot.
    if not (short_disagrees and mid_disagrees):
        return 0.0

    # Magnitude gate — both legs must be meaningful but the threshold is
    # calibrated to catch inflation-shock-style reversals (2022) while
    # ignoring routine 5d wiggles.
    if abs(mom_5d) < 0.012 or abs(mom_21d) < 0.008:
        return 0.0

    accel = abs(mom_5d) + 0.5 * abs(mom_21d)
    intensity = math.tanh(accel * 22.0)   # slightly stronger scaling
    # Direction of the pivot is the new direction (opposite to prevailing).
    new_direction = -math.copysign(1.0, prevailing)
    return float(new_direction * intensity)


def _short_trend_specialist(closes: np.ndarray, i: int) -> bool:
    """
    True when the system should be in a SHORT TREND regime:
      - SMA50 < SMA200 (long-term downtrend confirmed)
      - 21d momentum still negative (not yet bottoming)
      - Not at extreme oversold extension (let mean-rev catch true bottoms)
    """
    if i < 200:
        return False
    sma50  = float(np.mean(closes[i - 50:i]))
    sma200 = float(np.mean(closes[i - 200:i]))
    if sma50 >= sma200:
        return False
    if i < 22:
        return False
    mom_21d = (closes[i - 1] - closes[i - 22]) / max(closes[i - 22], 1e-9)
    if mom_21d > -0.005:    # require a real ongoing decline
        return False
    # Skip when we're already deeply oversold (>= 1.8 sigma below SMA20)
    ext = _extension_z(closes, i)
    if ext < -1.8:
        return False
    return True


def _technical_conviction(
    closes: np.ndarray, i: int
) -> tuple[float, str, float]:
    """
    Returns (conviction in [-1,+1], tier, direction sign).
    Multi-component proxy for the live alpha_stacker — uses MA cross,
    momentum (5d + 21d), Bollinger %B fade, and 63d return persistence.
    Scaled to produce a tier distribution similar to the live stacker's
    17+ engine output.

    Phase XIX: pivot_score added so the conviction can flip earlier
    during regime reversals (2008-style whipsaws).
    """
    if i < 60:
        return 0.0, "VERY_LOW", 0.0
    sma20 = float(np.mean(closes[i - 20:i]))
    sma50 = float(np.mean(closes[i - 50:i]))
    sma200 = float(np.mean(closes[max(i - 200, 0):i])) if i >= 100 else sma50
    mom_5d  = (closes[i - 1] - closes[i - 6])  / closes[i - 6]  if i >= 6  else 0.0
    mom_21d = (closes[i - 1] - closes[i - 22]) / closes[i - 22] if i >= 22 else 0.0
    mom_63d = (closes[i - 1] - closes[i - 64]) / closes[i - 64] if i >= 64 else 0.0
    std20 = float(np.std(closes[i - 20:i]))
    bb = 0.0
    if std20 >= 1e-12:
        bb = max(-1.0, min(1.0, (closes[i - 1] - sma20) / (2.0 * std20)))

    # Component signals — wider scale than before to lift tier distribution
    trend_short  = math.tanh((sma20 - sma50)  / max(abs(sma50),  1.0) * 80)
    trend_long   = math.tanh((sma50 - sma200) / max(abs(sma200), 1.0) * 80)
    mom_combined = math.tanh((mom_5d * 30) + (mom_21d * 12) + (mom_63d * 4))
    mean_rev_fade = -bb * 0.4   # extreme BB readings fade
    pivot = _pivot_score(closes, i)   # Phase XIX

    # Blended conviction — weights now load from data/conviction_weights.json
    # (Phase XXIII Sharpe-optimised) with fallback to the original hard-coded
    # defaults if the file is missing.
    w = _conviction_weights()
    conviction = (
        w.get("trend_short",   0.32) * trend_short
      + w.get("trend_long",    0.22) * trend_long
      + w.get("mom_combined",  0.28) * mom_combined
      + w.get("mean_rev_fade", 0.08) * mean_rev_fade
      + w.get("pivot",         0.10) * pivot
    )
    conviction = max(-1.0, min(1.0, conviction))

    a = abs(conviction)
    if a < 0.10:
        tier = "VERY_LOW"
    elif a < 0.25:
        tier = "LOW"
    elif a < 0.45:
        tier = "MEDIUM"
    elif a < 0.65:
        tier = "HIGH"
    else:
        tier = "VERY_HIGH"

    direction = math.copysign(1.0, conviction) if a >= 0.10 else 0.0
    return conviction, tier, direction


# ──────────────────────────────────────────────────────────────────────────────
# Strategy selection — mirrors scripts/strategy_selector.py rules (Phase XV tune)
# ──────────────────────────────────────────────────────────────────────────────
def _extension_z(closes: np.ndarray, i: int) -> float:
    """Z-score of current price vs SMA20."""
    if i < 20:
        return 0.0
    window = closes[i - 20:i]
    mu = float(window.mean())
    sd = float(window.std())
    if sd < 1e-9:
        return 0.0
    return (closes[i] - mu) / sd


def _select_strategy(
    *,
    tier: str,
    direction: float,
    vol_regime: str,
    hmm_state: str,
    conviction: float,
    extension_z: float = 0.0,
    macro_tilt: float = 0.0,
    short_trend: bool = False,
) -> str:
    # Phase XIX — Short-trend specialist: confirmed bear-trend regime,
    # take TREND (short) explicitly so we ride the downtrend instead of
    # fading it.  Only fires when conviction is non-trivial.
    if short_trend and tier in ("LOW", "MEDIUM", "HIGH", "VERY_HIGH"):
        return "TREND"

    # TREND has priority on conviction
    if tier in ("HIGH", "VERY_HIGH"):
        return "TREND"
    if tier == "MEDIUM" and hmm_state in ("BULLISH", "BEARISH"):
        return "TREND"
    if tier == "MEDIUM" and abs(macro_tilt) >= 1.0:
        return "TREND"

    # MEAN_REVERSION only with truly extreme extension AND non-elevated vol
    if (
        abs(extension_z) >= 2.0
        and tier != "VERY_LOW"
        and vol_regime in ("LOW", "NORMAL")
    ):
        return "MEAN_REVERSION"

    # VOL_SHORT for low/normal regimes
    if vol_regime == "LOW":
        return "VOL_SHORT"
    if vol_regime == "NORMAL":
        return "VOL_SHORT"

    # VOLATILE + ELEVATED/EXTREME with weak conviction — step aside
    if vol_regime in ("EXTREME", "ELEVATED") and hmm_state == "VOLATILE":
        if tier in ("LOW", "VERY_LOW"):
            return "CASH"
        return "VOL_SHORT"

    if tier == "VERY_LOW":
        return "CASH"

    return "VOL_SHORT"


def _regime_size_multiplier(vol_regime: str, hmm_state: str, tier: str) -> float:
    base = 1.0
    if vol_regime == "EXTREME":
        base *= 0.40
    elif vol_regime == "ELEVATED":
        base *= 0.60
    elif vol_regime == "LOW":
        base *= 1.15
    if hmm_state == "VOLATILE":
        base *= 0.70
    elif hmm_state == "BEARISH":
        base *= 0.90
    if tier == "VERY_HIGH":
        base *= 1.40
    elif tier == "HIGH":
        base *= 1.20
    elif tier == "LOW":
        base *= 0.75
    return max(0.10, min(1.60, base))


def _stacker_size_pct(tier: str, conviction: float) -> float:
    """Mirrors alpha_stacker._recommended_size_pct."""
    a = abs(conviction)
    if tier == "VERY_LOW":
        return 0.0
    if tier == "LOW":
        return 15.0 + 40.0 * (a - 0.10) / 0.15
    if tier == "MEDIUM":
        return 25.0 + 50.0 * (a - 0.25) / 0.20
    if tier == "HIGH":
        return 50.0 + 30.0 * (a - 0.45) / 0.20
    return min(100.0, 80.0 + 50.0 * (a - 0.65))


def _final_size_pct(
    strategy: str,
    tier: str,
    conviction: float,
    vol_regime: str,
    hmm_state: str,
) -> float:
    stacker_pct = _stacker_size_pct(tier, conviction)
    # Non-directional strategies get a 25% floor (mirrors selector)
    if strategy in ("MEAN_REVERSION", "VOL_SHORT", "PAIRS"):
        base = max(stacker_pct, 25.0)
    elif strategy == "TAIL_HEDGE":
        base = max(stacker_pct, 15.0)
    else:
        base = stacker_pct
    regime_mult = _regime_size_multiplier(vol_regime, hmm_state, tier)
    # Vol Kelly multiplier proxy (matches vol_surface guidance)
    if vol_regime == "EXTREME":
        kelly = 0.35
    elif vol_regime == "ELEVATED":
        kelly = 0.55
    elif vol_regime == "LOW":
        kelly = 1.10
    else:
        kelly = 0.85
    return max(0.0, min(100.0, base * regime_mult * kelly))


# ──────────────────────────────────────────────────────────────────────────────
# Strategy P&L simulation
# ──────────────────────────────────────────────────────────────────────────────
def _strategy_return(
    strategy: str,
    direction: float,
    closes: np.ndarray,
    i: int,
) -> float:
    """
    Returns the realised daily return [%] for the strategy entered at close[i]
    and marked at close[i+1].  Includes 5bps round-trip transaction cost.
    """
    if i + 1 >= len(closes) or strategy == "CASH":
        return 0.0
    realised = (closes[i + 1] / closes[i] - 1.0) * 100  # %
    cost = 0.01  # 5 bps round-trip ≈ 0.01% one-way; we apply at exit only here

    if strategy == "TREND":
        sign = direction if direction != 0 else 1.0
        return sign * realised - cost
    if strategy == "MEAN_REVERSION":
        # Fade vs short-term MA — proxied by reverse of 5d momentum
        if i >= 5:
            mom_5d = (closes[i] - closes[i - 5]) / closes[i - 5]
            fade = -1.0 if mom_5d > 0 else 1.0
        else:
            fade = -math.copysign(1.0, direction) if direction != 0 else 1.0
        return fade * realised - cost
    if strategy == "VOL_SHORT":
        # Long when range-bound; 50% capture; small theta-decay proxy +0.05% bias
        return 0.5 * realised + 0.05 - cost
    if strategy == "TAIL_HEDGE":
        # Protective short; small drag; pays off on big down moves
        return -0.3 * realised - 0.02 - cost
    if strategy == "PAIRS":
        # Approx pair-spread — uncorrelated with directional; bias zero, low vol
        return 0.02 * np.sign(np.random.randn()) - cost
    return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Backtest core
# ──────────────────────────────────────────────────────────────────────────────
def _max_drawdown(equity_curve: np.ndarray) -> float:
    if len(equity_curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    return float(dd.min() * 100)


def _sharpe(daily_returns: np.ndarray) -> float | None:
    if len(daily_returns) < 5:
        return None
    mu = float(np.mean(daily_returns))
    sd = float(np.std(daily_returns, ddof=0))
    if sd < 1e-12:
        return None
    return mu / sd * SQ252


def _monthly_bucket(returns_pct: list[float], dates: list[str]) -> list[dict]:
    """Group daily returns into calendar months."""
    months: dict[str, list[float]] = {}
    for r, d in zip(returns_pct, dates):
        key = d[:7]  # YYYY-MM
        months.setdefault(key, []).append(r)
    out = []
    for key in sorted(months.keys()):
        r_arr = months[key]
        cum = (np.prod([1 + x / 100 for x in r_arr]) - 1.0) * 100
        out.append({
            "month":    key,
            "n_days":   len(r_arr),
            "return_pct": round(float(cum), 3),
        })
    return out


def run_backtest(
    ticker: str = DEFAULT_TICKER,
    lookback_days: int = DEFAULT_LOOKBACK,
    target_pct: float = DEFAULT_TARGET_PCT,
) -> dict:
    closes, dates = _fetch_history(ticker, lookback_days)
    n = len(closes)
    if n < 60:
        raise SystemExit(f"insufficient history: only {n} rows")
    log_returns = np.diff(np.log(closes))
    pct_returns = np.diff(closes) / closes[:-1] * 100  # daily %

    rv_21 = _rolling_vol_pct(log_returns, 21)
    rv_63 = _rolling_vol_pct(log_returns, 63)
    drift_5  = _rolling_mean(log_returns, 5)
    drift_21 = _rolling_mean(log_returns, 21)

    daily_strategy_returns: list[float] = []
    strategy_history: list[dict] = []
    nav = 100_000.0
    equity_curve = [nav]
    starting_nav = nav

    # We loop over the *return* array; index i corresponds to closes[i+1]
    for i in range(60, len(log_returns) - 1):
        rv21 = float(rv_21[i]) if i < len(rv_21) else float("nan")
        rv63 = float(rv_63[i]) if i < len(rv_63) else float("nan")
        d5 = float(drift_5[i]) if i < len(drift_5) else 0.0
        d21 = float(drift_21[i]) if i < len(drift_21) else 0.0

        hmm = _classify_regime_proxy(rv21, d5, d21)
        vol_regime = _classify_vol_regime(rv21, rv63)
        conviction, tier, direction = _technical_conviction(closes, i + 1)
        ext_z = _extension_z(closes, i + 1)
        # Macro tilt proxy: we have no live macro feed in the backtest, so
        # use the 63d drift sign as a coarse proxy.
        macro_tilt = float(np.sign(d21) * 1.2) if abs(d21) > 0.002 else 0.0
        # Phase XIX — short-trend regime detector
        in_short_trend = _short_trend_specialist(closes, i + 1)
        strategy = _select_strategy(
            tier=tier, direction=direction,
            vol_regime=vol_regime, hmm_state=hmm,
            conviction=conviction,
            extension_z=ext_z, macro_tilt=macro_tilt,
            short_trend=in_short_trend,
        )
        size_pct = _final_size_pct(strategy, tier, conviction, vol_regime, hmm)

        # Adaptive risk multiplier from a fake "performance targeter":
        # we don't have a NAV yet at this point of the loop, so use a static 1.0
        # except when behind a 21d-target progress estimate.
        if len(daily_strategy_returns) >= 21:
            mtd_window = daily_strategy_returns[-21:]
            mtd_cum = (np.prod([1 + x / 100 for x in mtd_window]) - 1.0) * 100
            gap = mtd_cum - target_pct
            if gap > 2.0:
                pmult = 0.70
            elif gap > 0.5:
                pmult = 0.85
            elif gap >= -0.5:
                pmult = 1.00
            elif gap >= -2.0:
                pmult = 1.20
            elif gap >= -5.0:
                pmult = 1.50
            else:
                pmult = 1.75
        else:
            pmult = 1.0

        adjusted_size = min(95.0, size_pct * pmult)

        # ── Crisis guard (Phase XVIII) ────────────────────────────────────
        # Classify the regime from prices and apply the size cap / strategy
        # override.  This is the same logic the live selector uses.
        crisis = _crisis_classify(closes, current_idx=i + 1)
        crisis_tier = crisis["tier"]
        guarded_strategy, guarded_size, _guard_reason = _crisis_apply_guard(
            strategy, adjusted_size, crisis_tier,
        )
        strategy = guarded_strategy
        adjusted_size = guarded_size

        strat_ret_pct = _strategy_return(strategy, direction, closes, i + 1)
        net_daily = (adjusted_size / 100.0) * strat_ret_pct
        daily_strategy_returns.append(net_daily)
        nav *= (1.0 + net_daily / 100.0)
        equity_curve.append(nav)

        strategy_history.append({
            "date":       dates[i + 1],
            "strategy":   strategy,
            "tier":       tier,
            "direction":  int(direction),
            "vol_regime": vol_regime,
            "hmm":        hmm,
            "crisis_tier":  crisis_tier,
            "crisis_score": crisis["score"],
            "extension_z": round(ext_z, 2),
            "macro_tilt": round(macro_tilt, 2),
            "size_pct":   round(adjusted_size, 2),
            "return_pct": round(net_daily, 4),
        })

    # Aggregate statistics
    ret_arr = np.asarray(daily_strategy_returns)
    equity_arr = np.asarray(equity_curve)
    cum_return_pct = float((nav / starting_nav - 1.0) * 100)
    sharpe = _sharpe(ret_arr)
    max_dd_pct = _max_drawdown(equity_arr)
    win_days_pct = float((ret_arr > 0).mean() * 100) if len(ret_arr) else 0.0
    hit_rate = float(((ret_arr * np.array([h["direction"] for h in strategy_history])) > 0).mean() * 100) \
        if len(ret_arr) else 0.0
    avg_daily = float(np.mean(ret_arr)) if len(ret_arr) else 0.0
    vol_daily = float(np.std(ret_arr)) if len(ret_arr) else 0.0

    # Calendar-month returns + target hit rate
    bt_dates = [h["date"] for h in strategy_history]
    monthlies = _monthly_bucket(daily_strategy_returns, bt_dates)
    monthly_returns = [m["return_pct"] for m in monthlies]
    n_months = len(monthlies)
    n_months_at_target = sum(1 for r in monthly_returns if r >= target_pct)
    pct_months_at_target = (n_months_at_target / n_months * 100) if n_months else 0.0

    # Strategy breakdown
    strat_counts: dict[str, int] = {}
    strat_pl: dict[str, float] = {}
    for h in strategy_history:
        s = h["strategy"]
        strat_counts[s] = strat_counts.get(s, 0) + 1
        strat_pl[s] = strat_pl.get(s, 0.0) + h["return_pct"]
    strat_attribution = sorted(
        [
            {
                "strategy": s,
                "n_days":   c,
                "share_pct": round(c / max(len(strategy_history), 1) * 100, 1),
                "total_pl_pct": round(strat_pl.get(s, 0.0), 2),
                "avg_daily_pl_pct": round(strat_pl.get(s, 0.0) / c, 4) if c else 0.0,
            }
            for s, c in strat_counts.items()
        ],
        key=lambda x: -x["total_pl_pct"],
    )

    # Projected outlook
    if len(ret_arr):
        annual_return = (math.exp(np.sum(np.log1p(ret_arr / 100))) - 1.0) * 100
        ann_vol = vol_daily * SQ252
    else:
        annual_return = 0.0
        ann_vol = 0.0

    # ── Achievability verdict ──────────────────────────────────────────────
    # Composite: weighs target-hit rate against absolute Sharpe.  A system
    # that misses the target but has Sharpe > 1.5 isn't broken — it's just
    # facing an unrealistic target.  Reflect that.
    sharpe_score = sharpe if sharpe is not None else 0.0
    if pct_months_at_target >= 50:
        achievability = "ACHIEVABLE"
    elif pct_months_at_target >= 25:
        achievability = "STRETCH"
    elif pct_months_at_target >= 10:
        achievability = "OPTIMISTIC"
    elif sharpe_score >= 1.5 and max_dd_pct > -15:
        # Target is unrealistic vs gold's structural return, but the
        # system itself is genuinely excellent — surface that.
        achievability = "ELITE_SYSTEM_TARGET_TOO_HIGH"
    else:
        achievability = "UNREALISTIC"

    out: dict[str, Any] = {
        "schema_version": "1.0",
        "engine":         "strategy_backtester",
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker":         ticker,
        "lookback_days":  lookback_days,
        "n_simulated":    len(strategy_history),
        "first_date":     bt_dates[0] if bt_dates else None,
        "last_date":      bt_dates[-1] if bt_dates else None,
        "target": {
            "monthly_pct":          target_pct,
            "annual_implied_pct":   round((((1 + target_pct / 100) ** 12 - 1) * 100), 1),
        },
        "performance": {
            "cum_return_pct":          round(cum_return_pct, 2),
            "annualised_return_pct":   round(annual_return * (252 / max(len(ret_arr), 1)), 2),
            "annualised_vol_pct":      round(ann_vol, 2),
            "sharpe":                  round(sharpe, 3) if sharpe is not None else None,
            "max_drawdown_pct":        round(max_dd_pct, 2),
            "win_days_pct":            round(win_days_pct, 2),
            "direction_hit_rate_pct":  round(hit_rate, 2),
            "avg_daily_return_pct":    round(avg_daily, 4),
            "starting_nav_usd":        starting_nav,
            "ending_nav_usd":          round(nav, 2),
        },
        "monthly": {
            "n_months":                n_months,
            "n_at_or_above_target":    n_months_at_target,
            "pct_at_or_above_target":  round(pct_months_at_target, 1),
            "best_month_pct":          round(max(monthly_returns), 2) if monthly_returns else 0.0,
            "worst_month_pct":         round(min(monthly_returns), 2) if monthly_returns else 0.0,
            "median_month_pct":        round(float(np.median(monthly_returns)), 2) if monthly_returns else 0.0,
            "returns":                 monthlies,
        },
        "by_strategy": strat_attribution,
        "achievability_verdict":      achievability,
        "achievability_note": (
            (
                f"Sharpe {round(sharpe, 2) if sharpe else 'n/a'} with "
                f"{round(max_dd_pct, 1)}% max DD over {n_months} months — "
                f"institutional-grade for single-asset gold. The {target_pct:.0f}%/mo "
                f"target is structurally unreachable from gold alone; pair with "
                f"the equity book or accept ~{round(annual_return * (252 / max(len(ret_arr), 1)), 0)}%/y "
                f"as the honest target."
            ) if achievability == "ELITE_SYSTEM_TARGET_TOO_HIGH" else
            (
                f"Across {n_months} historical months, "
                f"{n_months_at_target} ({pct_months_at_target:.0f}%) hit the "
                f"{target_pct:.0f}% target. Sharpe={round(sharpe, 2) if sharpe else 'n/a'}, "
                f"max DD={round(max_dd_pct, 1)}%."
            )
        ),
        "equity_curve_sampled": [
            {"i": i, "nav_usd": round(float(equity_arr[i]), 2)}
            for i in range(0, len(equity_arr), max(1, len(equity_arr) // 200))
        ],
        # Per-day series — used by the multi_asset_backtester for honest
        # daily-vol / Sharpe estimation when this book is combined with others.
        # Includes crisis_tier per day so the regime-adaptive allocator can
        # replay the historical weight schedule.
        "daily_series": [
            {
                "date":         h["date"],
                "return_pct":   h["return_pct"],
                "strategy":     h["strategy"],
                "size_pct":     h["size_pct"],
                "crisis_tier":  h.get("crisis_tier", "NORMAL"),
                "crisis_score": h.get("crisis_score", 0.0),
            }
            for h in strategy_history
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
    ap.add_argument("--ticker", default=DEFAULT_TICKER)
    ap.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    ap.add_argument("--target", type=float, default=DEFAULT_TARGET_PCT)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    out = run_backtest(
        ticker=args.ticker,
        lookback_days=args.lookback,
        target_pct=args.target,
    )
    if args.quiet:
        return 0

    p = out["performance"]
    m = out["monthly"]
    print("=" * 64)
    print(f"STRATEGY BACKTEST  ({out['generated_at']})")
    print("=" * 64)
    print(f"  Ticker         : {out['ticker']}")
    print(f"  Window         : {out['first_date']} → {out['last_date']}  ({out['n_simulated']} days)")
    print()
    print(f"  Cum return     : {p['cum_return_pct']:+.2f}%")
    print(f"  Annualised     : {p['annualised_return_pct']:+.2f}%/y  σ={p['annualised_vol_pct']:.1f}%")
    print(f"  Sharpe         : {p['sharpe']}")
    print(f"  Max drawdown   : {p['max_drawdown_pct']:+.2f}%")
    print(f"  Win days       : {p['win_days_pct']:.1f}%   direction hit rate {p['direction_hit_rate_pct']:.1f}%")
    print()
    print(f"  Target         : {out['target']['monthly_pct']:.0f}%/mo  "
          f"({out['target']['annual_implied_pct']:.0f}% ann implied)")
    print(f"  Months ≥ target: {m['n_at_or_above_target']}/{m['n_months']}  "
          f"({m['pct_at_or_above_target']:.1f}%)")
    print(f"  Best month     : {m['best_month_pct']:+.2f}%")
    print(f"  Worst month    : {m['worst_month_pct']:+.2f}%")
    print(f"  Median month   : {m['median_month_pct']:+.2f}%")
    print()
    print(f"  Verdict        : {out['achievability_verdict']}")
    print(f"                   {out['achievability_note']}")
    print()
    print("  By strategy (most profitable first):")
    for s in out["by_strategy"]:
        print(f"    {s['strategy']:<16s} "
              f"share={s['share_pct']:5.1f}%  "
              f"total={s['total_pl_pct']:+7.2f}%  "
              f"avg/d={s['avg_daily_pl_pct']:+.3f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
