#!/usr/bin/env python3
"""
Alpha Attribution Engine
=========================
Decomposes strategy return potential into five independent alpha sources,
each driven by a non-overlapping signal generator applied to gold returns.
For every source, the engine produces a daily return stream by lagging the
signal one day (no lookahead) and multiplying by the next day's return.

Sources (all use only past information):
  1. lstm_momentum   sign of trailing 5d return (LSTM directional proxy)
  2. macro_overlay   -sign of trailing 5d DXY return (Perplexity macro proxy)
  3. regime_filter   vol-contraction gate (long when vol_21d < vol_63d)
  4. technical       SMA 20 / SMA 50 trend cross
  5. mean_reversion  Bollinger %B fade (oversold long, overbought short)

For each source the engine reports:
  - full-history Sharpe, ann return / vol, max drawdown, win rate
  - rolling windows: 21d / 63d / 252d
  - information ratio vs equal-weighted blend (active return / TE)
  - cumulative return

For the combined equal-weighted blend it reports:
  - Sharpe, return, vol
  - diversification ratio = sum-of-individual-vols / portfolio-vol
  - full correlation and covariance matrices (annualised)
  - source ranking by Sharpe and by IR

Output: data/alpha_attribution.json
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
OUTPUT_FILE = DATA_DIR / "alpha_attribution.json"
ALT_CSV = DATA_DIR / "alt_data.csv"

DEFAULT_TICKER = "GC=F"
DEFAULT_LOOKBACK = "5y"

WINDOWS = [21, 63, 252]
SQ252 = float(np.sqrt(252))

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _download_close(ticker: str, lookback: str) -> pd.Series:
    """Download a single ticker's adjusted close as a Series."""
    raw = yf.download(
        ticker, period=lookback, interval="1d",
        progress=False, auto_adjust=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    return raw["Close"].dropna()


def _fetch_panel(ticker: str, lookback: str) -> pd.DataFrame:
    """Fetch the price panel needed for signal generation."""
    if yf is None:
        raise ImportError("yfinance is required for alpha_attribution")
    gold = _download_close(ticker, lookback)
    dxy = _download_close("DX-Y.NYB", lookback)
    df = pd.DataFrame({"gold": gold, "dxy": dxy}).dropna()
    return df


# ---------------------------------------------------------------------------
# Signal generators (all causal, lag-1 applied later)
# ---------------------------------------------------------------------------
def _generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    g = df["gold"]
    r = g.pct_change()

    sigs = pd.DataFrame(index=df.index)

    # 1. LSTM momentum proxy
    r5 = g.pct_change(5)
    sigs["lstm_momentum"] = np.sign(r5).fillna(0)

    # 2. Macro overlay: invert DXY momentum
    dxy_r5 = df["dxy"].pct_change(5)
    sigs["macro_overlay"] = -np.sign(dxy_r5).fillna(0)

    # 3. Regime filter: vol contraction = long, expansion = short
    vol21 = r.rolling(21).std()
    vol63 = r.rolling(63).std()
    rf = np.where(vol21.notna() & vol63.notna(),
                  np.where(vol21 < vol63, 1.0, -1.0), 0.0)
    sigs["regime_filter"] = pd.Series(rf, index=df.index).astype(float)

    # 4. Technical: SMA 20 / 50 cross
    sma20 = g.rolling(20).mean()
    sma50 = g.rolling(50).mean()
    tech = np.where(sma20.notna() & sma50.notna(),
                    np.where(sma20 > sma50, 1.0, -1.0), 0.0)
    sigs["technical"] = pd.Series(tech, index=df.index).astype(float)

    # 5. Mean-reversion: Bollinger %B fade
    bb_sma = g.rolling(20).mean()
    bb_std = g.rolling(20).std()
    upper = bb_sma + 2 * bb_std
    lower = bb_sma - 2 * bb_std
    width = (upper - lower).replace(0, np.nan)
    pct_b = (g - lower) / width
    mr = np.where(pct_b < 0.2, 1.0, np.where(pct_b > 0.8, -1.0, 0.0))
    sigs["mean_reversion"] = pd.Series(mr, index=df.index).astype(float)

    return sigs.fillna(0.0)


def _compute_source_returns(df: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    r = df["gold"].pct_change().fillna(0)
    lagged = signals.shift(1).fillna(0)
    return lagged.mul(r, axis=0)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def _summarize(returns: pd.Series, name: str) -> dict:
    r = returns.dropna()
    if len(r) < 10:
        return {
            "name": name, "n_obs": int(len(r)),
            "ann_return_pct": 0.0, "ann_vol_pct": 0.0,
            "sharpe": 0.0, "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0, "cumulative_return_pct": 0.0,
        }
    ann_ret = float(r.mean() * 252)
    ann_vol = float(r.std() * SQ252)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
    cum = (1 + r).cumprod()
    rolling_max = cum.cummax()
    dd = float((cum / rolling_max - 1).min())
    win_rate = float((r > 0).mean())
    cumret = float(cum.iloc[-1] - 1)

    return {
        "name": name,
        "n_obs": int(len(r)),
        "ann_return_pct": round(ann_ret * 100, 3),
        "ann_vol_pct": round(ann_vol * 100, 3),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(dd * 100, 3),
        "win_rate_pct": round(win_rate * 100, 2),
        "cumulative_return_pct": round(cumret * 100, 3),
    }


def _window_summary(src_returns: pd.DataFrame, windows: list[int]) -> dict:
    out = {}
    for w in windows:
        recent = src_returns.tail(w)
        out[f"{w}d"] = {
            col: _summarize(recent[col], col) for col in src_returns.columns
        }
    return out


def _info_ratios(src_returns: pd.DataFrame) -> dict:
    eq = src_returns.mean(axis=1)
    out = {}
    for col in src_returns.columns:
        active = src_returns[col] - eq
        active_mean = float(active.mean() * 252)
        te = float(active.std() * SQ252)
        ir = active_mean / te if te > 1e-12 else 0.0
        out[col] = {
            "active_return_pct": round(active_mean * 100, 3),
            "tracking_error_pct": round(te * 100, 3),
            "information_ratio": round(ir, 3),
        }
    return out


def _combined_metrics(src_returns: pd.DataFrame) -> dict:
    eq = src_returns.mean(axis=1)
    base = _summarize(eq, "equal_weight_combined")

    individual_vols = src_returns.std()
    n_sources = max(len(individual_vols), 1)
    weighted_avg_vol_ann = float(individual_vols.mean() * SQ252)  # eq-weight
    sum_vols_ann = float(individual_vols.sum() * SQ252)
    portfolio_vol_ann = float(eq.std() * SQ252)
    # Choueifaty-Coignard diversification ratio: (w'σ)/σ_P with w_i = 1/N
    div_ratio = (
        weighted_avg_vol_ann / portfolio_vol_ann
        if portfolio_vol_ann > 1e-12 else 0.0
    )

    corr = src_returns.corr().round(3).fillna(0).to_dict()
    cov_ann = (src_returns.cov() * 252).round(6).fillna(0).to_dict()

    return {
        "equal_weight_summary": base,
        "diversification_ratio": round(div_ratio, 3),
        "weighted_avg_vol_pct": round(weighted_avg_vol_ann * 100, 3),
        "sum_of_individual_vols_pct": round(sum_vols_ann * 100, 3),
        "portfolio_vol_pct": round(portfolio_vol_ann * 100, 3),
        "n_sources": int(n_sources),
        "correlation_matrix": corr,
        "covariance_matrix_annualised": cov_ann,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_alpha_attribution(
    ticker: str = DEFAULT_TICKER,
    lookback: str = DEFAULT_LOOKBACK,
) -> dict:
    df = _fetch_panel(ticker, lookback)
    signals = _generate_signals(df)
    src_returns = _compute_source_returns(df, signals)

    full_summary = {
        col: _summarize(src_returns[col], col) for col in src_returns.columns
    }
    window_summary = _window_summary(src_returns, WINDOWS)
    info_ratios = _info_ratios(src_returns)
    combined = _combined_metrics(src_returns)

    ranked_sharpe = sorted(
        full_summary.values(), key=lambda x: x["sharpe"], reverse=True
    )
    ranked_ir = sorted(
        info_ratios.items(),
        key=lambda kv: kv[1]["information_ratio"],
        reverse=True,
    )

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker": ticker,
        "lookback": lookback,
        "n_obs": int(len(src_returns)),
        "sources": list(src_returns.columns),
        "full_history": full_summary,
        "rolling_windows": window_summary,
        "information_ratios": info_ratios,
        "combined": combined,
        "ranked_by_sharpe": [s["name"] for s in ranked_sharpe],
        "ranked_by_information_ratio": [k for k, _ in ranked_ir],
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
    print(f"  ALPHA ATTRIBUTION ENGINE -- {r['ticker']}")
    print(SEP)
    print(f"  Sources:        {len(r['sources'])}")
    print(f"  Observations:   {r['n_obs']}")
    print(f"  Lookback:       {r['lookback']}")
    print()

    print(f"  FULL-HISTORY METRICS BY SOURCE")
    print(f"  {'─' * 58}")
    for s in r["full_history"].values():
        marker = " >>" if s["sharpe"] >= 0.3 else "   "
        print(
            f"  {marker} {s['name']:<16s}  "
            f"Sharpe={s['sharpe']:+6.3f}  "
            f"Ret={s['ann_return_pct']:+7.2f}%  "
            f"Vol={s['ann_vol_pct']:5.2f}%  "
            f"DD={s['max_drawdown_pct']:6.1f}%"
        )
    print()

    print(f"  EQUAL-WEIGHTED BLEND")
    print(f"  {'─' * 58}")
    es = r["combined"]["equal_weight_summary"]
    c = r["combined"]
    print(f"  Sharpe:                {es['sharpe']:+.3f}")
    print(f"  Ann Return:            {es['ann_return_pct']:+.3f}%")
    print(f"  Ann Vol:               {es['ann_vol_pct']:.3f}%")
    print(f"  Max Drawdown:          {es['max_drawdown_pct']:.2f}%")
    print(f"  Diversification Ratio: {c['diversification_ratio']:.3f}x")
    print(f"  Avg Source Vol:        {c['weighted_avg_vol_pct']:.2f}%")
    print(f"  Portfolio Vol:         {c['portfolio_vol_pct']:.2f}%")
    print()

    print(f"  INFORMATION RATIO (vs equal-weight blend)")
    print(f"  {'─' * 58}")
    for src in r["sources"]:
        ir = r["information_ratios"][src]
        print(
            f"  {src:<16s}  IR={ir['information_ratio']:+6.3f}  "
            f"Active={ir['active_return_pct']:+7.2f}%  "
            f"TE={ir['tracking_error_pct']:5.2f}%"
        )
    print()

    print(f"  RANKED BY SHARPE: {', '.join(r['ranked_by_sharpe'])}")
    print(f"  RANKED BY IR:     {', '.join(r['ranked_by_information_ratio'])}")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpha Attribution Engine")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--lookback", default=DEFAULT_LOOKBACK)
    args = parser.parse_args()
    run_alpha_attribution(ticker=args.ticker, lookback=args.lookback)
