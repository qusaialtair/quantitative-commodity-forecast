#!/usr/bin/env python3
"""
Cross-Asset Correlation Monitor
================================
Tracks rolling correlations between gold and macro assets.
Detects regime shifts via correlation breakdowns.

Usage:
    python3 scripts/correlation_monitor.py
    python3 scripts/correlation_monitor.py --window 21
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

log = logging.getLogger("correlation_monitor")

TICKERS = [
    "GC=F",      # Gold futures
    "SI=F",      # Silver futures
    "DX-Y.NYB",  # Dollar Index
    "^VIX",      # VIX
    "^TNX",      # 10Y Treasury Yield
    "SPY",       # S&P 500
    "TLT",       # 20Y+ Treasury Bond ETF
    "GLD",       # Gold ETF (volume/flow proxy)
]

GOLD_TICKER = "GC=F"

DISPLAY_NAMES = {
    "SI=F":      "Silver (SI=F)",
    "DX-Y.NYB":  "Dollar (DXY)",
    "^VIX":      "VIX",
    "^TNX":      "10Y Yield (TNX)",
    "SPY":       "S&P 500 (SPY)",
    "TLT":       "Treasuries (TLT)",
    "GLD":       "Gold ETF (GLD)",
}

JSON_KEYS = {
    "SI=F":      "SI=F",
    "DX-Y.NYB":  "DXY",
    "^VIX":      "VIX",
    "^TNX":      "TNX",
    "SPY":       "SPY",
    "TLT":       "TLT",
    "GLD":       "GLD",
}

TYPICAL_RANGES: Dict[str, Tuple[float, float]] = {
    "SI=F":      (+0.70, +0.90),
    "DX-Y.NYB":  (-0.70, -0.40),
    "^VIX":      (+0.10, +0.30),
    "^TNX":      (-0.50, -0.30),
    "SPY":       (-0.20, +0.10),
    "TLT":       (+0.20, +0.50),
    "GLD":       (+0.85, +0.99),
}

ANOMALY_THRESHOLD = 0.3


def _fetch_prices(period: str = "1y") -> Optional[pd.DataFrame]:
    """Fetch adjusted close prices for all tracked assets."""
    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance not installed — run: pip install yfinance")
        return None

    try:
        raw = yf.download(TICKERS, period=period, auto_adjust=True, progress=False)
    except Exception as exc:
        log.error("yfinance download failed: %s", exc)
        return None

    if raw is None or raw.empty:
        log.error("yfinance returned empty dataframe")
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw

    prices = prices.dropna(how="all")
    if prices.empty:
        return None

    return prices


def fetch_correlation_matrix(
    window: int = 21,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Return (current_corr_21d, historical_corr_63d).
    Both are full NxN correlation matrices of daily log returns.
    """
    prices = _fetch_prices(period="1y")
    if prices is None:
        return None, None

    log_returns = np.log(prices / prices.shift(1)).dropna()

    if len(log_returns) < 63:
        log.warning("Insufficient data: only %d rows", len(log_returns))
        return None, None

    rolling_short = log_returns.rolling(window=window)
    corr_short = rolling_short.corr()

    rolling_long = log_returns.rolling(window=63)
    corr_long = rolling_long.corr()

    last_date = log_returns.index[-1]

    try:
        current_corr = corr_short.loc[last_date]
        historical_corr = corr_long.loc[last_date]
    except KeyError:
        log.error("Could not extract correlation matrices for last date")
        return None, None

    return current_corr, historical_corr


def detect_anomalies(
    current_corr: pd.DataFrame,
    historical_corr: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Flag gold correlations that have shifted by more than ANOMALY_THRESHOLD
    from their 63-day average. These are structural breaks.
    """
    anomalies: List[Dict[str, Any]] = []

    if GOLD_TICKER not in current_corr.index or GOLD_TICKER not in historical_corr.index:
        return anomalies

    for ticker in TICKERS:
        if ticker == GOLD_TICKER:
            continue
        if ticker not in current_corr.columns or ticker not in historical_corr.columns:
            continue

        try:
            curr_val = current_corr.loc[GOLD_TICKER, ticker]
            hist_val = historical_corr.loc[GOLD_TICKER, ticker]
        except KeyError:
            continue

        if pd.isna(curr_val) or pd.isna(hist_val):
            continue

        shift = curr_val - hist_val
        if abs(shift) >= ANOMALY_THRESHOLD:
            anomalies.append({
                "asset": ticker,
                "display_name": DISPLAY_NAMES.get(ticker, ticker),
                "current_corr": round(float(curr_val), 4),
                "historical_corr": round(float(hist_val), 4),
                "shift": round(float(shift), 4),
            })

    return anomalies


def compute_beta(
    gold_returns: pd.Series,
    spx_returns: pd.Series,
    window: int = 63,
) -> Optional[float]:
    """
    Rolling gold beta to SPX over the last `window` days.
    Beta = Cov(gold, spx) / Var(spx).
    """
    aligned = pd.concat([gold_returns, spx_returns], axis=1).dropna()
    if len(aligned) < window:
        return None

    tail = aligned.iloc[-window:]
    gold_col = tail.iloc[:, 0]
    spx_col = tail.iloc[:, 1]

    var_spx = spx_col.var()
    if var_spx == 0 or pd.isna(var_spx):
        return None

    cov = gold_col.cov(spx_col)
    beta = cov / var_spx
    return round(float(beta), 4)


def gold_silver_ratio(prices: Optional[pd.DataFrame] = None) -> Tuple[Optional[float], Optional[float]]:
    """
    Current gold/silver ratio and its z-score (1yr lookback).
    Returns (ratio, zscore) or (None, None) on failure.
    """
    if prices is None:
        prices = _fetch_prices(period="1y")
    if prices is None:
        return None, None

    if GOLD_TICKER not in prices.columns or "SI=F" not in prices.columns:
        return None, None

    gold = prices[GOLD_TICKER].dropna()
    silver = prices["SI=F"].dropna()

    aligned = pd.concat([gold, silver], axis=1).dropna()
    if aligned.empty:
        return None, None

    ratio_series = aligned.iloc[:, 0] / aligned.iloc[:, 1]
    ratio_series = ratio_series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(ratio_series) < 20:
        return None, None

    current_ratio = float(ratio_series.iloc[-1])
    mean = float(ratio_series.mean())
    std = float(ratio_series.std())

    if std == 0:
        return round(current_ratio, 2), 0.0

    zscore = (current_ratio - mean) / std
    return round(current_ratio, 2), round(zscore, 2)


def _classify_correlation(ticker: str, value: float) -> str:
    """Classify a correlation value as NORMAL or flag it relative to typical range."""
    lo, hi = TYPICAL_RANGES.get(ticker, (-1.0, 1.0))
    if lo <= value <= hi:
        return f"NORMAL -- typically {lo:+.1f} to {hi:+.1f}"
    if value < lo:
        return f"LOW -- typically {lo:+.1f} to {hi:+.1f}"
    return f"HIGH -- typically {lo:+.1f} to {hi:+.1f}"


def _classify_beta(beta: float) -> str:
    if beta < -0.15:
        return "STRONG SAFE-HAVEN"
    if beta < 0.05:
        return "NORMAL"
    if beta < 0.20:
        return "ELEVATED -- gold acting risk-on"
    return "DANGER -- gold correlated with equities"


def generate_report(window: int = 21) -> Dict[str, Any]:
    """
    Main entry point. Fetches all data, computes correlations, detects anomalies,
    prints a terminal report, and saves JSON to data/correlation_report.json.
    """
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    result: Dict[str, Any] = {
        "timestamp": now.isoformat(),
        "correlations_21d": {},
        "correlations_63d": {},
        "anomalies": [],
        "gold_silver_ratio": None,
        "gold_silver_ratio_zscore": None,
        "gold_beta_spx": None,
        "regime_signal": "UNKNOWN",
    }

    prices = _fetch_prices(period="1y")
    if prices is None:
        _print_error_report(date_str)
        _save_json(result)
        return result

    log_returns = np.log(prices / prices.shift(1)).dropna()

    current_corr, historical_corr = _compute_corr_from_returns(log_returns, window)
    if current_corr is None or historical_corr is None:
        _print_error_report(date_str)
        _save_json(result)
        return result

    corr_21d: Dict[str, float] = {}
    corr_63d: Dict[str, float] = {}

    for ticker in TICKERS:
        if ticker == GOLD_TICKER:
            continue
        json_key = JSON_KEYS.get(ticker, ticker)
        try:
            val_short = float(current_corr.loc[GOLD_TICKER, ticker])
            val_long = float(historical_corr.loc[GOLD_TICKER, ticker])
            if not pd.isna(val_short):
                corr_21d[json_key] = round(val_short, 4)
            if not pd.isna(val_long):
                corr_63d[json_key] = round(val_long, 4)
        except (KeyError, TypeError):
            continue

    result["correlations_21d"] = corr_21d
    result["correlations_63d"] = corr_63d

    anomalies = detect_anomalies(current_corr, historical_corr)
    result["anomalies"] = anomalies

    ratio, zscore = gold_silver_ratio(prices)
    result["gold_silver_ratio"] = ratio
    result["gold_silver_ratio_zscore"] = zscore

    beta = None
    if GOLD_TICKER in log_returns.columns and "SPY" in log_returns.columns:
        beta = compute_beta(log_returns[GOLD_TICKER], log_returns["SPY"], window=63)
    result["gold_beta_spx"] = beta

    if len(anomalies) > 0:
        result["regime_signal"] = "STRUCTURAL_BREAK"
    else:
        result["regime_signal"] = "NORMAL"

    _print_report(date_str, window, corr_21d, anomalies, ratio, zscore, beta)
    _save_json(result)

    return result


def _compute_corr_from_returns(
    log_returns: pd.DataFrame, window: int
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Compute rolling correlation matrices from pre-computed log returns."""
    if len(log_returns) < 63:
        return None, None

    rolling_short = log_returns.rolling(window=window)
    corr_short = rolling_short.corr()

    rolling_long = log_returns.rolling(window=63)
    corr_long = rolling_long.corr()

    last_date = log_returns.index[-1]

    try:
        current_corr = corr_short.loc[last_date]
        historical_corr = corr_long.loc[last_date]
    except KeyError:
        return None, None

    return current_corr, historical_corr


def _print_report(
    date_str: str,
    window: int,
    corr_21d: Dict[str, float],
    anomalies: List[Dict[str, Any]],
    ratio: Optional[float],
    zscore: Optional[float],
    beta: Optional[float],
) -> None:
    """Print formatted terminal report."""
    bar = "\u2501" * 60
    thin = "\u2500" * 34

    lines = [
        "",
        f"  {bar}",
        f"    CROSS-ASSET CORRELATION MONITOR -- {date_str}",
        f"  {bar}",
        "",
        f"    Gold Correlations ({window}-day rolling)",
        f"    {thin}",
    ]

    ticker_to_json = {v: k for k, v in JSON_KEYS.items()}
    ordered_tickers = ["SI=F", "DX-Y.NYB", "^VIX", "^TNX", "SPY", "TLT", "GLD"]

    for ticker in ordered_tickers:
        json_key = JSON_KEYS.get(ticker, ticker)
        display = DISPLAY_NAMES.get(ticker, ticker)
        val = corr_21d.get(json_key)
        if val is None:
            lines.append(f"    {display:<22s}:    N/A")
            continue
        classification = _classify_correlation(ticker, val)
        lines.append(f"    {display:<22s}:  {val:+.2f}  [{classification}]")

    lines.append("")

    n_anomalies = len(anomalies)
    if n_anomalies == 0:
        lines.append(f"    ANOMALIES DETECTED: 0")
    else:
        lines.append(f"    ANOMALIES DETECTED: {n_anomalies}")
        for a in anomalies:
            lines.append(
                f"      >> {a['display_name']}: shifted {a['shift']:+.2f} "
                f"(now {a['current_corr']:+.2f}, was {a['historical_corr']:+.2f})"
            )

    lines.append("")

    if ratio is not None and zscore is not None:
        lines.append(f"    Gold/Silver Ratio:  {ratio:.1f}  (z-score: {zscore:+.1f}s)")
    else:
        lines.append(f"    Gold/Silver Ratio:  N/A")

    if beta is not None:
        beta_class = _classify_beta(beta)
        lines.append(f"    Gold Beta to SPX:   {beta:+.2f}  [{beta_class}]")
    else:
        lines.append(f"    Gold Beta to SPX:   N/A")

    lines.append("")
    lines.append(f"  {bar}")
    lines.append("")

    print("\n".join(lines))


def _print_error_report(date_str: str) -> None:
    bar = "\u2501" * 60
    print(f"\n  {bar}")
    print(f"    CROSS-ASSET CORRELATION MONITOR -- {date_str}")
    print(f"  {bar}")
    print(f"    ERROR: Could not fetch market data. Check network connection.")
    print(f"  {bar}\n")


def _save_json(result: Dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "correlation_report.json"
    try:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        log.info("Saved report to %s", out_path)
    except OSError as exc:
        log.error("Failed to save report: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-Asset Correlation Monitor for Gold"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=21,
        help="Rolling correlation window in trading days (default: 21)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    generate_report(window=args.window)


if __name__ == "__main__":
    main()
