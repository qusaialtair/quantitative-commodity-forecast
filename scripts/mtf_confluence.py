#!/usr/bin/env python3
"""
Multi-Timeframe Confluence Scorer
====================================
Scores alignment across daily, weekly, and monthly timeframes.
When all three agree, conviction is highest. Divergence signals caution.

Signals per timeframe:
  - Trend direction (price vs EMA)
  - Momentum (RSI zone)
  - Mean reversion (Bollinger %B)

Confluence scoring:
  - 3/3 aligned = STRONG (score ±90-100)
  - 2/3 aligned = MODERATE (score ±50-70)
  - 1/3 or mixed = WEAK (score ±10-30)

Usage:
    python3 scripts/mtf_confluence.py
    python3 scripts/mtf_confluence.py --ticker GC=F
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
OUTPUT_FILE = DATA_DIR / "mtf_confluence.json"

LINE_W = 62
SEP = "\u2501" * LINE_W


def _fetch_multi_tf(ticker: str) -> dict[str, pd.DataFrame]:
    """Fetch daily, weekly, and monthly data."""
    if yf is None:
        raise ImportError("yfinance required")

    frames = {}
    for interval, period in [("1d", "2y"), ("1wk", "5y"), ("1mo", "10y")]:
        raw = yf.download(ticker, period=period, interval=interval,
                          progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        if not raw.empty:
            frames[interval] = raw
    return frames


def _compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, min_periods=span, adjust=False).mean()


def _compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta.clip(upper=0.0))
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return float(rsi.iloc[-1]) if not rsi.empty else 50.0


def _compute_bollinger_pctb(series: pd.Series, period: int = 20) -> float:
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    pctb = (series - lower) / (upper - lower)
    val = float(pctb.iloc[-1]) if not pctb.empty else 0.5
    return max(0.0, min(1.0, val))


def _score_timeframe(df: pd.DataFrame, label: str) -> dict:
    """Score a single timeframe. Returns dict with direction and scores."""
    close = df["Close"].dropna()
    if len(close) < 30:
        return {"label": label, "direction": 0, "trend": 0, "momentum": 0, "mean_rev": 0, "score": 0}

    current = float(close.iloc[-1])

    # Trend: price vs 20-period EMA
    ema20 = float(_compute_ema(close, 20).iloc[-1])
    ema50 = float(_compute_ema(close, 50).iloc[-1]) if len(close) >= 50 else ema20

    trend_score = 0.0
    if current > ema20 and ema20 > ema50:
        trend_score = 1.0  # strong uptrend
    elif current > ema20:
        trend_score = 0.5  # above short EMA
    elif current < ema20 and ema20 < ema50:
        trend_score = -1.0  # strong downtrend
    elif current < ema20:
        trend_score = -0.5  # below short EMA

    # Momentum: RSI
    rsi = _compute_rsi(close, 14)
    if rsi > 70:
        mom_score = 0.8  # overbought but still bullish
    elif rsi > 55:
        mom_score = 1.0  # healthy bullish momentum
    elif rsi > 45:
        mom_score = 0.0  # neutral
    elif rsi > 30:
        mom_score = -1.0  # bearish momentum
    else:
        mom_score = -0.8  # oversold but still bearish

    # Mean reversion: Bollinger %B
    pctb = _compute_bollinger_pctb(close, 20)
    if pctb > 0.8:
        mr_score = -0.3  # extended high, mean reversion risk
    elif pctb > 0.5:
        mr_score = 0.5   # upper half, still room
    elif pctb > 0.2:
        mr_score = -0.5  # lower half
    else:
        mr_score = 0.3   # deeply oversold, bounce potential

    # Combined direction for this timeframe
    combined = trend_score * 0.5 + mom_score * 0.3 + mr_score * 0.2
    direction = 1 if combined > 0.1 else (-1 if combined < -0.1 else 0)

    return {
        "label": label,
        "direction": direction,
        "trend": round(trend_score, 2),
        "momentum": round(mom_score, 2),
        "mean_rev": round(mr_score, 2),
        "score": round(combined, 3),
        "rsi": round(rsi, 1),
        "bollinger_pctb": round(pctb, 3),
        "price_vs_ema20_pct": round((current / ema20 - 1) * 100, 2),
    }


def compute_confluence(ticker: str = "GC=F") -> dict:
    """
    Compute multi-timeframe confluence score.
    Returns comprehensive result dict.
    """
    frames = _fetch_multi_tf(ticker)

    tf_scores = {}
    tf_map = {"1d": "daily", "1wk": "weekly", "1mo": "monthly"}

    for interval, label in tf_map.items():
        if interval in frames:
            tf_scores[label] = _score_timeframe(frames[interval], label)
        else:
            tf_scores[label] = {"label": label, "direction": 0, "trend": 0,
                                "momentum": 0, "mean_rev": 0, "score": 0}

    # Confluence calculation
    directions = [tf_scores[tf]["direction"] for tf in ["daily", "weekly", "monthly"]]
    scores = [tf_scores[tf]["score"] for tf in ["daily", "weekly", "monthly"]]

    # Count alignment
    bullish_count = sum(1 for d in directions if d > 0)
    bearish_count = sum(1 for d in directions if d < 0)
    neutral_count = sum(1 for d in directions if d == 0)

    # Weighted average (daily 50%, weekly 30%, monthly 20%)
    weights = [0.50, 0.30, 0.20]
    weighted_score = sum(s * w for s, w in zip(scores, weights))

    # Confluence level
    if bullish_count == 3:
        confluence_level = "STRONG_BULLISH"
        confluence_score = min(100, int(50 + abs(weighted_score) * 50))
    elif bearish_count == 3:
        confluence_level = "STRONG_BEARISH"
        confluence_score = max(-100, int(-50 - abs(weighted_score) * 50))
    elif bullish_count == 2:
        confluence_level = "MODERATE_BULLISH"
        confluence_score = int(30 + abs(weighted_score) * 40)
    elif bearish_count == 2:
        confluence_level = "MODERATE_BEARISH"
        confluence_score = int(-30 - abs(weighted_score) * 40)
    else:
        confluence_level = "MIXED"
        confluence_score = int(weighted_score * 30)

    # Clamp
    confluence_score = max(-100, min(100, confluence_score))

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker": ticker,
        "timeframes": tf_scores,
        "confluence": {
            "level": confluence_level,
            "score": confluence_score,
            "bullish_tfs": bullish_count,
            "bearish_tfs": bearish_count,
            "neutral_tfs": neutral_count,
            "weighted_score": round(weighted_score, 4),
        },
        "recommendation": _recommendation(confluence_level, confluence_score),
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2))

    _print_report(result)
    return result


def _recommendation(level: str, score: int) -> str:
    if "STRONG_BULLISH" in level:
        return "All timeframes aligned bullish — maximum conviction long"
    elif "STRONG_BEARISH" in level:
        return "All timeframes aligned bearish — avoid or reduce position"
    elif "MODERATE_BULLISH" in level:
        return "Majority bullish — consider adding with normal sizing"
    elif "MODERATE_BEARISH" in level:
        return "Majority bearish — tighten stops, reduce exposure"
    else:
        return "Mixed signals — reduce size, wait for alignment"


def _print_report(result: dict) -> None:
    print(f"\n{SEP}")
    print(f"  MULTI-TIMEFRAME CONFLUENCE -- {result['ticker']}")
    print(SEP)

    for tf in ["daily", "weekly", "monthly"]:
        t = result["timeframes"].get(tf, {})
        d = t.get("direction", 0)
        d_str = "BULL" if d > 0 else "BEAR" if d < 0 else "FLAT"
        d_color = "\033[32m" if d > 0 else "\033[31m" if d < 0 else "\033[33m"
        print(
            f"  {tf.upper():<10s}  "
            f"Dir: {d_color}{d_str:<5s}\033[0m  "
            f"Trend: {t.get('trend', 0):+.2f}  "
            f"Mom: {t.get('momentum', 0):+.2f}  "
            f"MR: {t.get('mean_rev', 0):+.2f}  "
            f"RSI: {t.get('rsi', 0):.0f}  "
            f"Score: {t.get('score', 0):+.3f}"
        )

    c = result["confluence"]
    print()
    print(f"  CONFLUENCE")
    print(f"  {'─' * 40}")
    print(f"  Level:          {c['level']}")
    print(f"  Score:          {c['score']:+d}/100")
    print(f"  Alignment:      {c['bullish_tfs']}B / {c['neutral_tfs']}N / {c['bearish_tfs']}Bear")
    print(f"  Weighted:       {c['weighted_score']:+.4f}")
    print()
    print(f"  >> {result['recommendation']}")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Timeframe Confluence Scorer")
    parser.add_argument("--ticker", default="GC=F")
    args = parser.parse_args()
    compute_confluence(ticker=args.ticker)
