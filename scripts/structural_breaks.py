#!/usr/bin/env python3
"""
Structural Break Detector
==========================
Three orthogonal tests on the gold return series, each surfaces a different
kind of regime shift:

  1. CUSUM (Brown-Durbin-Evans 1975)
        Standardised cumulative sum of demeaned returns. Crosses the
        ±1.358·√T·(t/T) boundary if the mean drifts away from the in-sample
        average. Surfaces persistent mean shifts.

  2. Binary segmentation on the mean
        Recursive Welch-t splits with a Bonferroni-corrected threshold.
        Returns up to 5 most-significant break dates.

  3. Variance regime breaks (rolling F-statistic)
        Compares pre- and post-window sample variance; flags windows where
        the variance ratio exceeds 2.5×. Catches GARCH-style volatility shifts.

For each detected break the engine reports the date, the magnitude (mean
change in bps, variance ratio), and days since the most recent break.

Output: data/structural_breaks.json
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
OUTPUT_FILE = DATA_DIR / "structural_breaks.json"

DEFAULT_TICKER = "GC=F"
DEFAULT_LOOKBACK = "5y"

CUSUM_CRIT_5PCT = 1.358          # Brown-Durbin-Evans 5% boundary multiplier
SEG_T_THRESHOLD = 3.5             # Bonferroni-rough Welch-t threshold
SEG_MIN_SEGMENT = 63              # minimum 63d on each side of a break
SEG_MAX_BREAKS = 5
VAR_WINDOW = 63
VAR_RATIO_THRESHOLD = 2.5

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _fetch_returns(ticker: str, lookback: str) -> pd.Series:
    if yf is None:
        raise ImportError("yfinance is required")
    raw = yf.download(
        ticker, period=lookback, interval="1d",
        progress=False, auto_adjust=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    close = raw["Close"].dropna()
    return close.pct_change().dropna()


# ---------------------------------------------------------------------------
# 1. CUSUM test
# ---------------------------------------------------------------------------
def cusum_test(returns: pd.Series) -> dict:
    """Brown-Durbin-Evans CUSUM on demeaned returns."""
    r = returns.values - returns.mean()
    sigma = returns.std()
    T = len(r)
    if T < 30 or sigma <= 0:
        return {"break_detected": False, "test_stat": 0.0,
                "critical_value": CUSUM_CRIT_5PCT,
                "break_date": None}
    cum = np.cumsum(r) / (sigma * np.sqrt(T))
    test_stat = float(np.abs(cum).max())
    break_idx = int(np.argmax(np.abs(cum)))
    break_date = str(returns.index[break_idx].date())
    return {
        "break_detected":  bool(test_stat > CUSUM_CRIT_5PCT),
        "test_stat":       round(test_stat, 4),
        "critical_value":  CUSUM_CRIT_5PCT,
        "break_date":      break_date,
        "break_idx":       break_idx,
    }


# ---------------------------------------------------------------------------
# 2. Binary segmentation on the mean
# ---------------------------------------------------------------------------
def _welch_t(x: np.ndarray, split: int) -> float:
    if split < 2 or split > len(x) - 2:
        return 0.0
    a, b = x[:split], x[split:]
    m1, m2 = a.mean(), b.mean()
    s1 = a.var(ddof=1) / len(a)
    s2 = b.var(ddof=1) / len(b)
    pooled = np.sqrt(s1 + s2)
    if pooled <= 1e-12:
        return 0.0
    return float(abs(m1 - m2) / pooled)


def binary_segmentation(returns: pd.Series) -> list:
    breaks: list[dict] = []

    def _split(start: int, end: int) -> None:
        if (end - start) < 2 * SEG_MIN_SEGMENT or len(breaks) >= SEG_MAX_BREAKS:
            return
        x = returns.iloc[start:end].values
        best_t_stat = 0.0
        best_idx = None
        for t in range(SEG_MIN_SEGMENT, len(x) - SEG_MIN_SEGMENT):
            s = _welch_t(x, t)
            if s > best_t_stat:
                best_t_stat = s
                best_idx = t
        if best_idx is not None and best_t_stat > SEG_T_THRESHOLD:
            abs_idx = start + best_idx
            breaks.append({
                "idx":         int(abs_idx),
                "date":        str(returns.index[abs_idx].date()),
                "t_stat":      round(float(best_t_stat), 3),
                "mean_before_bps": round(float(x[:best_idx].mean()) * 10000, 2),
                "mean_after_bps":  round(float(x[best_idx:].mean()) * 10000, 2),
                "delta_bps":   round(
                    float(x[best_idx:].mean() - x[:best_idx].mean()) * 10000, 2
                ),
            })
            _split(start, abs_idx)
            _split(abs_idx, end)

    _split(0, len(returns))
    breaks.sort(key=lambda b: b["t_stat"], reverse=True)
    return breaks


# ---------------------------------------------------------------------------
# 3. Variance regime breaks (rolling F)
# ---------------------------------------------------------------------------
def variance_breaks(returns: pd.Series, window: int = VAR_WINDOW) -> list:
    n = len(returns)
    breaks = []
    step = max(window // 2, 5)
    for i in range(window, n - window, step):
        a = returns.iloc[i - window:i]
        b = returns.iloc[i:i + window]
        v1 = float(a.var(ddof=1))
        v2 = float(b.var(ddof=1))
        if min(v1, v2) <= 0:
            continue
        ratio = max(v1, v2) / min(v1, v2)
        if ratio > VAR_RATIO_THRESHOLD:
            breaks.append({
                "idx":            int(i),
                "date":           str(returns.index[i].date()),
                "vol_before_pct": round(np.sqrt(v1 * 252) * 100, 3),
                "vol_after_pct":  round(np.sqrt(v2 * 252) * 100, 3),
                "variance_ratio": round(ratio, 3),
                "direction":      "EXPANSION" if v2 > v1 else "CONTRACTION",
            })
    return breaks


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_structural_breaks(
    ticker: str = DEFAULT_TICKER,
    lookback: str = DEFAULT_LOOKBACK,
) -> dict:
    returns = _fetch_returns(ticker, lookback)

    cusum = cusum_test(returns)
    mean_bs = binary_segmentation(returns)
    var_bs = variance_breaks(returns)

    # Most recent break across all detectors
    all_breaks = []
    if cusum.get("break_detected"):
        all_breaks.append((cusum["break_date"], "CUSUM"))
    for b in mean_bs:
        all_breaks.append((b["date"], "MEAN"))
    for b in var_bs:
        all_breaks.append((b["date"], "VAR"))

    most_recent = None
    days_since = None
    if all_breaks:
        all_breaks.sort()  # by date string (ISO format sorts correctly)
        most_recent_date = all_breaks[-1][0]
        most_recent_type = all_breaks[-1][1]
        most_recent = f"{most_recent_date} ({most_recent_type})"
        days_since = (returns.index[-1].date() - pd.to_datetime(most_recent_date).date()).days

    result = {
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker":         ticker,
        "lookback":       lookback,
        "n_obs":          int(len(returns)),
        "cusum":          cusum,
        "mean_breaks":    mean_bs,
        "variance_breaks":var_bs,
        "summary": {
            "cusum_break":           bool(cusum.get("break_detected")),
            "n_mean_breaks":         len(mean_bs),
            "n_variance_breaks":     len(var_bs),
            "most_recent_break":     most_recent,
            "days_since_last_break": days_since,
        },
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
    print(f"  STRUCTURAL BREAK DETECTOR -- {r['ticker']}")
    print(SEP)
    print(f"  Observations:   {r['n_obs']}")
    print(f"  Lookback:       {r['lookback']}")
    print()

    c = r["cusum"]
    cusum_flag = "BREAK" if c.get("break_detected") else "STABLE"
    print(f"  CUSUM (Brown-Durbin-Evans)")
    print(f"  {'─' * 40}")
    print(f"  Status:         {cusum_flag}")
    print(f"  Test stat:      {c.get('test_stat', 0):.4f}")
    print(f"  Critical 5%:    {c.get('critical_value', 0):.4f}")
    if c.get("break_detected"):
        print(f"  Break date:     {c.get('break_date')}")
    print()

    print(f"  MEAN BREAKS (binary segmentation)")
    print(f"  {'─' * 58}")
    if r["mean_breaks"]:
        for b in r["mean_breaks"]:
            print(
                f"  {b['date']}  t={b['t_stat']:>5.2f}  "
                f"before={b['mean_before_bps']:>+7.2f}bp/d  "
                f"after={b['mean_after_bps']:>+7.2f}bp/d  "
                f"Δ={b['delta_bps']:>+7.2f}"
            )
    else:
        print(f"  No mean breaks > t={SEG_T_THRESHOLD:.1f}")
    print()

    print(f"  VARIANCE BREAKS  (ratio > {VAR_RATIO_THRESHOLD}×)")
    print(f"  {'─' * 58}")
    if r["variance_breaks"]:
        for b in r["variance_breaks"]:
            print(
                f"  {b['date']}  ratio={b['variance_ratio']:>5.2f}  "
                f"vol {b['vol_before_pct']:>5.2f}% → {b['vol_after_pct']:>5.2f}%  "
                f"{b['direction']}"
            )
    else:
        print(f"  No variance breaks > {VAR_RATIO_THRESHOLD}× ratio")
    print()

    s = r["summary"]
    print(f"  SUMMARY")
    print(f"  {'─' * 40}")
    if s["most_recent_break"]:
        print(f"  Last break:     {s['most_recent_break']}  ({s['days_since_last_break']}d ago)")
    else:
        print(f"  Last break:     none detected")
    print(f"  Total breaks:   CUSUM={int(s['cusum_break'])}  "
          f"Mean={s['n_mean_breaks']}  Var={s['n_variance_breaks']}")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structural Break Detector")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--lookback", default=DEFAULT_LOOKBACK)
    args = parser.parse_args()
    run_structural_breaks(ticker=args.ticker, lookback=args.lookback)
