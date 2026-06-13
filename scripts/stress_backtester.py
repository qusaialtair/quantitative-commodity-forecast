#!/usr/bin/env python3
"""
Stress Backtester  (Phase XVII Stage 79)
=========================================
Replays the Phase XIV rule cascade (alpha_stacker / strategy_selector /
multi_strategy_trader) against multiple historical crisis windows to test
whether the +29%/y from `strategy_backtester` on 2024-2026 is real edge
or a single-regime fluke.

Crisis windows tested
---------------------
- 2008-09  : Global Financial Crisis
- 2011-12  : European sovereign debt crisis
- 2013     : Bernanke "taper tantrum"
- 2015-16  : China devaluation + commodity rout
- 2018     : "Vol-mageddon" + Q4 sell-off
- 2020     : COVID-19 crash + reflation
- 2022     : Inflation shock + bond rout
- 2024-26  : Recent tuning window (control)

Method
------
For each window, fetch closes from yfinance, apply the exact same rule
tree used by the live strategy_selector + the same conviction proxy used
by strategy_backtester, and roll forward one day at a time.  Returns
per-window stats + a regime-robustness verdict.

Output: data/stress_backtest.json

Verdict aggregation
-------------------
The system gets one of four labels:

    ROBUST              passes every window with Sharpe ≥ 0.5 and DD ≥ -25%
    REGIME_SENSITIVE    fails 1 or 2 windows but Sharpe positive overall
    REGIME_FRAGILE      fails 3+ windows; tuning is regime-specific
    OVERFIT             negative Sharpe in the majority of windows

Usage:
    python3 scripts/stress_backtester.py             # default 8 windows
    python3 scripts/stress_backtester.py --quiet
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.cache_layer import cached  # noqa: E402
from scripts.crisis_detector import (  # noqa: E402
    apply_guard as _crisis_apply_guard,
    classify_from_prices as _crisis_classify,
)

# Reuse the live rule tree from the main backtester so this stress test
# tracks any future tuning automatically.
from scripts.strategy_backtester import (  # noqa: E402
    _classify_regime_proxy,
    _classify_vol_regime,
    _extension_z,
    _final_size_pct,
    _select_strategy,
    _strategy_return,
    _technical_conviction,
)

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "stress_backtest.json"
SQ252 = float(np.sqrt(252))


# ──────────────────────────────────────────────────────────────────────────────
# Crisis windows — each entry is (label, start, end, regime_description)
# ──────────────────────────────────────────────────────────────────────────────
CRISIS_WINDOWS: list[dict[str, str]] = [
    {
        "label":  "2008 GFC",
        "start":  "2008-01-01",
        "end":    "2009-12-31",
        "regime": "deflationary crash + reflation rally",
    },
    {
        "label":  "2011 Euro crisis",
        "start":  "2011-07-01",
        "end":    "2012-06-30",
        "regime": "European sovereign debt + safe-haven gold spike",
    },
    {
        "label":  "2013 taper tantrum",
        "start":  "2013-05-01",
        "end":    "2013-12-31",
        "regime": "Bernanke shock + gold sell-off",
    },
    {
        "label":  "2015 China rout",
        "start":  "2015-08-01",
        "end":    "2016-02-29",
        "regime": "PBoC devaluation + commodity collapse",
    },
    {
        "label":  "2018 vol-mageddon",
        "start":  "2018-01-01",
        "end":    "2018-12-31",
        "regime": "Feb vol spike + Q4 risk-off",
    },
    {
        "label":  "2020 COVID",
        "start":  "2020-02-01",
        "end":    "2020-08-31",
        "regime": "pandemic crash + unprecedented stimulus",
    },
    {
        "label":  "2022 inflation rout",
        "start":  "2022-01-01",
        "end":    "2022-12-31",
        "regime": "60/40 down 17% — first since 1969",
    },
    {
        "label":  "2024-26 (tuning window)",
        "start":  "2024-08-15",
        "end":    "2026-05-14",
        "regime": "control — same period as strategy_backtester",
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Data fetch — one big history, sliced per window
# ──────────────────────────────────────────────────────────────────────────────
@cached(namespace="yfinance", ttl_seconds=12 * 3600)
def _fetch_full_history(ticker: str = "GC=F") -> tuple[np.ndarray, list[str]]:
    """Fetch as much GC=F history as yfinance has (typically back to ~2000)."""
    import yfinance as yf

    hist = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=True)
    if hist is None or hist.empty:
        raise SystemExit(f"no history for {ticker}")

    closes = hist["Close"].astype(float).values
    dates = [d.strftime("%Y-%m-%d") for d in hist.index]
    return closes, dates


def _slice_window(
    closes: np.ndarray, dates: list[str], start: str, end: str
) -> tuple[np.ndarray, list[str]]:
    """Return the closes + dates strictly within [start, end] (inclusive)."""
    if not dates:
        return np.array([]), []
    # Find index bounds; need to include warmup for SMA/momentum calcs
    start_idx = next((i for i, d in enumerate(dates) if d >= start), None)
    end_idx   = next((i for i, d in enumerate(dates) if d >  end), len(dates))
    if start_idx is None:
        return np.array([]), []
    # Include 200 days warmup before the window so SMAs are usable
    warmup_start = max(0, start_idx - 220)
    return closes[warmup_start:end_idx], dates[warmup_start:end_idx]


# ──────────────────────────────────────────────────────────────────────────────
# Simulation core — mirrors strategy_backtester.run_backtest exactly
# ──────────────────────────────────────────────────────────────────────────────
def _rolling_vol_pct(returns: np.ndarray, window: int = 21) -> np.ndarray:
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


def _simulate_window(
    closes: np.ndarray, dates: list[str],
    window_start: str, window_end: str,
    target_pct: float = 10.0,
) -> dict[str, Any]:
    """Run the full Phase XIV cascade over closes[i].  Skip the 220-day warmup
    when computing stats — we only count returns inside [window_start, window_end]."""
    if len(closes) < 100:
        return {"error": "insufficient history"}

    log_returns = np.diff(np.log(closes))
    pct_returns = np.diff(closes) / closes[:-1] * 100

    rv_21 = _rolling_vol_pct(log_returns, 21)
    rv_63 = _rolling_vol_pct(log_returns, 63)
    drift_5  = _rolling_mean(log_returns, 5)
    drift_21 = _rolling_mean(log_returns, 21)

    daily_returns: list[float] = []
    in_window_returns: list[float] = []
    strategy_counts: dict[str, int] = {}
    strategy_pl: dict[str, float] = {}
    equity_curve = [100_000.0]
    nav = 100_000.0
    in_window_curve: list[float] = []

    for i in range(60, len(log_returns) - 1):
        rv21 = float(rv_21[i]) if i < len(rv_21) else float("nan")
        rv63 = float(rv_63[i]) if i < len(rv_63) else float("nan")
        d5   = float(drift_5[i]) if i < len(drift_5) else 0.0
        d21  = float(drift_21[i]) if i < len(drift_21) else 0.0

        hmm = _classify_regime_proxy(rv21, d5, d21)
        vol_regime = _classify_vol_regime(rv21, rv63)
        conviction, tier, direction = _technical_conviction(closes, i + 1)
        ext_z = _extension_z(closes, i + 1)
        macro_tilt = float(np.sign(d21) * 1.2) if abs(d21) > 0.002 else 0.0

        strategy = _select_strategy(
            tier=tier, direction=direction,
            vol_regime=vol_regime, hmm_state=hmm,
            conviction=conviction,
            extension_z=ext_z, macro_tilt=macro_tilt,
        )
        size_pct = _final_size_pct(strategy, tier, conviction, vol_regime, hmm)

        # Performance multiplier (matches strategy_backtester)
        if len(daily_returns) >= 21:
            mtd_window = daily_returns[-21:]
            mtd_cum = (np.prod([1 + x / 100 for x in mtd_window]) - 1.0) * 100
            gap = mtd_cum - target_pct
            if gap > 2.0:    pmult = 0.70
            elif gap > 0.5:  pmult = 0.85
            elif gap >= -0.5:pmult = 1.00
            elif gap >= -2.0:pmult = 1.20
            elif gap >= -5.0:pmult = 1.50
            else:            pmult = 1.75
        else:
            pmult = 1.0

        adjusted = min(95.0, size_pct * pmult)

        # ── Crisis guard (Phase XVIII) — same logic the live selector uses ──
        crisis = _crisis_classify(closes, current_idx=i + 1)
        guarded_strategy, guarded_size, _ = _crisis_apply_guard(
            strategy, adjusted, crisis["tier"],
        )
        strategy = guarded_strategy
        adjusted = guarded_size

        strat_ret = _strategy_return(strategy, direction, closes, i + 1)
        net_daily = (adjusted / 100.0) * strat_ret
        daily_returns.append(net_daily)
        nav *= 1.0 + net_daily / 100.0
        equity_curve.append(nav)

        # Track per-strategy share + P&L
        d_iso = dates[i + 1]
        if window_start <= d_iso <= window_end:
            in_window_returns.append(net_daily)
            in_window_curve.append(nav)
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            strategy_pl[strategy] = strategy_pl.get(strategy, 0.0) + net_daily

    return _compute_window_stats(
        in_window_returns, in_window_curve,
        strategy_counts, strategy_pl,
    )


def _compute_window_stats(
    returns: list[float], curve: list[float],
    strat_counts: dict[str, int], strat_pl: dict[str, float],
) -> dict[str, Any]:
    n = len(returns)
    if n < 5:
        return {"n_days": n, "error": "insufficient in-window history"}

    ret = np.asarray(returns)
    cum = (np.prod(1 + ret / 100) - 1.0) * 100
    annualised = ((1 + cum / 100) ** (252 / n) - 1.0) * 100 if n > 0 else 0.0
    mu = float(ret.mean())
    sd = float(ret.std(ddof=0))
    sharpe = (mu / sd * SQ252) if sd > 1e-12 else None

    if curve:
        c_arr = np.asarray(curve)
        peak = np.maximum.accumulate(c_arr)
        dd = (c_arr - peak) / peak
        max_dd = float(dd.min() * 100)
    else:
        max_dd = 0.0

    win_days = float((ret > 0).mean() * 100)
    total_days = sum(strat_counts.values()) or 1
    by_strategy = [
        {
            "strategy": s,
            "n_days":   c,
            "share_pct":round(c / total_days * 100, 1),
            "total_pl_pct": round(strat_pl.get(s, 0.0), 2),
            "avg_daily_pl_pct": round(strat_pl.get(s, 0.0) / c, 4) if c else 0.0,
        }
        for s, c in strat_counts.items()
    ]
    by_strategy.sort(key=lambda x: -x["total_pl_pct"])

    return {
        "n_days":           n,
        "cum_return_pct":   round(float(cum), 2),
        "annualised_pct":   round(float(annualised), 2),
        "ann_vol_pct":      round(float(sd * SQ252), 2),
        "sharpe":           round(sharpe, 3) if sharpe is not None else None,
        "max_drawdown_pct": round(max_dd, 2),
        "win_days_pct":     round(win_days, 2),
        "by_strategy":      by_strategy,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Verdict aggregation
# ──────────────────────────────────────────────────────────────────────────────
def _window_verdict(stats: dict) -> str:
    """Per-window pass/fail."""
    if "error" in stats:
        return "INSUFFICIENT_DATA"
    sharpe = stats.get("sharpe") or 0.0
    dd     = stats.get("max_drawdown_pct") or 0.0
    if sharpe >= 1.0 and dd > -15:
        return "STRONG"
    if sharpe >= 0.5 and dd > -25:
        return "PASS"
    if sharpe > 0 and dd > -35:
        return "DEGRADED"
    return "FAIL"


def _aggregate_verdict(window_results: list[dict]) -> tuple[str, str]:
    """System-wide robustness verdict + human-readable note."""
    verdicts = [w["verdict"] for w in window_results if w.get("verdict") != "INSUFFICIENT_DATA"]
    n_total = len(verdicts)
    if n_total == 0:
        return "INSUFFICIENT_DATA", "No usable crisis windows."

    n_strong   = verdicts.count("STRONG")
    n_pass     = verdicts.count("PASS")
    n_degraded = verdicts.count("DEGRADED")
    n_fail     = verdicts.count("FAIL")
    sharpes = [w["sharpe"] for w in window_results if w.get("sharpe") is not None]
    avg_sharpe = float(np.mean(sharpes)) if sharpes else 0.0

    if n_fail == 0 and (n_strong + n_pass) == n_total and avg_sharpe >= 0.8:
        verdict = "ROBUST"
        note = (
            f"All {n_total} crisis windows pass; avg Sharpe {avg_sharpe:.2f}. "
            f"Strategy rules generalise across regimes."
        )
    elif n_fail <= 1 and avg_sharpe >= 0.5:
        verdict = "REGIME_SENSITIVE"
        note = (
            f"{n_total - n_fail}/{n_total} windows pass; avg Sharpe "
            f"{avg_sharpe:.2f}. {n_fail} regime(s) hurt the system but core edge holds."
        )
    elif n_fail >= 3 or avg_sharpe < 0.3:
        verdict = "REGIME_FRAGILE"
        note = (
            f"Only {n_total - n_fail}/{n_total} windows pass; avg Sharpe "
            f"{avg_sharpe:.2f}. Tuning appears regime-specific; re-evaluate "
            f"before live deployment."
        )
    elif avg_sharpe <= 0:
        verdict = "OVERFIT"
        note = (
            f"Average Sharpe {avg_sharpe:.2f} across windows; tuning is "
            f"a single-regime artefact."
        )
    else:
        verdict = "REGIME_SENSITIVE"
        note = f"{n_total - n_fail}/{n_total} windows pass; avg Sharpe {avg_sharpe:.2f}."

    return verdict, note


# ──────────────────────────────────────────────────────────────────────────────
# Public entry
# ──────────────────────────────────────────────────────────────────────────────
def run_stress_test(ticker: str = "GC=F") -> dict:
    closes, dates = _fetch_full_history(ticker)
    if not dates:
        raise SystemExit(f"no history for {ticker}")

    history_start = dates[0]
    history_end   = dates[-1]

    window_results: list[dict] = []
    for w in CRISIS_WINDOWS:
        # Skip windows outside available history
        if w["end"] < history_start or w["start"] > history_end:
            window_results.append({
                "label":   w["label"],
                "start":   w["start"],
                "end":     w["end"],
                "regime":  w["regime"],
                "verdict": "INSUFFICIENT_DATA",
                "note":    "outside available yfinance history",
            })
            continue

        sliced_closes, sliced_dates = _slice_window(closes, dates, w["start"], w["end"])
        if len(sliced_closes) < 60:
            window_results.append({
                "label":   w["label"],
                "start":   w["start"],
                "end":     w["end"],
                "regime":  w["regime"],
                "verdict": "INSUFFICIENT_DATA",
                "note":    f"only {len(sliced_closes)} bars in window",
            })
            continue

        stats = _simulate_window(sliced_closes, sliced_dates, w["start"], w["end"])
        verdict = _window_verdict(stats)
        window_results.append({
            "label":   w["label"],
            "start":   w["start"],
            "end":     w["end"],
            "regime":  w["regime"],
            "verdict": verdict,
            **stats,
        })

    aggregate_verdict, aggregate_note = _aggregate_verdict(window_results)

    # Aggregate stats
    valid = [w for w in window_results if "sharpe" in w and w.get("sharpe") is not None]
    avg_sharpe = float(np.mean([w["sharpe"] for w in valid])) if valid else 0.0
    worst_dd   = float(min([w["max_drawdown_pct"] for w in valid], default=0.0))
    best_window  = max(valid, key=lambda w: w["sharpe"]) if valid else None
    worst_window = min(valid, key=lambda w: w["sharpe"]) if valid else None

    out = {
        "schema_version": "1.0",
        "engine":         "stress_backtester",
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker":         ticker,
        "history_range":  {"start": history_start, "end": history_end},
        "n_windows":      len(window_results),
        "n_valid":        len(valid),
        "aggregate": {
            "verdict":       aggregate_verdict,
            "note":          aggregate_note,
            "avg_sharpe":    round(avg_sharpe, 3),
            "worst_max_dd":  round(worst_dd, 2),
            "best_window":   best_window["label"] if best_window else None,
            "worst_window":  worst_window["label"] if worst_window else None,
        },
        "windows": window_results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(out, indent=2, default=str))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticker", default="GC=F")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    out = run_stress_test(ticker=args.ticker)
    if args.quiet:
        return 0

    a = out["aggregate"]
    print("=" * 72)
    print(f"STRESS BACKTEST  ({out['generated_at']})")
    print("=" * 72)
    print(f"  Ticker        : {out['ticker']}")
    print(f"  Available     : {out['history_range']['start']} → {out['history_range']['end']}")
    print(f"  Windows       : {out['n_valid']}/{out['n_windows']} usable")
    print()
    print(f"  AGGREGATE VERDICT: {a['verdict']}")
    print(f"  {a['note']}")
    print()
    print(f"  Avg Sharpe    : {a['avg_sharpe']:+.3f}")
    print(f"  Worst max DD  : {a['worst_max_dd']:+.2f}%")
    print(f"  Best window   : {a['best_window']}")
    print(f"  Worst window  : {a['worst_window']}")
    print()
    print(f"  {'Window':<28s} {'Verdict':<14s} {'Ann':>8s} {'Sharpe':>7s} {'DD':>8s} {'Win%':>6s}")
    print(f"  {'-'*28} {'-'*14} {'-'*8} {'-'*7} {'-'*8} {'-'*6}")
    for w in out["windows"]:
        if "error" in w or w["verdict"] == "INSUFFICIENT_DATA":
            print(f"  {w['label']:<28s} {w['verdict']:<14s} {'-':>8s} {'-':>7s} {'-':>8s} {'-':>6s}")
            continue
        print(
            f"  {w['label']:<28s} {w['verdict']:<14s} "
            f"{w.get('annualised_pct', 0):>+7.2f}% "
            f"{w.get('sharpe', 0):>+7.2f} "
            f"{w.get('max_drawdown_pct', 0):>+7.2f}% "
            f"{w.get('win_days_pct', 0):>5.1f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
