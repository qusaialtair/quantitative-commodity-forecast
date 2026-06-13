#!/usr/bin/env python3
"""
Walk-Forward Validator  (Phase XXII Stage 83)
==============================================
Slides a 1-year rolling window through ~15 years of gold history and
runs the full Phase XIV-XXI rule cascade at each step.  Answers the
question: "Is the +24%/y tuning-window result generalisable, or did we
overfit to 2024-2026?"

Method
------
- Full history pulled from yfinance via the cached helper.
- Step the window forward 21 trading days at a time (≈ 1 month).
- At each step, replay the full simulation across the trailing 252
  trading days.  No parameter changes between steps — this is pure
  *validation*, not optimisation.
- Capture per-window stats: annualised return, Sharpe, max DD,
  direction hit rate.

Stability verdict
-----------------

    STABLE          75 %+ windows have positive Sharpe, avg Sharpe ≥ 0.5
    DRIFTING        50-75 % positive, std Sharpe < 1.0
    UNSTABLE        < 50 % positive OR std Sharpe > 1.5
    INSUFFICIENT_HISTORY

The std-of-Sharpe across windows is the key diagnostic: if it's small,
the rules behave consistently across regimes; if it's large, results
are dominated by a few good windows and the rest are noise.

Output: data/walk_forward_validator.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.cache_layer import cached  # noqa: E402
from scripts.strategy_backtester import (  # noqa: E402
    _classify_regime_proxy,
    _classify_vol_regime,
    _extension_z,
    _final_size_pct,
    _select_strategy,
    _short_trend_specialist,
    _strategy_return,
    _technical_conviction,
)
from scripts.crisis_detector import (  # noqa: E402
    apply_guard as _crisis_apply_guard,
    classify_from_prices as _crisis_classify,
)
from scripts.stress_backtester import _fetch_full_history  # noqa: E402

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "walk_forward_validator.json"

SQ252 = float(np.sqrt(252))
DEFAULT_WINDOW_DAYS = 252
DEFAULT_STEP_DAYS   = 21
DEFAULT_START_DATE  = "2010-01-01"
TARGET_PCT          = 10.0


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (mirror strategy_backtester's per-step logic)
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


def _simulate_segment(
    closes: np.ndarray, dates: list[str],
    start_idx: int, end_idx: int,
) -> dict:
    """
    Replay the cascade across closes[start_idx:end_idx].
    Needs at least 220 bars of warmup *before* start_idx for SMA200.
    """
    warmup_start = max(0, start_idx - 220)
    seg_closes = closes[warmup_start:end_idx]
    seg_dates  = dates[warmup_start:end_idx]
    if len(seg_closes) < 60:
        return {"n_days": 0, "error": "insufficient warmup"}

    log_returns = np.diff(np.log(seg_closes))
    pct_returns = np.diff(seg_closes) / seg_closes[:-1] * 100

    rv_21 = _rolling_vol_pct(log_returns, 21)
    rv_63 = _rolling_vol_pct(log_returns, 63)
    d5    = _rolling_mean(log_returns, 5)
    d21   = _rolling_mean(log_returns, 21)

    in_window_returns: list[float] = []
    daily_for_pmult:   list[float] = []
    curve:             list[float] = []
    nav = 100_000.0
    # The "window-internal" offset relative to seg_closes
    window_internal_start = start_idx - warmup_start

    for i in range(60, len(log_returns) - 1):
        v21 = float(rv_21[i]) if i < len(rv_21) else float("nan")
        v63 = float(rv_63[i]) if i < len(rv_63) else float("nan")
        m5  = float(d5[i])  if i < len(d5)  else 0.0
        m21 = float(d21[i]) if i < len(d21) else 0.0

        hmm = _classify_regime_proxy(v21, m5, m21)
        vol_regime = _classify_vol_regime(v21, v63)
        conviction, tier, direction = _technical_conviction(seg_closes, i + 1)
        ext_z = _extension_z(seg_closes, i + 1)
        macro_tilt = float(np.sign(m21) * 1.2) if abs(m21) > 0.002 else 0.0
        in_short_trend = _short_trend_specialist(seg_closes, i + 1)

        strategy = _select_strategy(
            tier=tier, direction=direction,
            vol_regime=vol_regime, hmm_state=hmm,
            conviction=conviction,
            extension_z=ext_z, macro_tilt=macro_tilt,
            short_trend=in_short_trend,
        )
        size_pct = _final_size_pct(strategy, tier, conviction, vol_regime, hmm)

        if len(daily_for_pmult) >= 21:
            mtd_window = daily_for_pmult[-21:]
            mtd_cum = (np.prod([1 + x / 100 for x in mtd_window]) - 1.0) * 100
            gap = mtd_cum - TARGET_PCT
            if gap > 2.0:    pmult = 0.70
            elif gap > 0.5:  pmult = 0.85
            elif gap >= -0.5:pmult = 1.00
            elif gap >= -2.0:pmult = 1.20
            elif gap >= -5.0:pmult = 1.50
            else:            pmult = 1.75
        else:
            pmult = 1.0

        adjusted = min(95.0, size_pct * pmult)
        crisis = _crisis_classify(seg_closes, current_idx=i + 1)
        guarded_strategy, guarded_size, _ = _crisis_apply_guard(
            strategy, adjusted, crisis["tier"],
        )
        strat_ret = _strategy_return(guarded_strategy, direction, seg_closes, i + 1)
        net_daily = (guarded_size / 100.0) * strat_ret
        daily_for_pmult.append(net_daily)
        nav *= 1.0 + net_daily / 100.0

        if i + 1 >= window_internal_start:
            in_window_returns.append(net_daily)
            curve.append(nav)

    return _compute_stats(in_window_returns, curve)


def _compute_stats(returns: list[float], curve: list[float]) -> dict:
    if len(returns) < 5:
        return {"n_days": len(returns), "error": "insufficient"}
    ret = np.asarray(returns)
    cum = (np.prod(1 + ret / 100) - 1.0) * 100
    n = len(ret)
    ann = ((1 + cum / 100) ** (252 / n) - 1.0) * 100 if n > 0 else 0.0
    mu = float(ret.mean())
    sd = float(ret.std(ddof=0))
    sharpe = (mu / sd * SQ252) if sd > 1e-12 else None
    if curve:
        c = np.asarray(curve)
        peak = np.maximum.accumulate(c)
        dd = (c - peak) / peak
        max_dd = float(dd.min() * 100)
    else:
        max_dd = 0.0
    win = float((ret > 0).mean() * 100)
    return {
        "n_days":           n,
        "cum_return_pct":   round(float(cum), 2),
        "annualised_pct":   round(float(ann), 2),
        "ann_vol_pct":      round(float(sd * SQ252), 2),
        "sharpe":           round(sharpe, 3) if sharpe is not None else None,
        "max_drawdown_pct": round(max_dd, 2),
        "win_days_pct":     round(win, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Verdict
# ──────────────────────────────────────────────────────────────────────────────
def _stability_verdict(window_stats: list[dict]) -> tuple[str, str]:
    valid = [w for w in window_stats if w.get("sharpe") is not None]
    if len(valid) < 5:
        return "INSUFFICIENT_HISTORY", "Fewer than 5 usable windows."

    sharpes = np.array([w["sharpe"] for w in valid], dtype=float)
    positive_share = float((sharpes > 0).mean() * 100)
    avg_sharpe     = float(sharpes.mean())
    std_sharpe     = float(sharpes.std(ddof=0))
    worst          = float(sharpes.min())
    best           = float(sharpes.max())

    if positive_share >= 75 and avg_sharpe >= 0.5 and std_sharpe < 1.0:
        verdict = "STABLE"
        note = (
            f"{positive_share:.0f}% of {len(valid)} 1y windows have positive Sharpe; "
            f"avg {avg_sharpe:+.2f} (σ {std_sharpe:.2f}).  Rules generalise well."
        )
    elif positive_share >= 50 and std_sharpe < 1.5:
        verdict = "DRIFTING"
        note = (
            f"{positive_share:.0f}% positive but σ-Sharpe {std_sharpe:.2f} is "
            f"meaningful; performance depends on regime."
        )
    else:
        verdict = "UNSTABLE"
        note = (
            f"Only {positive_share:.0f}% of windows positive; σ-Sharpe "
            f"{std_sharpe:.2f}.  Rules are regime-dependent."
        )
    return verdict, note


# ──────────────────────────────────────────────────────────────────────────────
# Public entry
# ──────────────────────────────────────────────────────────────────────────────
def run_walk_forward(
    window_days: int = DEFAULT_WINDOW_DAYS,
    step_days: int = DEFAULT_STEP_DAYS,
    start_date: str = DEFAULT_START_DATE,
    ticker: str = "GC=F",
) -> dict:
    closes, dates = _fetch_full_history(ticker)
    if not dates:
        raise SystemExit("no history")

    # Find index of start_date
    start_idx = next((i for i, d in enumerate(dates) if d >= start_date), None)
    if start_idx is None:
        raise SystemExit(f"start_date {start_date} not in history")

    # We need at least 220 bars of warmup before start_idx
    start_idx = max(start_idx, 220)

    windows: list[dict] = []
    idx = start_idx
    while idx + window_days < len(dates):
        end_idx = idx + window_days
        stats = _simulate_segment(closes, dates, idx, end_idx)
        windows.append({
            "start_date":  dates[idx],
            "end_date":    dates[end_idx - 1] if end_idx - 1 < len(dates) else dates[-1],
            **stats,
        })
        idx += step_days

    verdict, note = _stability_verdict(windows)
    valid = [w for w in windows if w.get("sharpe") is not None]
    sharpes = [w["sharpe"] for w in valid]
    anns    = [w["annualised_pct"] for w in valid if w.get("annualised_pct") is not None]
    dds     = [w["max_drawdown_pct"] for w in valid if w.get("max_drawdown_pct") is not None]

    out = {
        "schema_version": "1.0",
        "engine":         "walk_forward_validator",
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker":         ticker,
        "window_days":    window_days,
        "step_days":      step_days,
        "history_range":  {"start": dates[0], "end": dates[-1]},
        "n_windows":      len(windows),
        "n_valid":        len(valid),
        "verdict":        verdict,
        "note":           note,
        "stats": {
            "positive_share_pct": round(float(np.mean([1 if s > 0 else 0 for s in sharpes]) * 100), 1) if sharpes else 0.0,
            "avg_sharpe":         round(float(np.mean(sharpes)), 3) if sharpes else 0.0,
            "std_sharpe":         round(float(np.std(sharpes, ddof=0)), 3) if sharpes else 0.0,
            "min_sharpe":         round(float(np.min(sharpes)), 3) if sharpes else 0.0,
            "max_sharpe":         round(float(np.max(sharpes)), 3) if sharpes else 0.0,
            "avg_annualised_pct": round(float(np.mean(anns)), 2) if anns else 0.0,
            "median_annualised_pct": round(float(np.median(anns)), 2) if anns else 0.0,
            "worst_max_dd_pct":   round(float(np.min(dds)), 2) if dds else 0.0,
        },
        "windows": windows,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(out, indent=2, default=str))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW_DAYS)
    ap.add_argument("--step",   type=int, default=DEFAULT_STEP_DAYS)
    ap.add_argument("--start",  type=str, default=DEFAULT_START_DATE)
    ap.add_argument("--quiet",  action="store_true")
    args = ap.parse_args()

    out = run_walk_forward(
        window_days=args.window,
        step_days=args.step,
        start_date=args.start,
    )
    if args.quiet:
        return 0

    s = out["stats"]
    print("=" * 72)
    print(f"WALK-FORWARD VALIDATOR  ({out['generated_at']})")
    print("=" * 72)
    print(f"  Ticker / window / step : {out['ticker']} / {out['window_days']}d / {out['step_days']}d")
    print(f"  History                : {out['history_range']['start']} → {out['history_range']['end']}")
    print(f"  Windows produced       : {out['n_valid']} / {out['n_windows']}")
    print()
    print(f"  VERDICT: {out['verdict']}")
    print(f"  {out['note']}")
    print()
    print(f"  Positive share : {s['positive_share_pct']}%")
    print(f"  Avg Sharpe     : {s['avg_sharpe']:+.3f}  (σ {s['std_sharpe']:.3f})")
    print(f"  Sharpe range   : {s['min_sharpe']:+.2f} → {s['max_sharpe']:+.2f}")
    print(f"  Avg annualised : {s['avg_annualised_pct']:+.2f}%")
    print(f"  Median annual  : {s['median_annualised_pct']:+.2f}%")
    print(f"  Worst max DD   : {s['worst_max_dd_pct']:+.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
