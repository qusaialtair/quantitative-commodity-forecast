#!/usr/bin/env python3
"""
Multi-Asset Stress Backtester  (Phase XX Stage 81)
====================================================
Asks the question: does combining the active metals book with a passive
halal-equity overlay rescue the historical crisis windows where the
metals-only system FAILed (2008 GFC, 2022 inflation rout)?

Method
------
For each crisis window in `stress_backtester.CRISIS_WINDOWS`:

  1. Reuse the metals simulation from `stress_backtester._simulate_window`
     to get the active-book daily-return series during that window.
     (Includes Phase XIV alpha-stacker / selector + Phase XVIII crisis
     guard + Phase XIX pivot detector / short-trend specialist.)

  2. Pull SPY closes for the same window and compute equal-weight
     momentum-tilted halal-proxy returns.  We use SPY as the universal
     proxy because some tuning-window tickers (TSLA, NVDA) didn't exist
     in 2008/2011/2015 — a coherent halal basket isn't reconstructible
     for the deep past.  SPY is the standard "world equity" benchmark
     and the right control here.

  3. Combine at 40/60 metals/equity (matches `multi_asset_backtester`).

  4. Compare three books per window: metals-only, equity-only,
     combined.  Verdict per book + aggregate "diversification benefit"
     (does combined beat metals on Sharpe?).

Output: data/multi_asset_stress_backtest.json
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
from scripts.regime_adaptive_allocator import (  # noqa: E402
    adaptive_series,
    weights_for_tier,
)
from scripts.stress_backtester import (  # noqa: E402
    CRISIS_WINDOWS,
    _fetch_full_history,
    _slice_window,
    _simulate_window,
    _window_verdict,
)

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "multi_asset_stress_backtest.json"

DEFAULT_METALS_WEIGHT  = 0.40
DEFAULT_EQUITY_WEIGHT  = 0.60
SQ252 = float(np.sqrt(252))

EQUITY_TRADING_COST_BPS = 5
EQUITY_DAILY_DRAG_BPS   = 0.10   # 25 bps annual riba-filter conservatism


# ──────────────────────────────────────────────────────────────────────────────
# Data fetch — SPY as deep-history halal proxy
# ──────────────────────────────────────────────────────────────────────────────
@cached(namespace="yfinance", ttl_seconds=12 * 3600)
def _fetch_spy_history() -> tuple[np.ndarray, list[str]]:
    import yfinance as yf
    hist = yf.Ticker("SPY").history(period="max", interval="1d", auto_adjust=True)
    if hist is None or hist.empty:
        raise SystemExit("no SPY history")
    closes = hist["Close"].astype(float).values
    dates = [d.strftime("%Y-%m-%d") for d in hist.index]
    return closes, dates


# ──────────────────────────────────────────────────────────────────────────────
# Equity book simulation — simple SPY-with-momentum-tilt over a window
# ──────────────────────────────────────────────────────────────────────────────
def _simulate_equity_window(
    closes: np.ndarray, dates: list[str],
    window_start: str, window_end: str,
) -> tuple[list[float], list[float]]:
    """
    Daily returns + equity curve for an SPY momentum-tilted book over
    the requested window.  Returns ([daily_returns_pct], [equity_curve]).

    The "momentum tilt" is a simple proxy: long full SPY when 21d-mom
    > 0, half-weight when 21d-mom < 0.  This isn't a serious alpha
    model — it's an asymmetric defensive overlay (cuts equity exposure
    during downtrends) that mirrors what an active halal-equity ranker
    would do at the basket level.
    """
    if len(closes) < 30:
        return [], []

    pct_returns = np.diff(closes) / closes[:-1] * 100  # daily %
    in_window_returns: list[float] = []
    nav = 100_000.0
    in_window_curve: list[float] = []
    daily_drag = EQUITY_DAILY_DRAG_BPS / 100.0
    fb_iso = window_start

    for i in range(22, len(pct_returns)):
        date_idx_in_closes = i + 1
        if date_idx_in_closes >= len(dates):
            break
        d_iso = dates[date_idx_in_closes]
        # Momentum tilt: full weight when 21d-mom > 0, half-weight otherwise.
        mom_21d = (closes[i] - closes[i - 21]) / max(closes[i - 21], 1e-9)
        weight = 1.0 if mom_21d > 0 else 0.5

        # Daily P&L
        day_ret_pct = float(pct_returns[i]) * weight - daily_drag
        nav *= (1.0 + day_ret_pct / 100.0)

        if window_start <= d_iso <= window_end:
            in_window_returns.append(day_ret_pct)
            in_window_curve.append(nav)

    return in_window_returns, in_window_curve


# ──────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ──────────────────────────────────────────────────────────────────────────────
def _stats(returns: list[float], curve: list[float]) -> dict:
    if len(returns) < 5:
        return {"n_days": len(returns), "error": "insufficient"}

    ret = np.asarray(returns)
    cum = (np.prod(1 + ret / 100) - 1.0) * 100
    n = len(ret)
    annualised = ((1 + cum / 100) ** (252 / n) - 1.0) * 100
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

    return {
        "n_days":           n,
        "cum_return_pct":   round(float(cum), 2),
        "annualised_pct":   round(float(annualised), 2),
        "ann_vol_pct":      round(float(sd * SQ252), 2),
        "sharpe":           round(sharpe, 3) if sharpe is not None else None,
        "max_drawdown_pct": round(max_dd, 2),
        "win_days_pct":     round(win_days, 2),
    }


def _combine_returns(
    metals_returns: list[float], equity_returns: list[float],
    w_metals: float, w_equity: float,
    *,
    metals_tiers: list[str] | None = None,
    adaptive: bool = False,
) -> tuple[list[float], list[float], list[float]]:
    """
    Align the two series by index (pads shorter with zeros) and combine
    at the target weights.

    Returns (combined_returns_pct, equity_curve, w_metals_series).
    With ``adaptive=True`` and ``metals_tiers`` provided, weights are
    driven by `regime_adaptive_allocator` instead of the static values.
    """
    n = max(len(metals_returns), len(equity_returns))
    m_arr = np.zeros(n)
    e_arr = np.zeros(n)
    if metals_returns:
        m_arr[-len(metals_returns):] = metals_returns
    if equity_returns:
        e_arr[-len(equity_returns):] = equity_returns

    if adaptive and metals_tiers:
        # Align tiers to the combined index — pad earliest entries with NORMAL.
        tiers = ["NORMAL"] * n
        offset = n - len(metals_tiers)
        for i, t in enumerate(metals_tiers):
            idx = i + offset
            if 0 <= idx < n:
                tiers[idx] = t
        w_m_series, w_e_series = adaptive_series(tiers, smoothing=True)
    else:
        w_m_series = [w_metals] * n
        w_e_series = [w_equity] * n

    combined = [w_m_series[i] * m_arr[i] + w_e_series[i] * e_arr[i] for i in range(n)]
    nav = 100_000.0
    curve = []
    for r in combined:
        nav *= (1.0 + r / 100.0)
        curve.append(nav)
    return combined, curve, w_m_series


# ──────────────────────────────────────────────────────────────────────────────
# Verdict
# ──────────────────────────────────────────────────────────────────────────────
def _aggregate_verdict(window_results: list[dict]) -> tuple[str, str]:
    verdicts = [w["combined_verdict"] for w in window_results
                if w.get("combined_verdict") not in (None, "INSUFFICIENT_DATA")]
    n = len(verdicts)
    if n == 0:
        return "INSUFFICIENT_DATA", "No usable windows."

    n_strong = verdicts.count("STRONG")
    n_pass   = verdicts.count("PASS")
    n_deg    = verdicts.count("DEGRADED")
    n_fail   = verdicts.count("FAIL")
    sharpes = [w["combined"]["sharpe"] for w in window_results
               if w.get("combined", {}).get("sharpe") is not None]
    avg_sharpe = float(np.mean(sharpes)) if sharpes else 0.0

    n_diversification_helped = sum(
        1 for w in window_results
        if w.get("diversification_benefit", {}).get("sharpe_delta", 0) > 0.2
    )

    if n_fail == 0 and avg_sharpe >= 0.8:
        verdict = "ROBUST"
        note = (
            f"Combined book passes all {n} crisis windows; avg Sharpe "
            f"{avg_sharpe:.2f}.  Diversification rescued {n_diversification_helped} regime(s)."
        )
    elif n_fail <= 1 and avg_sharpe >= 0.5:
        verdict = "REGIME_SENSITIVE"
        note = (
            f"{n - n_fail}/{n} windows pass; avg Sharpe {avg_sharpe:.2f}. "
            f"Diversification rescued {n_diversification_helped} window(s)."
        )
    elif n_fail >= 3:
        verdict = "REGIME_FRAGILE"
        note = f"{n - n_fail}/{n} pass; avg Sharpe {avg_sharpe:.2f}. Diversification didn't save enough."
    else:
        verdict = "REGIME_SENSITIVE"
        note = f"{n - n_fail}/{n} pass; avg Sharpe {avg_sharpe:.2f}."

    return verdict, note


# ──────────────────────────────────────────────────────────────────────────────
# Public entry
# ──────────────────────────────────────────────────────────────────────────────
def run_multi_asset_stress(
    w_metals: float = DEFAULT_METALS_WEIGHT,
    w_equity: float = DEFAULT_EQUITY_WEIGHT,
    adaptive: bool = False,
) -> dict:
    metals_closes, metals_dates = _fetch_full_history("GC=F")
    spy_closes,    spy_dates    = _fetch_spy_history()

    window_results: list[dict] = []
    for w in CRISIS_WINDOWS:
        metals_sliced, metals_d = _slice_window(metals_closes, metals_dates, w["start"], w["end"])
        spy_sliced,    spy_d    = _slice_window(spy_closes,    spy_dates,    w["start"], w["end"])

        # Need both legs to have data inside the window
        if len(metals_sliced) < 60 or len(spy_sliced) < 30:
            window_results.append({
                "label":  w["label"],
                "start":  w["start"],
                "end":    w["end"],
                "regime": w["regime"],
                "combined_verdict": "INSUFFICIENT_DATA",
                "note":             "insufficient history in one leg",
            })
            continue

        # Metals simulation (full Phase XIV-XIX stack)
        metals_stats = _simulate_window(
            metals_sliced, metals_d, w["start"], w["end"],
        )
        if "error" in metals_stats:
            window_results.append({
                "label":  w["label"],
                "start":  w["start"],
                "end":    w["end"],
                "regime": w["regime"],
                "combined_verdict": "INSUFFICIENT_DATA",
                "note":             "metals simulation insufficient",
            })
            continue

        # Equity simulation
        eq_returns, eq_curve = _simulate_equity_window(
            spy_sliced, spy_d, w["start"], w["end"],
        )
        equity_stats = _stats(eq_returns, eq_curve)

        # Combined
        # Reconstruct daily metals returns + per-day crisis tiers from a
        # full re-simulation (we need both to drive the adaptive allocator).
        metals_returns, metals_tiers = _simulate_metals_returns_and_tiers(
            metals_sliced, metals_d, w["start"], w["end"],
        )
        combined_returns, combined_curve, w_m_series = _combine_returns(
            metals_returns, eq_returns, w_metals, w_equity,
            metals_tiers=metals_tiers, adaptive=adaptive,
        )
        combined_stats = _stats(combined_returns, combined_curve)
        avg_metals_weight = float(sum(w_m_series) / len(w_m_series)) if w_m_series else w_metals

        # Diversification benefit
        metals_sharpe   = metals_stats.get("sharpe") or 0.0
        combined_sharpe = combined_stats.get("sharpe") or 0.0
        metals_dd       = metals_stats.get("max_drawdown_pct") or 0.0
        combined_dd     = combined_stats.get("max_drawdown_pct") or 0.0
        sharpe_delta = combined_sharpe - metals_sharpe
        dd_delta     = combined_dd - metals_dd

        combined_verdict = _window_verdict(combined_stats)

        window_results.append({
            "label":   w["label"],
            "start":   w["start"],
            "end":     w["end"],
            "regime":  w["regime"],
            "metals":  metals_stats,
            "equity":  equity_stats,
            "combined": combined_stats,
            "combined_verdict": combined_verdict,
            "avg_metals_weight": round(avg_metals_weight, 3),
            "diversification_benefit": {
                "sharpe_delta":      round(sharpe_delta, 3),
                "max_dd_delta_pp":   round(dd_delta, 2),
                "rescued_a_fail":    metals_stats.get("sharpe", 0) < 0 and combined_sharpe > 0.3,
            },
        })

    aggregate_verdict, aggregate_note = _aggregate_verdict(window_results)

    valid = [w for w in window_results if "sharpe" in (w.get("combined") or {})
             and (w.get("combined") or {}).get("sharpe") is not None]
    avg_combined_sharpe = float(np.mean([w["combined"]["sharpe"] for w in valid])) if valid else 0.0
    avg_metals_sharpe   = float(np.mean([w["metals"]["sharpe"] for w in valid if w["metals"].get("sharpe") is not None])) if valid else 0.0

    out = {
        "schema_version": "1.0",
        "engine":         "multi_asset_stress_backtester",
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "weights":        {"metals": w_metals, "equity": w_equity, "adaptive": bool(adaptive)},
        "equity_proxy":   "SPY (momentum-tilted: full when 21d-mom>0, half otherwise)",
        "n_windows":      len(window_results),
        "n_valid":        len(valid),
        "aggregate": {
            "verdict":              aggregate_verdict,
            "note":                 aggregate_note,
            "avg_combined_sharpe":  round(avg_combined_sharpe, 3),
            "avg_metals_sharpe":    round(avg_metals_sharpe, 3),
            "avg_sharpe_lift":      round(avg_combined_sharpe - avg_metals_sharpe, 3),
            "n_windows_rescued":    sum(1 for w in window_results
                                        if (w.get("diversification_benefit") or {}).get("rescued_a_fail")),
        },
        "windows": window_results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(out, indent=2, default=str))
    return out


def _simulate_metals_returns_and_tiers(
    closes: np.ndarray, dates: list[str],
    window_start: str, window_end: str,
) -> tuple[list[float], list[str]]:
    """
    Run the same logic as `stress_backtester._simulate_window` but
    return (in_window_returns, in_window_crisis_tiers) instead of the
    aggregated stats.  Both are needed to drive the regime-adaptive
    allocator across the window.
    """
    from scripts.strategy_backtester import (
        _classify_regime_proxy,
        _classify_vol_regime,
        _extension_z,
        _final_size_pct,
        _select_strategy,
        _short_trend_specialist,
        _strategy_return,
        _technical_conviction,
    )
    from scripts.crisis_detector import (
        apply_guard as _crisis_apply_guard,
        classify_from_prices as _crisis_classify,
    )

    if len(closes) < 60:
        return [], []

    log_returns = np.diff(np.log(closes))
    pct_returns = np.diff(closes) / closes[:-1] * 100

    rv_21 = np.full_like(log_returns, np.nan, dtype=float)
    rv_63 = np.full_like(log_returns, np.nan, dtype=float)
    for i in range(21, len(log_returns)):
        rv_21[i] = float(np.std(log_returns[i - 21:i], ddof=0)) * SQ252 * 100
    for i in range(63, len(log_returns)):
        rv_63[i] = float(np.std(log_returns[i - 63:i], ddof=0)) * SQ252 * 100

    def _rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
        out = np.full_like(arr, np.nan, dtype=float)
        csum = np.cumsum(np.insert(arr, 0, 0))
        out[w - 1:] = (csum[w:] - csum[:-w]) / w
        return out
    drift_5  = _rolling_mean(log_returns, 5)
    drift_21 = _rolling_mean(log_returns, 21)

    daily_returns: list[float] = []
    in_window: list[float] = []
    in_window_tiers: list[str] = []
    target_pct = 10.0

    for i in range(60, len(log_returns) - 1):
        rv21 = float(rv_21[i]) if i < len(rv_21) else float("nan")
        rv63 = float(rv_63[i]) if i < len(rv_63) else float("nan")
        d5   = float(drift_5[i])  if i < len(drift_5)  else 0.0
        d21  = float(drift_21[i]) if i < len(drift_21) else 0.0

        hmm = _classify_regime_proxy(rv21, d5, d21)
        vol_regime = _classify_vol_regime(rv21, rv63)
        conviction, tier, direction = _technical_conviction(closes, i + 1)
        ext_z = _extension_z(closes, i + 1)
        macro_tilt = float(np.sign(d21) * 1.2) if abs(d21) > 0.002 else 0.0
        in_short_trend = _short_trend_specialist(closes, i + 1)

        strategy = _select_strategy(
            tier=tier, direction=direction,
            vol_regime=vol_regime, hmm_state=hmm,
            conviction=conviction,
            extension_z=ext_z, macro_tilt=macro_tilt,
            short_trend=in_short_trend,
        )
        size_pct = _final_size_pct(strategy, tier, conviction, vol_regime, hmm)

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
        crisis = _crisis_classify(closes, current_idx=i + 1)
        guarded_strategy, guarded_size, _ = _crisis_apply_guard(
            strategy, adjusted, crisis["tier"],
        )
        strat_ret = _strategy_return(guarded_strategy, direction, closes, i + 1)
        net_daily = (guarded_size / 100.0) * strat_ret
        daily_returns.append(net_daily)

        d_iso = dates[i + 1]
        if window_start <= d_iso <= window_end:
            in_window.append(net_daily)
            in_window_tiers.append(crisis["tier"])

    return in_window, in_window_tiers


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metals-weight", type=float, default=DEFAULT_METALS_WEIGHT)
    ap.add_argument("--equity-weight", type=float, default=DEFAULT_EQUITY_WEIGHT)
    ap.add_argument("--adaptive", action="store_true",
                    help="Use regime-adaptive weights driven by per-day crisis tier")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    out = run_multi_asset_stress(
        w_metals=args.metals_weight,
        w_equity=args.equity_weight,
        adaptive=args.adaptive,
    )
    if args.quiet:
        return 0

    a = out["aggregate"]
    print("=" * 80)
    print(f"MULTI-ASSET STRESS BACKTEST  ({out['generated_at']})")
    print("=" * 80)
    print(f"  Weights       : {out['weights']['metals']*100:.0f}% metals / "
          f"{out['weights']['equity']*100:.0f}% equity ({out['equity_proxy']})")
    print(f"  Windows       : {out['n_valid']}/{out['n_windows']} usable")
    print()
    print(f"  AGGREGATE VERDICT: {a['verdict']}")
    print(f"  {a['note']}")
    print()
    print(f"  Avg combined Sharpe : {a['avg_combined_sharpe']:+.3f}")
    print(f"  Avg metals Sharpe   : {a['avg_metals_sharpe']:+.3f}")
    print(f"  Lift from equity    : {a['avg_sharpe_lift']:+.3f}")
    print(f"  Windows rescued     : {a['n_windows_rescued']}")
    print()
    print(f"  {'Window':<28s}  {'metals':>20s}  {'combined':>20s}  {'Δ':>10s}")
    print(f"  {'Sharpe / DD':<28s}  {'-'*20}  {'-'*20}  {'-'*10}")
    for w in out["windows"]:
        if w.get("combined_verdict") == "INSUFFICIENT_DATA":
            print(f"  {w['label']:<28s}  insufficient data")
            continue
        m = w.get("metals", {})
        c = w.get("combined", {})
        db = w.get("diversification_benefit", {})
        print(
            f"  {w['label']:<28s}  "
            f"{m.get('sharpe', 0):>+6.2f} / {m.get('max_drawdown_pct', 0):>+6.1f}%  "
            f"{c.get('sharpe', 0):>+6.2f} / {c.get('max_drawdown_pct', 0):>+6.1f}%  "
            f"{db.get('sharpe_delta', 0):>+6.2f}  "
            f"{'✓RESCUED' if db.get('rescued_a_fail') else ''}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
