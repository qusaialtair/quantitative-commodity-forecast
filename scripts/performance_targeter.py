#!/usr/bin/env python3
"""
Performance Targeter  (Phase XIV Stage 74)
===========================================
Tracks NAV progress versus the monthly performance target and emits a
"risk_multiplier" the downstream sizing logic respects.

Default target: 10% monthly  (≈ 215 bps / trading day, ≈ 124% annualised).
This is an ambitious target — most institutional shops target 10-20% annual,
so 10% monthly will require active, conviction-weighted multi-strategy
deployment.  The targeter does NOT promise the return; it just helps the
sizing engine dial aggression so we *attempt* the target while respecting
drawdown limits.

Logic per UTC trading day:

    elapsed_fraction   = trading_days_elapsed / trading_days_in_month
    expected_progress  = target_monthly_pct × elapsed_fraction
    actual_progress    = current_month_return_pct
    gap_pct            = actual_progress - expected_progress

    risk_multiplier:
        gap >  +2.0  → 0.70  (well ahead, throttle)
        gap >  +0.5  → 0.85  (ahead, mild throttle)
        |gap| ≤ 0.5  → 1.00  (on track)
        gap <  -0.5  → 1.20  (behind, lean in)
        gap <  -2.0  → 1.50  (well behind, max aggression — capped)
        gap <  -5.0  → 1.75  (very behind, but still capped by risk floors)

    Hard ceilings:
        - drawdown tier {ELEVATED, SEVERE, CRITICAL} forces multiplier ≤ 0.75
        - data_quality FAIL forces multiplier = 0.0  (paired with strategy=CASH)
        - 21-day realised vol > 35% forces multiplier ≤ 0.90 (defensive)

Reads:
    data/pnl_tracker.json     for NAV history (we look at month-to-date)
    data/drawdown_controller.json for ceiling
    data/data_quality.json    for kill-switch
    data/vol_surface.json     for vol-defensive cap
    data/portfolio.json + data/virtual_account.json + data/pipeline_state.json
        as redundant NAV sources

Output: data/performance_targeter.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "performance_targeter.json"

DEFAULT_MONTHLY_TARGET_PCT = 10.0
TRADING_DAYS_PER_MONTH = 21
ANNUAL_TARGET_FROM_MONTHLY = lambda m: ((1.0 + m / 100.0) ** 12 - 1.0) * 100.0


def _load(name: str) -> dict:
    p = DATA_DIR / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _safe(v, d=0.0) -> float:
    try:
        if v is None:
            return d
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return d
        return f
    except Exception:
        return d


def _month_first_business_day(today: date) -> date:
    d = today.replace(day=1)
    while d.weekday() >= 5:  # Sat=5, Sun=6
        d = d.replace(day=d.day + 1)
    return d


def _business_days_between(start: date, end: date) -> int:
    if end < start:
        return 0
    n = 0
    d = start
    while d <= end:
        if d.weekday() < 5:
            n += 1
        d = date.fromordinal(d.toordinal() + 1)
    return n


NAV_HISTORY_CSV = DATA_DIR / "nav_history.csv"


def _read_mtd_return() -> tuple[float, float, float, int]:
    """
    Returns (mtd_return_pct, mtd_nav_start, mtd_nav_now, n_history_rows).

    Uses data/nav_history.csv (the canonical NAV time-series).
    """
    if not NAV_HISTORY_CSV.exists():
        pn = _load("pnl_tracker.json")
        nav_now = _safe(pn.get("latest_nav_usd"), 0.0)
        return 0.0, nav_now, nav_now, 0

    try:
        rows = []
        with NAV_HISTORY_CSV.open() as f:
            header = f.readline().strip().split(",")
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < len(header):
                    continue
                row = dict(zip(header, parts))
                rows.append(row)
        if not rows:
            return 0.0, 0.0, 0.0, 0
    except Exception:
        return 0.0, 0.0, 0.0, 0

    rows.sort(key=lambda r: r.get("date", ""))
    today = date.today()
    first_business = _month_first_business_day(today)
    fb_iso = first_business.isoformat()
    in_month = [r for r in rows if r.get("date", "") >= fb_iso]
    if not in_month:
        nav_now = _safe(rows[-1].get("nav_usd"), 0.0)
        return 0.0, nav_now, nav_now, len(rows)
    nav_start = _safe(in_month[0].get("nav_usd"), 0.0)
    nav_now = _safe(in_month[-1].get("nav_usd"), 0.0)
    if nav_start <= 0:
        return 0.0, nav_start, nav_now, len(in_month)
    ret_pct = (nav_now / nav_start - 1.0) * 100.0
    return ret_pct, nav_start, nav_now, len(in_month)


def _classify_track(gap_pct: float) -> str:
    if gap_pct > 2.0:
        return "AHEAD"
    if gap_pct > 0.5:
        return "ON_TRACK_AHEAD"
    if gap_pct >= -0.5:
        return "ON_TRACK"
    if gap_pct >= -2.0:
        return "BEHIND"
    if gap_pct >= -5.0:
        return "WELL_BEHIND"
    return "CRITICALLY_BEHIND"


def _multiplier_from_gap(gap_pct: float) -> float:
    if gap_pct > 2.0:
        return 0.70
    if gap_pct > 0.5:
        return 0.85
    if gap_pct >= -0.5:
        return 1.00
    if gap_pct >= -2.0:
        return 1.20
    if gap_pct >= -5.0:
        return 1.50
    return 1.75


def run_performance_targeter(target_monthly_pct: float | None = None) -> dict:
    target = float(target_monthly_pct or DEFAULT_MONTHLY_TARGET_PCT)
    today = date.today()
    first_business = _month_first_business_day(today)
    days_elapsed = max(1, _business_days_between(first_business, today))
    elapsed_fraction = min(1.0, days_elapsed / TRADING_DAYS_PER_MONTH)

    mtd_return_pct, mtd_start_nav, mtd_now_nav, n_hist = _read_mtd_return()
    expected_progress = target * elapsed_fraction
    gap = mtd_return_pct - expected_progress
    track = _classify_track(gap)
    raw_mult = _multiplier_from_gap(gap)

    # Ceilings
    ceilings_applied = []
    capped = raw_mult

    dd = _load("drawdown_controller.json")
    dd_tier = (dd.get("tier_name") or dd.get("tier") or "NORMAL").upper()
    if dd_tier in ("ELEVATED", "SEVERE", "CRITICAL"):
        if capped > 0.75:
            ceilings_applied.append(f"drawdown_tier={dd_tier} → ≤0.75")
            capped = 0.75

    dq = _load("data_quality.json")
    if (dq.get("overall_status") or "").upper() == "FAIL":
        ceilings_applied.append("data_quality FAIL → 0.0")
        capped = 0.0

    vs = _load("vol_surface.json")
    rv = (vs.get("term_structure") or {}).get("realised_vol", {})
    rv_21d = _safe(rv.get("21d_pct"), 0.0)
    if rv_21d > 35.0:
        if capped > 0.90:
            ceilings_applied.append(f"21d-RV={rv_21d:.1f}% > 35% → ≤0.90")
            capped = 0.90

    # Daily progress on a compounding pace
    daily_target_pct = ((1.0 + target / 100.0) ** (1.0 / TRADING_DAYS_PER_MONTH) - 1.0) * 100.0
    annual_target = ANNUAL_TARGET_FROM_MONTHLY(target)

    # Projected month-end (if current pace persists)
    if days_elapsed > 0:
        projected_full_month = mtd_return_pct * TRADING_DAYS_PER_MONTH / days_elapsed
    else:
        projected_full_month = 0.0

    out = {
        "schema_version": "1.0",
        "engine":         "performance_targeter",
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": {
            "monthly_pct":        round(target, 2),
            "daily_pct_compound": round(daily_target_pct, 3),
            "annualised_pct":     round(annual_target, 1),
            "trading_days_in_month": TRADING_DAYS_PER_MONTH,
        },
        "progress": {
            "month_first_business_day": first_business.isoformat(),
            "trading_days_elapsed":     days_elapsed,
            "elapsed_fraction":         round(elapsed_fraction, 4),
            "expected_progress_pct":    round(expected_progress, 3),
            "actual_progress_pct":      round(mtd_return_pct, 3),
            "gap_pct":                  round(gap, 3),
            "track_status":             track,
            "mtd_start_nav_usd":        round(mtd_start_nav, 2),
            "mtd_current_nav_usd":      round(mtd_now_nav, 2),
            "projected_full_month_pct": round(projected_full_month, 2),
            "n_history_rows_in_month":  n_hist,
        },
        "risk_multiplier": {
            "raw":               round(raw_mult, 3),
            "final":             round(capped, 3),
            "ceilings_applied":  ceilings_applied,
            "n_ceilings_applied": len(ceilings_applied),
        },
        "context": {
            "drawdown_tier":   dd_tier,
            "data_quality_status": (dq.get("overall_status") or "UNKNOWN").upper(),
            "rv_21d_pct":      round(rv_21d, 2),
        },
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(out, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--target", type=float, default=None,
        help="Override monthly target % (default 10.0)",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    out = run_performance_targeter(target_monthly_pct=args.target)
    if args.quiet:
        return 0
    t = out["target"]
    p = out["progress"]
    r = out["risk_multiplier"]
    print("=" * 64)
    print(f"PERFORMANCE TARGETER  ({out['generated_at']})")
    print("=" * 64)
    print(f"  Target            : {t['monthly_pct']:.2f}%/mo  "
          f"({t['daily_pct_compound']:.3f}%/d compounding ≈ {t['annualised_pct']:.0f}% ann)")
    print(f"  Day               : {p['trading_days_elapsed']}/{t['trading_days_in_month']} "
          f"({p['elapsed_fraction']*100:.1f}% of month)")
    print(f"  Expected MTD      : {p['expected_progress_pct']:+.3f}%")
    print(f"  Actual MTD        : {p['actual_progress_pct']:+.3f}%")
    print(f"  Gap               : {p['gap_pct']:+.3f}%  → {p['track_status']}")
    print(f"  Projected MoM     : {p['projected_full_month_pct']:+.2f}% if pace persists")
    print()
    print(f"  Risk multiplier   : raw={r['raw']:.2f}  final={r['final']:.2f}")
    if r["ceilings_applied"]:
        print(f"  Ceilings applied  : {', '.join(r['ceilings_applied'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
