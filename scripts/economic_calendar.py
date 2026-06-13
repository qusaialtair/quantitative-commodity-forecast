#!/usr/bin/env python3
"""
Economic Event Calendar  (Phase XIII Stage 67)
================================================
Tracks high-impact macro events (FOMC, CPI, NFP, GDP, PMI) and emits a
"position guard" signal that blocks or sizes-down new exposure inside a
configurable pre/post-event cooldown window.

Event list is built from US official 2026 schedules; can be extended.

For each event the engine computes:
  - days_until        signed, negative = past
  - in_blackout       True if within [-cooldown_pre, +cooldown_post]
  - severity          HIGH (FOMC, CPI), MEDIUM (NFP, GDP), LOW (PMI)
  - action_recommend  HOLD / SIZE_DOWN / NORMAL

Top-level output:
  position_guard      most restrictive across all near-term events
  next_event          earliest upcoming event with countdown
  blocked_today       True if any HIGH event in blackout

Output: data/economic_calendar.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, date, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "economic_calendar.json"

# Cooldown windows (calendar days) per severity
COOLDOWN_PRE = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
COOLDOWN_POST = {"HIGH": 1, "MEDIUM": 1, "LOW": 0}

# 2026 US event schedule (best-effort published dates)
EVENTS_2026 = [
    # FOMC rate decisions (8 / year)
    ("2026-01-28", "FOMC_DECISION", "HIGH"),
    ("2026-03-18", "FOMC_DECISION", "HIGH"),
    ("2026-04-29", "FOMC_DECISION", "HIGH"),
    ("2026-06-17", "FOMC_DECISION", "HIGH"),
    ("2026-07-29", "FOMC_DECISION", "HIGH"),
    ("2026-09-16", "FOMC_DECISION", "HIGH"),
    ("2026-10-28", "FOMC_DECISION", "HIGH"),
    ("2026-12-09", "FOMC_DECISION", "HIGH"),
    # CPI (monthly, mid-month US BLS)
    ("2026-01-13", "CPI_RELEASE", "HIGH"),
    ("2026-02-11", "CPI_RELEASE", "HIGH"),
    ("2026-03-11", "CPI_RELEASE", "HIGH"),
    ("2026-04-14", "CPI_RELEASE", "HIGH"),
    ("2026-05-13", "CPI_RELEASE", "HIGH"),
    ("2026-06-10", "CPI_RELEASE", "HIGH"),
    ("2026-07-15", "CPI_RELEASE", "HIGH"),
    ("2026-08-12", "CPI_RELEASE", "HIGH"),
    ("2026-09-15", "CPI_RELEASE", "HIGH"),
    ("2026-10-14", "CPI_RELEASE", "HIGH"),
    ("2026-11-12", "CPI_RELEASE", "HIGH"),
    ("2026-12-10", "CPI_RELEASE", "HIGH"),
    # NFP (first Friday of each month, ~)
    ("2026-01-09", "NFP_RELEASE", "MEDIUM"),
    ("2026-02-06", "NFP_RELEASE", "MEDIUM"),
    ("2026-03-06", "NFP_RELEASE", "MEDIUM"),
    ("2026-04-03", "NFP_RELEASE", "MEDIUM"),
    ("2026-05-01", "NFP_RELEASE", "MEDIUM"),
    ("2026-06-05", "NFP_RELEASE", "MEDIUM"),
    ("2026-07-02", "NFP_RELEASE", "MEDIUM"),
    ("2026-08-07", "NFP_RELEASE", "MEDIUM"),
    ("2026-09-04", "NFP_RELEASE", "MEDIUM"),
    ("2026-10-02", "NFP_RELEASE", "MEDIUM"),
    ("2026-11-06", "NFP_RELEASE", "MEDIUM"),
    ("2026-12-04", "NFP_RELEASE", "MEDIUM"),
    # GDP advance estimate (quarterly, end of month after quarter close)
    ("2026-01-29", "GDP_ADVANCE", "MEDIUM"),
    ("2026-04-30", "GDP_ADVANCE", "MEDIUM"),
    ("2026-07-30", "GDP_ADVANCE", "MEDIUM"),
    ("2026-10-29", "GDP_ADVANCE", "MEDIUM"),
    # ISM Manufacturing PMI (first business day of month)
    ("2026-01-02", "ISM_PMI", "LOW"),
    ("2026-02-02", "ISM_PMI", "LOW"),
    ("2026-03-02", "ISM_PMI", "LOW"),
    ("2026-04-01", "ISM_PMI", "LOW"),
    ("2026-05-01", "ISM_PMI", "LOW"),
    ("2026-06-01", "ISM_PMI", "LOW"),
    ("2026-07-01", "ISM_PMI", "LOW"),
    ("2026-08-03", "ISM_PMI", "LOW"),
    ("2026-09-01", "ISM_PMI", "LOW"),
    ("2026-10-01", "ISM_PMI", "LOW"),
    ("2026-11-02", "ISM_PMI", "LOW"),
    ("2026-12-01", "ISM_PMI", "LOW"),
]

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Calculations
# ---------------------------------------------------------------------------
def _in_blackout(days_until: int, severity: str) -> bool:
    if days_until > 0 and days_until <= COOLDOWN_PRE.get(severity, 0):
        return True
    if days_until < 0 and abs(days_until) <= COOLDOWN_POST.get(severity, 0):
        return True
    return days_until == 0  # event today


def _action_for_event(event: dict) -> str:
    if not event["in_blackout"]:
        return "NORMAL"
    if event["severity"] == "HIGH":
        return "HOLD"
    if event["severity"] == "MEDIUM":
        return "SIZE_DOWN"
    return "NORMAL"


def run_economic_calendar(lookahead_days: int = 14) -> dict:
    today = date.today()
    cutoff = today + timedelta(days=lookahead_days)

    events = []
    for ev_date_str, kind, severity in EVENTS_2026:
        try:
            ev_date = date.fromisoformat(ev_date_str)
        except Exception:
            continue
        days_until = (ev_date - today).days
        if days_until < -7 or days_until > lookahead_days + 7:
            continue
        in_blackout = _in_blackout(days_until, severity)
        ev = {
            "date":         ev_date_str,
            "kind":         kind,
            "severity":     severity,
            "days_until":   days_until,
            "in_blackout":  in_blackout,
        }
        ev["action"] = _action_for_event(ev)
        events.append(ev)

    # Sort by absolute proximity
    events.sort(key=lambda e: abs(e["days_until"]))

    # Find earliest UPCOMING event
    upcoming = [e for e in events if e["days_until"] >= 0]
    next_event = upcoming[0] if upcoming else None

    # Position guard = most restrictive active blackout
    blackout_events = [e for e in events if e["in_blackout"]]
    if any(e["severity"] == "HIGH" for e in blackout_events):
        position_guard = "HOLD_NEW_POSITIONS"
        guard_severity = "HIGH"
    elif any(e["severity"] == "MEDIUM" for e in blackout_events):
        position_guard = "SIZE_DOWN_NEW_POSITIONS"
        guard_severity = "MEDIUM"
    else:
        position_guard = "NORMAL"
        guard_severity = "LOW"

    result = {
        "generated_at":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "today":            today.isoformat(),
        "lookahead_days":   lookahead_days,
        "events":           events,
        "n_events_window":  len(events),
        "n_blackout_active":len(blackout_events),
        "blocked_today":    any(e["days_until"] == 0 and e["severity"] == "HIGH" for e in events),
        "position_guard":   position_guard,
        "guard_severity":   guard_severity,
        "next_event":       next_event,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2, default=str))
    _print_report(result)
    return result


def _print_report(r: dict) -> None:
    guard_color = {
        "NORMAL":                  "\033[32;1m",
        "SIZE_DOWN_NEW_POSITIONS": "\033[33m",
        "HOLD_NEW_POSITIONS":      "\033[31;1m",
    }.get(r["position_guard"], "\033[0m")

    print(f"\n{SEP}\n  ECONOMIC EVENT CALENDAR\n{SEP}")
    print(f"  Today:           {r['today']}")
    print(f"  Position guard:  {guard_color}{r['position_guard']}\033[0m")
    print(f"  Blocked today:   {r['blocked_today']}")
    print(f"  Events in {r['lookahead_days']}d window: {r['n_events_window']}")
    print(f"  Active blackouts:{r['n_blackout_active']}")
    print()

    ne = r.get("next_event")
    if ne:
        print(f"  NEXT EVENT")
        print(f"    {ne['date']}  {ne['kind']}  [{ne['severity']}]  T-{ne['days_until']}d")
    print()

    if r["events"]:
        print(f"  WINDOW (±7d ±lookahead)")
        print(f"  {'─' * 60}")
        for ev in r["events"][:12]:
            mark = " ▶" if ev["in_blackout"] else "  "
            sev_color = {"HIGH": "\033[31m", "MEDIUM": "\033[33m", "LOW": "\033[36m"}.get(ev["severity"], "")
            print(
                f"  {mark} {ev['date']}  "
                f"T{ev['days_until']:+d}d  "
                f"{sev_color}{ev['severity']:<7s}\033[0m  "
                f"{ev['kind']:<14s}  →  {ev['action']}"
            )
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Economic Calendar")
    parser.add_argument("--lookahead", type=int, default=14)
    args = parser.parse_args()
    run_economic_calendar(lookahead_days=args.lookahead)
