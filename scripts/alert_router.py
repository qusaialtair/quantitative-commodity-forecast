#!/usr/bin/env python3
"""
Alert Router  (Phase IX Stage 48)
===================================
Centralised event-to-channel dispatcher. Subscribes to the daily
pipeline_state.json + DCC / drawdown / regime files and sends a notification
when a watch condition fires.

Channels:
  - Telegram  (uses existing scripts/telegram_notifier.send_urgent)
  - Console   (always; for development)
  - Email     (stub — wire SMTP creds via env if needed)

Watch conditions (configurable):
  - DD escalation:        drawdown_tier transitions to CAUTION / DEFENSIVE / etc.
  - Regime change:        HMM state transitions (e.g. BULLISH → VOLATILE)
  - Structural break:     CUSUM flagged
  - Geopolitical event:   priority HIGH
  - DCC stress:           any pair flagged stressed
  - Vol regime escalation: vol regime transitions to ELEVATED / EXTREME
  - Pipeline ABORTED

State persistence: data/alert_state.json (only fires on transitions).

Output: data/alert_router.json (last firing decisions)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "alert_router.json"
STATE_FILE = DATA_DIR / "alert_state.json"
PIPELINE_STATE = DATA_DIR / "pipeline_state.json"

LINE_W = 62
SEP = "━" * LINE_W


def _safe_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _load_state() -> dict:
    return _safe_json(STATE_FILE)


def _save_state(state: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def _send_telegram(title: str, body: str, urgent: bool = False) -> bool:
    try:
        if urgent:
            from scripts.telegram_notifier import send_urgent
            send_urgent(title, body, context="alert_router")
        else:
            from scripts.telegram_notifier import send_heartbeat
            send_heartbeat({"alert": title, "body": body})
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Watchers
# ---------------------------------------------------------------------------
def _check_pipeline_status(ps: dict, prev_state: dict) -> list:
    alerts = []
    status = ps.get("pipeline_status")
    last = prev_state.get("pipeline_status")
    if status == "ABORTED" and last != "ABORTED":
        alerts.append({
            "severity": "CRITICAL",
            "title":    "Pipeline ABORTED",
            "body":     ps.get("abort_reason", "no reason given"),
        })
    return alerts


def _check_drawdown(ps: dict, prev_state: dict) -> list:
    alerts = []
    dd = ps.get("drawdown_tier", {})
    tier_now = dd.get("tier_name")
    tier_last = prev_state.get("drawdown_tier_name")
    if tier_now and tier_last and tier_now != tier_last:
        worse = ["CAUTION", "DEFENSIVE", "CRITICAL", "EMERGENCY"]
        if tier_now in worse:
            alerts.append({
                "severity": "HIGH",
                "title":    f"Drawdown tier → {tier_now}",
                "body":     (
                    f"Was {tier_last}; now {tier_now}. "
                    f"Sizing × {dd.get('sizing_multiplier', 1.0):.2f}; "
                    f"DD {dd.get('current_dd_pct', 0):+.2f}%. "
                    f"Action: {dd.get('action', 'n/a')}"
                ),
            })
    return alerts


def _check_regime(ps: dict, prev_state: dict) -> list:
    alerts = []
    state_now = ps.get("regime", {}).get("hmm_state")
    state_last = prev_state.get("hmm_state")
    if state_now and state_last and state_now != state_last:
        alerts.append({
            "severity": "MEDIUM",
            "title":    f"HMM regime → {state_now}",
            "body":     f"Was {state_last}; now {state_now}.",
        })
    return alerts


def _check_structural_break(_: dict, prev_state: dict) -> list:
    alerts = []
    sb = _safe_json(DATA_DIR / "structural_breaks.json")
    if not sb:
        return alerts
    cusum = sb.get("cusum", {})
    is_break = cusum.get("break_detected", False)
    was_break = prev_state.get("cusum_break", False)
    if is_break and not was_break:
        alerts.append({
            "severity": "HIGH",
            "title":    "CUSUM break detected",
            "body":     f"Test stat {cusum.get('test_stat', 0):.3f} > "
                        f"{cusum.get('critical_value', 0):.3f}; "
                        f"break date {cusum.get('break_date', 'n/a')}",
        })
    return alerts


def _check_geopolitical(_: dict, prev_state: dict) -> list:
    alerts = []
    ge = _safe_json(DATA_DIR / "geopolitical_events.json")
    if not ge:
        return alerts
    priority = ge.get("priority")
    last_priority = prev_state.get("geo_priority")
    if priority == "HIGH" and last_priority != "HIGH":
        alerts.append({
            "severity": "HIGH",
            "title":    "Geopolitical event",
            "body":     f"Geo regime {ge.get('regime')}; "
                        f"score {ge.get('current_score', 0):.2f}; "
                        f"Δ {ge.get('delta_dod', 0):+.3f}",
        })
    return alerts


def _check_dcc_stress(_: dict, prev_state: dict) -> list:
    alerts = []
    dcc = _safe_json(DATA_DIR / "dcc_garch.json")
    if not dcc:
        return alerts
    n_stress = dcc.get("n_stressed", 0)
    last = prev_state.get("dcc_n_stressed", 0)
    if n_stress > 0 and n_stress > last:
        alerts.append({
            "severity": "MEDIUM",
            "title":    f"DCC correlation stress ({n_stress} pairs)",
            "body":     "Stressed: " + ", ".join(dcc.get("stressed_pairs", [])[:3]),
        })
    return alerts


def _check_vol_surface(_: dict, prev_state: dict) -> list:
    alerts = []
    vs = _safe_json(DATA_DIR / "vol_surface.json")
    if not vs:
        return alerts
    regime = vs.get("vol_regime")
    last = prev_state.get("vol_regime")
    if regime in ("ELEVATED", "EXTREME") and last not in ("ELEVATED", "EXTREME"):
        alerts.append({
            "severity": "MEDIUM" if regime == "ELEVATED" else "HIGH",
            "title":    f"Vol regime → {regime}",
            "body":     f"Was {last or 'n/a'}; now {regime} "
                        f"(rv_21d {vs.get('term_structure', {}).get('rv_21d', 0):.2f}%)",
        })
    return alerts


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def run_alert_router(send: bool = True) -> dict:
    ps = _safe_json(PIPELINE_STATE)
    prev_state = _load_state()

    alerts = []
    alerts.extend(_check_pipeline_status(ps, prev_state))
    alerts.extend(_check_drawdown(ps, prev_state))
    alerts.extend(_check_regime(ps, prev_state))
    alerts.extend(_check_structural_break(ps, prev_state))
    alerts.extend(_check_geopolitical(ps, prev_state))
    alerts.extend(_check_dcc_stress(ps, prev_state))
    alerts.extend(_check_vol_surface(ps, prev_state))

    delivery_log = []
    for a in alerts:
        urgent = a["severity"] in ("HIGH", "CRITICAL")
        delivered = False
        if send:
            delivered = _send_telegram(a["title"], a["body"], urgent=urgent)
        # Always print console
        print(f"  [{a['severity']:>8s}] {a['title']}")
        print(f"             {a['body']}")
        delivery_log.append({
            **a,
            "telegram_sent": delivered,
            "ts":            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })

    # Update state
    new_state = {
        "pipeline_status":   ps.get("pipeline_status"),
        "drawdown_tier_name":ps.get("drawdown_tier", {}).get("tier_name"),
        "hmm_state":         ps.get("regime", {}).get("hmm_state"),
        "cusum_break":       _safe_json(DATA_DIR / "structural_breaks.json")
                                 .get("cusum", {}).get("break_detected", False),
        "geo_priority":      _safe_json(DATA_DIR / "geopolitical_events.json")
                                 .get("priority"),
        "dcc_n_stressed":    _safe_json(DATA_DIR / "dcc_garch.json").get("n_stressed", 0),
        "vol_regime":        _safe_json(DATA_DIR / "vol_surface.json").get("vol_regime"),
        "last_check":        datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _save_state(new_state)

    result = {
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_alerts":       len(alerts),
        "alerts":         delivery_log,
        "previous_state": prev_state,
        "new_state":      new_state,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alert Router")
    parser.add_argument("--no-send", action="store_true",
                        help="Don't send to Telegram; just print")
    args = parser.parse_args()
    print(f"\n{SEP}\n  ALERT ROUTER\n{SEP}")
    r = run_alert_router(send=not args.no_send)
    if r["n_alerts"] == 0:
        print(f"  No state changes — no alerts to dispatch.")
    print(f"\n  Saved: {OUTPUT_FILE}")
    print(SEP)
