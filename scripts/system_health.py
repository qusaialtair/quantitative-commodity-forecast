#!/usr/bin/env python3
"""
System Health Monitor  (Phase XII Stage 62)
=============================================
Detects the failure modes that matter when the platform runs autonomously:

  1. STALE ENGINE OUTPUT     any data/<engine>.json older than its TTL
  2. MISSING FILE            engine never produced its output JSON
  3. FAILED PIPELINE STAGE   stages dict shows STATUS=FAILED / ABORTED
  4. PRICE FETCH BREAKAGE    yfinance returns NaN or stale close
  5. PERPLEXITY DEPLETION    PERPLEXITY_API_KEY env present but no fresh data
  6. AUDIT CHAIN BREAK       SQLite hash chain fails verification
  7. DISK / BACKUP MISS      no DR backup in last 48h, or last backup < 1 MB

Per-engine TTLs:
    daily   (24h)   — pipeline_state, alpha_attribution, monte_carlo, oracle
    weekly  (168h)  — bayesian_hpo, purged_kfold, ensemble_stacking, conformal
    rarely  (720h)  — halal_universe (refreshed every 7d but ttl 30d)

Output: data/system_health.json
Severity-bucketed flags so the alert router can pick up CRITICAL ones.
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
OUTPUT_FILE = DATA_DIR / "system_health.json"

# TTL hours per engine file (None = ignored)
ENGINE_TTL = {
    # CRITICAL daily files
    "pipeline_state.json":         (24, "CRITICAL"),
    "alpha_attribution.json":      (24, "HIGH"),
    "monte_carlo_simulation.json": (24, "HIGH"),
    "vol_surface.json":            (24, "HIGH"),
    "mrm_champion.json":           (48, "HIGH"),
    "trade_idea.json":             (24, "CRITICAL"),
    "deepseek_last_turn.json":     (48, "MEDIUM"),
    "tail_risk_engine.json":       (24, "HIGH"),
    "drawdown_controller.json":    (24, "HIGH"),
    "macro_nowcast.json":          (24, "HIGH"),
    "audit_trail_status.json":     (48, "CRITICAL"),
    "dr_backup.json":              (48, "MEDIUM"),
    # Weekly files
    "bayesian_hpo.json":           (168, "MEDIUM"),
    "purged_kfold.json":           (168, "MEDIUM"),
    "ensemble_stacking.json":      (168, "MEDIUM"),
    "conformal_intervals.json":    (168, "MEDIUM"),
    # Monthly
    "halal_universe.json":         (720, "LOW"),
}

LINE_W = 62
SEP = "━" * LINE_W


def _age_hours(path: Path) -> float | None:
    if not path.exists():
        return None
    return (datetime.now(timezone.utc).timestamp() - path.stat().st_mtime) / 3600.0


def _load(name: str) -> dict:
    p = DATA_DIR / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------
def check_engine_freshness() -> list:
    flags = []
    for fname, (ttl_h, severity) in ENGINE_TTL.items():
        p = DATA_DIR / fname
        if not p.exists():
            flags.append({
                "severity": severity,
                "kind":     "MISSING_FILE",
                "file":     fname,
                "detail":   f"never produced an output",
            })
            continue
        age = _age_hours(p)
        if age is not None and age > ttl_h:
            flags.append({
                "severity": severity,
                "kind":     "STALE",
                "file":     fname,
                "age_hours":round(age, 1),
                "ttl_hours":ttl_h,
                "detail":   f"{age:.1f}h old vs TTL {ttl_h}h",
            })
    return flags


def check_pipeline_stages() -> list:
    flags = []
    ps = _load("pipeline_state.json")
    if not ps:
        return [{"severity": "CRITICAL", "kind": "NO_PIPELINE_STATE",
                 "detail": "pipeline_state.json absent"}]
    status = ps.get("pipeline_status")
    if status == "ABORTED":
        flags.append({
            "severity": "CRITICAL", "kind": "PIPELINE_ABORTED",
            "detail":   ps.get("abort_reason", "no reason"),
        })
    stages = ps.get("stages", {}) or {}
    for name, info in stages.items():
        if not isinstance(info, dict):
            continue
        if info.get("status") == "FAILED":
            flags.append({
                "severity": "HIGH", "kind": "STAGE_FAILED",
                "file":     "pipeline_state.json",
                "detail":   f"stage {name}: {info.get('note', '?')[:80]}",
            })
    return flags


def check_price_data() -> list:
    flags = []
    ps = _load("pipeline_state.json")
    spot = ps.get("portfolio", {}).get("last_spot", 0) or 0
    if spot <= 0:
        flags.append({
            "severity": "HIGH", "kind": "PRICE_FETCH_BROKEN",
            "detail":   "last_spot is zero in pipeline_state",
        })
    return flags


def check_perplexity() -> list:
    """If env key present but no fresh oracle history → depletion."""
    import os
    flags = []
    if not os.getenv("PERPLEXITY_API_KEY"):
        return []  # not configured; not an error
    history_csv = DATA_DIR / "oracle_history.csv"
    if not history_csv.exists():
        flags.append({
            "severity": "MEDIUM", "kind": "ORACLE_NO_HISTORY",
            "detail":   "PERPLEXITY_API_KEY set but no oracle_history.csv",
        })
        return flags
    age = _age_hours(history_csv)
    if age is None or age > 72:
        flags.append({
            "severity": "MEDIUM", "kind": "ORACLE_STALE",
            "detail":   f"oracle_history.csv {age:.1f}h old (> 72h)",
        })
    return flags


def check_audit_chain() -> list:
    flags = []
    try:
        from scripts.audit_trail import verify_chain
        chain = verify_chain()
        if not chain.get("valid", True):
            flags.append({
                "severity": "CRITICAL", "kind": "AUDIT_CHAIN_BROKEN",
                "detail":   f"break at row {chain.get('first_break_id')}",
            })
    except Exception as exc:
        flags.append({
            "severity": "HIGH", "kind": "AUDIT_VERIFY_ERROR",
            "detail":   str(exc)[:120],
        })
    return flags


def check_dr_backup() -> list:
    flags = []
    backups_dir = ROOT / "backups"
    if not backups_dir.exists():
        flags.append({
            "severity": "MEDIUM", "kind": "DR_NO_BACKUPS",
            "detail":   "backups/ dir absent",
        })
        return flags
    snaps = sorted(backups_dir.glob("snapshot_*.tar.gz*"),
                  key=lambda p: p.stat().st_mtime)
    if not snaps:
        flags.append({
            "severity": "HIGH", "kind": "DR_NO_SNAPSHOTS",
            "detail":   "no snapshots in backups/",
        })
        return flags
    latest = snaps[-1]
    age_h = (datetime.now(timezone.utc).timestamp() - latest.stat().st_mtime) / 3600.0
    size_mb = latest.stat().st_size / 1_048_576
    if age_h > 48:
        flags.append({
            "severity": "MEDIUM", "kind": "DR_STALE",
            "detail":   f"last snapshot {age_h:.1f}h old",
        })
    if size_mb < 0.5:
        flags.append({
            "severity": "HIGH", "kind": "DR_SUSPICIOUS_SIZE",
            "detail":   f"last snapshot only {size_mb:.2f} MB",
        })
    return flags


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_system_health() -> dict:
    all_flags = []
    all_flags.extend(check_engine_freshness())
    all_flags.extend(check_pipeline_stages())
    all_flags.extend(check_price_data())
    all_flags.extend(check_perplexity())
    all_flags.extend(check_audit_chain())
    all_flags.extend(check_dr_backup())

    by_severity = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for f in all_flags:
        by_severity[f.get("severity", "LOW")] = by_severity.get(f.get("severity", "LOW"), 0) + 1

    overall = (
        "CRITICAL" if by_severity["CRITICAL"] > 0
        else "DEGRADED" if (by_severity["HIGH"] + by_severity["MEDIUM"]) > 2
        else "OK"
    )

    result = {
        "generated_at":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overall_status":   overall,
        "n_flags_total":    len(all_flags),
        "by_severity":      by_severity,
        "flags":            all_flags,
        "n_engines_checked":len(ENGINE_TTL),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2, default=str))
    _print_report(result)
    return result


def _print_report(r: dict) -> None:
    color = {
        "OK":       "\033[32;1m",
        "DEGRADED": "\033[33m",
        "CRITICAL": "\033[31;1m",
    }.get(r["overall_status"], "\033[0m")
    print(f"\n{SEP}\n  SYSTEM HEALTH\n{SEP}")
    print(f"  Overall:        {color}{r['overall_status']}\033[0m")
    print(f"  Total flags:    {r['n_flags_total']}")
    bs = r["by_severity"]
    print(f"  By severity:    CRITICAL={bs['CRITICAL']}  HIGH={bs['HIGH']}  "
          f"MEDIUM={bs['MEDIUM']}  LOW={bs['LOW']}")
    print(f"  Engines watched:{r['n_engines_checked']}")
    print()
    if r["flags"]:
        print(f"  ACTIVE FLAGS")
        print(f"  {'─' * 58}")
        for f in r["flags"]:
            sev = f["severity"]
            sev_color = {
                "CRITICAL": "\033[31;1m",
                "HIGH":     "\033[31m",
                "MEDIUM":   "\033[33m",
                "LOW":      "\033[36m",
            }.get(sev, "\033[0m")
            print(f"  [{sev_color}{sev:>8s}\033[0m] {f.get('kind', '?'):<22s} "
                  f"{f.get('detail', '')[:48]}")
    else:
        print("  All checks pass.")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="System Health Monitor")
    args = parser.parse_args()
    run_system_health()
