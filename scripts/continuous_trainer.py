#!/usr/bin/env python3
"""
Continuous Training Orchestrator  (Phase XI Stage 60)
=======================================================
Background training loop that keeps the institutional stack learning.
Driven by a state file (data/training_state.json) that tracks when each
training task last ran; the orchestrator decides which tasks are due and
executes them, then logs every run to the audit trail.

Schedule:
  - Daily      Bayesian HPO refresh on the signal-generator hyperparams
               LSTM 1-step fine-tune (delegated to daily_trainer.py if present)
               RL Q-learning refresh
  - Weekly     Purged K-fold stability check
               Ensemble stacking refit
               MRM champion-challenger evaluation
  - On-demand  Conformal-interval recalibration

Auto-promotion:
  If MRM returns "PROMOTE", record the promotion as an audit event and
  update mrm_state.json's champion.

Usage:
    python3 scripts/continuous_trainer.py             # run what's due now
    python3 scripts/continuous_trainer.py --force     # ignore due-dates
    python3 scripts/continuous_trainer.py --weekly    # only weekly tasks
    python3 scripts/continuous_trainer.py --status    # show schedule

Output: data/continuous_trainer.json (last run summary)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "continuous_trainer.json"
STATE_FILE = DATA_DIR / "training_state.json"

# Task → (interval_hours, runner function name)
TASKS = {
    "bayesian_hpo":         {"interval_h": 24,  "module": "scripts.bayesian_hpo",       "func": "run_bayesian_hpo"},
    "rl_sizing":            {"interval_h": 24,  "module": "scripts.rl_sizing_agent",    "func": "run_rl_sizing"},
    "purged_kfold":         {"interval_h": 168, "module": "scripts.purged_kfold",       "func": "run_purged_kfold"},
    "ensemble_stacking":    {"interval_h": 168, "module": "scripts.ensemble_stacking",  "func": "run_ensemble_stacking"},
    "conformal_intervals":  {"interval_h": 168, "module": "scripts.conformal_intervals","func": "run_conformal_intervals"},
    "mrm_champion":         {"interval_h": 24,  "module": "scripts.mrm_champion",       "func": "run_mrm"},
}

LINE_W = 62
SEP = "━" * LINE_W


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def _hours_since(ts_iso: str | None) -> float:
    if not ts_iso:
        return float("inf")
    try:
        t = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - t
        return delta.total_seconds() / 3600.0
    except Exception:
        return float("inf")


def _is_due(task: str, state: dict) -> bool:
    last_ts = state.get(task, {}).get("last_run")
    interval = TASKS[task]["interval_h"]
    return _hours_since(last_ts) >= interval


def _run_task(task: str) -> dict:
    spec = TASKS[task]
    module_name = spec["module"]
    func_name = spec["func"]
    t_start = datetime.now(timezone.utc)
    try:
        import importlib
        mod = importlib.import_module(module_name)
        fn = getattr(mod, func_name)
        result = fn()
        t_end = datetime.now(timezone.utc)
        return {
            "task":       task,
            "status":     "OK",
            "start_ts":   t_start.isoformat(timespec="seconds"),
            "end_ts":     t_end.isoformat(timespec="seconds"),
            "duration_s": round((t_end - t_start).total_seconds(), 1),
        }
    except Exception as exc:
        t_end = datetime.now(timezone.utc)
        return {
            "task":       task,
            "status":     "FAILED",
            "start_ts":   t_start.isoformat(timespec="seconds"),
            "end_ts":     t_end.isoformat(timespec="seconds"),
            "duration_s": round((t_end - t_start).total_seconds(), 1),
            "error":      str(exc)[:200],
        }


def _log_audit(event_type: str, payload: dict) -> None:
    try:
        from scripts.audit_trail import record
        record(event_type, payload)
    except Exception:
        pass


def run_orchestrator(
    force: bool = False,
    only_weekly: bool = False,
    only_daily: bool = False,
) -> dict:
    state = _load_state()
    runs = []

    for task in TASKS:
        spec = TASKS[task]
        is_weekly = spec["interval_h"] >= 168
        if only_weekly and not is_weekly:
            continue
        if only_daily and is_weekly:
            continue
        due = force or _is_due(task, state)
        if not due:
            runs.append({
                "task": task,
                "status": "SKIPPED",
                "reason": "not due yet",
                "last_run": state.get(task, {}).get("last_run"),
                "hours_since": round(_hours_since(state.get(task, {}).get("last_run")), 1),
            })
            continue

        print(f"  ▶ running {task}...", flush=True)
        result = _run_task(task)
        runs.append(result)

        # Update state
        state[task] = {
            "last_run":   result["end_ts"],
            "status":     result["status"],
            "duration_s": result["duration_s"],
        }
        _log_audit("TRAINING_RUN", {
            "task":       task,
            "status":     result["status"],
            "duration_s": result["duration_s"],
        })

    # Check for MRM promotion after MRM task ran
    promotion_event = None
    if any(r["task"] == "mrm_champion" and r["status"] == "OK" for r in runs):
        try:
            mrm = json.loads((DATA_DIR / "mrm_champion.json").read_text())
            if mrm.get("decision") == "PROMOTE":
                promotion_event = {
                    "ts":              datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "previous_champion": state.get("champion_history", {}).get("current"),
                    "new_champion":    mrm.get("new_champion"),
                    "score_delta":     mrm.get("score_delta"),
                }
                state.setdefault("champion_history", {})["current"] = mrm.get("new_champion")
                state["champion_history"].setdefault("log", []).append(promotion_event)
                _log_audit("CHAMPION_PROMOTION", promotion_event)
        except Exception:
            pass

    _save_state(state)

    successful = sum(1 for r in runs if r["status"] == "OK")
    failed = sum(1 for r in runs if r["status"] == "FAILED")
    skipped = sum(1 for r in runs if r["status"] == "SKIPPED")

    result = {
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_tasks":        len(TASKS),
        "n_run":          successful + failed,
        "n_successful":   successful,
        "n_failed":       failed,
        "n_skipped":      skipped,
        "runs":           runs,
        "promotion":      promotion_event,
        "state":          state,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2, default=str))
    _print_report(result)
    return result


def _print_report(r: dict) -> None:
    print(f"\n{SEP}\n  CONTINUOUS TRAINING ORCHESTRATOR\n{SEP}")
    print(f"  Total tasks: {r['n_tasks']}")
    print(f"  Ran:         {r['n_run']}  (successful: {r['n_successful']}  failed: {r['n_failed']})")
    print(f"  Skipped:     {r['n_skipped']}")
    print()
    print(f"  TASK STATUS")
    print(f"  {'─' * 58}")
    for run in r["runs"]:
        status_color = {
            "OK":      "\033[32m",
            "FAILED":  "\033[31m",
            "SKIPPED": "\033[90m",
        }.get(run["status"], "\033[0m")
        if run["status"] == "SKIPPED":
            since = run.get("hours_since", 0)
            print(f"  {run['task']:<22s}  {status_color}{run['status']:<8s}\033[0m  {since:.1f}h since last")
        else:
            print(f"  {run['task']:<22s}  {status_color}{run['status']:<8s}\033[0m  {run.get('duration_s', 0):.1f}s")
    print()
    if r["promotion"]:
        print(f"  ⭐ CHAMPION PROMOTION DETECTED")
        print(f"    {r['promotion']['previous_champion']} → {r['promotion']['new_champion']}  "
              f"(Δ {r['promotion']['score_delta']:+.4f})")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


def _print_status() -> None:
    state = _load_state()
    print(f"\n{SEP}\n  TRAINING SCHEDULE STATUS\n{SEP}")
    print(f"  {'task':<22s}  {'interval':>9s}  {'last_run':>22s}  {'hours_since':>11s}  due?")
    for task, spec in TASKS.items():
        last_ts = state.get(task, {}).get("last_run", "never")
        since = _hours_since(state.get(task, {}).get("last_run"))
        due = "YES" if since >= spec["interval_h"] else "no"
        since_str = f"{since:.1f}" if since < 9999 else "never"
        print(f"  {task:<22s}  {spec['interval_h']:>7d}h  {last_ts:>22s}  {since_str:>11s}  {due}")
    print(SEP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continuous Training Orchestrator")
    parser.add_argument("--force", action="store_true",
                        help="Run all tasks regardless of due-dates")
    parser.add_argument("--weekly", action="store_true",
                        help="Only run weekly tasks (interval ≥ 168h)")
    parser.add_argument("--daily", action="store_true",
                        help="Only run daily tasks (interval < 168h)")
    parser.add_argument("--status", action="store_true",
                        help="Print schedule status and exit")
    args = parser.parse_args()

    if args.status:
        _print_status()
    else:
        run_orchestrator(
            force=args.force, only_weekly=args.weekly, only_daily=args.daily,
        )
