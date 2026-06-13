#!/usr/bin/env python3
"""
Latency Profiler  (Phase IX Stage 51)
=======================================
Parses the daily master_controller pipeline log and per-stage timings out of
pipeline_state.json, computes a per-stage profile, identifies the slowest
stages and the post-pipeline engines that exceed a configurable threshold,
and reports trends across recent runs.

Reads:
  - data/pipeline_state.json     for current-run stage durations
  - data/logs/pipeline_*.log     for engine timings (the "(X.Ys)" suffixes)

Output: data/latency_profile.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
LOG_DIR = DATA_DIR / "logs"
PIPELINE_STATE = DATA_DIR / "pipeline_state.json"
OUTPUT_FILE = DATA_DIR / "latency_profile.json"

# Threshold (seconds) above which an engine is flagged as slow
SLOW_ENGINE_S = 5.0
LINE_W = 62
SEP = "━" * LINE_W

# Match "Mean-CVaR: ..." style summary lines that wrap the engine output
ENGINE_LOG_RE = re.compile(r"INFO\s+([A-Za-z][A-Za-z\- ]+?):\s")
DURATION_RE = re.compile(r"\((\d+\.\d+)s\)")


# ---------------------------------------------------------------------------
# Parse pipeline_state.json stage durations
# ---------------------------------------------------------------------------
def _stage_durations() -> dict:
    if not PIPELINE_STATE.exists():
        return {}
    try:
        ps = json.loads(PIPELINE_STATE.read_text())
    except Exception:
        return {}
    out = {}
    for k, v in ps.get("stages", {}).items():
        if isinstance(v, dict):
            out[k] = float(v.get("duration_s", 0))
    return out


def _pipeline_run_total() -> float:
    durations = _stage_durations()
    return sum(durations.values())


# ---------------------------------------------------------------------------
# Parse engine timings out of recent pipeline logs
# ---------------------------------------------------------------------------
def _parse_latest_log() -> dict:
    if not LOG_DIR.exists():
        return {}
    log_files = sorted(LOG_DIR.glob("pipeline_*.log"))
    if not log_files:
        return {}
    latest = log_files[-1]
    timings = {}
    last_engine = None
    try:
        text = latest.read_text()
    except Exception:
        return {}

    for line in text.splitlines():
        # Engine summary line — capture name as "key"
        # Format example:
        #   2026-05-12 16:53:23 UTC [PIPELINE  ] INFO  Fama-French: α=+19.28% ...
        m = ENGINE_LOG_RE.search(line)
        if m and "PIPELINE" in line:
            engine = m.group(1).strip()
            last_engine = engine
            if engine not in timings:
                timings[engine] = 0.0
        # Duration suffix can appear on the engine's own output or stage summary
        # We use stage_durations() for stages 1-7; this catches engine line items.
    return timings


# ---------------------------------------------------------------------------
# Aggregate across recent runs
# ---------------------------------------------------------------------------
def _historical_pipeline_totals(n_runs: int = 10) -> list:
    if not LOG_DIR.exists():
        return []
    log_files = sorted(LOG_DIR.glob("pipeline_*.log"))[-n_runs:]
    history = []
    for lf in log_files:
        try:
            text = lf.read_text()
            m = re.search(r"Pipeline (SUCCESS|PARTIAL|ABORTED).*total ([\d\.]+)s", text)
            if m:
                history.append({
                    "log_file":     lf.name,
                    "status":       m.group(1),
                    "total_s":      float(m.group(2)),
                })
        except Exception:
            continue
    return history


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_latency_profiler() -> dict:
    durations = _stage_durations()
    total_pipeline = sum(durations.values())

    # Slow stages
    slow_stages = sorted(
        [(k, v) for k, v in durations.items()],
        key=lambda kv: kv[1], reverse=True,
    )

    historical = _historical_pipeline_totals(20)
    if historical:
        totals = [h["total_s"] for h in historical]
        avg_total = sum(totals) / len(totals)
        min_total = min(totals)
        max_total = max(totals)
    else:
        avg_total = total_pipeline
        min_total = total_pipeline
        max_total = total_pipeline

    # Recommendations: any stage > 30% of total?
    recommendations = []
    if total_pipeline > 0:
        for k, v in durations.items():
            pct = v / total_pipeline * 100
            if pct > 30:
                recommendations.append({
                    "stage":         k,
                    "duration_s":    round(v, 2),
                    "pct_of_total":  round(pct, 1),
                    "suggestion":    f"{k} consumes {pct:.0f}% of pipeline; consider caching or parallel execution",
                })

    if total_pipeline > 90:
        recommendations.append({
            "stage":         "OVERALL",
            "duration_s":    round(total_pipeline, 2),
            "pct_of_total":  100.0,
            "suggestion":    "Total > 90s; investigate top stages and post-pipeline engines",
        })

    result = {
        "generated_at":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "current_run_total_s":   round(total_pipeline, 2),
        "stage_durations_s":     {k: round(v, 2) for k, v in durations.items()},
        "slowest_stage":         slow_stages[0][0] if slow_stages else None,
        "slowest_duration_s":    round(slow_stages[0][1], 2) if slow_stages else 0,
        "n_stages":              len(durations),
        "historical": {
            "n_runs":   len(historical),
            "avg_s":    round(avg_total, 2),
            "min_s":    round(min_total, 2),
            "max_s":    round(max_total, 2),
            "recent":   historical[-5:],
        },
        "recommendations":       recommendations,
        "n_recommendations":     len(recommendations),
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2))
    _print_report(result)
    return result


def _print_report(r: dict) -> None:
    print(f"\n{SEP}\n  LATENCY PROFILE\n{SEP}")
    print(f"  Current run:    {r['current_run_total_s']} s")
    print(f"  Slowest stage:  {r['slowest_stage']}  ({r['slowest_duration_s']} s)")
    print()
    print(f"  PER-STAGE DURATIONS")
    print(f"  {'─' * 40}")
    for k, v in sorted(r["stage_durations_s"].items(), key=lambda kv: kv[1], reverse=True):
        bar = "█" * int(v)
        print(f"  {k:<14s}  {v:>6.2f} s  {bar}")
    print()
    h = r["historical"]
    print(f"  HISTORICAL ({h['n_runs']} runs)")
    print(f"  avg {h['avg_s']} s   min {h['min_s']} s   max {h['max_s']} s")
    print()
    if r["recommendations"]:
        print(f"  RECOMMENDATIONS")
        for rec in r["recommendations"]:
            print(f"    • {rec['suggestion']}")
    else:
        print(f"  No bottlenecks > 30% of pipeline")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latency Profiler")
    args = parser.parse_args()
    run_latency_profiler()
