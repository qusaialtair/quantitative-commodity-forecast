#!/usr/bin/env python3
"""
Operator Runbook  (Phase XII Stage 66)
========================================
A single-page summary the operator can read in 60 seconds every morning.
Assembles the data the trader actually acts on, in plain English:

  - Today's trade idea + size + entry / stop / target
  - DeepSeek briefing (first paragraph)
  - System health summary
  - Position reconciliation status
  - Top 3 risk flags
  - Day's checklist (TWS logged in, halal universe fresh, etc.)

Output: data/operator_runbook.md  (also saved as JSON for the UI to consume).
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
RUNBOOK_MD = DATA_DIR / "operator_runbook.md"
RUNBOOK_JSON = DATA_DIR / "operator_runbook.json"

LINE_W = 62
SEP = "━" * LINE_W


def _load(name: str) -> dict:
    p = DATA_DIR / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def run_operator_runbook() -> dict:
    ps = _load("pipeline_state.json")
    ti = _load("trade_idea.json")
    tb = _load("trade_basket.json")
    ds = _load("deepseek_last_turn.json")
    sh = _load("system_health.json")
    pr = _load("position_reconciliation.json")
    mr = _load("metals_rebalancer.json")
    ar = _load("alert_router.json")

    tc = ti.get("trade_card", {})
    flags = ti.get("risk_flags", []) or []

    # Briefing first paragraph (split on double newline)
    answer = (ds.get("answer") or "").strip()
    first_para = answer.split("\n\n", 1)[0] if answer else "(no briefing yet)"

    # Checklist
    checklist = []
    # 1. Halal universe fresh
    hu = _load("halal_universe.json")
    if hu.get("generated_at"):
        try:
            t = datetime.fromisoformat(hu["generated_at"].replace("Z", "+00:00"))
            age_d = (datetime.now(timezone.utc) - t).days
            checklist.append({
                "label":  "Halal universe screened",
                "ok":     age_d < 14,
                "note":   f"{age_d}d ago, {hu.get('n_passing', 0)} tickers passing",
            })
        except Exception:
            pass
    # 2. Audit chain valid
    at = _load("audit_trail_status.json")
    if at:
        checklist.append({
            "label": "Audit chain valid",
            "ok":    at.get("chain_valid", False),
            "note":  f"{at.get('n_total', 0)} rows",
        })
    # 3. DR backup recent
    dr = _load("dr_backup.json")
    if dr:
        size = dr.get("snapshot", {}).get("size_mb", 0)
        checklist.append({
            "label": "DR backup recent",
            "ok":    size > 0.5,
            "note":  f"{size} MB; {dr.get('n_snapshots', 0)} retained",
        })
    # 4. Position reconciliation OK
    checklist.append({
        "label": "Positions reconciled",
        "ok":    pr.get("status") == "OK",
        "note":  pr.get("status", "n/a"),
    })
    # 5. System health OK
    checklist.append({
        "label": "System health OK",
        "ok":    sh.get("overall_status") == "OK",
        "note":  f"{sh.get('n_flags_total', 0)} flags",
    })

    # Construct markdown
    lines = []
    lines.append(f"# Operator Runbook — {ps.get('run_date', '—')}")
    lines.append("")
    lines.append(f"*Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}*")
    lines.append("")

    # 1. Trade card
    lines.append("## Today's Trade Idea")
    lines.append("")
    side = tc.get("side", "—")
    ticker = tc.get("ticker", "—")
    lines.append(f"**{side} {ticker}**  ·  size {tc.get('size_pct', 0):.2f}%  ·  "
                 f"conviction {tc.get('conviction', '—')}")
    if tc.get("entry_price") is not None:
        lines.append(f"- Entry: ${tc.get('entry_price', 0):,.2f}")
    if tc.get("stop_price") is not None:
        lines.append(f"- Stop: ${tc.get('stop_price', 0):,.2f}  "
                     f"({tc.get('stop_distance_pct', 0):.2f}% below)")
    if tc.get("target_price") is not None:
        lines.append(f"- Target: ${tc.get('target_price', 0):,.2f}  "
                     f"({tc.get('target_horizon_days', 5)}d horizon)")
    lines.append(f"- IBKR-ready: {ti.get('ibkr_ready', False)}")
    lines.append(f"- Champion signal: {tc.get('champion_signal', '—')}")
    lines.append("")

    # 2. Basket
    if tb.get("long_basket"):
        lines.append("## Long Basket (top picks)")
        lines.append("")
        for b in tb["long_basket"][:5]:
            lines.append(
                f"- **{b['ticker']}** {b['weight_pct']:.2f}%  · "
                f"mom_21 {b['mom_21_pct']:+.2f}% · vol {b['ann_vol_pct']:.1f}% · z {b['tilted_score']:+.2f}"
            )
        lines.append("")

    # 3. Risk flags
    lines.append("## Risk Flags")
    lines.append("")
    if flags:
        for f in flags:
            lines.append(f"- ⚠ {f}")
    else:
        lines.append("- No active risk flags from the trade idea generator.")
    sh_flags = sh.get("flags", [])
    critical = [f for f in sh_flags if f.get("severity") in ("CRITICAL", "HIGH")]
    if critical:
        lines.append("")
        lines.append("**System health (critical/high)**")
        for f in critical[:5]:
            lines.append(f"- [{f.get('severity', '?')}] {f.get('kind', '?')}: {f.get('detail', '')}")
    lines.append("")

    # 4. Briefing
    lines.append("## DeepSeek Briefing  (first paragraph)")
    lines.append("")
    lines.append(first_para)
    lines.append("")

    # 5. Metals rebalancer
    if mr.get("candidate_action"):
        lines.append("## Physical Metals Rebalancer")
        lines.append("")
        lines.append(f"Candidate action: **{mr.get('candidate_action', '—')}**")
        if mr.get("candidate_target"):
            lines.append(f"Target: ${mr.get('candidate_target', 0):,.2f}")
        if mr.get("n_open_trades", 0) > 0:
            lines.append(f"Open rebalance trades: {mr.get('n_open_trades', 0)}")
        lines.append("")

    # 6. Checklist
    lines.append("## Pre-Trade Checklist")
    lines.append("")
    for c in checklist:
        mark = "✅" if c["ok"] else "❌"
        lines.append(f"- {mark} {c['label']} — {c['note']}")
    lines.append("")

    # 7. Action of the day
    lines.append("## Today's Action")
    lines.append("")
    if side == "HOLD" or not ti.get("ibkr_ready", False):
        lines.append("- **Stand down.** No actionable trade today.")
    else:
        lines.append(f"- Open TWS / IB Gateway.")
        lines.append(f"- Run: `python3 scripts/ibkr_adapter.py --buy {ticker} "
                     f"--qty <calculated from size_usd ${tc.get('size_usd', 0):,.0f}> --live`")
        lines.append(f"- Set stop at ${tc.get('stop_price', 0):,.2f}.")
        lines.append("- Append the fill to `data/fill_log.jsonl` so slippage tracker updates.")
    lines.append("")

    body = "\n".join(lines)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUNBOOK_MD.write_text(body)

    # JSON for the UI
    summary = {
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_date":       ps.get("run_date"),
        "trade_card":     tc,
        "risk_flags":     flags,
        "first_paragraph_briefing": first_para,
        "checklist":      checklist,
        "n_checklist_ok": sum(1 for c in checklist if c["ok"]),
        "n_checklist":    len(checklist),
        "alerts_dispatched": ar.get("n_alerts", 0),
        "metals_action":  mr.get("candidate_action"),
        "markdown_path":  str(RUNBOOK_MD),
    }
    RUNBOOK_JSON.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n{SEP}\n  OPERATOR RUNBOOK\n{SEP}")
    print(f"  Markdown:    {RUNBOOK_MD}")
    print(f"  JSON:        {RUNBOOK_JSON}")
    print(f"  Checklist:   {summary['n_checklist_ok']} / {summary['n_checklist']} ✅")
    print(f"  Risk flags:  {len(flags)}")
    print(f"  Trade:       {side} {ticker} ({tc.get('conviction', '—')})")
    print(SEP)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Operator Runbook")
    args = parser.parse_args()
    run_operator_runbook()
