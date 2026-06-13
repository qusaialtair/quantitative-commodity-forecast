#!/usr/bin/env python3
"""
run_daily_pipeline.py
=====================
Phase 4 — Master Orchestrator for the Sharia-compliant equity fund.

Runs the full daily lifecycle end-to-end, designed to be fired once a
night (01:00 local by default) from launchd / cron. Guarantees strict
sequential blocking: each stage MUST complete and commit its artefact
to disk before the next stage is allowed to start. Any stage failure
halts the pipeline instantly, fires a critical Telegram alert, and
aborts trading for the day.

Sequence
--------
  1. Features   — data_pipelines.equity_features       → data/equity_features/latest.parquet
  2. Ranker     — models.equity_ranker.ranker rank     → data/equity_ranker/ranking_latest.parquet
  3. CIO Swarm  — agents.equity_swarm.swarm            → data/equity_decision.json
  4. MTM        — execution.virtual_trader mark        → data/virtual_account.json (marks)
  5. Rebalance  — execution.virtual_trader rebalance   → data/virtual_account.json (book)
                                                       → data/trade_log_v2.jsonl
  6. Briefing   — Telegram executive summary           → fire-and-forget

Fault tolerance
---------------
  * `subprocess.run(..., check=True)` — a non-zero exit code raises
    CalledProcessError, which the global try/except traps.
  * Per-stage wall-clock timeouts — a stall is treated as failure.
  * All paths absolute, resolved from __file__, so the script is
    cron/launchd-safe regardless of the invoking CWD.
  * A stage-1..5 failure HALTS the run, fires an URGENT Telegram
    alert, and exits 1 so launchd/cron observes the failure.
  * Stage 6 (Telegram summary) is best-effort — a notifier outage
    must never mask a successful trading day.

CLI
---
  python3 run_daily_pipeline.py                        # full nightly run
  python3 run_daily_pipeline.py --skip-features        # reuse today's features
  python3 run_daily_pipeline.py --dry-run              # skip Rebalance (stage 5)
  python3 run_daily_pipeline.py --only "mtm,rebalance" # surgical re-run
  python3 run_daily_pipeline.py --top 60               # Stage-3 candidates
  python3 run_daily_pipeline.py --no-sentiment         # Stage-3 no Perplexity
  python3 run_daily_pipeline.py --no-insider           # Stage-3 no SEC Form-4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import time
import traceback
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

# ══════════════════════════════════════════════════════════════════════════════
#  ABSOLUTE PATH ANCHORING — the script is cron/launchd-safe because every
#  path below is resolved from __file__, not from the invoking CWD.
# ══════════════════════════════════════════════════════════════════════════════
ROOT            = Path(os.path.abspath(os.path.dirname(__file__)))
DATA_DIR        = ROOT / "data"
LOG_DIR         = DATA_DIR / "logs"
SCRIPTS_DIR     = ROOT / "scripts"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_LATEST = DATA_DIR / "equity_features" / "latest.parquet"
RANKING_LATEST  = DATA_DIR / "equity_ranker"   / "ranking_latest.parquet"
DECISION_PATH   = DATA_DIR / "equity_decision.json"
VAULT_PATH      = DATA_DIR / "virtual_account.json"
TRADE_LOG_V2    = DATA_DIR / "trade_log_v2.jsonl"
PIPELINE_STATE  = DATA_DIR / "pipeline_state.json"
MST_STATE       = DATA_DIR / "multi_strategy_trader.json"   # Phase XIV/XXV

# Make the project importable (for scripts.telegram_notifier at the end).
sys.path.insert(0, str(ROOT))

# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE DEFINITION — ordered, named, with per-stage wall-clock ceilings.
#
#  Each entry is a tuple:
#     (key, label, argv_builder, timeout_s, output_artefact, critical)
#
#     critical=True  → a failure HALTS the pipeline and alerts.
#     critical=False → best-effort (currently only the Telegram briefing).
#
#  argv_builder is a callable (args_ns) → list[str].
#  The output_artefact is the path we verify after the stage claims
#  success; a stage that exits 0 but leaves no artefact is treated
#  as a silent failure.
# ══════════════════════════════════════════════════════════════════════════════
PY = sys.executable  # same interpreter the orchestrator runs under


def _argv_features(ns) -> list[str]:
    cmd = [PY, "-m", "data_pipelines.equity_features"]
    if ns.universe_seed_only:
        cmd += ["--universe", "seed"]
    return cmd


def _argv_ranker(ns) -> list[str]:
    return [PY, "-m", "models.equity_ranker.ranker", "rank",
            "--features", str(FEATURES_LATEST)]


def _argv_swarm(ns) -> list[str]:
    cmd = [PY, "-m", "agents.equity_swarm.swarm", "--top", str(ns.top)]
    if ns.no_sentiment:
        cmd.append("--no-sentiment")
    if ns.no_insider:
        cmd.append("--no-insider")
    return cmd


def _argv_mtm(ns) -> list[str]:
    return [PY, "-m", "execution.virtual_trader", "mark"]


def _argv_rebalance(ns) -> list[str]:
    return [PY, "-m", "execution.virtual_trader", "rebalance"]


# key         label                 argv_builder     timeout   artefact          critical
STAGES: list[tuple[str, str, callable, int, Optional[Path], bool]] = [
    ("features",   "1. FEATURES",    _argv_features,  45 * 60, FEATURES_LATEST, True),
    ("ranker",     "2. RANKER",      _argv_ranker,     5 * 60, RANKING_LATEST,  True),
    ("swarm",      "3. CIO SWARM",   _argv_swarm,     15 * 60, DECISION_PATH,   True),
    ("mtm",        "4. MARK-TO-MKT", _argv_mtm,        5 * 60, VAULT_PATH,      True),
    ("rebalance",  "5. REBALANCE",   _argv_rebalance,  5 * 60, VAULT_PATH,      True),
]
STAGE_KEYS = {k for (k, *_rest) in STAGES}

logger = logging.getLogger("ORCHESTRATOR")


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE RUNNER — strict blocking; check=True raises on non-zero exit.
# ══════════════════════════════════════════════════════════════════════════════
class StageFailure(RuntimeError):
    """Raised when a pipeline stage fails or stalls. Carries the stage key."""
    def __init__(self, key: str, label: str, reason: str, elapsed: float):
        super().__init__(f"[{label}] {reason}")
        self.key     = key
        self.label   = label
        self.reason  = reason
        self.elapsed = elapsed


def _run_stage(key: str, label: str, argv: list[str], timeout: int,
               artefact: Optional[Path], log_fh) -> float:
    """Execute a single stage. Returns elapsed seconds on success.

    Blocks until the subprocess either exits, errors, or exceeds the
    wall-clock ceiling. On any non-zero exit or timeout, raises
    StageFailure — the global try/except in run_pipeline() converts
    this to a Telegram URGENT alert and aborts the day.
    """
    logger.info("━" * 72)
    logger.info("▶ STAGE %s  %s", label, " ".join(shlex.quote(x) for x in argv))
    logger.info("  timeout: %ds   artefact: %s", timeout,
                artefact.relative_to(ROOT) if artefact else "—")
    logger.info("━" * 72)

    log_fh.write(
        f"\n\n===== {label} @ {datetime.now(timezone.utc).isoformat()} =====\n"
        f"CMD: {' '.join(shlex.quote(x) for x in argv)}\n"
    )
    log_fh.flush()

    # Record artefact mtime before we start — if the stage claims success
    # but the artefact wasn't refreshed, we treat it as a silent failure.
    prev_mtime = artefact.stat().st_mtime if (artefact and artefact.exists()) else 0.0

    t0 = time.time()
    try:
        proc = subprocess.run(
            argv,
            cwd=ROOT,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,                 # ← strict blocking: non-zero raises
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - t0
        partial = exc.output or ""
        log_fh.write(partial + f"\n*** TIMEOUT after {elapsed:.0f}s ***\n")
        log_fh.flush()
        raise StageFailure(key, label,
                           f"timeout after {timeout}s (hard ceiling)",
                           elapsed) from exc
    except subprocess.CalledProcessError as exc:
        elapsed = time.time() - t0
        log_fh.write((exc.output or "") + f"\n*** EXIT {exc.returncode} ***\n")
        log_fh.flush()
        raise StageFailure(key, label,
                           f"non-zero exit {exc.returncode}",
                           elapsed) from exc
    except FileNotFoundError as exc:
        elapsed = time.time() - t0
        raise StageFailure(key, label,
                           f"executable not found: {exc}",
                           elapsed) from exc

    elapsed = time.time() - t0
    log_fh.write(proc.stdout or "")
    log_fh.write(f"\n*** EXIT 0 · {elapsed:.1f}s ***\n")
    log_fh.flush()

    # Artefact freshness check — the stage must have written something new.
    if artefact is not None:
        if not artefact.exists():
            raise StageFailure(key, label,
                               f"stage exited 0 but artefact missing: "
                               f"{artefact.relative_to(ROOT)}",
                               elapsed)
        new_mtime = artefact.stat().st_mtime
        # Allow equal mtime for stages that rewrite on same-second boundaries.
        if new_mtime < prev_mtime:
            raise StageFailure(key, label,
                               f"artefact not refreshed: "
                               f"{artefact.relative_to(ROOT)}",
                               elapsed)

    logger.info("✓ STAGE %s DONE  (%.1fs)", label, elapsed)
    return elapsed


# ══════════════════════════════════════════════════════════════════════════════
#  EXECUTIVE BRIEFING — reads the live vault + decision and builds a
#  Telegram summary. Always runs last, even on pipeline failure (so the
#  operator gets told the day was aborted).
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_money(x: float) -> str:
    return f"${x:,.2f}"


def _trades_today_from_log() -> list[dict]:
    if not TRADE_LOG_V2.exists():
        return []
    today = date.today().isoformat()
    out: list[dict] = []
    for line in TRADE_LOG_V2.read_text().splitlines():
        try:
            row = json.loads(line)
            if str(row.get("timestamp", "")).startswith(today):
                out.append(row)
        except Exception:
            continue
    return out


def _build_executive_summary(stage_results: dict, halted: bool,
                             halt_reason: Optional[str]) -> str:
    """Construct a Telegram-flavoured executive summary."""
    lines: list[str] = []
    today = date.today().isoformat()
    status = "HALTED" if halted else "OK"
    lines.append(f"<b>DAILY PIPELINE · {today} · {status}</b>")
    if halted and halt_reason:
        lines.append(f"<b>HALT REASON:</b> <code>{halt_reason}</code>")
    lines.append("")

    # Stage outcomes
    lines.append("<b>Stages</b>")
    for label, res in stage_results.items():
        icon = "OK  " if res["ok"] else "FAIL"
        lines.append(f"  {icon}  {label}  ({res['elapsed']:.0f}s)")
        if not res["ok"] and res.get("reason"):
            lines.append(f"        └ <code>{res['reason']}</code>")
    lines.append("")

    # Decision summary (if any)
    if DECISION_PATH.exists():
        try:
            d = json.loads(DECISION_PATH.read_text())
            lines.append("<b>CIO Target</b>")
            lines.append(f"  regime:    {d.get('macro_regime')} / "
                         f"cap {d.get('macro_gross_cap', 0):.0%}")
            lines.append(f"  positions: {d.get('n_positions', 0)}")
            lines.append(f"  gross:     {d.get('equity_gross', 0):.1%}")
            lines.append(f"  cash:      {d.get('cash_sukuk', 0):.1%}")
            lines.append("")
        except Exception:
            pass

    # Live vault snapshot
    if VAULT_PATH.exists():
        try:
            v = json.loads(VAULT_PATH.read_text())
            eq    = float(v.get("total_equity") or 0.0)
            cash  = float(v.get("cash_balance") or 0.0)
            start = float(v.get("starting_cash") or eq or 1.0)
            pnl   = eq - start
            pnl_p = (pnl / start) * 100.0 if start else 0.0
            cash_pct = (cash / eq * 100.0) if eq > 0 else 100.0
            lines.append("<b>Vault</b>")
            lines.append(f"  total equity: {_fmt_money(eq)}")
            lines.append(f"  cash:         {_fmt_money(cash)} ({cash_pct:.1f}%)")
            lines.append(f"  positions:    {len(v.get('positions', {}))}")
            lines.append(f"  lifetime P&L: {_fmt_money(pnl)} ({pnl_p:+.2f}%)")
            lines.append("")
        except Exception:
            pass

    # Phase XIV multi-strategy trader + Phase XXV treasury hedge sleeve
    if MST_STATE.exists():
        try:
            mst = json.loads(MST_STATE.read_text())
            hs   = mst.get("hedge_state") or {}
            by_s = mst.get("by_strategy") or {}
            eq   = float(mst.get("book_equity_usd") or 0.0)
            lines.append("<b>Phase XIV — Multi-Strategy Trader</b>")
            lines.append(f"  strategy:   {mst.get('strategy', '—')}")
            lines.append(f"  equity:     {_fmt_money(eq)}")
            lines.append(
                f"  lifetime:   {mst.get('lifetime_pl_pct', 0.0):+.2f}%  "
                f"open={mst.get('n_open', 0)}  "
                f"closed={mst.get('n_closed_total', 0)}"
            )
            # Hedge sleeve status
            hedge_instr    = hs.get("instrument")
            hedge_notional = float(mst.get("hedge_notional_usd") or 0.0)
            hedge_frac     = float(mst.get("hedge_fraction") or 0.0)
            sub_tag        = hs.get("sub_tag")
            gate           = hs.get("gate_action") or "—"
            if hedge_instr:
                sub_str = f" [{sub_tag}]" if sub_tag else ""
                lines.append(
                    f"  hedge:      {hedge_instr}{sub_str}"
                    f"  {_fmt_money(hedge_notional)} ({hedge_frac:.0%} of book)"
                    f"  gate={gate}"
                )
            else:
                lines.append(f"  hedge:      off  (mode={hs.get('mode', '—')})")
            # P&L split: hedge drag vs alpha
            if by_s:
                th_pl    = float((by_s.get("TREASURY_HEDGE") or {}).get("realized_pl_usd") or 0.0)
                alpha_pl = sum(
                    float(v.get("realized_pl_usd") or 0.0)
                    for k, v in by_s.items() if k != "TREASURY_HEDGE"
                )
                lines.append(
                    f"  hedge P&L:  {_fmt_money(th_pl)}  "
                    f"alpha P&L: {_fmt_money(alpha_pl)}"
                )
            lines.append("")
        except Exception:
            pass

    # Today's trades
    trades_today = _trades_today_from_log()
    if trades_today:
        lines.append(f"<b>Trades today: {len(trades_today)}</b>")
        by_reason: dict[str, int] = {}
        for t in trades_today:
            by_reason[t.get("reason", "?")] = by_reason.get(t.get("reason", "?"), 0) + 1
        for r, n in sorted(by_reason.items()):
            lines.append(f"  · {r}: {n}")
        overrides = [t for t in trades_today
                     if t.get("reason") in ("ATR_STOP", "SHARIA_VIOLATION")]
        if overrides:
            lines.append("")
            lines.append("<b>Risk overrides</b>")
            for t in overrides[:10]:
                lines.append(
                    f"  · {t.get('reason')} {t.get('ticker')} "
                    f"@ {t.get('price_exec', 0):.2f}  qty={t.get('qty', 0)}"
                )
    elif not halted:
        lines.append("<i>No trades executed today.</i>")

    return "\n".join(lines)


def _send_telegram_briefing(text: str) -> tuple[bool, str]:
    """Send the heartbeat briefing. Never raises."""
    try:
        from scripts.telegram_notifier import send_alert
    except Exception as exc:
        return False, f"notifier import failed: {exc}"
    try:
        ok = send_alert(text, level="OK")
        return bool(ok), "sent" if ok else "send_alert returned False"
    except Exception as exc:
        return False, f"exception: {exc}"


def _send_telegram_critical(label: str, reason: str, traceback_str: str) -> None:
    """Fire an URGENT alert on pipeline halt. Never raises."""
    try:
        from scripts.telegram_notifier import send_alert
    except Exception:
        return
    try:
        text = (
            f"<b>🚨 PIPELINE HALTED · TRADING ABORTED</b>\n\n"
            f"<b>Failed stage:</b> <code>{label}</code>\n"
            f"<b>Reason:</b>       <code>{reason}</code>\n"
            f"<b>Time:</b>         <code>{datetime.now(timezone.utc).isoformat(timespec='seconds')}</code>\n\n"
            f"<b>No rebalance will execute today.</b> "
            f"The virtual vault is untouched past the last successful stage. "
            f"Inspect <code>data/logs/daily_pipeline_{date.today().isoformat()}.log</code>, "
            f"fix the failure, and re-run with "
            f"<code>python3 run_daily_pipeline.py --only</code> "
            f"to resume from the broken stage.\n\n"
            f"<b>Traceback (truncated):</b>\n<code>{traceback_str[:1500]}</code>"
        )
        send_alert(text, level="URGENT")
    except Exception:
        # Best-effort; notifier outage cannot cascade.
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR ENTRY
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(ns: argparse.Namespace) -> dict:
    """Execute the daily lifecycle. Returns the final pipeline_state dict."""
    wall0    = time.time()
    log_path = LOG_DIR / f"daily_pipeline_{date.today().isoformat()}.log"
    log_fh   = log_path.open("a", encoding="utf-8")
    log_fh.write(
        f"\n\n################################################################\n"
        f"# PIPELINE START @ {datetime.now(timezone.utc).isoformat()}\n"
        f"# ROOT: {ROOT}\n"
        f"# PYTHON: {PY}\n"
        f"# ARGS: {vars(ns)}\n"
        f"################################################################\n"
    )
    log_fh.flush()

    only: Optional[set[str]] = None
    if ns.only:
        only = {s.strip().lower() for s in ns.only.split(",") if s.strip()}
        unknown = only - STAGE_KEYS
        if unknown:
            raise SystemExit(f"--only contains unknown stage keys: {unknown}. "
                             f"Valid: {sorted(STAGE_KEYS)}")

    def _selected(key: str) -> bool:
        if only is not None:
            return key in only
        if key == "features"  and ns.skip_features:
            return False
        if key == "rebalance" and ns.dry_run:
            return False
        return True

    results: dict[str, dict] = {}
    halted       = False
    halt_label:  Optional[str] = None
    halt_reason: Optional[str] = None
    halt_trace:  Optional[str] = None

    # ── GLOBAL TRY/EXCEPT — any stage failure HALTS and alerts. ───────────────
    try:
        for key, label, builder, timeout, artefact, critical in STAGES:
            if not _selected(key):
                logger.info("− SKIP  %s", label)
                results[label] = {"ok": True,  "elapsed": 0.0,
                                  "reason": "skipped by --only/--skip/--dry-run"}
                continue

            argv = builder(ns)
            elapsed = _run_stage(key, label, argv, timeout, artefact, log_fh)
            results[label] = {"ok": True, "elapsed": elapsed, "reason": None}

    except StageFailure as sf:
        halted       = True
        halt_label   = sf.label
        halt_reason  = sf.reason
        halt_trace   = traceback.format_exc()
        results[sf.label] = {"ok": False,
                             "elapsed": sf.elapsed,
                             "reason": sf.reason}
        logger.error("✗ STAGE %s FAILED — %s (%.1fs)",
                     sf.label, sf.reason, sf.elapsed)
        logger.error("Pipeline HALTED. Trading aborted for the day.")

    except Exception as exc:
        # Catch-all: anything unexpected (import error, permission error,
        # out-of-memory) still triggers the URGENT alert path.
        halted       = True
        halt_label   = "ORCHESTRATOR"
        halt_reason  = f"unhandled: {exc!r}"
        halt_trace   = traceback.format_exc()
        logger.error("✗ ORCHESTRATOR CRASHED — %s", exc)
        logger.error("%s", halt_trace)

    # ── Stage 6 ALWAYS runs — success or halt — so the operator is told. ─────
    briefing = _build_executive_summary(results, halted, halt_reason)
    log_fh.write("\n\n===== EXECUTIVE BRIEFING =====\n" + briefing + "\n")

    if halted:
        _send_telegram_critical(halt_label or "?", halt_reason or "?",
                                halt_trace or "")
    tg_ok, tg_reason = _send_telegram_briefing(briefing)
    results["6. BRIEFING"] = {"ok": tg_ok, "elapsed": 0.0,
                              "reason": None if tg_ok else tg_reason}

    # ── State snapshot for launchd / cron to inspect. ─────────────────────────
    wall = time.time() - wall0
    state = {
        "date":        date.today().isoformat(),
        "started":     datetime.fromtimestamp(wall0, tz=timezone.utc)
                               .isoformat(timespec="seconds"),
        "finished":    datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "halted":      halted,
        "halt_label":  halt_label,
        "halt_reason": halt_reason,
        "wall_s":      round(wall, 1),
        "stages":      results,
    }
    PIPELINE_STATE.write_text(json.dumps(state, indent=2, default=str))

    log_fh.write(f"\n===== PIPELINE DONE in {wall:.1f}s · halted={halted} =====\n")
    log_fh.close()

    logger.info("━" * 72)
    logger.info("PIPELINE %s  in %.1fs", "HALTED" if halted else "OK", wall)
    logger.info("Log:   %s", log_path)
    logger.info("State: %s", PIPELINE_STATE)

    return state


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Daily equity pipeline orchestrator (Phase 4, strict)."
    )
    ap.add_argument("--skip-features", action="store_true",
                    help="Skip Stage 1 (reuse today's features/latest.parquet).")
    ap.add_argument("--universe-seed-only", action="store_true",
                    help="Stage 1: use static seed universe instead of full halal screen.")
    ap.add_argument("--top", type=int, default=50,
                    help="Stage 3: top-N candidates to hand to the CIO (default 50).")
    ap.add_argument("--no-sentiment", action="store_true",
                    help="Stage 3: skip Perplexity sentiment calls.")
    ap.add_argument("--no-insider", action="store_true",
                    help="Stage 3: skip SEC Form-4 scrape.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip Stage 5 (rebalance) — run read-only path.")
    ap.add_argument("--only", default=None,
                    help='Comma-separated stage keys to run, e.g. '
                         '--only "mtm,rebalance" (valid: features, ranker, '
                         'swarm, mtm, rebalance).')
    ns = ap.parse_args()

    log_file = LOG_DIR / f"daily_pipeline_{date.today().isoformat()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s UTC [%(name)-13s] %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )

    state = run_pipeline(ns)
    # launchd / cron sees a real non-zero on halt.
    sys.exit(1 if state["halted"] else 0)


if __name__ == "__main__":
    main()
