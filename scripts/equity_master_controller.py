#!/usr/bin/env python3
"""
scripts/equity_master_controller.py
=====================================
Daily pipeline orchestrator for the Halal Equity trading system.

Execution order (strictly sequential — every gate is HARD ABORT):
    Stage 1  equity_screener.py   — Universe build + VAM/SMA200 screen → top 50
    Stage 2  equity_logic.py      — 3-agent DeepSeek Investment Committee → top 3
    Stage 3  alpaca_trader.py     — Rebalance Alpaca paper account to target weights

Abort logic:
    ANY stage non-zero exit code → immediate abort.
    Remaining stages are NEVER executed.
    Telegram URGENT alert fires instantly with the failed script + exit code.
    equity_pipeline_state.json is written with status=ABORTED so the UI reflects it.

Outputs:
    data/logs/equity_pipeline_YYYY-MM-DD.log  — structured daily audit log (UTC)
    data/equity_pipeline_state.json           — atomic UI snapshot (written on
                                                 completion OR abort)

CLI
---
  python3 scripts/equity_master_controller.py              # full run
  python3 scripts/equity_master_controller.py --force      # ignore all daily caches
  python3 scripts/equity_master_controller.py --dry-run    # all stages skipped; state written
  python3 scripts/equity_master_controller.py --skip-trade # run screener + logic, skip Alpaca
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── Paths ─────────────────────────────────────────────────────────────────────
LOG_DIR        = ROOT / "data" / "logs"
EQUITY_STATE   = ROOT / "data" / "equity_pipeline_state.json"
CANDIDATES     = ROOT / "data" / "halal_discovery_candidates.json"
DECISION       = ROOT / "data" / "equity_decision.json"
ALPACA_SUMMARY = ROOT / "data" / "alpaca_daily_summary.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

_LOG_FMT  = "%(asctime)s UTC  %(levelname)-7s  %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

# ── Stage definitions ─────────────────────────────────────────────────────────
# Each tuple: (stage_key, script_path, human_label)
_STAGES = [
    ("screener",      "scripts/equity_screener.py",            "Equity Screener"),
    ("equity_logic",  "scripts/equity_logic.py",               "Investment Committee"),
    ("alpaca_trader", "scripts/virtual_trader.py",             "Virtual Trader"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging(run_date: str) -> logging.Logger:
    log_path = LOG_DIR / f"equity_pipeline_{run_date}.log"
    root = logging.getLogger()
    root.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format=_LOG_FMT,
        datefmt=_DATE_FMT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    for _lib in ("yfinance", "urllib3", "peewee", "charset_normalizer",
                 "openai", "httpx", "alpaca"):
        logging.getLogger(_lib).setLevel(logging.ERROR)
    logger = logging.getLogger("equity_master")
    logger.info("Log file: %s", log_path)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _atomic_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.rename(tmp, path)


def _safe_read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Telegram helpers
# ─────────────────────────────────────────────────────────────────────────────

def _alert_stage_failure(stage_label: str, script: str, exit_code: int,
                          stderr_tail: str) -> None:
    """Fire a Telegram alert when a stage aborts the pipeline."""
    try:
        from scripts.telegram_notifier import send_alert
        context = (
            f"script={script}  exit_code={exit_code}\n"
            f"stderr: {stderr_tail[:300]}"
        )
        send_alert(
            "\n".join([
                "<b>PIPELINE HALTED — Equity Stage Failed</b>",
                "",
                f"<b>Stage:</b> <code>{stage_label}</code>",
                f"<b>Exit:</b>  <code>{exit_code}</code>",
                f"<b>Detail:</b> <code>{context[:400]}</code>",
            ]),
            level="URGENT",
        )
    except Exception as exc:
        logging.getLogger("equity_master").warning(
            "Telegram urgent alert failed: %s", exc
        )


def _send_success_heartbeat(state: dict) -> None:
    """Fire the daily completion heartbeat after a successful full run."""
    try:
        from scripts.telegram_notifier import send_alert

        today    = state.get("run_date", "")
        regime   = state.get("committee", {}).get("market_regime", "UNKNOWN")
        spsk_or  = state.get("committee", {}).get("spsk_override", False)
        equity   = state.get("execution", {}).get("account_equity_post_liquidation", 0.0)
        sells    = state.get("execution", {}).get("liquidations", [])
        picks    = state.get("committee", {}).get("selections", [])
        dur      = state.get("total_duration_s", 0.0)

        lines = [
            f"✅ <b>Equity Pipeline Complete — {today}</b>",
            f"<b>Regime:</b> <code>{regime}</code>   "
            f"<b>SPSK:</b> <code>{'YES' if spsk_or else 'no'}</code>   "
            f"<b>Equity:</b> <code>${equity:,.0f}</code>",
            "",
            "<b>Portfolio:</b>",
        ]
        for p in picks:
            lines.append(
                f"  #{p['rank']} <code>{p['ticker']}</code>  "
                f"{int(p.get('weight', 0) * 100)}%  "
                f"VAM {p.get('vam_score', 0):.2f}  "
                f"<i>{p.get('sector', '')}</i>"
            )

        if sells:
            lines += ["", "<b>Liquidated:</b>"]
            for s in sells:
                lines.append(f"  SELL <code>{s['ticker']}</code>  "
                             f"{s.get('qty', 0):.2f} shares")

        lines.append(f"\n<b>Runtime:</b> <code>{dur:.0f}s</code>")
        lines.append(f"<i>{_now_utc()}</i>")

        send_alert("\n".join(lines), level="OK")
    except Exception as exc:
        logging.getLogger("equity_master").warning(
            "Telegram success heartbeat failed: %s", exc
        )


def _send_abort_heartbeat(state: dict, stage_label: str, exit_code: int) -> None:
    """Fire an abort summary heartbeat (distinct from the per-stage urgent alert)."""
    try:
        from scripts.telegram_notifier import send_alert
        send_alert(
            f"🚨 <b>Equity Pipeline ABORTED</b> — {state.get('run_date', '')}\n"
            f"<b>Failed stage:</b> <code>{stage_label}</code>  "
            f"<b>Exit code:</b> <code>{exit_code}</code>\n"
            f"<i>{_now_utc()}</i>",
            level="URGENT",
        )
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# State builder — assembles equity_pipeline_state.json from output files
# ─────────────────────────────────────────────────────────────────────────────

def _build_state(
    today: str,
    stage_results: dict[str, dict],
    pipeline_status: str,
    abort_reason: str,
    total_s: float,
) -> dict:
    """
    Read the three output JSON files and compile the single UI state document.
    Safe — all reads are wrapped; missing keys return sensible defaults.
    """
    screener_data  = _safe_read_json(CANDIDATES)
    decision_data  = _safe_read_json(DECISION)
    alpaca_data    = _safe_read_json(ALPACA_SUMMARY)

    # ── Screener block ────────────────────────────────────────────────────────
    screener_block = {
        "run_date":        screener_data.get("run_date", ""),
        "universe_size":   screener_data.get("universe_size", 0),
        "candidates_n":    screener_data.get("candidates_n", 0),
        "screen_params":   screener_data.get("screen_params", {}),
        "sukuk_benchmark": screener_data.get("sukuk_benchmark", {}),
    }
    # Derive SMA200 survivors from candidates (all trend_aligned=True passed gate)
    candidates_list = screener_data.get("candidates", [])
    screener_block["sma200_survivors"] = sum(
        1 for c in candidates_list if c.get("trend_aligned", True)
    )

    # ── Committee block ───────────────────────────────────────────────────────
    committee_block = {
        "decision_date":  decision_data.get("decision_date", ""),
        "market_regime":  decision_data.get("market_regime", "UNKNOWN"),
        "spsk_override":  decision_data.get("spsk_override", False),
        "weights_sum":    decision_data.get("weights_sum", 0.0),
        "selections":     decision_data.get("selections", []),
        "quant_top10":    decision_data.get("quant_top10", []),
        "excluded_tickers": decision_data.get("excluded_tickers", []),
        "macro_context":  decision_data.get("macro_context", {}),
        "cio_reasoning":  decision_data.get("cio_reasoning", ""),
    }

    # ── Execution block ───────────────────────────────────────────────────────
    execution_block = {
        "run_date":              alpaca_data.get("run_date", ""),
        "pipeline_status":       alpaca_data.get("pipeline_status", "UNKNOWN"),
        "dry_run":               alpaca_data.get("dry_run", False),
        "account_equity_post_liquidation": alpaca_data.get("account_equity_post_liquidation", 0.0),
        "deployable_usd":        alpaca_data.get("deployable_usd", 0.0),
        "liquidations":          alpaca_data.get("liquidations", []),
        "new_positions":         alpaca_data.get("new_positions", []),
        "sell_orders_filled":    alpaca_data.get("sell_orders_filled", False),
    }

    return {
        "run_date":         today,
        "run_timestamp":    _now_utc(),
        "pipeline_status":  pipeline_status,
        "abort_reason":     abort_reason,
        "stages":           stage_results,
        "screener":         screener_block,
        "committee":        committee_block,
        "execution":        execution_block,
        "total_duration_s": round(total_s, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_stage(
    logger: logging.Logger,
    stage_key: str,
    script: str,
    label: str,
    extra_args: list[str],
    dry_run: bool,
) -> tuple[bool, dict]:
    """
    Execute one pipeline stage via subprocess.run().

    Returns:
        (success: bool, result: dict)  where result contains status/duration/exit_code.

    On dry_run: skips execution and returns a synthetic OK result.
    """
    if dry_run:
        logger.info("[DRY RUN] Skipping %s", label)
        return True, {"status": "DRY_RUN", "duration_s": 0.0, "exit_code": 0}

    cmd = [sys.executable, str(ROOT / script)] + extra_args
    logger.info("─" * 60)
    logger.info("STAGE START  %-22s  cmd: %s", label, " ".join(cmd))
    t0 = time.time()

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    duration = round(time.time() - t0, 1)
    exit_code = proc.returncode

    # Always surface stdout/stderr to the master log
    if proc.stdout.strip():
        for line in proc.stdout.strip().splitlines():
            logger.info("  [%s] %s", stage_key, line)
    if proc.stderr.strip():
        for line in proc.stderr.strip().splitlines():
            logger.warning("  [%s] STDERR: %s", stage_key, line)

    result = {
        "duration_s": duration,
        "exit_code":  exit_code,
    }

    if exit_code == 0:
        logger.info("STAGE OK     %-22s  %.1fs", label, duration)
        result["status"] = "OK"
        return True, result
    else:
        logger.error(
            "STAGE FAILED %-22s  exit=%d  %.1fs",
            label, exit_code, duration,
        )
        result["status"] = "FAILED"
        return False, result


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(force: bool = False, dry_run: bool = False, skip_trade: bool = False) -> int:
    """
    Orchestrate all three equity pipeline stages.
    Returns 0 on SUCCESS, 1 on ABORTED/PARTIAL.
    """
    today   = date.today().isoformat()
    logger  = _setup_logging(today)
    t_start = time.time()

    logger.info("=" * 60)
    logger.info("EQUITY PIPELINE START  %s", _now_utc())
    logger.info("force=%s  dry_run=%s  skip_trade=%s", force, dry_run, skip_trade)
    logger.info("=" * 60)

    # Build per-stage extra args
    stage_args: dict[str, list[str]] = {
        "screener":      ["--force"] if force else [],
        "equity_logic":  ["--force"] if force else [],
        "alpaca_trader": ["--force"] if force else [],
    }

    stage_results:   dict[str, dict] = {}
    pipeline_status  = "SUCCESS"
    abort_reason     = ""
    aborted_label    = ""
    aborted_code     = 0

    active_stages = [
        s for s in _STAGES
        if not (s[0] == "alpaca_trader" and skip_trade)
    ]

    # ── Sequential stage execution ────────────────────────────────────────────
    for stage_key, script, label in active_stages:
        success, result = _run_stage(
            logger, stage_key, script, label,
            stage_args.get(stage_key, []),
            dry_run,
        )
        stage_results[stage_key] = result

        if not success:
            pipeline_status = "ABORTED"
            abort_reason    = (
                f"{label} exited with code {result['exit_code']}"
            )
            aborted_label = label
            aborted_code  = result["exit_code"]

            # ── HARD ABORT ────────────────────────────────────────────────────
            logger.error("PIPELINE ABORTED — %s", abort_reason)

            # Mark remaining stages as skipped
            remaining = [s for s in active_stages
                         if s[0] not in stage_results]
            for sk, _, lbl in remaining:
                stage_results[sk] = {"status": "SKIPPED", "duration_s": 0.0,
                                     "exit_code": None}
                logger.info("  SKIPPED  %s  (upstream abort)", lbl)

            # Telegram URGENT alert
            _alert_stage_failure(
                label, script, aborted_code,
                stderr_tail="(see equity_pipeline log for details)",
            )
            break

    # If skip_trade was set, mark alpaca_trader as skipped in results
    if skip_trade and "alpaca_trader" not in stage_results:
        stage_results["alpaca_trader"] = {
            "status": "SKIPPED", "duration_s": 0.0, "exit_code": None
        }

    total_s = round(time.time() - t_start, 1)

    # ── Build + write state ───────────────────────────────────────────────────
    state = _build_state(today, stage_results, pipeline_status, abort_reason, total_s)

    if not dry_run:
        _atomic_write(EQUITY_STATE, state)
        logger.info("Wrote equity_pipeline_state.json")

    # ── Telegram final message ────────────────────────────────────────────────
    if not dry_run:
        if pipeline_status == "SUCCESS":
            _send_success_heartbeat(state)
        else:
            _send_abort_heartbeat(state, aborted_label, aborted_code)

    # ── Console summary ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("EQUITY PIPELINE %s  |  %.0fs total", pipeline_status, total_s)
    for sk, res in stage_results.items():
        label = next((lbl for key, _, lbl in _STAGES if key == sk), sk)
        logger.info(
            "  %-22s  %-8s  %.1fs",
            label,
            res.get("status", "UNKNOWN"),
            res.get("duration_s", 0.0),
        )
    if pipeline_status == "SUCCESS":
        picks = state.get("committee", {}).get("selections", [])
        for p in picks:
            logger.info(
                "  %-6s  %.0f%%  VAM %.2f",
                p["ticker"],
                p.get("weight", 0) * 100,
                p.get("vam_score", 0),
            )
    logger.info("=" * 60)

    return 0 if pipeline_status == "SUCCESS" else 1


# ─────────────────────────────────────────────────────────────────────────────
# launchd installer  (macOS — runs pipeline nightly at 22:00 UTC)
# ─────────────────────────────────────────────────────────────────────────────

def _install_launchd() -> None:
    """
    Write a launchd plist to ~/Library/LaunchAgents/ and load it.
    Schedules equity_master_controller.py to run every day at 22:00 UTC
    (US market closes 21:00 UTC; UAE morning check is ~06:00 UTC+4).

    Usage:
        python3 scripts/equity_master_controller.py --install-launchd
    """
    import subprocess as _sp

    python  = sys.executable
    script  = str(Path(__file__).resolve())
    workdir = str(ROOT)
    label   = "com.equity.master_controller"
    plist   = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    logfile = str(ROOT / "data" / "logs" / "equity_launchd.log")

    # 22:00 UTC = local hour depends on system timezone.
    # Force UTC via TZ env var so launchd always fires at the right moment.
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>             <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python}</string>
        <string>{script}</string>
    </array>
    <key>WorkingDirectory</key>  <string>{workdir}</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>          <integer>22</integer>
        <key>Minute</key>        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>   <string>{logfile}</string>
    <key>StandardErrorPath</key> <string>{logfile}</string>
    <key>RunAtLoad</key>         <false/>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>TZ</key>
        <string>UTC</string>
    </dict>
</dict>
</plist>
"""
    plist.parent.mkdir(parents=True, exist_ok=True)
    plist.write_text(xml)
    print(f"plist written → {plist}")

    # Unload any prior version, then load fresh
    _sp.run(["launchctl", "unload", str(plist)], capture_output=True)
    result = _sp.run(["launchctl", "load",   str(plist)],
                     capture_output=True, text=True)
    if result.returncode == 0:
        print("launchd job loaded — equity pipeline will run nightly at 22:00 UTC.")
        print(f"  Log  → {logfile}")
        print(f"  Stop → launchctl unload {plist}")
    else:
        print(f"launchctl load failed: {result.stderr}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Halal Equity Pipeline — daily orchestrator"
    )
    parser.add_argument(
        "--force",           action="store_true",
        help="Pass --force to all sub-scripts (ignore their daily caches)",
    )
    parser.add_argument(
        "--dry-run",         action="store_true",
        help="Skip all subprocess calls; write a synthetic state file",
    )
    parser.add_argument(
        "--skip-trade",      action="store_true",
        help="Run screener + committee only; skip Alpaca execution",
    )
    parser.add_argument(
        "--install-launchd", action="store_true",
        help="Install macOS launchd job to run pipeline nightly at 22:00 UTC",
    )
    args = parser.parse_args()

    if args.install_launchd:
        _install_launchd()
        return

    exit_code = run(
        force      = args.force,
        dry_run    = args.dry_run,
        skip_trade = args.skip_trade,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
