#!/usr/bin/env python3
"""
scripts/system_health_check.py
===============================
Manual diagnostic script — run before burn-in or after any system change.

Checks:
  1. API connectivity  — Alpaca paper API + DeepSeek chat endpoint
  2. File integrity    — halal_discovery_candidates.json + equity_pipeline_state.json
  3. Schedule log      — last 10 lines of launchd_cron.log; flag errors / exit-1
  4. UI reachability   — Streamlit on localhost:8501

Usage:
    python3 scripts/system_health_check.py
    python3 scripts/system_health_check.py --verbose   # print full file summaries
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

# ── Bootstrap ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

# ── ANSI colours (graceful fallback on Windows) ───────────────────────────────
_TTY = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text

def OK(msg: str)   -> str: return _c("32;1", f"  PASS  ") + msg
def FAIL(msg: str) -> str: return _c("31;1", f"  FAIL  ") + msg
def WARN(msg: str) -> str: return _c("33;1", f"  WARN  ") + msg
def INFO(msg: str) -> str: return _c("36",   f"  INFO  ") + msg
def HDR(msg: str)  -> str: return _c("1",    msg)

_results: list[tuple[str, bool | None]] = []   # (label, pass/fail/None=warn)


def record(label: str, passed: bool | None, detail: str = "") -> None:
    _results.append((label, passed))
    if passed is True:
        print(OK(label) + (f"  — {detail}" if detail else ""))
    elif passed is False:
        print(FAIL(label) + (f"  — {detail}" if detail else ""))
    else:
        print(WARN(label) + (f"  — {detail}" if detail else ""))


# ══════════════════════════════════════════════════════════════════════════════
# Check 1 — Virtual account integrity (replaces Alpaca API check)
# ══════════════════════════════════════════════════════════════════════════════

def check_alpaca(verbose: bool) -> None:
    """Check virtual_account.json health (Alpaca replaced by Virtual Trader)."""
    print(HDR("\n[1/4] Virtual Account"))

    va_path = ROOT / "data" / "virtual_account.json"

    if not va_path.exists():
        record("virtual_account.json exists", False,
               "Run: python3 scripts/migrate_from_alpaca.py")
        return

    size_kb = va_path.stat().st_size / 1024
    if size_kb < 0.05:
        record("virtual_account.json exists", False, f"file is empty ({size_kb:.1f} KB)")
        return

    try:
        data      = json.loads(va_path.read_text())
        cash      = float(data.get("cash_balance", 0.0))
        positions = data.get("positions", {})
        migrated  = data.get("migrated_at", "unknown")
        updated   = data.get("last_updated", "unknown")

        # Compute mark-to-market equity (use avg_cost as proxy — no live fetch here)
        pos_value = sum(
            float(p["qty"]) * float(p.get("avg_cost", 0))
            for p in positions.values()
        )
        approx_equity = cash + pos_value

        record("virtual_account.json exists", True,
               f"{size_kb:.1f} KB  |  migrated={migrated[:10]}")
        record("Virtual account balance", True,
               f"equity≈${approx_equity:,.2f}  |  cash=${cash:,.2f}  |  "
               f"positions={len(positions)}  |  last_updated={updated[:10]}")

        if verbose and positions:
            for sym, p in positions.items():
                print(INFO(f"  {sym:<8}  qty={p['qty']:.4f}  avg_cost=${p.get('avg_cost', 0):.2f}"))

    except Exception as exc:
        record("virtual_account.json readable", False, str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# Check 2 — DeepSeek API connectivity
# ══════════════════════════════════════════════════════════════════════════════

def check_deepseek(verbose: bool) -> None:
    print(HDR("\n[2/4] DeepSeek API (equity logic engine)"))
    ds_key = os.getenv("DEEPSEEK_API_KEY", "")

    if not ds_key:
        record("DEEPSEEK_API_KEY in .env", False, "key not set")
        return

    record("DEEPSEEK_API_KEY present", True, f"{ds_key[:8]}…")

    try:
        from openai import OpenAI
        t0     = time.time()
        client = OpenAI(api_key=ds_key, base_url="https://api.deepseek.com")
        resp   = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
            temperature=0.0,
            max_tokens=5,
            timeout=20,
        )
        ms   = int((time.time() - t0) * 1000)
        body = resp.choices[0].message.content or ""
        record("DeepSeek API reachable", True, f"{ms} ms  |  response='{body.strip()}'")
    except Exception as exc:
        record("DeepSeek API reachable", False, str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# Check 3 — Critical file integrity
# ══════════════════════════════════════════════════════════════════════════════

def check_files(verbose: bool) -> None:
    print(HDR("\n[3/4] Pipeline File Integrity"))

    files_to_check = [
        {
            "path":        ROOT / "data" / "halal_discovery_candidates.json",
            "label":       "halal_discovery_candidates.json",
            "date_key":    "run_date",
            "size_key":    "candidates_n",
            "size_label":  "candidates",
        },
        {
            "path":        ROOT / "data" / "equity_pipeline_state.json",
            "label":       "equity_pipeline_state.json",
            "date_key":    "run_date",
            "size_key":    None,
            "size_label":  None,
        },
        {
            "path":        ROOT / "data" / "equity_decision.json",
            "label":       "equity_decision.json",
            "date_key":    "decision_date",
            "size_key":    None,
            "size_label":  None,
        },
        {
            "path":        ROOT / "data" / "alpaca_daily_summary.json",
            "label":       "alpaca_daily_summary.json",
            "date_key":    "run_date",
            "size_key":    None,
            "size_label":  None,
        },
    ]

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for spec in files_to_check:
        path: Path = spec["path"]

        if not path.exists():
            record(spec["label"], False, "file not found")
            continue

        size_kb = path.stat().st_size / 1024
        if size_kb < 0.05:
            record(spec["label"], False, f"file is empty ({size_kb:.1f} KB)")
            continue

        try:
            data     = json.loads(path.read_text())
            run_date = data.get(spec["date_key"], "unknown")
            stale    = run_date != today

            detail_parts = [f"run_date={run_date}", f"{size_kb:.1f} KB"]
            if spec["size_key"]:
                n = data.get(spec["size_key"])
                if n is not None:
                    detail_parts.append(f"{n} {spec['size_label']}")

            detail = "  |  ".join(detail_parts)

            if stale:
                record(spec["label"], None, f"STALE — {detail}  (today={today})")
            else:
                record(spec["label"], True, detail)

            # Schema spot-checks
            if spec["label"] == "equity_pipeline_state.json":
                _check_pipeline_state_schema(data, verbose)

            if spec["label"] == "halal_discovery_candidates.json":
                cands = data.get("candidates", [])
                if cands:
                    top = cands[0]
                    required = {"ticker", "vam_score", "return_6m", "vol_6m_ann", "sma200"}
                    missing  = required - set(top.keys())
                    if missing:
                        record("  Schema: candidates fields", None, f"missing keys: {missing}")
                    else:
                        record("  Schema: candidates fields", True,
                               f"top ticker={top['ticker']}  VAM={top['vam_score']:.3f}")

        except json.JSONDecodeError as exc:
            record(spec["label"], False, f"JSON parse error: {exc}")
        except Exception as exc:
            record(spec["label"], False, str(exc))


def _check_pipeline_state_schema(data: dict, verbose: bool) -> None:
    """Spot-check equity_pipeline_state.json schema."""
    required_top = {"run_date", "pipeline_status", "committee", "execution"}
    missing      = required_top - set(data.keys())
    if missing:
        record("  Schema: pipeline_state top-level keys", False, f"missing: {missing}")
    else:
        record("  Schema: pipeline_state top-level keys", True)

    committee = data.get("committee", {})
    selections = committee.get("selections", [])
    regime     = committee.get("market_regime", "UNKNOWN")
    spsk_or    = committee.get("spsk_override", False)
    record("  Schema: committee.selections", True if selections else None,
           f"{len(selections)} picks  |  regime={regime}  |  spsk_override={spsk_or}")

    execution = data.get("execution", {})
    eq = execution.get("account_equity_post_liquidation", 0.0)
    ps = execution.get("pipeline_status", "UNKNOWN")
    record("  Schema: execution block",
           True if ps == "SUCCESS" else None,
           f"status={ps}  |  equity=${eq:,.2f}")

    if verbose and selections:
        for p in selections:
            print(INFO(
                f"    #{p.get('rank',0)}  {p.get('ticker','?'):6s}  "
                f"weight={p.get('weight',0):.0%}  sector={p.get('sector','?')}"
            ))


# ══════════════════════════════════════════════════════════════════════════════
# Check 4 — launchd schedule log
# ══════════════════════════════════════════════════════════════════════════════

_ERROR_PATTERNS = [
    "error",
    "exit code 1",
    "exit 1",
    "traceback",
    "exception",
    "failed",
    "killed",
    "abort",
]

def check_schedule_log(verbose: bool) -> None:
    print(HDR("\n[4/4] launchd Schedule Log"))

    log_path = ROOT / "data" / "logs" / "launchd_cron.log"

    if not log_path.exists():
        record("launchd_cron.log exists", None,
               f"not found at {log_path} — launchd may not have fired yet")
        return

    size_kb = log_path.stat().st_size / 1024
    record("launchd_cron.log exists", True, f"{size_kb:.1f} KB")

    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except Exception as exc:
        record("launchd_cron.log readable", False, str(exc))
        return

    tail = lines[-10:] if len(lines) >= 10 else lines
    record("launchd_cron.log last lines", True, f"showing last {len(tail)} of {len(lines)} lines")

    print()
    flagged: list[tuple[int, str]] = []

    for i, line in enumerate(tail, start=max(1, len(lines) - 9)):
        low = line.lower()
        is_bad = any(pat in low for pat in _ERROR_PATTERNS)
        if is_bad:
            flagged.append((i, line))
            prefix = _c("31;1", f"  L{i:>4}  !! ")
        else:
            prefix = _c("2",    f"  L{i:>4}     ")
        print(prefix + line)

    print()
    if flagged:
        record("launchd_cron.log clean",
               False,
               f"{len(flagged)} suspicious line(s) flagged — review above")
    else:
        record("launchd_cron.log clean", True, "no errors or exit-1 patterns detected")


# ══════════════════════════════════════════════════════════════════════════════
# Check 5 — Streamlit UI reachability
# ══════════════════════════════════════════════════════════════════════════════

_STREAMLIT_PORTS = [8501, 8502, 8503]

def check_ui(verbose: bool) -> None:
    print(HDR("\n[5/4] Streamlit UI"))   # labelled 5 but fits after schedule log

    reached = False
    for port in _STREAMLIT_PORTS:
        url = f"http://localhost:{port}/"
        try:
            t0   = time.time()
            resp = requests.get(url, timeout=4, allow_redirects=True)
            ms   = int((time.time() - t0) * 1000)
            if resp.status_code < 500:
                record(f"Streamlit UI on port {port}", True,
                       f"HTTP {resp.status_code}  |  {ms} ms")
                reached = True
                break
            else:
                record(f"Streamlit UI on port {port}", None,
                       f"HTTP {resp.status_code}")
        except requests.exceptions.ConnectionError:
            # Not running on this port — try next
            continue
        except requests.exceptions.Timeout:
            record(f"Streamlit UI on port {port}", None, "timeout (>4 s)")
            continue
        except Exception as exc:
            record(f"Streamlit UI on port {port}", None, str(exc))

    if not reached:
        record("Streamlit UI reachable", False,
               f"not responding on ports {_STREAMLIT_PORTS}  "
               "— run: streamlit run ui/app.py")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

def print_summary() -> int:
    passed  = sum(1 for _, v in _results if v is True)
    warned  = sum(1 for _, v in _results if v is None)
    failed  = sum(1 for _, v in _results if v is False)
    total   = len(_results)

    print()
    print("─" * 64)
    print(HDR(f"System Health Summary — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"))
    print(f"  {_c('32;1', f'PASS  {passed:>2}')} / {total}   "
          f"{_c('33;1', f'WARN  {warned:>2}')}   "
          f"{_c('31;1', f'FAIL  {failed:>2}')}")
    print("─" * 64)

    if failed == 0 and warned == 0:
        print(_c("32;1", "  All checks passed. System ready for burn-in."))
    elif failed == 0:
        print(_c("33;1", "  Warnings present — review before proceeding."))
    else:
        print(_c("31;1", f"  {failed} failure(s) detected — do NOT start burn-in until resolved."))

    print()
    return 1 if failed > 0 else 0


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gold Trading AI — System Health Check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print extended details (account info, pick breakdown, etc.)")
    args = parser.parse_args()

    print("=" * 64)
    print(HDR("  Gold Trading AI — System Health Check"))
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 64)

    check_alpaca(args.verbose)
    check_deepseek(args.verbose)
    check_files(args.verbose)
    check_schedule_log(args.verbose)
    check_ui(args.verbose)

    exit_code = print_summary()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
