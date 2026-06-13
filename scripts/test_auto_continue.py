#!/usr/bin/env python3
"""
Self-test for scripts/auto_continue.py
========================================
Verifies every component of the auto-continue pipeline without typing into
your live Claude Code session.

Tests:
  1. Duration parsing (5h, 30m, 90s, 1d)
  2. Target-time parsing (HH:MM, ISO, past-time-rolls-to-tomorrow)
  3. Terminal detection
  4. osascript availability
  5. End-to-end keystroke into TextEdit  (--keystroke only)

Usage:
    python3 scripts/test_auto_continue.py
    python3 scripts/test_auto_continue.py --keystroke
"""
from __future__ import annotations

import datetime as dt
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.auto_continue import (
    _osascript,
    detect_front_terminal,
    parse_duration,
    parse_target_time,
    send_keystroke,
)

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"


def t(label: str, ok: bool, detail: str = "") -> bool:
    sym = PASS if ok else FAIL
    msg = f"  {sym} {label}"
    if detail:
        msg += f"  — {detail}"
    print(msg)
    return ok


def test_duration_parsing() -> bool:
    cases = [
        ("5h", 5 * 3600),
        ("30m", 30 * 60),
        ("90s", 90),
        ("1d", 86400),
        ("2.5h", 2.5 * 3600),
        ("60", 60),
    ]
    ok = True
    for s, expected in cases:
        try:
            actual = parse_duration(s).total_seconds()
            if abs(actual - expected) > 0.001:
                ok = t(f"parse_duration({s!r})", False,
                       f"got {actual}, expected {expected}")
            else:
                t(f"parse_duration({s!r})", True, f"= {actual}s")
        except Exception as e:
            ok = t(f"parse_duration({s!r})", False, f"exception: {e}")
    return ok


def test_time_parsing() -> bool:
    ok = True
    # HH:MM in the future today
    now = dt.datetime.now()
    future_hh = (now.hour + 1) % 24
    s = f"{future_hh:02d}:00"
    try:
        target = parse_target_time(s)
        if target > now:
            t(f"parse_target_time({s!r}) future", True, f"→ {target}")
        else:
            ok = t(f"parse_target_time({s!r}) future", False, f"→ {target} (not in future)")
    except Exception as e:
        ok = t(f"parse_target_time({s!r}) future", False, f"exception: {e}")

    # Past HH:MM should roll to tomorrow
    past_hh = (now.hour - 1) % 24
    s2 = f"{past_hh:02d}:00"
    try:
        target = parse_target_time(s2)
        rolled = target.date() > now.date()
        ok = t(f"parse_target_time({s2!r}) past→tomorrow", rolled,
                f"→ {target} (today is {now.date()})") and ok
    except Exception as e:
        ok = t(f"parse_target_time({s2!r})", False, f"exception: {e}") and ok

    # ISO datetime
    iso = "2030-01-01T12:00:00"
    try:
        target = parse_target_time(iso)
        good = target.year == 2030 and target.month == 1
        ok = t(f"parse_target_time({iso!r})", good, f"→ {target}") and ok
    except Exception as e:
        ok = t(f"parse_target_time({iso!r})", False, f"exception: {e}") and ok
    return ok


def test_osascript() -> bool:
    code, out = _osascript('return "auto_continue_ok"')
    return t("osascript executes", code == 0 and "auto_continue_ok" in out,
             f"output: {out!r}")


def test_detect_terminal() -> bool:
    app = detect_front_terminal()
    return t("detect_front_terminal", bool(app),
             f"detected: {app!r}")


def test_keystroke_into_textedit() -> bool:
    """End-to-end keystroke test using TextEdit as a benign target."""
    print("\n  ---- end-to-end keystroke test ----")
    print("  Will: open TextEdit → send 'auto_continue verification ok'")
    print("        → read back → close without saving.")
    print("  Counting down 5s — DO NOT switch focus.")
    for i in range(5, 0, -1):
        print(f"    {i}...", flush=True)
        time.sleep(1)

    # Open TextEdit with a new document
    code, out = _osascript('''
        tell application "TextEdit"
            activate
            if (count of documents) = 0 then
                make new document
            else
                make new document
            end if
        end tell
    ''')
    if code != 0:
        return t("open TextEdit", False, out)
    time.sleep(1.0)

    test_phrase = "auto_continue verification ok"
    send_keystroke(test_phrase, "TextEdit", hold_delay=0.8)
    time.sleep(1.0)

    # Read back
    code, content = _osascript('''
        tell application "TextEdit"
            get text of front document
        end tell
    ''')

    # Close without saving
    _osascript('''
        tell application "TextEdit"
            close front document saving no
        end tell
    ''')

    landed = test_phrase in content
    return t("keystroke landed in TextEdit", landed,
             f"got: {content[:80]!r}")


def main() -> int:
    print("\nAUTO-CONTINUE SELF-TEST")
    print("=" * 50)

    print("\n[ duration parsing ]")
    p1 = test_duration_parsing()

    print("\n[ time parsing ]")
    p2 = test_time_parsing()

    print("\n[ osascript availability ]")
    p3 = test_osascript()

    print("\n[ terminal detection ]")
    p4 = test_detect_terminal()

    keystroke_ok = True
    if "--keystroke" in sys.argv:
        print("\n[ keystroke pipeline ]")
        keystroke_ok = test_keystroke_into_textedit()
    else:
        print("\n[ keystroke pipeline ]")
        print("  (skipped — re-run with --keystroke for end-to-end test)")
        print("  When you do, focus will jump to TextEdit briefly.")

    print("\n" + "=" * 50)
    all_ok = p1 and p2 and p3 and p4 and keystroke_ok
    if all_ok:
        print(f"  {PASS} ALL TESTS PASSED")
        print("  scripts/auto_continue.py is ready to use.")
        return 0
    print(f"  {FAIL} SOME TESTS FAILED")
    print("  Most common cause: macOS Accessibility permission not granted.")
    print("  Fix: System Settings → Privacy & Security → Accessibility")
    print("       → add Terminal/iTerm2/your terminal app.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
