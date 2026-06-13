#!/usr/bin/env python3
"""
Auto-Continue for Claude Code (macOS)
======================================
When your Claude Code 5-hour usage window hits its cap, this script waits
until the reset moment and types "continue" (or any prompt you choose) into
the focused Terminal / iTerm window so work resumes without you having to
remember.

Usage
-----
    # Default: wait 5 hours from now, send "continue", loop forever
    python3 scripts/auto_continue.py

    # Wake at a specific clock time (today, or tomorrow if past)
    python3 scripts/auto_continue.py --at 14:30

    # Wake at an ISO datetime
    python3 scripts/auto_continue.py --at 2026-05-08T14:30

    # One-shot, no loop
    python3 scripts/auto_continue.py --in 5h --once

    # Custom prompt
    python3 scripts/auto_continue.py --prompt "continue improving per the grand plan"

    # Specify which terminal app to keystroke into
    python3 scripts/auto_continue.py --app iTerm2     # or "Terminal", "Ghostty", "Warp"

Setup (one-time)
----------------
macOS requires Accessibility permission for the app running this script
(Terminal.app or iTerm2). Grant via:
    System Settings -> Privacy & Security -> Accessibility -> add Terminal/iTerm2

If you have not granted it, the first keystroke call will prompt you.

Behavior
--------
- Default cycle is 5 hours (Claude Code's standard usage window).
- A 30-second buffer is added so the limit is fully reset before sending.
- Before each keystroke the script activates the configured terminal app and
  waits a short delay so focus is established.
- Press Ctrl+C to cancel.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = ROOT / "logs" / "auto_continue.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_CYCLE_HOURS = 5
RESET_BUFFER_SECONDS = 30  # Cushion past the limit reset moment
LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg: str, end: str = "\n", to_file: bool = True) -> None:
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, end=end, flush=True)
    if to_file:
        try:
            LOG_FILE.write_text(
                (LOG_FILE.read_text() if LOG_FILE.exists() else "") + line + "\n"
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Time parsing
# ---------------------------------------------------------------------------
_DUR_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd])\s*$", re.I)


def parse_duration(s: str) -> dt.timedelta:
    """Parse '5h', '30m', '90s', '1d' or a bare number of seconds."""
    s = s.strip()
    if s.isdigit():
        return dt.timedelta(seconds=int(s))
    m = _DUR_RE.match(s)
    if not m:
        raise ValueError(f"Unrecognized duration: {s!r}")
    n, unit = float(m.group(1)), m.group(2).lower()
    return dt.timedelta(
        seconds=n * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    )


def parse_target_time(s: str) -> dt.datetime:
    """Accepts ISO datetime, HH:MM, or HH:MM:SS."""
    s = s.strip()
    try:
        return dt.datetime.fromisoformat(s)
    except ValueError:
        pass
    m = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$", s)
    if not m:
        raise ValueError(f"Cannot parse time: {s!r}")
    h = int(m.group(1))
    mi = int(m.group(2))
    sec = int(m.group(3)) if m.group(3) else 0
    now = dt.datetime.now()
    target = now.replace(hour=h, minute=mi, second=sec, microsecond=0)
    if target <= now:
        target += dt.timedelta(days=1)
    return target


# ---------------------------------------------------------------------------
# Terminal detection and keystroke
# ---------------------------------------------------------------------------
def _osascript(script: str) -> tuple[int, str]:
    proc = subprocess.run(
        ["osascript", "-e", script], capture_output=True, text=True
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def detect_front_terminal() -> str:
    """Best-effort guess of which terminal app is currently active."""
    code, out = _osascript(
        'tell application "System Events" to get name of first process '
        "whose frontmost is true"
    )
    if code == 0 and out:
        name = out.strip()
        # Normalize known terminal app process names
        norm = {
            "Terminal": "Terminal",
            "iTerm2": "iTerm2",
            "iTerm": "iTerm2",
            "Ghostty": "Ghostty",
            "Warp": "Warp",
            "Alacritty": "Alacritty",
            "Hyper": "Hyper",
            "kitty": "kitty",
            "WezTerm": "WezTerm",
        }
        return norm.get(name, name)
    return "Terminal"


def activate_app(app_name: str) -> None:
    """Bring the given terminal app to the front."""
    # iTerm has both 'iTerm' (app) and 'iTerm2' (process). 'tell application "iTerm"' works.
    target = "iTerm" if app_name == "iTerm2" else app_name
    # Escape app name for AppleScript
    safe = target.replace('"', '\\"')
    _osascript(f'tell application "{safe}" to activate')


def send_keystroke(prompt: str, app_name: str, hold_delay: float = 0.6) -> None:
    """
    Activate the configured terminal and type the prompt followed by Return.
    Uses System Events keystroke; safer than relying on app-specific scripting
    which behaves differently across terminal apps.
    """
    activate_app(app_name)
    time.sleep(hold_delay)

    # Escape double-quotes and backslashes for AppleScript string literal
    safe_prompt = prompt.replace("\\", "\\\\").replace('"', '\\"')

    script = f'''
    tell application "System Events"
        keystroke "{safe_prompt}"
        delay 0.25
        key code 36
    end tell
    '''
    code, out = _osascript(script)
    if code != 0:
        log(f"!! osascript failed (code {code}): {out}")
        log("   You probably need to grant Accessibility permission to your terminal app.")
        log("   System Settings -> Privacy & Security -> Accessibility")
    else:
        log(f"-> Sent prompt to {app_name}: {prompt!r}")


# ---------------------------------------------------------------------------
# Wait loop with live countdown
# ---------------------------------------------------------------------------
def wait_until(target: dt.datetime, label: str = "next continue") -> None:
    last_render = 0
    while True:
        remaining = (target - dt.datetime.now()).total_seconds()
        if remaining <= 0:
            sys.stdout.write("\r" + " " * 78 + "\r")
            sys.stdout.flush()
            return
        h, rem = divmod(int(remaining), 3600)
        m, s = divmod(rem, 60)
        # Refresh the countdown line every second
        if time.monotonic() - last_render >= 1.0:
            sys.stdout.write(
                f"\r  {label}: {h:02d}:{m:02d}:{s:02d} remaining "
                f"(target {target.strftime('%H:%M:%S')})"
            )
            sys.stdout.flush()
            last_render = time.monotonic()
        # Sleep adaptively: long sleeps when far out, short near the end
        time.sleep(min(remaining, 1.0 if remaining < 60 else 5.0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Auto-continue Claude Code when usage limit resets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--at",
        help="Reset clock time (HH:MM, HH:MM:SS, or ISO). "
        "If past today, assumed tomorrow.",
    )
    g.add_argument(
        "--in",
        dest="in_dur",
        help="Wait this much from now: e.g. 5h, 30m, 90s",
    )
    p.add_argument(
        "--prompt",
        default="continue",
        help="Text to type into the terminal (default: 'continue')",
    )
    p.add_argument(
        "--app",
        default=None,
        help="Terminal app to keystroke into. Auto-detected if omitted. "
        "Examples: Terminal, iTerm2, Ghostty, Warp",
    )
    p.add_argument(
        "--cycle",
        default=f"{DEFAULT_CYCLE_HOURS}h",
        help="Cycle length when looping (default 5h, matches Claude Code window)",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Send a single continue and exit (default: loop forever)",
    )
    p.add_argument(
        "--no-buffer",
        action="store_true",
        help="Skip the 30-second post-reset cushion",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without sending any keystrokes",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cycle = parse_duration(args.cycle)
    buffer_secs = 0 if args.no_buffer else RESET_BUFFER_SECONDS

    # Determine first trigger time
    now = dt.datetime.now()
    if args.at:
        first = parse_target_time(args.at)
    elif args.in_dur:
        first = now + parse_duration(args.in_dur)
    else:
        first = now + cycle
    first += dt.timedelta(seconds=buffer_secs)

    # Detect terminal
    app_name = args.app or detect_front_terminal()

    # Banner
    log(SEP, to_file=False)
    log("  AUTO-CONTINUE FOR CLAUDE CODE", to_file=False)
    log(SEP, to_file=False)
    log(f"  Target terminal app   : {app_name}")
    log(f"  Prompt to send        : {args.prompt!r}")
    log(f"  First continue at     : {first.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Loop cycle            : {cycle} ({'one-shot' if args.once else 'looping'})")
    log(f"  Reset buffer          : {buffer_secs}s")
    log(f"  Log file              : {LOG_FILE}")
    log(SEP, to_file=False)
    log("  Tip: keep your Claude Code window focused or minimize all other apps")
    log("  so the keystroke lands in the right place. Ctrl+C to cancel.")
    log("")

    if args.dry_run:
        log("  [dry run] would send 'continue' at the time above. Exiting.")
        return 0

    # Trap Ctrl+C cleanly
    def _bye(signum, frame):
        log("\n  Cancelled by user.")
        sys.exit(0)
    signal.signal(signal.SIGINT, _bye)

    target = first
    iteration = 0
    while True:
        iteration += 1
        wait_until(target, label=f"continue #{iteration}")
        send_keystroke(args.prompt, app_name)

        if args.once:
            return 0

        target = dt.datetime.now() + cycle + dt.timedelta(seconds=buffer_secs)
        log(f"   next continue scheduled at {target.strftime('%Y-%m-%d %H:%M:%S')}")
        log("")


if __name__ == "__main__":
    raise SystemExit(main())
