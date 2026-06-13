#!/usr/bin/env python3
"""Persist ALPACA_* credentials from the environment into .env (never hardcode keys)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"

api_key = os.environ.get("ALPACA_API_KEY", "").strip()
secret_key = os.environ.get("ALPACA_SECRET_KEY", "").strip()

if not api_key or not secret_key:
    print(
        "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in the environment, then re-run.",
        file=sys.stderr,
    )
    sys.exit(1)

lines: list[str] = []
if ENV_FILE.exists():
    lines = ENV_FILE.read_text().splitlines(keepends=True)

filtered = [
    line for line in lines
    if not line.startswith(("ALPACA_API_KEY=", "ALPACA_SECRET_KEY="))
]
if filtered and not filtered[-1].endswith("\n"):
    filtered[-1] += "\n"

filtered.append(f"ALPACA_API_KEY={api_key}\n")
filtered.append(f"ALPACA_SECRET_KEY={secret_key}\n")
ENV_FILE.write_text("".join(filtered))
print(f"Updated {ENV_FILE.name} with Alpaca credentials from the environment.")
