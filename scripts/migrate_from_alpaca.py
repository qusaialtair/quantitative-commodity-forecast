#!/usr/bin/env python3
"""
scripts/migrate_from_alpaca.py
================================
One-time migration script.  Reads current positions + cash from the Alpaca
paper account and saves them into data/virtual_account.json so the
Virtual Trader can take over immediately.

Run once:
    python3 scripts/migrate_from_alpaca.py

If Alpaca is already locked, the script will fail gracefully and write a
zero-position virtual account seeded with the last known equity from
data/alpaca_daily_summary.json so the rest of the system keeps running.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

VIRTUAL_ACCOUNT_FILE = ROOT / "data" / "virtual_account.json"
DAILY_SUMMARY_FILE   = ROOT / "data" / "alpaca_daily_summary.json"

_API_KEY    = os.getenv("ALPACA_API_KEY",    "")
_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.rename(tmp, path)


def _fallback_from_summary() -> dict:
    """
    If Alpaca is unreachable, seed virtual account from the last daily summary
    (best-effort).  Positions are unknown so we start with cash only.
    """
    equity = 100_000.0
    if DAILY_SUMMARY_FILE.exists():
        try:
            s = json.loads(DAILY_SUMMARY_FILE.read_text())
            equity = float(s.get("account_equity_post_liquidation", equity))
            print(f"  Seeding from last daily summary: equity=${equity:,.2f}")
        except Exception:
            pass

    return {
        "migrated_at":     _now_utc(),
        "migrated_from":   "fallback_summary",
        "note":            "Alpaca unreachable — seeded from alpaca_daily_summary.json. Positions unknown.",
        "cash_balance":    equity,
        "positions":       {},
        "last_updated":    _now_utc(),
    }


def migrate() -> None:
    print("=" * 60)
    print("  Alpaca → Virtual Account Migration")
    print(f"  {_now_utc()}")
    print("=" * 60)

    # ── Attempt live Alpaca read ──────────────────────────────────────────────
    if not _API_KEY or not _SECRET_KEY:
        print("  WARN: Alpaca keys not found in .env — using fallback.")
        payload = _fallback_from_summary()
    else:
        try:
            from alpaca.trading.client import TradingClient

            print("  Connecting to Alpaca paper API…")
            client = TradingClient(_API_KEY, _SECRET_KEY, paper=True)

            acct   = client.get_account()
            cash   = float(acct.cash)
            equity = float(acct.equity)
            print(f"  Connected.  Equity=${equity:,.2f}  Cash=${cash:,.2f}")

            all_pos = client.get_all_positions()
            positions: dict[str, dict] = {}
            for pos in all_pos:
                symbol   = pos.symbol
                qty      = float(pos.qty)
                avg_cost = float(pos.avg_entry_price)
                mkt_val  = float(pos.market_value) if hasattr(pos, "market_value") else 0.0
                positions[symbol] = {
                    "qty":       round(qty, 6),
                    "avg_cost":  round(avg_cost, 4),
                    "mkt_value": round(mkt_val, 2),
                    "migrated":  True,
                }
                print(f"    {symbol:<8}  qty={qty:.4f}  avg_cost=${avg_cost:.2f}  mkt_val=${mkt_val:,.2f}")

            if not positions:
                print("  No open positions — virtual account starts as cash only.")

            payload = {
                "migrated_at":   _now_utc(),
                "migrated_from": "alpaca_paper",
                "cash_balance":  round(cash, 2),
                "equity_at_migration": round(equity, 2),
                "positions":     positions,
                "last_updated":  _now_utc(),
            }

        except Exception as exc:
            print(f"  WARN: Alpaca unreachable ({exc}) — using fallback.")
            payload = _fallback_from_summary()

    # ── Write virtual_account.json ────────────────────────────────────────────
    VIRTUAL_ACCOUNT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(VIRTUAL_ACCOUNT_FILE, payload)

    total_pos_val = sum(
        p.get("mkt_value", 0) for p in payload["positions"].values()
    )
    total_equity = payload["cash_balance"] + total_pos_val

    print()
    print(f"  Positions migrated : {len(payload['positions'])}")
    print(f"  Cash balance       : ${payload['cash_balance']:,.2f}")
    print(f"  Position value     : ${total_pos_val:,.2f}")
    print(f"  Total equity       : ${total_equity:,.2f}")
    print(f"  Written to         : {VIRTUAL_ACCOUNT_FILE}")
    print()
    print("  Migration complete.  Virtual Trader is now in control.")
    print("=" * 60)


if __name__ == "__main__":
    migrate()
