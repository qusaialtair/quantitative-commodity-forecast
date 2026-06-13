#!/usr/bin/env python3
"""
Compliance / Audit Trail  (Phase IX Stage 49)
===============================================
Append-only SQLite log with a cryptographic hash chain — every signal,
sizing decision, order, and risk event is recorded so that the platform
is regulatory-ready.

Schema:

  audit_events
  ┌────────────────┬───────────┬─────────────────────────────────────────┐
  │ id             │ INTEGER PK│ autoincrement                           │
  │ ts_utc         │ TEXT      │ ISO 8601 UTC                            │
  │ event_type     │ TEXT      │ SIGNAL / DECISION / ORDER / RISK / ...  │
  │ ticker         │ TEXT      │ asset symbol or NULL                    │
  │ payload_json   │ TEXT      │ full event body                         │
  │ prev_hash      │ TEXT      │ SHA-256 of previous row                 │
  │ row_hash       │ TEXT      │ SHA-256(prev_hash + canonical_payload)  │
  └────────────────┴───────────┴─────────────────────────────────────────┘

Operations:
  - record(event_type, payload, ticker=None) → row_id, row_hash
  - verify_chain()                            → bool, first_break_idx
  - tail(limit)                               → recent rows
  - count_by_type()                           → dict

Storage: data/audit_trail.db

Output (CLI): data/audit_trail_status.json with chain verification + counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
DB_FILE = DATA_DIR / "audit_trail.db"
OUTPUT_FILE = DATA_DIR / "audit_trail_status.json"

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Schema + helpers
# ---------------------------------------------------------------------------
def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            event_type TEXT NOT NULL,
            ticker TEXT,
            payload_json TEXT NOT NULL,
            prev_hash TEXT NOT NULL,
            row_hash TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON audit_events(ts_utc)")
    conn.commit()
    return conn


def _last_hash(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT row_hash FROM audit_events ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        return "0" * 64
    return row["row_hash"]


def _hash_row(prev_hash: str, ts: str, event_type: str, ticker: str | None,
              payload_json: str) -> str:
    """Hash spec: SHA256(prev_hash | ts | event_type | ticker | payload_json)."""
    msg = f"{prev_hash}|{ts}|{event_type}|{ticker or ''}|{payload_json}"
    return hashlib.sha256(msg.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def record(event_type: str, payload: dict, ticker: str | None = None) -> dict:
    """Append a new event. Returns row id + hash."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload_json = json.dumps(payload, sort_keys=True, default=str)
    conn = _connect()
    try:
        prev_hash = _last_hash(conn)
        row_hash = _hash_row(prev_hash, ts, event_type, ticker, payload_json)
        cur = conn.execute(
            "INSERT INTO audit_events "
            "(ts_utc, event_type, ticker, payload_json, prev_hash, row_hash) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (ts, event_type, ticker, payload_json, prev_hash, row_hash),
        )
        conn.commit()
        return {
            "id":        cur.lastrowid,
            "ts_utc":    ts,
            "row_hash":  row_hash,
            "prev_hash": prev_hash,
        }
    finally:
        conn.close()


def verify_chain() -> dict:
    """Re-hash every row and confirm the chain is unbroken."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, ts_utc, event_type, ticker, payload_json, prev_hash, row_hash "
            "FROM audit_events ORDER BY id"
        ).fetchall()
    finally:
        conn.close()

    prev_hash = "0" * 64
    for r in rows:
        expected = _hash_row(
            prev_hash, r["ts_utc"], r["event_type"], r["ticker"], r["payload_json"]
        )
        if expected != r["row_hash"] or r["prev_hash"] != prev_hash:
            return {
                "valid":            False,
                "first_break_id":   int(r["id"]),
                "expected_hash":    expected,
                "actual_hash":      r["row_hash"],
                "n_rows_verified":  int(r["id"]) - 1,
                "n_total":          len(rows),
            }
        prev_hash = r["row_hash"]
    return {
        "valid":            True,
        "first_break_id":   None,
        "n_total":          len(rows),
        "last_hash":        prev_hash,
    }


def tail(limit: int = 10) -> list:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, ts_utc, event_type, ticker FROM audit_events "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def count_by_type() -> dict:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT event_type, COUNT(*) AS n FROM audit_events GROUP BY event_type"
        ).fetchall()
    finally:
        conn.close()
    return {r["event_type"]: int(r["n"]) for r in rows}


def status_snapshot() -> dict:
    """Generate a snapshot JSON used by master_controller + UI."""
    chain = verify_chain()
    by_type = count_by_type()
    recent = tail(5)
    snapshot = {
        "generated_at":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "db_path":       str(DB_FILE),
        "chain_valid":   chain["valid"],
        "first_break_id":chain.get("first_break_id"),
        "n_total":       chain["n_total"],
        "last_hash":     chain.get("last_hash"),
        "counts_by_type":by_type,
        "recent":        recent,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(snapshot, indent=2, default=str))
    return snapshot


# ---------------------------------------------------------------------------
# Auto-record from pipeline state
# ---------------------------------------------------------------------------
def record_pipeline_snapshot() -> list:
    """Read pipeline_state.json and record any new SIGNAL/DECISION/RISK events."""
    ps_path = DATA_DIR / "pipeline_state.json"
    if not ps_path.exists():
        return []
    try:
        ps = json.loads(ps_path.read_text())
    except Exception:
        return []

    recorded = []

    # Decision event
    cm = ps.get("committee", {})
    if cm.get("action_taken"):
        recorded.append(record(
            "DECISION",
            {
                "action":           cm.get("action_taken"),
                "quant_conviction": cm.get("quant_conviction"),
                "macro_conviction": cm.get("macro_conviction"),
                "veto_active":      cm.get("veto_active"),
                "oracle_score":     cm.get("oracle_score"),
                "regime":           ps.get("regime", {}).get("hmm_state"),
                "run_date":         ps.get("run_date"),
            },
            ticker=ps.get("ticker"),
        ))

    # Risk event (sizing)
    risk = ps.get("risk", {})
    if risk.get("target_weight") is not None:
        recorded.append(record(
            "RISK",
            {
                "target_weight":  risk.get("target_weight"),
                "deploy_usd":     risk.get("deploy_usd"),
                "var_95_daily":   risk.get("var_95_daily"),
                "var_override":   risk.get("var_override"),
                "run_date":       ps.get("run_date"),
            },
            ticker=ps.get("ticker"),
        ))

    return recorded


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Trail")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--tail", type=int, default=0)
    parser.add_argument("--counts", action="store_true")
    parser.add_argument("--record-pipeline", action="store_true",
                        help="Append SIGNAL+DECISION+RISK rows from pipeline_state.json")
    parser.add_argument("--status", action="store_true",
                        help="Write status snapshot to audit_trail_status.json")
    args = parser.parse_args()

    print(f"\n{SEP}\n  AUDIT TRAIL\n{SEP}")
    print(f"  DB: {DB_FILE}")

    if args.record_pipeline:
        rs = record_pipeline_snapshot()
        for r in rs:
            print(f"  Recorded #{r['id']} hash={r['row_hash'][:12]}...")
        if not rs:
            print(f"  Nothing to record (no committee/risk data).")

    if args.verify or not (args.tail or args.counts or args.record_pipeline or args.status):
        chain = verify_chain()
        print(f"\n  CHAIN VERIFICATION")
        print(f"  Rows:    {chain['n_total']}")
        print(f"  Valid:   {chain['valid']}")
        if chain["valid"]:
            print(f"  Last hash: {chain.get('last_hash', '')[:16]}...")
        else:
            print(f"  ⚠ break at row id {chain.get('first_break_id')}")

    if args.tail:
        rows = tail(args.tail)
        print(f"\n  LAST {len(rows)} EVENTS")
        for r in rows:
            print(f"    #{r['id']:<6d}  {r['ts_utc']}  "
                  f"{r['event_type']:<10s}  {r['ticker'] or '-'}")

    if args.counts:
        c = count_by_type()
        print(f"\n  COUNTS BY TYPE")
        for k, v in c.items():
            print(f"    {k:<14s}  {v}")

    if args.status:
        s = status_snapshot()
        print(f"\n  Status snapshot written: {OUTPUT_FILE}")
        print(f"  Total rows: {s['n_total']}  chain_valid: {s['chain_valid']}")

    print(SEP)


if __name__ == "__main__":
    main()
