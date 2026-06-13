#!/usr/bin/env python3
"""
Position Reconciliation  (Phase XII Stage 64)
===============================================
Diffs the broker's view of the operator's positions against the internal
shadow book. Any drift between the two ledgers is a hard operational
problem: failed fills, missed cancels, manual overrides, or a corrupt
shadow_book.db.

Sources:
  - IBKR adapter `get_positions()`  (LIVE if connected, else DRY_RUN empty)
  - data/shadow_book.db `portfolio_state` table  (last-known internal book)
  - data/portfolio.json                          (equity-book ledger)
  - data/ibkr_audit.jsonl                        (recent simulated orders)

Outputs per side: tickers in broker only, tickers in shadow only, and
quantity / cost-basis mismatches.

Output: data/position_reconciliation.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "position_reconciliation.json"
SHADOW_DB = DATA_DIR / "shadow_book.db"

QTY_TOLERANCE = 0.001  # absolute share tolerance

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load_broker_positions(use_ibkr: bool = True) -> list[dict]:
    if not use_ibkr:
        return []
    try:
        from scripts.ibkr_adapter import IBKRClient
        client = IBKRClient(dry_run=False)
        connected = client.connect()
        if not connected:
            return []
        try:
            return client.get_positions() or []
        finally:
            client.disconnect()
    except Exception:
        return []


def _load_shadow_metals() -> list[dict]:
    """Read shadow_book.db's portfolio_state for the metals book."""
    out = []
    if not SHADOW_DB.exists():
        return out
    try:
        conn = sqlite3.connect(SHADOW_DB)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM portfolio_state WHERE id = 1"
        ).fetchone()
        conn.close()
        if not row:
            return out
        gold_oz = float(row["gold_oz"] or 0)
        avg_entry = float(row["avg_entry"] or 0) if "avg_entry" in row.keys() else 0
        last_spot = float(row["last_spot"] or 0) if "last_spot" in row.keys() else 0
        if gold_oz > 1e-6:
            out.append({
                "ticker":   "GC=F",
                "quantity": gold_oz,
                "avg_cost": avg_entry,
                "source":   "shadow_book",
                "value_usd":gold_oz * last_spot,
            })
    except Exception:
        pass
    return out


def _load_shadow_equities() -> list[dict]:
    """Read portfolio.json for equity positions."""
    out = []
    pj = DATA_DIR / "portfolio.json"
    if not pj.exists():
        return out
    try:
        data = json.loads(pj.read_text())
        if not isinstance(data, list):
            return out
        for entry in data:
            tick = entry.get("ticker")
            qty = float(entry.get("shares") or entry.get("qty") or 0)
            if tick and qty > 1e-6:
                out.append({
                    "ticker":   tick,
                    "quantity": qty,
                    "avg_cost": float(entry.get("avg_cost") or 0),
                    "source":   "portfolio_json",
                    "value_usd":float(entry.get("value_usd") or qty * float(entry.get("price_usd") or 0)),
                })
    except Exception:
        pass
    return out


def _load_pending_audit_orders() -> list[dict]:
    """Read last 50 IBKR audit events to surface un-reconciled simulated orders."""
    audit = DATA_DIR / "ibkr_audit.jsonl"
    if not audit.exists():
        return []
    try:
        lines = audit.read_text().strip().split("\n")[-50:]
        out = []
        for line in lines:
            try:
                row = json.loads(line)
                if row.get("event") in ("ORDER_SIMULATED", "ORDER_SUBMITTED"):
                    out.append({
                        "ts":      row.get("ts"),
                        "ticker":  row.get("ticker"),
                        "side":    row.get("side"),
                        "qty":     row.get("qty"),
                        "mode":    row.get("mode"),
                    })
            except Exception:
                continue
        return out
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------
def _diff(broker: list[dict], shadow: list[dict]) -> dict:
    broker_map = {p["ticker"]: p for p in broker}
    shadow_map = {p["ticker"]: p for p in shadow}

    only_broker = [t for t in broker_map if t not in shadow_map]
    only_shadow = [t for t in shadow_map if t not in broker_map]
    common = [t for t in broker_map if t in shadow_map]

    qty_mismatches = []
    cost_mismatches = []
    for t in common:
        b = broker_map[t]
        s = shadow_map[t]
        bq = float(b.get("quantity", 0))
        sq = float(s.get("quantity", 0))
        if abs(bq - sq) > QTY_TOLERANCE:
            qty_mismatches.append({
                "ticker":  t,
                "broker":  bq,
                "shadow":  sq,
                "delta":   bq - sq,
            })
        bc = float(b.get("avg_cost", 0))
        sc = float(s.get("avg_cost", 0))
        if bc > 0 and sc > 0 and abs(bc - sc) / sc > 0.01:  # 1% cost-basis drift
            cost_mismatches.append({
                "ticker":  t,
                "broker":  bc,
                "shadow":  sc,
                "delta":   bc - sc,
            })

    return {
        "n_broker":         len(broker),
        "n_shadow":         len(shadow),
        "n_common":         len(common),
        "only_in_broker":   only_broker,
        "only_in_shadow":   only_shadow,
        "qty_mismatches":   qty_mismatches,
        "cost_mismatches":  cost_mismatches,
        "n_drift_total":    (
            len(only_broker) + len(only_shadow) +
            len(qty_mismatches) + len(cost_mismatches)
        ),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_reconciler(use_ibkr: bool = True) -> dict:
    broker = _load_broker_positions(use_ibkr=use_ibkr)
    shadow = _load_shadow_metals() + _load_shadow_equities()
    diff = _diff(broker, shadow)
    pending = _load_pending_audit_orders()

    status = "OK"
    if diff["n_drift_total"] > 0:
        status = "DRIFT" if diff["n_drift_total"] <= 2 else "MAJOR_DRIFT"

    result = {
        "generated_at":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ibkr_mode":         "LIVE" if (use_ibkr and broker) else "DRY_RUN",
        "status":            status,
        "diff":              diff,
        "broker_positions":  broker,
        "shadow_positions":  shadow,
        "recent_pending_audit_orders": pending[-10:],
        "n_pending_audit":   len(pending),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2, default=str))
    _print_report(result)
    return result


def _print_report(r: dict) -> None:
    color = {"OK": "\033[32;1m", "DRIFT": "\033[33m", "MAJOR_DRIFT": "\033[31;1m"}.get(r["status"], "\033[0m")
    print(f"\n{SEP}\n  POSITION RECONCILIATION\n{SEP}")
    print(f"  IBKR mode:       {r['ibkr_mode']}")
    print(f"  Status:          {color}{r['status']}\033[0m")
    print(f"  Broker positions:{r['diff']['n_broker']}")
    print(f"  Shadow positions:{r['diff']['n_shadow']}")
    print(f"  In common:       {r['diff']['n_common']}")
    print(f"  Drift total:     {r['diff']['n_drift_total']}")
    print()
    d = r["diff"]
    if d["only_in_broker"]:
        print(f"  ⚠ IN BROKER ONLY: {', '.join(d['only_in_broker'])}")
    if d["only_in_shadow"]:
        print(f"  ⚠ IN SHADOW ONLY: {', '.join(d['only_in_shadow'])}")
    for m in d["qty_mismatches"]:
        print(f"  ⚠ QTY {m['ticker']}: broker={m['broker']}  shadow={m['shadow']}  Δ={m['delta']:+.4f}")
    for m in d["cost_mismatches"]:
        print(f"  ⚠ COST {m['ticker']}: broker=${m['broker']:.2f}  shadow=${m['shadow']:.2f}  Δ={m['delta']:+.2f}")
    if r["status"] == "OK":
        print(f"  All positions reconciled.")
    print()
    print(f"  Pending audit orders (last 10): {len(r['recent_pending_audit_orders'])}")
    for p in r["recent_pending_audit_orders"][:5]:
        print(f"    {p.get('ts', '?')[:19]}  {p.get('side', '?')} {p.get('ticker', '?')} x{p.get('qty', 0)}  [{p.get('mode', '?')}]")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Position Reconciliation")
    parser.add_argument("--no-ibkr", action="store_true",
                        help="Skip IBKR fetch (use shadow only)")
    args = parser.parse_args()
    run_reconciler(use_ibkr=not args.no_ibkr)
