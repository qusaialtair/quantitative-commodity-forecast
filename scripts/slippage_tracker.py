#!/usr/bin/env python3
"""
Slippage / Fill Quality Tracker  (Phase XII Stage 65)
=======================================================
Compares expected (pre-trade) prices against realised fills logged in the
IBKR audit trail and produces a rolling fill-quality scorecard.

Sources:
  data/ibkr_audit.jsonl       every simulated/live order with a price_hint
  data/fill_log.jsonl         (optional) operator-provided realised fills
                              format: {ts, ticker, side, qty, fill_price, order_id}

Metrics:
  - per-ticker slippage_bps  weighted average
  - per-side  slippage_bps  buy vs sell
  - rolling 30d slippage
  - venue tag (if present in audit row)
  - hit rate of within-spread fills

Output: data/slippage_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "slippage_report.json"
AUDIT_FILE = DATA_DIR / "ibkr_audit.jsonl"
FILL_LOG = DATA_DIR / "fill_log.jsonl"

LINE_W = 62
SEP = "━" * LINE_W


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    except Exception:
        return []
    return rows


# ---------------------------------------------------------------------------
# Pair audit orders to fills
# ---------------------------------------------------------------------------
def _pair_orders_to_fills(orders: list[dict], fills: list[dict]) -> list[dict]:
    """Match each order to a fill by ticker + side + nearest ts, ±60min."""
    paired = []
    used_fills = set()
    for o in orders:
        if o.get("event") not in ("ORDER_SUBMITTED", "ORDER_SIMULATED"):
            continue
        if not o.get("price_hint") and not o.get("limit_price"):
            continue
        expected_price = o.get("limit_price") or o.get("price_hint")
        o_ticker = (o.get("ticker") or "").upper()
        o_side = (o.get("side") or "").upper()
        try:
            o_ts = datetime.fromisoformat((o.get("ts") or "").replace("Z", "+00:00"))
        except Exception:
            continue
        best_idx = None
        best_dt = timedelta(minutes=60)
        for i, f in enumerate(fills):
            if i in used_fills:
                continue
            if (f.get("ticker") or "").upper() != o_ticker:
                continue
            if (f.get("side") or "").upper() != o_side:
                continue
            try:
                f_ts = datetime.fromisoformat((f.get("ts") or "").replace("Z", "+00:00"))
            except Exception:
                continue
            dt = abs(f_ts - o_ts)
            if dt < best_dt:
                best_dt = dt
                best_idx = i
        if best_idx is None:
            continue
        used_fills.add(best_idx)
        fill = fills[best_idx]
        fill_price = float(fill.get("fill_price") or 0)
        if fill_price <= 0 or expected_price <= 0:
            continue
        # Slippage in bps. For BUY, positive slip = paid more than expected.
        if o_side == "BUY":
            slippage_bps = (fill_price - expected_price) / expected_price * 10_000
        else:
            slippage_bps = (expected_price - fill_price) / expected_price * 10_000
        paired.append({
            "ts":             fill.get("ts"),
            "ticker":         o_ticker,
            "side":           o_side,
            "qty":            float(o.get("qty") or 0),
            "expected_price": float(expected_price),
            "fill_price":     fill_price,
            "slippage_bps":   round(float(slippage_bps), 3),
            "mode":           o.get("mode"),
            "venue":          fill.get("venue"),
        })
    return paired


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _aggregate(pairs: list[dict]) -> dict:
    if not pairs:
        return {
            "n_pairs":              0,
            "avg_slippage_bps":     None,
            "max_slippage_bps":     None,
            "by_side":              {},
            "by_ticker":            {},
        }
    avg = sum(p["slippage_bps"] for p in pairs) / len(pairs)
    mx = max(p["slippage_bps"] for p in pairs)
    mn = min(p["slippage_bps"] for p in pairs)

    by_side = {}
    for side in ("BUY", "SELL"):
        subset = [p for p in pairs if p["side"] == side]
        if subset:
            by_side[side] = {
                "n":      len(subset),
                "avg":    round(sum(s["slippage_bps"] for s in subset) / len(subset), 3),
                "max":    round(max(s["slippage_bps"] for s in subset), 3),
            }
    by_ticker = {}
    for p in pairs:
        t = p["ticker"]
        by_ticker.setdefault(t, []).append(p["slippage_bps"])
    by_ticker = {
        t: {
            "n":   len(v),
            "avg": round(sum(v) / len(v), 3),
            "max": round(max(v), 3),
        } for t, v in by_ticker.items()
    }
    return {
        "n_pairs":          len(pairs),
        "avg_slippage_bps": round(avg, 3),
        "max_slippage_bps": round(mx, 3),
        "min_slippage_bps": round(mn, 3),
        "by_side":          by_side,
        "by_ticker":        by_ticker,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_slippage() -> dict:
    audit_rows = _read_jsonl(AUDIT_FILE)
    fill_rows = _read_jsonl(FILL_LOG)

    pairs = _pair_orders_to_fills(audit_rows, fill_rows)
    agg_all = _aggregate(pairs)

    # 30d rolling
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    recent_pairs = []
    for p in pairs:
        try:
            p_ts = datetime.fromisoformat(p["ts"].replace("Z", "+00:00"))
            if p_ts >= cutoff:
                recent_pairs.append(p)
        except Exception:
            continue
    agg_30d = _aggregate(recent_pairs)

    n_audit_orders = sum(
        1 for r in audit_rows
        if r.get("event") in ("ORDER_SUBMITTED", "ORDER_SIMULATED")
    )
    n_fills_unmatched = len(fill_rows) - len(pairs)

    result = {
        "generated_at":         datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_audit_orders":       n_audit_orders,
        "n_fills_recorded":     len(fill_rows),
        "n_paired":             len(pairs),
        "n_unmatched_fills":    max(0, n_fills_unmatched),
        "fill_match_rate_pct":  round(len(pairs) / max(n_audit_orders, 1) * 100, 2),
        "all_time":             agg_all,
        "rolling_30d":          agg_30d,
        "warning":               (
            "No fills logged yet. Operator must append rows to data/fill_log.jsonl "
            "after live IBKR runs to enable slippage analytics."
            if not fill_rows else None
        ),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2, default=str))
    _print_report(result)
    return result


def _print_report(r: dict) -> None:
    print(f"\n{SEP}\n  SLIPPAGE / FILL QUALITY\n{SEP}")
    print(f"  Audit orders:        {r['n_audit_orders']}")
    print(f"  Fills recorded:      {r['n_fills_recorded']}")
    print(f"  Paired:              {r['n_paired']}")
    print(f"  Fill match rate:     {r['fill_match_rate_pct']:.1f}%")
    print()
    if r.get("warning"):
        print(f"  ⚠ {r['warning']}")
        print(SEP)
        return
    a = r["all_time"]
    if a["n_pairs"]:
        print(f"  ALL-TIME")
        print(f"    Avg slippage:  {a['avg_slippage_bps']:+.2f} bps")
        print(f"    Range:         {a['min_slippage_bps']:+.2f} → {a['max_slippage_bps']:+.2f} bps")
        for side, m in a["by_side"].items():
            print(f"    {side:<4s}  n={m['n']:>3d}  avg={m['avg']:+.2f} bps  max={m['max']:+.2f} bps")
    r30 = r["rolling_30d"]
    if r30["n_pairs"]:
        print(f"\n  ROLLING 30d")
        print(f"    Avg slippage:  {r30['avg_slippage_bps']:+.2f} bps  ({r30['n_pairs']} fills)")
    if a.get("by_ticker"):
        print(f"\n  PER-TICKER (top 5 by avg slippage)")
        sorted_t = sorted(a["by_ticker"].items(), key=lambda kv: abs(kv[1]["avg"]), reverse=True)
        for t, m in sorted_t[:5]:
            print(f"    {t:<6s}  n={m['n']:>3d}  avg={m['avg']:+.2f} bps  max={m['max']:+.2f} bps")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slippage Tracker")
    args = parser.parse_args()
    run_slippage()
