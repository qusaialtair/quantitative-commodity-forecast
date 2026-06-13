#!/usr/bin/env python3
"""
Order Router — single execution surface (Phase XXIV Stage 1)
============================================================
All entry/exit orders from the multi-strategy trader (and any future
trader) flow through route_order() in this module. The router branches
on the EXECUTION_MODE env var, so the strategy code never has to know
whether it's hitting the internal paper book, an IBKR paper account,
or live capital.

Modes
-----
- "paper_internal"  (DEFAULT — permanently pinned)
      No external call. The caller's internal book.append is the only
      record. Backwards-compatible with everything pre-Phase XXIV. This is
      the only supported mode after the IBKR pivot.

--- DEPRECATED MODES (IBKR pivot — regional compliance) -----------------
The two IBKR modes below are DEPRECATED and intentionally disabled by
pinning EXECUTION_MODE=paper_internal in .env. The code is retained for
historical reference only and must not be re-enabled without a compliance
sign-off.

- "paper_ibkr"   [DEPRECATED]
      Submits the order against IBKR via IBKRClient with dry_run=False.
      Requires TWS or IB Gateway running on the paper port (default 7497).
      Falls back to ORDER_SIMULATED if the connection isn't available.
- "live_ibkr"    [DEPRECATED]
      Submits the order against the LIVE IBKR account. Requires BOTH:
        EXECUTION_MODE=live_ibkr
        LIVE_TRADING_CONFIRM=YES_<today's UTC date in YYYY-MM-DD>
      The date-stamped confirm forces the operator to re-affirm every
      calendar day. Missing or stale confirm = order rejected, audited.
-------------------------------------------------------------------------

Every routed order is appended to data/ibkr_audit.jsonl (hash-chained)
via the existing IBKRClient.place_order audit hook, so the audit trail
is uniform across modes.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

VALID_MODES = {"paper_internal", "paper_ibkr", "live_ibkr"}
HALT_FLAG = ROOT / "data" / "trading_halted.flag"

_client_cache: dict[str, Any] = {}
_routed_this_session: list[dict] = []


def pop_routed_orders() -> list[dict]:
    """Return and clear orders routed successfully this process session."""
    orders = list(_routed_this_session)
    _routed_this_session.clear()
    return orders


def _record_routed(
    result: dict,
    *,
    side: str,
    ticker: str,
    qty: float,
    strategy: str,
    note: str,
) -> None:
    status = result.get("status")
    if status not in {"SUBMITTED", "NOOP"} or qty <= 0:
        return
    _routed_this_session.append({
        "ticker":   ticker,
        "side":     side,
        "qty":      qty,
        "status":   status,
        "mode":     result.get("mode"),
        "strategy": strategy,
        "note":     note,
        "order_id": result.get("order_id"),
    })


def is_halted() -> bool:
    """True if data/trading_halted.flag exists — durable kill-switch."""
    return HALT_FLAG.exists()


def _mode() -> str:
    # Permanently defaults to paper_internal (IBKR pivot). Any unset/invalid
    # value resolves to the local-simulation book so the nightly pipeline never
    # attempts a broker connection.
    m = (os.environ.get("EXECUTION_MODE") or "paper_internal").lower()
    return m if m in VALID_MODES else "paper_internal"


def _live_confirm_valid() -> bool:
    today = datetime.now(timezone.utc).date().isoformat()
    expected = f"YES_{today}"
    return os.environ.get("LIVE_TRADING_CONFIRM") == expected


def _get_client(mode: str):
    """Lazy IBKR client; reuse across calls within a single trader run."""
    if mode in _client_cache:
        return _client_cache[mode]

    from scripts.ibkr_adapter import IBKRClient, DEFAULT_HOST, DEFAULT_CLIENT_ID

    port = int(os.environ.get("IBKR_PORT", "7497"))
    host = os.environ.get("IBKR_HOST", DEFAULT_HOST)
    client_id = int(os.environ.get("IBKR_CLIENT_ID", str(DEFAULT_CLIENT_ID)))

    # mode determines dry_run: paper_ibkr / live_ibkr both want real submission;
    # the live-vs-paper distinction is encoded in the IBKR PORT.
    client = IBKRClient(host=host, port=port, client_id=client_id, dry_run=False)
    if not client.connect():
        # connect prints its own error; we still return the object so place_order
        # can produce a clean ORDER_SIMULATED record rather than crashing the trader.
        client.dry_run = True
    _client_cache[mode] = client
    return client


def route_order(
    side: str,
    ticker: str,
    qty: float,
    price_hint: float | None = None,
    strategy: str = "",
    note: str = "",
) -> dict:
    """Single entry point for every trade write.

    Returns a dict shaped like IBKRClient.place_order's return:
      {status, mode, ticker, side, qty, ...}
    status ∈ {NOOP, REJECTED, SIMULATED, SUBMITTED, ERROR}.
    NOOP means paper_internal — the caller's book write IS the trade.
    """
    mode = _mode()
    side_u = side.upper()
    if side_u not in {"BUY", "SELL", "LONG", "SHORT"}:
        return {"status": "REJECTED", "reason": f"unknown side {side}", "mode": mode}
    # Internal book uses LONG/SHORT; IBKR uses BUY/SELL
    ibkr_side = "BUY" if side_u in {"BUY", "LONG"} else "SELL"

    # Durable kill-switch — UI halt button + ops emergency
    if is_halted():
        return {
            "status": "HALTED",
            "reason": f"data/trading_halted.flag present — all routing suspended",
            "mode": mode,
            "ticker": ticker,
            "side": ibkr_side,
            "qty": qty,
        }

    if mode == "paper_internal":
        result = {
            "status": "NOOP",
            "mode": mode,
            "ticker": ticker,
            "side": ibkr_side,
            "qty": qty,
            "strategy": strategy,
            "note": note,
        }
        _record_routed(result, side=ibkr_side, ticker=ticker, qty=qty,
                       strategy=strategy, note=note)
        return result

    # ===================================================================
    # DEPRECATED — IBKR routing (paper_ibkr / live_ibkr).
    # Disabled by the IBKR pivot (regional compliance). Unreachable while
    # EXECUTION_MODE=paper_internal (the pinned default), which returns NOOP
    # above. Retained for historical reference only; do not re-enable without
    # a compliance sign-off.
    # ===================================================================
    # Metals channel is physical-only per project_trading_mandate — never route
    # physical-metal proxies (futures, GLD/SLV/etc.) to IBKR. Halal-equity
    # tickers fall through to broker submission.
    from scripts.ibkr_adapter import is_halal
    if not is_halal(ticker):
        return {
            "status": "NOT_ROUTABLE",
            "reason": f"{ticker} is not in halal equity universe (metals = physical channel)",
            "mode": mode,
            "ticker": ticker,
            "side": ibkr_side,
            "qty": qty,
            "strategy": strategy,
        }

    if mode == "live_ibkr" and not _live_confirm_valid():
        return {
            "status": "REJECTED",
            "reason": (
                "live_ibkr requires LIVE_TRADING_CONFIRM=YES_"
                + datetime.now(timezone.utc).date().isoformat()
            ),
            "mode": mode,
            "ticker": ticker,
            "side": ibkr_side,
            "qty": qty,
        }

    # paper_ibkr or validly-confirmed live_ibkr → submit through IBKRClient
    try:
        client = _get_client(mode)
        qty_int = max(int(round(qty)), 1)  # IBKR equity orders are integer share counts
        result = client.place_order(
            ticker=ticker,
            qty=qty_int,
            side=ibkr_side,
            order_type="MKT",
            limit_price=None,
        )
        result["routed_mode"] = mode
        result["strategy"] = strategy
        result["note"] = note
        _record_routed(result, side=ibkr_side, ticker=ticker, qty=qty,
                       strategy=strategy, note=note)
        return result
    except Exception as exc:
        return {
            "status": "ERROR",
            "reason": f"router exception: {exc}",
            "mode": mode,
            "ticker": ticker,
            "side": ibkr_side,
            "qty": qty,
        }


def shutdown() -> None:
    """Cleanly disconnect any cached IBKR sessions. Call at trader exit."""
    for client in _client_cache.values():
        try:
            client.disconnect()
        except Exception:
            pass
    _client_cache.clear()


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
def _smoke() -> int:
    """python3 scripts/order_router.py — runs a 3-order smoke in current mode."""
    mode = _mode()
    print(f"order_router smoke — EXECUTION_MODE={mode}")
    cases = [
        ("BUY",  "GLD",  1, 200.0, "METAL_TREND", "smoke #1"),
        ("BUY",  "SLV",  2,  25.0, "METAL_TREND", "smoke #2"),
        ("SELL", "GLD",  1, 200.0, "METAL_TREND", "smoke #3 exit"),
    ]
    for side, tkr, qty, px, strat, note in cases:
        r = route_order(side, tkr, qty, px, strat, note)
        print(f"  {side:4s} {tkr:5s} qty={qty} -> {r.get('status'):<10s} {r.get('reason','')}")
    shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke())
