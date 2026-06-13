#!/usr/bin/env python3
"""
Interactive Brokers Adapter
=============================
ib_insync-backed client that bridges the institutional signal stack to the
IBKR TWS / IB Gateway. Implements the Phase IX Stage 47 capability from the
grand master plan: live execution on halal equities with a hard pre-trade
gate.

Safety defaults:
  - DRY_RUN mode by default; no orders are sent without --live.
  - Halal pre-trade gate: ticker must appear in the halal universe screener
    output (data/halal_universe.json) or be explicitly whitelisted.
  - Per-ticker max-position cap and per-day order limit.
  - Every order — submitted or simulated — gets appended to
    data/ibkr_audit.jsonl with a chained hash (Stage 49 wires this further).

Usage (smoke test, no order):
    python3 scripts/ibkr_adapter.py --probe
    python3 scripts/ibkr_adapter.py --positions

Live trading (requires TWS / IB Gateway running, paper or live):
    python3 scripts/ibkr_adapter.py --buy SNOW --qty 1 --live
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

DATA_DIR = ROOT / "data"
AUDIT_FILE = DATA_DIR / "ibkr_audit.jsonl"
STATE_FILE = DATA_DIR / "ibkr_state.json"
HALAL_FILE = DATA_DIR / "halal_universe.json"

# Defaults
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7497   # TWS paper trading default; 7496 = TWS live; 4002 = Gateway paper
DEFAULT_CLIENT_ID = 17

# Risk caps
MAX_POSITION_PCT = 20.0     # max position size as % of portfolio
MAX_ORDERS_PER_DAY = 25     # circuit-breaker
LARGE_ORDER_USD = 25_000.0  # require operator confirmation above this

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Halal gate
# ---------------------------------------------------------------------------
def _load_halal_universe() -> set[str]:
    """Load the halal-compliant ticker set from screener output."""
    if HALAL_FILE.exists():
        try:
            data = json.loads(HALAL_FILE.read_text())
            if isinstance(data, dict) and "tickers" in data:
                return set(data["tickers"])
            if isinstance(data, list):
                return set(data)
        except Exception:
            pass
    # Fallback whitelist (operator-confirmed halal-friendly tech / industrials)
    return {
        "SNOW", "PLTR", "NOW", "INTU", "HUBS", "ZS", "EL", "EXPE", "ENPH",
        "QGEN", "MU", "FORM", "ON", "GLD", "SLV", "BAR", "IAU", "GLDM",
    }


def is_halal(ticker: str) -> bool:
    return ticker.upper() in _load_halal_universe()


# ---------------------------------------------------------------------------
# Audit trail (Stage 49 will deepen this)
# ---------------------------------------------------------------------------
def _last_audit_hash() -> str:
    if not AUDIT_FILE.exists():
        return "0" * 64
    try:
        with open(AUDIT_FILE, "rb") as f:
            lines = f.readlines()
        if not lines:
            return "0" * 64
        last = json.loads(lines[-1])
        return last.get("hash", "0" * 64)
    except Exception:
        return "0" * 64


def _audit_log(event: dict) -> None:
    """Append a hash-chained event row to data/ibkr_audit.jsonl."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prev_hash = _last_audit_hash()
    payload = json.dumps(event, sort_keys=True, default=str)
    h = hashlib.sha256((prev_hash + payload).encode("utf-8")).hexdigest()
    record = {**event, "prev_hash": prev_hash, "hash": h}
    with open(AUDIT_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def _today_order_count() -> int:
    if not AUDIT_FILE.exists():
        return 0
    today = datetime.now(timezone.utc).date().isoformat()
    count = 0
    try:
        with open(AUDIT_FILE) as f:
            for line in f:
                r = json.loads(line)
                if r.get("event") in ("ORDER_SUBMITTED", "ORDER_SIMULATED") \
                   and r.get("date") == today:
                    count += 1
    except Exception:
        pass
    return count


# ---------------------------------------------------------------------------
# IBKR connection
# ---------------------------------------------------------------------------
class IBKRClient:
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        client_id: int = DEFAULT_CLIENT_ID,
        dry_run: bool = True,
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.dry_run = dry_run
        self._ib = None

    def connect(self) -> bool:
        try:
            import ib_insync
        except ImportError:
            print(
                "  [IBKR] ib_insync not installed. Install with: "
                "pip install ib_insync"
            )
            return False
        if self.dry_run:
            print(f"  [IBKR] DRY-RUN mode — skipping real connection to "
                  f"{self.host}:{self.port}")
            return True
        try:
            self._ib = ib_insync.IB()
            self._ib.connect(self.host, self.port, clientId=self.client_id, timeout=10)
            return self._ib.isConnected()
        except Exception as exc:
            print(f"  [IBKR] Connection failed: {exc}")
            print("  [IBKR] Is TWS / IB Gateway running? Port 7497 (paper) "
                  "or 7496 (live)?")
            return False

    def disconnect(self) -> None:
        if self._ib is not None:
            try:
                self._ib.disconnect()
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Account / positions
    # -----------------------------------------------------------------------
    def get_account_summary(self) -> dict:
        if self.dry_run or self._ib is None:
            return {
                "mode":          "DRY_RUN",
                "account":       "DU0000000",
                "net_liquidation_usd": 100_000.0,
                "cash_usd":      100_000.0,
                "buying_power":  200_000.0,
            }
        try:
            summary = self._ib.accountSummary()
            account = self._ib.managedAccounts()[0] if self._ib.managedAccounts() else "n/a"
            tags = {item.tag: item.value for item in summary}
            return {
                "mode":          "LIVE",
                "account":       account,
                "net_liquidation_usd": float(tags.get("NetLiquidation", 0)),
                "cash_usd":      float(tags.get("TotalCashValue", 0)),
                "buying_power":  float(tags.get("BuyingPower", 0)),
            }
        except Exception as exc:
            return {"mode": "ERROR", "error": str(exc)}

    def get_positions(self) -> list:
        if self.dry_run or self._ib is None:
            return []
        try:
            return [
                {
                    "ticker":   p.contract.symbol,
                    "quantity": float(p.position),
                    "avg_cost": float(p.avgCost),
                    "exchange": p.contract.exchange,
                    "currency": p.contract.currency,
                }
                for p in self._ib.positions()
            ]
        except Exception:
            return []

    # -----------------------------------------------------------------------
    # Pre-trade gate
    # -----------------------------------------------------------------------
    def pretrade_gate(self, ticker: str, qty: int, price_hint: float | None = None) -> tuple[bool, str]:
        ticker = ticker.upper()
        # Halal compliance
        if not is_halal(ticker):
            return False, f"{ticker} not in halal universe"

        # Daily order limit
        n_today = _today_order_count()
        if n_today >= MAX_ORDERS_PER_DAY:
            return False, f"daily order limit reached ({n_today}/{MAX_ORDERS_PER_DAY})"

        # Position size cap (only enforceable if we know NAV + price)
        if price_hint and price_hint > 0:
            acct = self.get_account_summary()
            nlv = acct.get("net_liquidation_usd", 0)
            if nlv > 0:
                notional = qty * price_hint
                pct = notional / nlv * 100
                if pct > MAX_POSITION_PCT:
                    return False, (
                        f"position {pct:.1f}% would exceed cap "
                        f"{MAX_POSITION_PCT:.1f}% of NAV"
                    )
                if notional > LARGE_ORDER_USD and qty > 1:
                    # Large order — require explicit force flag in caller
                    return False, (
                        f"order ${notional:,.0f} > ${LARGE_ORDER_USD:,.0f} "
                        f"requires --force-large"
                    )
        return True, "ok"

    # -----------------------------------------------------------------------
    # Order placement
    # -----------------------------------------------------------------------
    def place_order(
        self, ticker: str, qty: int, side: str = "BUY",
        order_type: str = "MKT", limit_price: float | None = None,
        force_large: bool = False,
    ) -> dict:
        ticker = ticker.upper()
        side = side.upper()

        # Price hint for the pre-trade gate
        price_hint = limit_price
        if price_hint is None:
            try:
                import yfinance as yf
                raw = yf.download(ticker, period="2d", interval="1d",
                                  progress=False, auto_adjust=True)
                price_hint = float(raw["Close"].dropna().iloc[-1])
            except Exception:
                price_hint = None

        ok, reason = self.pretrade_gate(ticker, qty, price_hint)
        if not ok and not (force_large and "force-large" in reason):
            event = {
                "event":      "ORDER_REJECTED",
                "ts":         datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "date":       datetime.now(timezone.utc).date().isoformat(),
                "ticker":     ticker,
                "side":       side,
                "qty":        qty,
                "reason":     reason,
                "mode":       "DRY_RUN" if self.dry_run else "LIVE",
            }
            _audit_log(event)
            return {"status": "REJECTED", "reason": reason, **event}

        if self.dry_run or self._ib is None:
            event = {
                "event":      "ORDER_SIMULATED",
                "ts":         datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "date":       datetime.now(timezone.utc).date().isoformat(),
                "ticker":     ticker,
                "side":       side,
                "qty":        qty,
                "order_type": order_type,
                "limit_price":limit_price,
                "price_hint": price_hint,
                "mode":       "DRY_RUN",
            }
            _audit_log(event)
            return {"status": "SIMULATED", **event}

        # Real submission
        try:
            import ib_insync
            contract = ib_insync.Stock(ticker, "SMART", "USD")
            self._ib.qualifyContracts(contract)
            if order_type == "LMT" and limit_price:
                order = ib_insync.LimitOrder(side, qty, limit_price)
            else:
                order = ib_insync.MarketOrder(side, qty)
            trade = self._ib.placeOrder(contract, order)
            self._ib.sleep(0.5)
            event = {
                "event":      "ORDER_SUBMITTED",
                "ts":         datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "date":       datetime.now(timezone.utc).date().isoformat(),
                "ticker":     ticker,
                "side":       side,
                "qty":        qty,
                "order_type": order_type,
                "limit_price":limit_price,
                "order_id":   trade.order.orderId,
                "mode":       "LIVE",
            }
            _audit_log(event)
            return {"status": "SUBMITTED", **event}
        except Exception as exc:
            event = {
                "event":      "ORDER_ERROR",
                "ts":         datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "date":       datetime.now(timezone.utc).date().isoformat(),
                "ticker":     ticker,
                "side":       side,
                "qty":        qty,
                "error":      str(exc),
                "mode":       "LIVE",
            }
            _audit_log(event)
            return {"status": "ERROR", **event}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_status(client: IBKRClient, label: str) -> None:
    print(f"\n{SEP}\n  IBKR ADAPTER -- {label}\n{SEP}")
    acct = client.get_account_summary()
    print(f"  Mode:           {acct.get('mode')}")
    print(f"  Account:        {acct.get('account')}")
    print(f"  Net Liquidation: ${acct.get('net_liquidation_usd', 0):,.2f}")
    print(f"  Cash:            ${acct.get('cash_usd', 0):,.2f}")
    print(f"  Buying Power:    ${acct.get('buying_power', 0):,.2f}")
    print()
    positions = client.get_positions()
    if positions:
        print(f"  POSITIONS ({len(positions)})")
        for p in positions:
            print(f"    {p['ticker']:<8s} qty={p['quantity']:>8.2f}  "
                  f"avg_cost=${p['avg_cost']:.2f}  {p['exchange']}")
    else:
        print(f"  POSITIONS: none")
    print()
    print(f"  HALAL UNIVERSE: {len(_load_halal_universe())} tickers")
    print(f"  Today's orders: {_today_order_count()} / {MAX_ORDERS_PER_DAY}")
    print(SEP)


def main() -> None:
    parser = argparse.ArgumentParser(description="IBKR Adapter")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--client-id", type=int, default=DEFAULT_CLIENT_ID)
    parser.add_argument("--live", action="store_true",
                        help="Send real orders. Default is DRY_RUN.")
    parser.add_argument("--probe", action="store_true",
                        help="Test connection only")
    parser.add_argument("--positions", action="store_true",
                        help="Print account + positions")
    parser.add_argument("--buy", default=None, help="Ticker to buy")
    parser.add_argument("--sell", default=None, help="Ticker to sell")
    parser.add_argument("--qty", type=int, default=1)
    parser.add_argument("--limit", type=float, default=None,
                        help="Limit price (omit for market)")
    parser.add_argument("--force-large", action="store_true",
                        help="Bypass large-order gate")
    args = parser.parse_args()

    dry_run = not args.live

    client = IBKRClient(
        host=args.host, port=args.port, client_id=args.client_id, dry_run=dry_run,
    )

    if not client.connect():
        print("  [IBKR] connect() returned False — aborting.")
        sys.exit(1)

    try:
        if args.probe:
            _print_status(client, "PROBE")
        elif args.positions:
            _print_status(client, "POSITIONS")
        elif args.buy:
            order_type = "LMT" if args.limit else "MKT"
            res = client.place_order(
                args.buy, args.qty, "BUY", order_type,
                limit_price=args.limit, force_large=args.force_large,
            )
            print(f"\n  Order result: {json.dumps(res, indent=2)}")
        elif args.sell:
            order_type = "LMT" if args.limit else "MKT"
            res = client.place_order(
                args.sell, args.qty, "SELL", order_type,
                limit_price=args.limit, force_large=args.force_large,
            )
            print(f"\n  Order result: {json.dumps(res, indent=2)}")
        else:
            _print_status(client, "DEFAULT")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
