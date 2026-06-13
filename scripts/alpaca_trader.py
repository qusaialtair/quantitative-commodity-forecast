#!/usr/bin/env python3
"""
scripts/alpaca_trader.py
========================
Phase 4 — Alpaca Paper Trading Execution Engine

Reads today's CIO decision from data/equity_decision.json and rebalances the
Alpaca paper-trading account to match the target weights exactly.

Execution flow (synchronous — no fire-and-forget):
  1. Load equity_decision.json  →  today's Top-3 tickers + weights
  2. Daily cache check          →  skip if already executed today (--force bypasses)
  3. GET /v2/positions          →  identify stale positions (not in today's Top-3)
  4. Submit Market SELL orders  →  liquidate all stale positions via close_position()
  5. Poll every 2 s (30 s max)  →  wait for all sells to reach FILLED status
  6. GET /v2/account            →  read fresh account equity (post-liquidation)
  7. Submit notional BUY orders →  equity × weight for each of the 3 picks
     └─ Fractional-share strategy:
        PRIMARY  → notional order (Alpaca auto-computes fractional qty)
        FALLBACK → if 422 Unprocessable, re-submit as qty = math.floor($ / price)
  8. Append each event to data/trade_log.jsonl  (permanent audit trail)
  9. Write data/alpaca_daily_summary.json       (today's run summary — overwrites)
 10. Telegram heartbeat

CLI
---
  python3 scripts/alpaca_trader.py              # run (skip if already executed today)
  python3 scripts/alpaca_trader.py --force      # re-run even if today's summary exists
  python3 scripts/alpaca_trader.py --dry-run    # log what would happen, no real orders
  python3 scripts/alpaca_trader.py --status     # print current Alpaca positions + equity
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import yfinance as yf
from dotenv import load_dotenv

# ── Root & env ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

# ── Paths ─────────────────────────────────────────────────────────────────────
DECISION_FILE      = ROOT / "data" / "equity_decision.json"
TRADE_LOG_FILE     = ROOT / "data" / "trade_log.jsonl"
DAILY_SUMMARY_FILE = ROOT / "data" / "alpaca_daily_summary.json"
TRADE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Alpaca credentials ────────────────────────────────────────────────────────
_API_KEY    = os.getenv("ALPACA_API_KEY",    "")
_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
_PAPER_URL  = "https://paper-api.alpaca.markets"

# ── Execution constants ───────────────────────────────────────────────────────
_POLL_INTERVAL_S  = 2      # seconds between order-status polls
_POLL_TIMEOUT_S   = 30     # max seconds to wait for sell fills
_MIN_NOTIONAL     = 1.0    # skip orders below $1 (Alpaca minimum)
_CASH_BUFFER_PCT  = 0.995  # deploy 99.5% of equity — leave tiny buffer for fees

# ── Logging ───────────────────────────────────────────────────────────────────
for _lib in ("yfinance", "urllib3", "peewee", "charset_normalizer", "httpx", "alpaca"):
    logging.getLogger(_lib).setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("alpaca_trader")


# ─────────────────────────────────────────────────────────────────────────────
# Alpaca client bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def _build_client():
    """Build and return a paper-trading TradingClient. Raises if keys are missing."""
    from alpaca.trading.client import TradingClient

    if not _API_KEY or not _SECRET_KEY:
        raise RuntimeError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env"
        )
    return TradingClient(_API_KEY, _SECRET_KEY, paper=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _today() -> str:
    return date.today().isoformat()


def _atomic_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.rename(tmp, path)


def _append_trade_log(event: dict) -> None:
    """Append a single JSON event line to trade_log.jsonl (permanent audit trail)."""
    try:
        with TRADE_LOG_FILE.open("a") as f:
            f.write(json.dumps({**event, "timestamp": _now_utc()}) + "\n")
    except Exception as exc:
        logger.warning("trade_log write failed: %s", exc)


def _fetch_price_yf(ticker: str) -> float | None:
    """Fallback: fetch latest close price from yfinance for whole-share sizing."""
    try:
        hist = yf.Ticker(ticker).history(period="2d")
        if hist.empty:
            return None
        return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Daily cache
# ─────────────────────────────────────────────────────────────────────────────

def _load_daily_cache() -> dict | None:
    """Return today's summary dict if it exists, else None."""
    try:
        if DAILY_SUMMARY_FILE.exists():
            d = json.loads(DAILY_SUMMARY_FILE.read_text())
            if d.get("run_date") == _today():
                return d
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Sell logic
# ─────────────────────────────────────────────────────────────────────────────

def _liquidate_stale(client, current_positions: dict, target_tickers: set[str], dry_run: bool) -> list[str]:
    """
    Submit Market SELL orders for every held ticker not in target_tickers.
    Returns list of order IDs submitted (empty in dry-run mode).
    """
    from alpaca.trading.enums import OrderSide

    stale = [tk for tk in current_positions if tk not in target_tickers]
    order_ids: list[str] = []

    if not stale:
        logger.info("No stale positions to liquidate.")
        return order_ids

    for ticker in stale:
        pos  = current_positions[ticker]
        qty  = float(pos.qty)
        mval = float(pos.market_value) if hasattr(pos, "market_value") else 0.0
        logger.info("SELL  %-6s  %.4f shares  (~$%.2f)%s", ticker, qty, mval,
                    "  [DRY RUN]" if dry_run else "")
        _append_trade_log({
            "date":         _today(),
            "event":        "SELL",
            "ticker":       ticker,
            "qty":          qty,
            "market_value": mval,
            "reason":       "not_in_top3",
            "order_id":     "DRY_RUN" if dry_run else None,
            "status":       "dry_run" if dry_run else "submitted",
        })
        if not dry_run:
            try:
                order = client.close_position(ticker)
                order_ids.append(str(order.id))
                _append_trade_log({
                    "date":     _today(),
                    "event":    "SELL_SUBMITTED",
                    "ticker":   ticker,
                    "order_id": str(order.id),
                    "status":   str(order.status),
                })
                logger.info("  Order submitted: %s", order.id)
            except Exception as exc:
                logger.error("  Failed to close %s: %s", ticker, exc)
                _append_trade_log({
                    "date":    _today(),
                    "event":   "SELL_ERROR",
                    "ticker":  ticker,
                    "error":   str(exc),
                })

    return order_ids


# ─────────────────────────────────────────────────────────────────────────────
# Polling loop — wait for sells to fill
# ─────────────────────────────────────────────────────────────────────────────

def _poll_fills(client, order_ids: list[str], timeout_s: int = _POLL_TIMEOUT_S) -> bool:
    """
    Poll order status every _POLL_INTERVAL_S seconds until all orders reach
    FILLED (or a terminal state). Returns True if all filled within timeout.
    """
    from alpaca.trading.enums import OrderStatus

    if not order_ids:
        return True

    terminal = {
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.EXPIRED,
        OrderStatus.REPLACED,
        OrderStatus.REJECTED,
        OrderStatus.DONE_FOR_DAY,
    }

    deadline   = time.time() + timeout_s
    remaining  = set(order_ids)
    all_filled = True

    logger.info("Polling %d sell order(s) — timeout %ds…", len(order_ids), timeout_s)

    while remaining and time.time() < deadline:
        time.sleep(_POLL_INTERVAL_S)
        done_this_round: list[str] = []

        for oid in list(remaining):
            try:
                order = client.get_order_by_id(oid)
                status = order.status
                if status == OrderStatus.FILLED:
                    logger.info("  ✓ Order %s FILLED (%s)", oid[:8], order.symbol)
                    _append_trade_log({
                        "date":     _today(),
                        "event":    "SELL_FILLED",
                        "ticker":   order.symbol,
                        "order_id": oid,
                        "status":   "filled",
                    })
                    done_this_round.append(oid)
                elif status in terminal:
                    logger.warning("  ✗ Order %s terminal status: %s", oid[:8], status)
                    _append_trade_log({
                        "date":     _today(),
                        "event":    "SELL_TERMINAL",
                        "ticker":   order.symbol,
                        "order_id": oid,
                        "status":   str(status),
                    })
                    done_this_round.append(oid)
                    all_filled = False
                else:
                    logger.debug("  … Order %s status: %s", oid[:8], status)
            except Exception as exc:
                logger.warning("  Poll error for order %s: %s", oid[:8], exc)

        remaining -= set(done_this_round)

    if remaining:
        logger.warning(
            "SELL TIMEOUT: %d order(s) not yet filled after %ds — "
            "proceeding with whatever cash has settled. IDs: %s",
            len(remaining), timeout_s,
            [oid[:8] for oid in remaining],
        )
        _append_trade_log({
            "date":    _today(),
            "event":   "SELL_TIMEOUT_WARNING",
            "pending_orders": list(remaining),
            "timeout_s": timeout_s,
        })
        return False

    return all_filled


# ─────────────────────────────────────────────────────────────────────────────
# Buy logic
# ─────────────────────────────────────────────────────────────────────────────

def _submit_buy(client, ticker: str, target_usd: float, dry_run: bool) -> dict:
    """
    Submit a single BUY order. Tries notional first; falls back to whole shares.
    Returns a result dict with order_id, method, status.
    """
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    if target_usd < _MIN_NOTIONAL:
        logger.warning("BUY  %-6s  $%.2f < minimum — skipped", ticker, target_usd)
        return {"ticker": ticker, "target_usd": target_usd, "status": "skipped_below_min"}

    logger.info("BUY   %-6s  notional=$%.2f%s", ticker, target_usd,
                "  [DRY RUN]" if dry_run else "")

    if dry_run:
        _append_trade_log({
            "date":       _today(),
            "event":      "BUY",
            "ticker":     ticker,
            "notional":   target_usd,
            "method":     "notional",
            "order_id":   "DRY_RUN",
            "status":     "dry_run",
        })
        return {"ticker": ticker, "target_usd": target_usd, "order_id": "DRY_RUN",
                "method": "notional", "status": "dry_run"}

    # ── Attempt 1: notional order (supports fractional shares) ────────────────
    try:
        order = client.submit_order(
            MarketOrderRequest(
                symbol=ticker,
                notional=round(target_usd, 2),
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
        )
        result = {
            "ticker":     ticker,
            "target_usd": target_usd,
            "order_id":   str(order.id),
            "method":     "notional",
            "status":     str(order.status),
        }
        _append_trade_log({"date": _today(), "event": "BUY_SUBMITTED", **result})
        logger.info("  Order submitted: %s  (notional)", order.id)
        return result

    except Exception as notional_exc:
        logger.warning("  Notional order failed for %s (%s) — trying whole shares…",
                       ticker, notional_exc)

    # ── Attempt 2: whole-share fallback ───────────────────────────────────────
    price = _fetch_price_yf(ticker)
    if not price or price <= 0:
        logger.error("  Cannot determine price for %s — BUY skipped", ticker)
        _append_trade_log({
            "date": _today(), "event": "BUY_ERROR", "ticker": ticker,
            "error": "price unavailable for whole-share fallback",
        })
        return {"ticker": ticker, "target_usd": target_usd, "status": "error_no_price"}

    qty = math.floor(target_usd / price)
    if qty < 1:
        logger.warning("  $%.2f / $%.2f = %.2f shares — floor=0, skipped",
                       target_usd, price, target_usd / price)
        _append_trade_log({
            "date": _today(), "event": "BUY_SKIPPED", "ticker": ticker,
            "reason": "qty_floor_zero", "target_usd": target_usd, "price": price,
        })
        return {"ticker": ticker, "target_usd": target_usd, "status": "skipped_qty_zero"}

    try:
        order = client.submit_order(
            MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
        )
        result = {
            "ticker":     ticker,
            "target_usd": target_usd,
            "order_id":   str(order.id),
            "method":     f"whole_shares (qty={qty}, price≈${price:.2f})",
            "status":     str(order.status),
        }
        _append_trade_log({"date": _today(), "event": "BUY_SUBMITTED", **result})
        logger.info("  Order submitted: %s  (qty=%d whole shares)", order.id, qty)
        return result

    except Exception as whole_exc:
        logger.error("  BUY failed for %s (both methods): %s", ticker, whole_exc)
        _append_trade_log({
            "date": _today(), "event": "BUY_ERROR", "ticker": ticker,
            "error": str(whole_exc),
        })
        return {"ticker": ticker, "target_usd": target_usd, "status": "error",
                "error": str(whole_exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Telegram heartbeat
# ─────────────────────────────────────────────────────────────────────────────

def _send_heartbeat(summary: dict) -> None:
    try:
        from scripts.telegram_notifier import send_alert

        status     = summary.get("pipeline_status", "UNKNOWN")
        equity     = summary.get("account_equity_post_liquidation", 0.0)
        regime     = summary.get("market_regime", "UNKNOWN")
        spsk_or    = summary.get("spsk_override", False)
        sells      = summary.get("liquidations", [])
        buys       = summary.get("new_positions", [])
        icon       = "✅" if status == "SUCCESS" else ("⚠️" if status == "PARTIAL" else "🚨")

        lines = [
            f"{icon} <b>Alpaca Paper Trader — {summary.get('run_date', '')}</b>",
            f"<b>Status:</b> <code>{status}</code>   "
            f"<b>Regime:</b> <code>{regime}</code>   "
            f"<b>SPSK:</b> <code>{'YES' if spsk_or else 'no'}</code>",
            f"<b>Account equity:</b> <code>${equity:,.2f}</code>",
            "",
        ]

        if sells:
            lines.append("<b>Liquidated:</b>")
            for s in sells:
                lines.append(f"  SELL <code>{s['ticker']}</code>  "
                             f"{s.get('qty', 0):.2f} shares")

        if buys:
            lines.append("<b>New positions:</b>")
            for b in buys:
                lines.append(
                    f"  BUY  <code>{b['ticker']}</code>  "
                    f"{int(b.get('weight', 0)*100)}%  "
                    f"${b.get('target_usd', 0):,.0f}"
                    + (f"  <code>{b.get('order_id', '')[:8]}</code>" if b.get('order_id') and b['order_id'] != 'DRY_RUN' else "")
                )

        lines.append(f"\n<i>{_now_utc()}</i>")
        level = "OK" if status == "SUCCESS" else "WARNING"
        send_alert("\n".join(lines), level=level)
    except Exception as exc:
        logger.warning("Telegram heartbeat failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Status display helper
# ─────────────────────────────────────────────────────────────────────────────

def _print_status(client) -> None:
    """Print current Alpaca account equity and positions."""
    account = client.get_all_positions()
    acct    = client.get_account()
    equity  = float(acct.equity)
    cash    = float(acct.cash)

    print(f"\nAlpaca Paper Account")
    print(f"  Equity:    ${equity:,.2f}")
    print(f"  Cash:      ${cash:,.2f}")
    print(f"  Positions: {len(account)}\n")
    if account:
        print(f"  {'Ticker':<8} {'Qty':>10} {'Mkt Value':>12} {'Unreal P&L':>12}")
        print("  " + "-" * 46)
        for pos in account:
            print(
                f"  {pos.symbol:<8} "
                f"{float(pos.qty):>10.4f} "
                f"${float(pos.market_value):>11,.2f} "
                f"${float(pos.unrealized_pl):>11,.2f}"
            )
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(force: bool = False, dry_run: bool = False) -> dict:
    """
    Full rebalancing pipeline. Returns the daily summary dict.
    """
    today = _today()

    # ── 1. Daily cache check ──────────────────────────────────────────────────
    if not force and not dry_run:
        cached = _load_daily_cache()
        if cached:
            logger.info("Already executed today — skipping (use --force to override).")
            return cached

    # ── 2. Load CIO decision ──────────────────────────────────────────────────
    if not DECISION_FILE.exists():
        raise FileNotFoundError(
            f"{DECISION_FILE} not found. "
            "Run: python3 scripts/equity_logic.py"
        )
    decision = json.loads(DECISION_FILE.read_text())

    decision_date = decision.get("decision_date", "")
    if decision_date and decision_date != today:
        logger.warning(
            "equity_decision.json is from %s (today is %s). "
            "Consider re-running equity_logic.py first.",
            decision_date, today,
        )

    selections  = decision.get("selections", [])
    regime      = decision.get("market_regime", "NEUTRAL")
    spsk_or     = decision.get("spsk_override", False)
    pipeline_st = "SUCCESS"

    if not selections:
        raise ValueError("No selections in equity_decision.json — aborting.")

    target_tickers = {s["ticker"] for s in selections}
    logger.info(
        "CIO decision: %s  regime=%s  spsk_override=%s",
        ", ".join(sorted(target_tickers)), regime, spsk_or,
    )

    # ── 3. Build Alpaca client ────────────────────────────────────────────────
    client = _build_client()

    # ── 4. Get current positions ──────────────────────────────────────────────
    current_positions: dict = {}
    try:
        all_pos = client.get_all_positions()
        current_positions = {p.symbol: p for p in all_pos}
        logger.info("Current positions: %s",
                    list(current_positions.keys()) or "(none)")
    except Exception as exc:
        logger.error("Failed to fetch positions: %s", exc)
        pipeline_st = "PARTIAL"

    # ── 5. Liquidate stale positions ──────────────────────────────────────────
    sell_order_ids = _liquidate_stale(
        client, current_positions, target_tickers, dry_run
    )

    liquidations = [
        {
            "ticker":       tk,
            "qty":          float(current_positions[tk].qty),
            "market_value": float(current_positions[tk].market_value)
                            if hasattr(current_positions[tk], "market_value") else 0.0,
            "reason":       "not_in_top3",
        }
        for tk in current_positions
        if tk not in target_tickers
    ]

    # ── 6. Poll until sells are filled ───────────────────────────────────────
    sells_ok = True
    if sell_order_ids and not dry_run:
        sells_ok = _poll_fills(client, sell_order_ids)
        if not sells_ok:
            pipeline_st = "PARTIAL"

    # ── 7. Read fresh account equity ─────────────────────────────────────────
    account_equity = 10_000.0   # safe default (paper account starting balance)
    try:
        acct           = client.get_account()
        account_equity = float(acct.equity)
        logger.info("Account equity (post-liquidation): $%s", f"{account_equity:,.2f}")
    except Exception as exc:
        logger.error("Failed to read account equity: %s", exc)
        pipeline_st = "PARTIAL"

    deployable = account_equity * _CASH_BUFFER_PCT
    logger.info("Deployable capital (%.1f%% of equity): $%s",
                _CASH_BUFFER_PCT * 100, f"{deployable:,.2f}")

    # ── 8. Submit BUY orders ─────────────────────────────────────────────────
    new_positions: list[dict] = []
    for sel in selections:
        ticker     = sel["ticker"]
        weight     = float(sel.get("weight", 0.0))
        target_usd = round(deployable * weight, 2)
        vam        = sel.get("vam_score", 0.0)

        logger.info(
            "Target  %-6s  weight=%.0f%%  $%.2f",
            ticker, weight * 100, target_usd,
        )

        buy_result = _submit_buy(client, ticker, target_usd, dry_run)
        new_positions.append({
            **buy_result,
            "weight":     weight,
            "vam_score":  vam,
            "sector":     sel.get("sector", "Unknown"),
        })

        if buy_result.get("status") in ("error", "error_no_price"):
            pipeline_st = "PARTIAL"

    # ── 9. Build summary ──────────────────────────────────────────────────────
    summary: dict = {
        "run_date":             today,
        "run_timestamp":        _now_utc(),
        "pipeline_status":      pipeline_st,
        "dry_run":              dry_run,
        "market_regime":        regime,
        "spsk_override":        spsk_or,
        "account_equity_post_liquidation": account_equity,
        "deployable_usd":       round(deployable, 2),
        "liquidations":         liquidations,
        "sell_orders_filled":   sells_ok,
        "new_positions":        new_positions,
        "decision_date":        decision_date,
    }

    _append_trade_log({"date": today, "event": "SUMMARY", **summary})

    # ── 10. Write daily summary ───────────────────────────────────────────────
    if not dry_run:
        _atomic_write(DAILY_SUMMARY_FILE, summary)
        logger.info("Wrote %s", DAILY_SUMMARY_FILE.name)

    # ── 11. Telegram heartbeat ────────────────────────────────────────────────
    if not dry_run:
        _send_heartbeat(summary)

    # ── Console summary ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Alpaca Trader — %s  [%s]%s", today, pipeline_st,
                "  DRY RUN" if dry_run else "")
    logger.info("Equity: $%s  |  Regime: %s  |  SPSK: %s",
                f"{account_equity:,.2f}", regime, spsk_or)
    for p in new_positions:
        logger.info(
            "  %-6s  %.0f%%  $%s  %s",
            p["ticker"],
            p.get("weight", 0) * 100,
            f"{p.get('target_usd', 0):,.0f}",
            p.get("status", ""),
        )
    logger.info("Liquidated: %s",
                [lq["ticker"] for lq in liquidations] or "(none)")
    logger.info("=" * 60)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Alpaca Paper Trading Execution Engine"
    )
    parser.add_argument(
        "--force",   action="store_true",
        help="Re-run even if today's summary already exists",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log all planned orders without submitting anything to Alpaca",
    )
    parser.add_argument(
        "--status",  action="store_true",
        help="Print current Alpaca positions and account equity, then exit",
    )
    args = parser.parse_args()

    if not _API_KEY or not _SECRET_KEY:
        logger.error(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env"
        )
        sys.exit(1)

    client = _build_client()

    if args.status:
        _print_status(client)
        return

    try:
        summary = run(force=args.force, dry_run=args.dry_run)
        success = summary.get("pipeline_status") in ("SUCCESS", "PARTIAL")
        sys.exit(0 if success else 1)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
