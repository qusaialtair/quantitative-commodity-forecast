#!/usr/bin/env python3
"""
scripts/virtual_trader.py
==========================
Drop-in replacement for alpaca_trader.py.  All order execution is local —
no brokerage API required.  Prices come from yfinance; the account state
lives in data/virtual_account.json.

Execution flow (mirrors alpaca_trader.py exactly):
  1. Load equity_decision.json  →  today's Top-3 tickers + weights
  2. Daily cache check          →  skip if already executed today (--force bypasses)
  3. Read virtual_account.json  →  current positions + cash
  4. SELL stale positions       →  cash += qty × price × (1 − COMMISSION)
  5. Compute total equity       →  cash + mark-to-market remaining positions
  6. BUY target positions       →  qty = floor(equity × weight / (price × (1 + COMMISSION)))
  7. Append each event to data/trade_log.jsonl
  8. Write data/alpaca_daily_summary.json  (same schema — UI reads this)
  9. Telegram heartbeat

CLI
---
  python3 scripts/virtual_trader.py              # run (skip if already executed today)
  python3 scripts/virtual_trader.py --force      # re-run even if today's summary exists
  python3 scripts/virtual_trader.py --dry-run    # log without touching virtual_account.json
  python3 scripts/virtual_trader.py --status     # print current virtual portfolio
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.file_io import atomic_write_json, read_json

# ── Paths ─────────────────────────────────────────────────────────────────────
DECISION_FILE        = ROOT / "data" / "equity_decision.json"
VIRTUAL_ACCOUNT_FILE = ROOT / "data" / "virtual_account.json"
TRADE_LOG_FILE       = ROOT / "data" / "trade_log.jsonl"
DAILY_SUMMARY_FILE   = ROOT / "data" / "alpaca_daily_summary.json"
TRADE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Execution constants ───────────────────────────────────────────────────────
COMMISSION      = 0.0005   # 0.05% virtual commission per trade (both sides)
_CASH_BUFFER    = 0.995    # deploy 99.5% of equity — leave buffer
_MIN_NOTIONAL   = 1.0      # skip orders below $1
_PRICE_RETRIES  = 3        # yfinance fetch retries

# ── Logging ───────────────────────────────────────────────────────────────────
for _lib in ("yfinance", "urllib3", "peewee", "charset_normalizer", "httpx"):
    logging.getLogger(_lib).setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("virtual_trader")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _today() -> str:
    return date.today().isoformat()


def _atomic_write(path: Path, data: dict) -> None:
    atomic_write_json(path, data)


def _append_trade_log(event: dict) -> None:
    try:
        with TRADE_LOG_FILE.open("a") as f:
            f.write(json.dumps({**event, "timestamp": _now_utc()}) + "\n")
    except Exception as exc:
        logger.warning("trade_log write failed: %s", exc)


def _fetch_price(ticker: str) -> float | None:
    """Fetch latest close price via yfinance with retries."""
    import yfinance as yf

    for attempt in range(_PRICE_RETRIES):
        try:
            hist = yf.Ticker(ticker).history(period="2d")
            if not hist.empty:
                return float(hist["Close"].dropna().iloc[-1])
        except Exception as exc:
            logger.warning("Price fetch attempt %d failed for %s: %s", attempt + 1, ticker, exc)
            if attempt < _PRICE_RETRIES - 1:
                time.sleep(2 ** attempt)
    return None


def _load_virtual_account() -> dict:
    if not VIRTUAL_ACCOUNT_FILE.exists():
        raise FileNotFoundError(
            f"{VIRTUAL_ACCOUNT_FILE} not found.\n"
            "Run: python3 scripts/migrate_from_alpaca.py"
        )
    return read_json(VIRTUAL_ACCOUNT_FILE)


def _load_daily_cache() -> dict | None:
    try:
        if DAILY_SUMMARY_FILE.exists():
            d = read_json(DAILY_SUMMARY_FILE)
            if d.get("run_date") == _today():
                return d
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Status display
# ─────────────────────────────────────────────────────────────────────────────

def _print_status() -> None:
    acct = _load_virtual_account()
    cash = float(acct["cash_balance"])
    pos  = acct.get("positions", {})

    total_mkt = 0.0
    rows = []
    for sym, p in pos.items():
        qty  = float(p["qty"])
        price = _fetch_price(sym) or float(p.get("avg_cost", 0))
        mkt   = qty * price
        pnl   = (price - float(p["avg_cost"])) / float(p["avg_cost"]) * 100 if p["avg_cost"] else 0
        total_mkt += mkt
        rows.append((sym, qty, price, mkt, pnl))

    equity = cash + total_mkt
    print(f"\nVirtual Account (migrated from Alpaca)")
    print(f"  Equity:    ${equity:,.2f}")
    print(f"  Cash:      ${cash:,.2f}")
    print(f"  Positions: {len(pos)}\n")
    if rows:
        print(f"  {'Ticker':<8} {'Qty':>10} {'Price':>10} {'Mkt Value':>12} {'P&L':>8}")
        print("  " + "-" * 52)
        for sym, qty, price, mkt, pnl in rows:
            print(f"  {sym:<8} {qty:>10.4f} ${price:>9,.2f} ${mkt:>11,.2f} {pnl:>+7.1f}%")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Sell logic
# ─────────────────────────────────────────────────────────────────────────────

def _sell_stale(
    positions: dict,
    cash: float,
    target_tickers: set[str],
    prices: dict[str, float],
    dry_run: bool,
) -> tuple[dict, float, list[dict]]:
    """
    Liquidate all positions not in target_tickers.
    Returns (updated_positions, updated_cash, liquidations_list).
    Commission applied: proceeds = qty × price × (1 − COMMISSION).
    """
    stale = [tk for tk in list(positions.keys()) if tk not in target_tickers]
    liquidations: list[dict] = []

    for ticker in stale:
        pos   = positions[ticker]
        qty   = float(pos["qty"])
        price = prices.get(ticker)

        if price is None:
            logger.warning("SELL  %-6s  no price available — skipping", ticker)
            _append_trade_log({
                "date": _today(), "event": "SELL_SKIPPED",
                "ticker": ticker, "reason": "no_price",
            })
            continue

        proceeds = qty * price * (1.0 - COMMISSION)
        commission_usd = qty * price * COMMISSION

        logger.info(
            "SELL  %-6s  %.4f sh  @ $%.2f  → $%.2f  (comm $%.2f)%s",
            ticker, qty, price, proceeds, commission_usd,
            "  [DRY RUN]" if dry_run else "",
        )

        _append_trade_log({
            "date":           _today(),
            "event":          "SELL",
            "ticker":         ticker,
            "qty":            round(qty, 6),
            "price":          round(price, 4),
            "proceeds":       round(proceeds, 2),
            "commission_usd": round(commission_usd, 4),
            "reason":         "not_in_top3",
            "status":         "dry_run" if dry_run else "filled",
        })

        liquidations.append({
            "ticker":     ticker,
            "qty":        round(qty, 4),
            "price":      round(price, 4),
            "market_value": round(proceeds, 2),
            "reason":     "not_in_top3",
        })

        if not dry_run:
            cash += proceeds
            del positions[ticker]

    return positions, cash, liquidations


# ─────────────────────────────────────────────────────────────────────────────
# Buy logic
# ─────────────────────────────────────────────────────────────────────────────

def _buy_position(
    ticker: str,
    target_usd: float,
    positions: dict,
    cash: float,
    prices: dict[str, float],
    dry_run: bool,
) -> tuple[dict, float, dict]:
    """
    Buy `target_usd` of `ticker`.
    Effective buy price = price × (1 + COMMISSION).
    Uses fractional quantities (no whole-share rounding for stocks).
    Returns (updated_positions, updated_cash, buy_result).
    """
    if target_usd < _MIN_NOTIONAL:
        logger.warning("BUY  %-6s  $%.2f < minimum — skipped", ticker, target_usd)
        return positions, cash, {
            "ticker": ticker, "target_usd": target_usd, "status": "skipped_below_min",
        }

    price = prices.get(ticker)
    if price is None or price <= 0:
        logger.error("BUY  %-6s  no price available — skipped", ticker)
        _append_trade_log({
            "date": _today(), "event": "BUY_SKIPPED",
            "ticker": ticker, "reason": "no_price",
        })
        return positions, cash, {
            "ticker": ticker, "target_usd": target_usd, "status": "error_no_price",
        }

    effective_price = price * (1.0 + COMMISSION)
    qty             = target_usd / effective_price          # fractional
    commission_usd  = qty * price * COMMISSION
    actual_cost     = qty * effective_price

    # Clamp to available cash
    if actual_cost > cash:
        qty         = cash / effective_price
        actual_cost = cash
        commission_usd = qty * price * COMMISSION
        logger.warning(
            "BUY  %-6s  clamped to available cash ($%.2f)",
            ticker, cash,
        )

    logger.info(
        "BUY   %-6s  %.4f sh  @ $%.2f (eff)  cost $%.2f  comm $%.2f%s",
        ticker, qty, effective_price, actual_cost, commission_usd,
        "  [DRY RUN]" if dry_run else "",
    )

    result = {
        "ticker":          ticker,
        "target_usd":      round(target_usd, 2),
        "qty":             round(qty, 6),
        "price":           round(price, 4),
        "effective_price": round(effective_price, 4),
        "actual_cost":     round(actual_cost, 2),
        "commission_usd":  round(commission_usd, 4),
        "order_id":        f"VIRT-{_today()}-{ticker}",
        "method":          "virtual_fractional",
        "status":          "dry_run" if dry_run else "filled",
    }

    _append_trade_log({"date": _today(), "event": "BUY", **result})

    if not dry_run:
        cash -= actual_cost
        if ticker in positions:
            # Average-down / average-up: weighted average cost
            old_qty  = float(positions[ticker]["qty"])
            old_cost = float(positions[ticker]["avg_cost"])
            new_qty  = old_qty + qty
            new_cost = (old_qty * old_cost + qty * price) / new_qty
            positions[ticker] = {
                "qty":       round(new_qty, 6),
                "avg_cost":  round(new_cost, 4),
            }
        else:
            positions[ticker] = {
                "qty":      round(qty, 6),
                "avg_cost": round(price, 4),
            }

    return positions, cash, result


# ─────────────────────────────────────────────────────────────────────────────
# Telegram heartbeat
# ─────────────────────────────────────────────────────────────────────────────

def _send_heartbeat(summary: dict) -> None:
    try:
        from scripts.telegram_notifier import send_alert

        status  = summary.get("pipeline_status", "UNKNOWN")
        equity  = summary.get("account_equity_post_liquidation", 0.0)
        regime  = summary.get("market_regime", "UNKNOWN")
        spsk_or = summary.get("spsk_override", False)
        sells   = summary.get("liquidations", [])
        buys    = summary.get("new_positions", [])
        icon    = "✅" if status == "SUCCESS" else ("⚠️" if status == "PARTIAL" else "🚨")

        lines = [
            f"{icon} <b>Virtual Trader — {summary.get('run_date', '')}</b>",
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
                             f"{s.get('qty', 0):.4f} sh  @${s.get('price', 0):,.2f}")
        if buys:
            lines.append("<b>New positions:</b>")
            for b in buys:
                lines.append(
                    f"  BUY  <code>{b['ticker']}</code>  "
                    f"{int(b.get('weight', 0) * 100)}%  "
                    f"${b.get('target_usd', 0):,.0f}"
                )
        lines.append(f"\n<i>{_now_utc()}</i>")
        level = "OK" if status == "SUCCESS" else "WARNING"
        send_alert("\n".join(lines), level=level)
    except Exception as exc:
        logger.warning("Telegram heartbeat failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(force: bool = False, dry_run: bool = False) -> dict:
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
            f"{DECISION_FILE} not found. Run: python3 scripts/equity_logic.py"
        )
    decision = read_json(DECISION_FILE)

    decision_date = decision.get("decision_date", "")
    if decision_date and decision_date != today:
        logger.warning(
            "equity_decision.json is from %s (today is %s). "
            "Consider re-running equity_logic.py first.",
            decision_date, today,
        )

    selections     = decision.get("selections", [])
    regime         = decision.get("market_regime", "NEUTRAL")
    spsk_or        = decision.get("spsk_override", False)
    pipeline_st    = "SUCCESS"

    if not selections:
        raise ValueError("No selections in equity_decision.json — aborting.")

    target_tickers = {s["ticker"] for s in selections}
    logger.info(
        "CIO decision: %s  regime=%s  spsk_override=%s",
        ", ".join(sorted(target_tickers)), regime, spsk_or,
    )

    # ── 3. Load virtual account ───────────────────────────────────────────────
    acct      = _load_virtual_account()
    cash      = float(acct["cash_balance"])
    positions = {k: dict(v) for k, v in acct.get("positions", {}).items()}

    logger.info(
        "Virtual account: cash=$%s  positions=%s",
        f"{cash:,.2f}", list(positions.keys()) or "(none)",
    )

    # ── 4. Fetch all needed prices ────────────────────────────────────────────
    all_tickers = set(positions.keys()) | target_tickers
    logger.info("Fetching prices for: %s", sorted(all_tickers))
    prices: dict[str, float] = {}
    for tk in all_tickers:
        p = _fetch_price(tk)
        if p:
            prices[tk] = p
            logger.info("  %-8s $%.4f", tk, p)
        else:
            logger.warning("  %-8s price unavailable", tk)

    # ── 5. Sell stale positions ───────────────────────────────────────────────
    positions, cash, liquidations = _sell_stale(
        positions, cash, target_tickers, prices, dry_run
    )

    # ── 5b. Partial-sell overlapping positions that exceed their target weight ─
    # Handles the case where a ticker is kept (still a target) but currently
    # represents more portfolio weight than the new allocation requires.
    # Example: FORM was 100% of portfolio; new target is 40% → sell the 60% excess
    # so that cash exists to fund the other target buys.
    _held_est = sum(
        float(p["qty"]) * prices.get(sym, float(p.get("avg_cost", 0)))
        for sym, p in positions.items()
    )
    _deployable_est = (cash + _held_est) * _CASH_BUFFER
    for sel in selections:
        tk     = sel["ticker"]
        weight = float(sel.get("weight", 0.0))
        if tk not in positions:
            continue
        price = prices.get(tk)
        if price is None:
            continue
        existing_qty = float(positions[tk]["qty"])
        existing_mkt = existing_qty * price
        target_val   = _deployable_est * weight
        if existing_mkt > target_val + 0.01:      # +$0.01 epsilon avoids float noise
            qty_to_sell    = (existing_mkt - target_val) / price
            proceeds       = qty_to_sell * price * (1.0 - COMMISSION)
            commission_usd = qty_to_sell * price * COMMISSION
            logger.info(
                "SELL  %-6s  %.4f sh  @ $%.2f  (excess rebalance)  → $%.2f%s",
                tk, qty_to_sell, price, proceeds,
                "  [DRY RUN]" if dry_run else "",
            )
            _append_trade_log({
                "date": _today(), "event": "SELL_EXCESS",
                "ticker": tk, "qty": round(qty_to_sell, 6),
                "price": round(price, 4), "proceeds": round(proceeds, 2),
                "commission_usd": round(commission_usd, 4),
                "reason": "rebalance_excess",
                "status": "dry_run" if dry_run else "filled",
            })
            liquidations.append({
                "ticker": tk, "qty": round(qty_to_sell, 4),
                "price": round(price, 4), "market_value": round(proceeds, 2),
                "reason": "rebalance_excess",
            })
            if not dry_run:
                cash += proceeds
                new_qty = existing_qty - qty_to_sell
                positions[tk]["qty"] = round(max(new_qty, 0.0), 6)

    # ── 6. Compute total equity & deployable capital ──────────────────────────
    held_value = sum(
        float(p["qty"]) * prices.get(sym, float(p.get("avg_cost", 0)))
        for sym, p in positions.items()
    )
    total_equity = cash + held_value
    deployable   = total_equity * _CASH_BUFFER

    logger.info(
        "Post-sell equity: $%s  (cash=$%s  held=$%s)  deployable=$%s",
        f"{total_equity:,.2f}", f"{cash:,.2f}",
        f"{held_value:,.2f}", f"{deployable:,.2f}",
    )

    # ── 7. Buy target positions ───────────────────────────────────────────────
    new_positions: list[dict] = []
    for sel in selections:
        ticker     = sel["ticker"]
        weight     = float(sel.get("weight", 0.0))
        vam        = sel.get("vam_score", 0.0)

        # Delta-buy: deduct current market value of any existing stake so we
        # don't double-buy tickers that survived the _sell_stale pass.
        full_target_usd = round(deployable * weight, 2)
        existing_mkt = 0.0
        if ticker in positions:
            existing_qty  = float(positions[ticker].get("qty", 0))
            existing_mkt  = existing_qty * prices.get(ticker, float(positions[ticker].get("avg_cost", 0)))
        target_usd = max(0.0, round(full_target_usd - existing_mkt, 2))

        logger.info(
            "Target  %-6s  weight=%.0f%%  full=$%.2f  existing=$%.2f  delta=$%.2f",
            ticker, weight * 100, full_target_usd, existing_mkt, target_usd,
        )

        positions, cash, buy_result = _buy_position(
            ticker, target_usd, positions, cash, prices, dry_run
        )

        new_positions.append({
            **buy_result,
            "weight":    weight,
            "vam_score": vam,
            "sector":    sel.get("sector", "Unknown"),
        })

        if buy_result.get("status") in ("error_no_price", "error"):
            pipeline_st = "PARTIAL"

    # ── 8. Persist updated virtual account ────────────────────────────────────
    if not dry_run:
        updated_acct = {
            **acct,
            "cash_balance":  round(cash, 2),
            "positions":     positions,
            "last_updated":  _now_utc(),
        }
        _atomic_write(VIRTUAL_ACCOUNT_FILE, updated_acct)
        logger.info("virtual_account.json updated.")

    # ── 9. Build final equity figure ──────────────────────────────────────────
    final_held = sum(
        float(p["qty"]) * prices.get(sym, float(p.get("avg_cost", 0)))
        for sym, p in positions.items()
    ) if not dry_run else held_value

    account_equity = cash + final_held if not dry_run else total_equity

    # ── 10. Write daily summary (same schema as alpaca_trader) ────────────────
    summary: dict = {
        "run_date":                       today,
        "run_timestamp":                  _now_utc(),
        "pipeline_status":                pipeline_st,
        "dry_run":                        dry_run,
        "broker":                         "virtual",
        "market_regime":                  regime,
        "spsk_override":                  spsk_or,
        "account_equity_post_liquidation": round(account_equity, 2),
        "deployable_usd":                 round(deployable, 2),
        "liquidations":                   liquidations,
        "sell_orders_filled":             True,
        "new_positions":                  new_positions,
        "decision_date":                  decision_date,
        "commission_rate":                COMMISSION,
    }

    _append_trade_log({"date": today, "event": "SUMMARY", **summary})

    if not dry_run:
        _atomic_write(DAILY_SUMMARY_FILE, summary)
        logger.info("Wrote %s", DAILY_SUMMARY_FILE.name)
        _send_heartbeat(summary)

    # ── Console summary ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Virtual Trader — %s  [%s]%s", today, pipeline_st,
                "  DRY RUN" if dry_run else "")
    logger.info("Equity: $%s  |  Regime: %s  |  SPSK: %s",
                f"{account_equity:,.2f}", regime, spsk_or)
    for p in new_positions:
        logger.info(
            "  %-6s  %.0f%%  $%s  %s",
            p["ticker"], p.get("weight", 0) * 100,
            f"{p.get('target_usd', 0):,.0f}",
            p.get("status", ""),
        )
    logger.info("Liquidated: %s", [lq["ticker"] for lq in liquidations] or "(none)")
    logger.info("=" * 60)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Virtual Broker — drop-in for Alpaca")
    parser.add_argument("--force",   action="store_true",
                        help="Re-run even if today's summary already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log all planned trades without touching virtual_account.json")
    parser.add_argument("--status",  action="store_true",
                        help="Print current virtual portfolio and exit")
    args = parser.parse_args()

    if args.status:
        _print_status()
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
