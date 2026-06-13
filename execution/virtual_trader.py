#!/usr/bin/env python3
"""
execution/virtual_trader.py
===========================
Phase 3 — Virtual Execution & Risk Engine for the Sharia-compliant equity
fund. Bridges the AI's target portfolio (data/equity_decision.json) with
the actual book (data/virtual_account.json).

Execution order — FIXED, NOT NEGOTIABLE:

    Phase A  RISK OVERRIDES
      · 2×ATR trailing-stop sweep  — ruthless exit, overrides any AI hold
      · Sharia-violation sweep     — strict AAOIFI regression → forced exit
    Phase B  EXITS (sells before buys)
      · Any holding absent from the new target book is liquidated
    Phase C  TRIMS
      · Holdings above their target weight are partially sold
    Phase D  BUYS
      · New entries + top-ups to target, funded from phases A–C cash

All sells execute before any buy so cash freed up by liquidation is
available to fund new positions. A 5 bps slippage cost is applied as a
price impact on every fill (buy above close, sell below).

Trailing stop math
------------------
At every rebalance / mark-to-market:
    highest_price_since_entry = max(prev_highest, today's High)
    trailing_stop_price       = highest_price_since_entry
                                − (atr_stop_multiplier × atr_at_entry)
If current close ≤ trailing_stop_price, the position is force-sold with
reason="ATR_STOP" even if the AI still wants to hold it.

State schema (data/virtual_account.json)
----------------------------------------
{
  "schema_version": "1.0",
  "generated_at":   ISO8601,
  "starting_cash":  1000000.0,
  "cash_balance":   352018.42,
  "total_equity":   1000842.71,        // cash + Σ(qty × current_price)
  "positions": {
    "AAPL": {
      "ticker":                    "AAPL",
      "qty":                       123.4567,
      "avg_entry_price":           180.50,
      "current_price":             183.12,
      "market_value":              22601.73,
      "atr_at_entry":              3.45,      // frozen at purchase, Wilder 14
      "atr_stop_multiplier":       2.0,       // frozen at purchase
      "highest_price_since_entry": 184.90,
      "trailing_stop_price":       178.00,
      "entry_timestamp":           ISO8601,
      "last_mark_timestamp":       ISO8601,
      "sharia_pass_at_entry":      true
    }
  },
  "last_rebalance": ISO8601
}

Trade log appended to data/trade_log_v2.jsonl — one JSON per line, never
overwritten. Every line has reason ∈ {REBALANCE_ENTRY, REBALANCE_EXIT,
REBALANCE_TRIM, REBALANCE_TOPUP, ATR_STOP, SHARIA_VIOLATION}.

CLI
---
  python3 -m execution.virtual_trader rebalance
  python3 -m execution.virtual_trader rebalance --dry-run
  python3 -m execution.virtual_trader mark                     # mark-to-market only
  python3 -m execution.virtual_trader status
  python3 -m execution.virtual_trader bootstrap --cash 1000000
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR           = ROOT / "data"
DECISION_PATH      = DATA_DIR / "equity_decision.json"
STATE_PATH         = DATA_DIR / "virtual_account.json"
TRADE_LOG_PATH     = DATA_DIR / "trade_log_v2.jsonl"
RANKING_PATH       = DATA_DIR / "equity_ranker" / "ranking_latest.parquet"
DATA_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("VTRADER")


# ══════════════════════════════════════════════════════════════════════════════
#  Constants — risk + execution policy
# ══════════════════════════════════════════════════════════════════════════════

SLIPPAGE_BPS         = 5.0      # 0.05% price impact per fill, each side
ATR_PERIOD           = 14       # Wilder's — matches Stage 1 feature engine
ATR_STOP_MULTIPLIER  = 2.0      # stop = highest − 2·ATR_entry
ATR_LOOKBACK_BARS    = 60       # history window when computing fresh ATR
MIN_NOTIONAL         = 5.0      # skip orders below $5 notional (noise)
QTY_PRECISION        = 6        # fractional-share precision
DEFAULT_START_CASH   = 1_000_000.0


# ══════════════════════════════════════════════════════════════════════════════
#  State model
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    ticker: str
    qty: float
    avg_entry_price: float
    current_price: float
    atr_at_entry: float
    atr_stop_multiplier: float = ATR_STOP_MULTIPLIER
    highest_price_since_entry: float = 0.0
    trailing_stop_price: float = 0.0
    entry_timestamp: str = ""
    last_mark_timestamp: str = ""
    sharia_pass_at_entry: bool = True

    @property
    def market_value(self) -> float:
        return self.qty * self.current_price

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["market_value"] = round(self.market_value, 2)
        # clean rounding for JSON readability
        for k in ("qty",):
            d[k] = round(d[k], QTY_PRECISION)
        for k in ("avg_entry_price", "current_price", "atr_at_entry",
                  "highest_price_since_entry", "trailing_stop_price"):
            d[k] = round(d[k], 4)
        return d


# ══════════════════════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.rename(path)


def _append_trade_log(entry: dict) -> None:
    with TRADE_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def _wilder_atr(ohlc: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    """Wilder's ATR(period) — identical smoothing to Stage 1 feature engine."""
    if ohlc is None or len(ohlc) < period + 1:
        return float("nan")
    high, low, close = ohlc["High"], ohlc["Low"], ohlc["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_series = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return float(atr_series.iloc[-1])


# ══════════════════════════════════════════════════════════════════════════════
#  Market data — one yfinance call per rebalance, sliced per ticker
# ══════════════════════════════════════════════════════════════════════════════

def fetch_bars(tickers: list[str], bars: int = ATR_LOOKBACK_BARS) -> dict[str, pd.DataFrame]:
    """Batch-download daily OHLC for every ticker we care about."""
    if not tickers:
        return {}
    import yfinance as yf
    tickers = sorted(set(tickers))
    period = f"{max(bars, ATR_PERIOD + 5) + 10}d"   # yfinance calendar days → trading days
    try:
        raw = yf.download(tickers, period=period, interval="1d",
                          auto_adjust=True, progress=False, group_by="ticker",
                          threads=True)
    except Exception as exc:
        logger.warning("yfinance batch download failed: %s", exc)
        return {}

    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    has_multi = isinstance(raw.columns, pd.MultiIndex)
    for t in tickers:
        try:
            sub = raw[t] if has_multi else raw
            sub = sub.dropna(how="all")
            if not {"Open", "High", "Low", "Close"}.issubset(sub.columns):
                continue
            if sub.empty:
                continue
            out[t] = sub.tail(bars)
        except Exception:
            continue
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Load / persist state
# ══════════════════════════════════════════════════════════════════════════════

def load_state() -> dict:
    if not STATE_PATH.exists():
        return {
            "schema_version": "1.0",
            "generated_at":   _now(),
            "starting_cash":  DEFAULT_START_CASH,
            "cash_balance":   DEFAULT_START_CASH,
            "total_equity":   DEFAULT_START_CASH,
            "positions":      {},
            "last_rebalance": None,
        }
    state = json.loads(STATE_PATH.read_text())
    # Back-fill any newly added schema fields.
    state.setdefault("schema_version", "1.0")
    state.setdefault("positions", {})
    return state


def persist_state(state: dict) -> None:
    state["generated_at"] = _now()
    state["total_equity"] = round(
        state["cash_balance"] + sum(p["qty"] * p["current_price"]
                                     for p in state["positions"].values()),
        2,
    )
    _atomic_write(STATE_PATH, state)


def load_decision() -> dict:
    if not DECISION_PATH.exists():
        raise FileNotFoundError(f"No decision file at {DECISION_PATH}. "
                                "Run the swarm first.")
    return json.loads(DECISION_PATH.read_text())


def load_ranking() -> pd.DataFrame:
    """Used for the Sharia regression check — live strict gate."""
    if not RANKING_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(RANKING_PATH)
    if df.index.name != "ticker":
        df = df.set_index("ticker") if "ticker" in df.columns else df
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Order primitives — slippage and cash bookkeeping live here ONLY
# ══════════════════════════════════════════════════════════════════════════════

def _fill_price(close: float, side: str) -> float:
    """Apply slippage. BUY fills above close, SELL fills below."""
    s = SLIPPAGE_BPS / 1e4
    if side == "BUY":
        return close * (1.0 + s)
    if side == "SELL":
        return close * (1.0 - s)
    raise ValueError(side)


def _record_trade(state: dict, *, ticker: str, side: str, qty: float,
                  close: float, reason: str, extra: dict | None = None) -> dict:
    exec_price = _fill_price(close, side)
    notional   = qty * exec_price

    if side == "BUY":
        state["cash_balance"] -= notional
    else:
        state["cash_balance"] += notional

    entry = {
        "timestamp":    _now(),
        "ticker":       ticker,
        "side":         side,
        "reason":       reason,
        "qty":          round(qty, QTY_PRECISION),
        "price_close":  round(close, 4),
        "slippage_bps": SLIPPAGE_BPS,
        "price_exec":   round(exec_price, 4),
        "notional":     round(notional, 2),
        "cash_after":   round(state["cash_balance"], 2),
    }
    if extra:
        entry.update(extra)
    _append_trade_log(entry)
    logger.info("  %-4s %-6s  qty=%10.4f  @$%.4f  notional=$%12.2f  reason=%s",
                side, ticker, qty, exec_price, notional, reason)
    return entry


# ══════════════════════════════════════════════════════════════════════════════
#  Phase A — mark to market + trailing-stop update
# ══════════════════════════════════════════════════════════════════════════════

def mark_to_market(state: dict, bars: dict[str, pd.DataFrame]) -> None:
    """Refresh current_price, highest_price_since_entry, trailing_stop_price."""
    now = _now()
    for tkr, p in state["positions"].items():
        ohlc = bars.get(tkr)
        if ohlc is None or ohlc.empty:
            logger.warning("No market data for held %s — keeping stale mark.", tkr)
            continue
        last_close = float(ohlc["Close"].iloc[-1])
        last_high  = float(ohlc["High"].iloc[-1])
        p["current_price"] = round(last_close, 4)

        # Highest watermark uses the daily HIGH, not close — industry-standard
        # trailing-stop behavior. Prevents a volatile intraday spike from being
        # missed by a close-only watermark.
        prev_high = p.get("highest_price_since_entry") or p["avg_entry_price"]
        p["highest_price_since_entry"] = round(max(prev_high, last_high), 4)

        atr = p["atr_at_entry"]
        mult = p.get("atr_stop_multiplier", ATR_STOP_MULTIPLIER)
        p["trailing_stop_price"] = round(
            p["highest_price_since_entry"] - mult * atr, 4
        )
        p["market_value"] = round(p["qty"] * last_close, 2)
        p["last_mark_timestamp"] = now


def sweep_stop_outs(state: dict, bars: dict[str, pd.DataFrame],
                    live_sharia: pd.Series | None) -> list[str]:
    """Phase A — force-sell anything breaching its 2×ATR stop or failing
    the live Sharia regression check. Overrides AI hold commands."""
    evicted: list[str] = []
    for tkr in list(state["positions"].keys()):
        p = state["positions"][tkr]
        ohlc = bars.get(tkr)
        if ohlc is None or ohlc.empty:
            continue
        close = float(ohlc["Close"].iloc[-1])

        # ─── 2×ATR trailing stop ─────────────────────────────────────────────
        if close <= p["trailing_stop_price"]:
            _record_trade(
                state, ticker=tkr, side="SELL", qty=p["qty"], close=close,
                reason="ATR_STOP",
                extra={
                    "atr_at_entry":       p["atr_at_entry"],
                    "stop_multiplier":    p["atr_stop_multiplier"],
                    "trailing_stop":      p["trailing_stop_price"],
                    "highest_seen":       p["highest_price_since_entry"],
                    "pnl_per_share":      round(close - p["avg_entry_price"], 4),
                },
            )
            del state["positions"][tkr]
            evicted.append(tkr)
            continue

        # ─── Sharia regression ───────────────────────────────────────────────
        if live_sharia is not None and tkr in live_sharia.index:
            if not bool(live_sharia.loc[tkr]):
                _record_trade(
                    state, ticker=tkr, side="SELL", qty=p["qty"], close=close,
                    reason="SHARIA_VIOLATION",
                    extra={"pnl_per_share": round(close - p["avg_entry_price"], 4)},
                )
                del state["positions"][tkr]
                evicted.append(tkr)
    return evicted


# ══════════════════════════════════════════════════════════════════════════════
#  Phase B — liquidate positions not in target
#  Phase C — trim overweight
#  Phase D — buys (new + top-ups)
# ══════════════════════════════════════════════════════════════════════════════

def _target_from_decision(decision: dict, live_sharia: pd.Series | None) -> dict[str, dict]:
    """ticker → {weight, name, sector}. Sharia-non-compliant rows dropped."""
    out: dict[str, dict] = {}
    for p in (decision.get("positions") or []):
        t = p.get("ticker")
        if not t or (p.get("weight") or 0.0) <= 0.0:
            continue
        # Defense-in-depth: never route a non-compliant ticker to execution
        # even if the CIO somehow sized one.
        if live_sharia is not None and t in live_sharia.index:
            if not bool(live_sharia.loc[t]):
                logger.warning(
                    "Target position %s dropped — fails live strict Sharia gate.",
                    t,
                )
                continue
        out[t] = {
            "weight":     float(p["weight"]),
            "name":       p.get("name", t),
            "sector":     p.get("sector", "Unknown"),
        }
    return out


def rebalance(state: dict, decision: dict,
              bars: dict[str, pd.DataFrame],
              live_sharia: pd.Series | None) -> dict:
    """Run phases B → C → D using the refreshed state and market data."""
    targets = _target_from_decision(decision, live_sharia)
    target_tkrs = set(targets.keys())
    held_tkrs   = set(state["positions"].keys())

    # ── Phase B — exit positions not in the new target book ─────────────────
    for tkr in list(held_tkrs - target_tkrs):
        p = state["positions"][tkr]
        ohlc = bars.get(tkr)
        if ohlc is None or ohlc.empty:
            logger.warning("Cannot exit %s — no price data. Leaving stale.", tkr)
            continue
        close = float(ohlc["Close"].iloc[-1])
        _record_trade(
            state, ticker=tkr, side="SELL", qty=p["qty"], close=close,
            reason="REBALANCE_EXIT",
            extra={"pnl_per_share": round(close - p["avg_entry_price"], 4)},
        )
        del state["positions"][tkr]

    # Recompute total equity after exits — trims and buys are sized off this.
    total_equity = state["cash_balance"] + sum(
        p["qty"] * p["current_price"] for p in state["positions"].values()
    )

    # ── Phase C — trim positions that exceed their target weight ────────────
    for tkr in list(state["positions"].keys()):
        if tkr not in targets:
            continue
        p = state["positions"][tkr]
        ohlc = bars.get(tkr)
        if ohlc is None or ohlc.empty:
            continue
        close = float(ohlc["Close"].iloc[-1])

        dollar_target  = total_equity * targets[tkr]["weight"]
        dollar_current = p["qty"] * close
        delta_dollars  = dollar_current - dollar_target

        if delta_dollars > MIN_NOTIONAL:
            sell_qty = round(delta_dollars / close, QTY_PRECISION)
            sell_qty = min(sell_qty, p["qty"])
            if sell_qty * close < MIN_NOTIONAL:
                continue
            _record_trade(
                state, ticker=tkr, side="SELL", qty=sell_qty, close=close,
                reason="REBALANCE_TRIM",
                extra={"target_weight": targets[tkr]["weight"]},
            )
            p["qty"] = round(p["qty"] - sell_qty, QTY_PRECISION)
            if p["qty"] <= 1e-9:
                del state["positions"][tkr]

    # ── Phase D — buy new entries + top-ups ─────────────────────────────────
    for tkr, meta in targets.items():
        ohlc = bars.get(tkr)
        if ohlc is None or ohlc.empty:
            logger.warning("Cannot buy %s — no price data.", tkr)
            continue
        close = float(ohlc["Close"].iloc[-1])
        dollar_target  = total_equity * meta["weight"]

        if tkr in state["positions"]:
            p = state["positions"][tkr]
            dollar_current = p["qty"] * close
            delta_dollars  = dollar_target - dollar_current
            if delta_dollars < MIN_NOTIONAL:
                continue
            # Honor cash — never overdraft.
            delta_dollars = min(delta_dollars, state["cash_balance"])
            if delta_dollars < MIN_NOTIONAL:
                continue
            exec_px  = _fill_price(close, "BUY")
            buy_qty  = round(delta_dollars / exec_px, QTY_PRECISION)
            if buy_qty * exec_px < MIN_NOTIONAL:
                continue
            _record_trade(
                state, ticker=tkr, side="BUY", qty=buy_qty, close=close,
                reason="REBALANCE_TOPUP",
                extra={"target_weight": meta["weight"]},
            )
            # Weighted-average the entry price over the combined lot.
            new_qty = p["qty"] + buy_qty
            p["avg_entry_price"] = round(
                (p["qty"] * p["avg_entry_price"] + buy_qty * exec_px) / new_qty, 4
            )
            p["qty"] = round(new_qty, QTY_PRECISION)
            p["current_price"] = round(close, 4)
            p["highest_price_since_entry"] = max(
                p.get("highest_price_since_entry", 0.0), close,
            )
            # atr_at_entry stays frozen at the original entry — by design.
            p["trailing_stop_price"] = round(
                p["highest_price_since_entry"] - p["atr_stop_multiplier"] * p["atr_at_entry"],
                4,
            )
            continue

        # New entry — compute ATR at entry and freeze it.
        atr = _wilder_atr(ohlc, ATR_PERIOD)
        if not math.isfinite(atr) or atr <= 0:
            logger.warning("Skip %s — ATR(%d) undefined (insufficient bars).",
                           tkr, ATR_PERIOD)
            continue

        dollar_budget = min(dollar_target, state["cash_balance"])
        if dollar_budget < MIN_NOTIONAL:
            continue
        exec_px = _fill_price(close, "BUY")
        buy_qty = round(dollar_budget / exec_px, QTY_PRECISION)
        notional = buy_qty * exec_px
        if notional < MIN_NOTIONAL:
            continue

        _record_trade(
            state, ticker=tkr, side="BUY", qty=buy_qty, close=close,
            reason="REBALANCE_ENTRY",
            extra={"target_weight": meta["weight"], "atr_at_entry": round(atr, 4)},
        )
        pos = Position(
            ticker=tkr, qty=buy_qty, avg_entry_price=round(exec_px, 4),
            current_price=round(close, 4), atr_at_entry=round(atr, 4),
            atr_stop_multiplier=ATR_STOP_MULTIPLIER,
            highest_price_since_entry=round(close, 4),
            trailing_stop_price=round(close - ATR_STOP_MULTIPLIER * atr, 4),
            entry_timestamp=_now(), last_mark_timestamp=_now(),
            sharia_pass_at_entry=True,
        )
        state["positions"][tkr] = pos.to_dict()

    state["last_rebalance"] = _now()
    return state


# ══════════════════════════════════════════════════════════════════════════════
#  Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_rebalance(dry_run: bool = False) -> dict:
    state    = load_state()
    decision = load_decision()
    ranking  = load_ranking()
    live_sharia = (ranking["sharia_pass"].astype(bool)
                   if not ranking.empty and "sharia_pass" in ranking.columns
                   else None)

    target_tkrs = [p["ticker"] for p in (decision.get("positions") or [])
                   if p.get("ticker")]
    held_tkrs   = list(state["positions"].keys())
    need_bars   = sorted(set(target_tkrs) | set(held_tkrs))

    logger.info("Loading market data for %d tickers (held=%d, target=%d)…",
                len(need_bars), len(held_tkrs), len(target_tkrs))
    bars = fetch_bars(need_bars)

    # Phase A — mark, then stop-out / Sharia sweep
    mark_to_market(state, bars)
    evicted = sweep_stop_outs(state, bars, live_sharia)
    if evicted:
        logger.info("Phase A evicted %d positions: %s",
                    len(evicted), ", ".join(evicted))

    # Phases B/C/D
    state = rebalance(state, decision, bars, live_sharia)

    if not dry_run:
        persist_state(state)
        logger.info("State → %s", STATE_PATH)
        logger.info("Trade log → %s", TRADE_LOG_PATH)
    else:
        logger.info("DRY RUN — state not persisted.")
    return state


def run_mark() -> dict:
    """Mark-to-market only — no trades, no rebalance. Used by the UI."""
    state = load_state()
    ranking = load_ranking()
    live_sharia = (ranking["sharia_pass"].astype(bool)
                   if not ranking.empty and "sharia_pass" in ranking.columns
                   else None)
    if not state["positions"]:
        persist_state(state)
        return state

    bars = fetch_bars(list(state["positions"].keys()))
    mark_to_market(state, bars)
    # Also surface would-be stops, but do NOT execute — operator must run
    # `rebalance` to actually sell.
    breaches = []
    for t, p in state["positions"].items():
        if p["current_price"] <= p["trailing_stop_price"]:
            breaches.append(t)
    if breaches:
        logger.warning("ATR stop breach (not executed in mark): %s",
                       ", ".join(breaches))
    persist_state(state)
    return state


def print_status(state: dict) -> None:
    pos = state.get("positions", {})
    eq  = state.get("total_equity", 0.0)
    cash = state.get("cash_balance", 0.0)
    invested = eq - cash
    print(f"\n  TOTAL EQUITY  ${eq:,.2f}")
    print(f"  CASH          ${cash:,.2f}    ({cash/eq*100 if eq else 0:.1f}%)")
    print(f"  INVESTED      ${invested:,.2f}  ({invested/eq*100 if eq else 0:.1f}%)")
    print(f"  POSITIONS     {len(pos)}\n")
    if not pos:
        return
    rows = sorted(pos.values(), key=lambda p: -p["qty"] * p["current_price"])
    print(f"  {'TKR':<6} {'QTY':>10}  {'ENTRY':>10}  {'LAST':>10}  "
          f"{'HIGH':>10}  {'STOP':>10}  {'MV':>12}  {'PnL':>10}")
    for p in rows:
        mv   = p["qty"] * p["current_price"]
        pnl  = (p["current_price"] - p["avg_entry_price"]) * p["qty"]
        print(f"  {p['ticker']:<6} {p['qty']:>10.4f}  "
              f"{p['avg_entry_price']:>10.2f}  {p['current_price']:>10.2f}  "
              f"{p['highest_price_since_entry']:>10.2f}  "
              f"{p['trailing_stop_price']:>10.2f}  ${mv:>11,.2f}  ${pnl:>9,.2f}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Virtual Execution & Risk Engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_rb = sub.add_parser("rebalance", help="Run full rebalance")
    ap_rb.add_argument("--dry-run", action="store_true",
                       help="Compute trades but do not persist state.")

    sub.add_parser("mark", help="Mark-to-market only (no trades)")

    ap_st = sub.add_parser("status", help="Print current account state")
    ap_st.add_argument("--json", action="store_true",
                       help="Emit raw state JSON instead of table.")

    ap_bs = sub.add_parser("bootstrap", help="Initialise a fresh virtual account")
    ap_bs.add_argument("--cash", type=float, default=DEFAULT_START_CASH)
    ap_bs.add_argument("--force", action="store_true",
                       help="Overwrite existing state.")

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s UTC [%(name)-7s] %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.cmd == "rebalance":
        state = run_rebalance(dry_run=args.dry_run)
        print_status(state)

    elif args.cmd == "mark":
        state = run_mark()
        print_status(state)

    elif args.cmd == "status":
        state = load_state()
        if args.json:
            print(json.dumps(state, indent=2, default=str))
        else:
            print_status(state)

    elif args.cmd == "bootstrap":
        if STATE_PATH.exists() and not args.force:
            print(f"State already exists at {STATE_PATH}. Use --force to overwrite.")
            sys.exit(1)
        state = {
            "schema_version": "1.0",
            "generated_at":   _now(),
            "starting_cash":  float(args.cash),
            "cash_balance":   float(args.cash),
            "total_equity":   float(args.cash),
            "positions":      {},
            "last_rebalance": None,
        }
        _atomic_write(STATE_PATH, state)
        print(f"Bootstrapped fresh virtual account with ${args.cash:,.2f} at {STATE_PATH}")


if __name__ == "__main__":
    main()
