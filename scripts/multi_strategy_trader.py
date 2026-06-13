#!/usr/bin/env python3
"""
Multi-Strategy Paper Trader  (Phase XIV Stage 75)
==================================================
Active paper-trading book that listens to the Strategy Selector and executes
the chosen strategy across the metals + halal-equity universe.  Unlike the
existing virtual_trader (equity-only) and shadow_trader (gold passive), this
trader rotates between five strategy classes based on regime + conviction:

    TREND          ride conviction (longs and shorts)
    MEAN_REVERSION fade extended moves
    PAIRS          cointegrated spread trades (consumes pairs_trader.json)
    VOL_SHORT      synthetic short-vol via in-the-money calls (paper)
    TAIL_HEDGE     synthetic short via inverse exposure
    CASH           stand aside

The trader is intentionally separated from the existing books — it writes to
`data/phase14_book.json` and `data/phase14_nav.csv` so the operator can
compare the active multi-strategy book against the passive metals + equity
books.  Mark-to-market every run.

Position sizing:
    target_usd = book_equity × (strategy_selector.final_size_pct / 100)

Entry rules per strategy:
    TREND           open at current spot; trailing 2.0×ATR stop; +1.5% target
    MEAN_REVERSION  open against direction at z-extreme; -0.5×ATR stop;
                    +0.8% target
    PAIRS           open long/short the pair legs in dollar-balanced sizes
                    when an actionable signal exists
    VOL_SHORT       synthetic: open short-vol-equivalent long position 50%
                    of target size with tight 1.0×ATR stop (proxy)
    TAIL_HEDGE      open synthetic short 30% of book; +1.0% take on -3% move
    CASH            close all open trades

Exit rules:
    - Take-profit / stop / 5-trading-day timeout
    - On strategy change, close prior-strategy positions before opening new
    - Daily MTM updates open P&L

Mark-to-market via yfinance.  Commission 5 bps both sides (matches
virtual_trader convention).

Output:
    data/phase14_book.json   open + closed trades + book equity
    data/phase14_nav.csv     daily NAV history (date,nav_usd,cash,open_pl)
    data/multi_strategy_trader.json  one-line summary for pipeline_state
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
BOOK_FILE = DATA_DIR / "phase14_book.json"
NAV_CSV   = DATA_DIR / "phase14_nav.csv"
SUMMARY_FILE = DATA_DIR / "multi_strategy_trader.json"

STARTING_CAPITAL = 100_000.0
COMMISSION_BPS = 5  # 0.05% per side
TIMEOUT_DAYS = 5
DEFAULT_TICKER = "GC=F"
DEFAULT_SHORT_PROXY = "DUST"  # leveraged inverse gold for paper TAIL_HEDGE; falls back if unavailable

# ── Phase XXV: Treasury hedge sleeve ──────────────────────────────────────────
# The hedge sleeve is a 6th allocation that coexists with the alpha strategies.
# It is tagged with this strategy name in the book so it is (a) excluded from
# alpha exit/rotation logic and (b) cleanly separable for P&L attribution.
HEDGE_STRATEGY = "TREASURY_HEDGE"
# Hard ceiling on the sleeve as a fraction of book equity. Alpha risk-budgets
# are scaled by (1 - hedge_fraction) so total exposure never exceeds 100%.
HEDGE_MAX_FRACTION = 0.20
# Durable kill-switch path (shared with order_router).
HALT_FLAG = ROOT / "data" / "trading_halted.flag"
# Phase XXV-F: stress-eval output that gates LOCAL_ACTIVE.
STRESS_EVAL_FILE = DATA_DIR / "treasury_overlay_stress_eval.json"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _load(name: str) -> dict:
    p = DATA_DIR / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _safe(v: Any, d: float = 0.0) -> float:
    try:
        if v is None:
            return d
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return d
        return f
    except Exception:
        return d


def _today_iso() -> str:
    return date.today().isoformat()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _commission(notional_usd: float) -> float:
    return abs(notional_usd) * COMMISSION_BPS / 10_000


def _fetch_price(ticker: str) -> float | None:
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=True)
        if hist is None or hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Book management
# ──────────────────────────────────────────────────────────────────────────────
def _new_book() -> dict:
    return {
        "schema_version":  "1.0",
        "starting_capital": STARTING_CAPITAL,
        "cash_usd":         STARTING_CAPITAL,
        "open_trades":      [],
        "closed_trades":    [],
        "last_strategy":    None,
        "last_run":         None,
        "n_runs":           0,
        "hedge_state":      None,
    }


def _load_book() -> dict:
    if not BOOK_FILE.exists():
        return _new_book()
    try:
        book = json.loads(BOOK_FILE.read_text())
        if not isinstance(book, dict):
            return _new_book()
        # back-fill missing fields
        for k, v in _new_book().items():
            book.setdefault(k, v)
        if not isinstance(book.get("open_trades"), list):
            book["open_trades"] = []
        if not isinstance(book.get("closed_trades"), list):
            book["closed_trades"] = []
        return book
    except Exception:
        return _new_book()


def _save_book(book: dict) -> None:
    """Atomic write — tmp + os.replace so readers never see a partial book."""
    BOOK_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = BOOK_FILE.with_suffix(".json.tmp")
    payload = json.dumps(book, indent=2, default=str)
    tmp.write_text(payload)
    os.replace(tmp, BOOK_FILE)


def _safe_book_equity(book: dict) -> float:
    """Cash + open notionals for sizing; floored at zero to avoid div-by-zero."""
    cash = _safe(book.get("cash_usd"))
    notionals = sum(
        _safe(t.get("notional_usd"))
        for t in (book.get("open_trades") or [])
    )
    return max(0.0, cash + notionals)


def _book_equity(book: dict, marks: dict[str, float]) -> tuple[float, float]:
    """Returns (total_equity_usd, open_pl_usd) given current marks."""
    open_pl = 0.0
    for t in book["open_trades"]:
        ticker = t["ticker"]
        mark = marks.get(ticker)
        if mark is None or t.get("qty") is None:
            continue
        side = 1 if t["side"] == "LONG" else -1
        open_pl += (mark - t["entry_price"]) * t["qty"] * side
    return book["cash_usd"] + sum(
        t.get("notional_usd", 0.0) for t in book["open_trades"]
    ) + open_pl, open_pl


def _close_trade(book: dict, trade: dict, exit_price: float, reason: str) -> None:
    qty = trade["qty"]
    side = 1 if trade["side"] == "LONG" else -1
    notional_exit = qty * exit_price
    fee = _commission(notional_exit)
    pl = (exit_price - trade["entry_price"]) * qty * side - fee - trade.get("entry_fee", 0.0)
    book["cash_usd"] += trade.get("notional_usd", 0.0) + pl
    closed = {
        **trade,
        "exit_price":  round(exit_price, 4),
        "exit_at":     _now_iso(),
        "exit_reason": reason,
        "pl_usd":      round(pl, 2),
        "pl_pct":      round(pl / max(trade.get("notional_usd", 1.0), 1.0) * 100, 3),
        "exit_fee":    round(fee, 4),
    }
    book["closed_trades"].append(closed)

    # Phase XXIV [DEPRECATED — IBKR pivot]: opposite-side broker order to flatten
    # the position. Inert while EXECUTION_MODE=paper_internal (pinned default):
    # route_order returns NOOP and the book write above is the trade of record.
    # Retained for reference; do not re-enable IBKR without a compliance sign-off.
    try:
        from scripts.order_router import route_order
        exit_side = "SHORT" if trade["side"] == "LONG" else "LONG"
        rr = route_order(
            exit_side, trade["ticker"], qty, exit_price,
            trade.get("strategy", ""), note=f"exit:{reason}",
        )
        closed["ibkr_exit_status"] = rr.get("status")
        if rr.get("status") == "SUBMITTED":
            closed["ibkr_exit_order_id"] = rr.get("order_id")
    except Exception as exc:
        closed["ibkr_exit_status"] = "ROUTER_ERROR"
        closed["ibkr_exit_error"] = str(exc)


def _close_all(book: dict, marks: dict[str, float], reason: str) -> int:
    n = 0
    remaining = []
    for t in book["open_trades"]:
        # Phase XXV: the treasury hedge sleeve is managed independently and must
        # survive alpha strategy rotations (incl. rotations to CASH).
        if t.get("strategy") == HEDGE_STRATEGY:
            remaining.append(t)
            continue
        mark = marks.get(t["ticker"])
        if mark is None:
            remaining.append(t)
            continue
        _close_trade(book, t, mark, reason)
        n += 1
    book["open_trades"] = remaining
    return n


def _open_trade(
    book: dict,
    ticker: str,
    side: str,
    notional_usd: float,
    entry_price: float,
    strategy: str,
    stop_price: float | None,
    target_price: float | None,
    note: str = "",
    local_only: bool = False,
    sub_tag: str | None = None,
) -> dict:
    notional_usd = min(notional_usd, book["cash_usd"])
    if notional_usd < 100.0 or entry_price <= 0:
        return {}
    qty = notional_usd / entry_price
    fee = _commission(notional_usd)
    book["cash_usd"] -= notional_usd + fee
    trade = {
        "trade_id":     f"P14-{int(datetime.now(timezone.utc).timestamp()*1000)}-{ticker}",
        "ticker":       ticker,
        "strategy":     strategy,
        "side":         side,
        "entry_price":  round(entry_price, 4),
        "entry_at":     _now_iso(),
        "entry_date":   _today_iso(),
        "qty":          round(qty, 6),
        "notional_usd": round(notional_usd, 2),
        "entry_fee":    round(fee, 4),
        "stop_price":   round(stop_price, 4) if stop_price else None,
        "target_price": round(target_price, 4) if target_price else None,
        "note":         note,
    }
    if sub_tag:
        trade["sub_tag"] = sub_tag
    book["open_trades"].append(trade)

    # Phase XXV: local-only sleeves (e.g. TREASURY_HEDGE) are written strictly
    # to the internal paper_internal book and must NEVER touch the order router.
    if local_only:
        return trade

    # Phase XXIV [DEPRECATED — IBKR pivot]: route to external broker if
    # EXECUTION_MODE != paper_internal. Inert while EXECUTION_MODE=paper_internal
    # (pinned default): route_order returns NOOP and the internal book write
    # above is the trade of record. Metals tickers return NOT_ROUTABLE
    # (physical-only channel); halal equities formerly submitted through the
    # IBKR pretrade_gate + place_order audit chain. Do not re-enable IBKR
    # without a compliance sign-off.
    try:
        from scripts.order_router import route_order
        rr = route_order(side, ticker, qty, entry_price, strategy, note=note)
        trade["ibkr_status"] = rr.get("status")
        if rr.get("status") == "SUBMITTED":
            trade["ibkr_order_id"] = rr.get("order_id")
    except Exception as exc:
        trade["ibkr_status"] = "ROUTER_ERROR"
        trade["ibkr_error"] = str(exc)

    return trade


def _days_open(trade: dict) -> int:
    try:
        entry = date.fromisoformat(trade["entry_date"])
        return (date.today() - entry).days
    except Exception:
        return 0


def _check_exits(book: dict, marks: dict[str, float]) -> int:
    n = 0
    remaining = []
    for t in book["open_trades"]:
        # Phase XXV: hedge sleeve trades are exempt from ATR stops / profit
        # targets / 5-day timeout — they exit only on a macro regime transition
        # or Sharia-gate change, handled by _execute_treasury_hedge().
        if t.get("strategy") == HEDGE_STRATEGY:
            remaining.append(t)
            continue
        mark = marks.get(t["ticker"])
        if mark is None:
            remaining.append(t)
            continue
        side = 1 if t["side"] == "LONG" else -1
        # Stop check
        if t.get("stop_price") is not None:
            stop_hit = (
                (side == 1 and mark <= t["stop_price"]) or
                (side == -1 and mark >= t["stop_price"])
            )
            if stop_hit:
                _close_trade(book, t, mark, "STOP_HIT")
                n += 1
                continue
        # Target check
        if t.get("target_price") is not None:
            target_hit = (
                (side == 1 and mark >= t["target_price"]) or
                (side == -1 and mark <= t["target_price"])
            )
            if target_hit:
                _close_trade(book, t, mark, "TARGET_HIT")
                n += 1
                continue
        # Timeout
        if _days_open(t) >= TIMEOUT_DAYS:
            _close_trade(book, t, mark, "TIMEOUT")
            n += 1
            continue
        remaining.append(t)
    book["open_trades"] = remaining
    return n


# ──────────────────────────────────────────────────────────────────────────────
# Strategy executors
# ──────────────────────────────────────────────────────────────────────────────
# Phase XXVII: annualised vol the book is willing to run at full size.
# When blended realised vol exceeds this, position notionals are scaled by
# (target / realised) — classic vol targeting. Override via env for tuning.
VOL_TARGET_ANN_PCT = _safe(os.environ.get("QCTF_VOL_TARGET_ANN_PCT"), 22.0)
VOL_BREAKER_FLOOR = 0.25   # never cut below 25% of intended size
DEFAULT_RV_PCT = 20.0


def _rv_from_vol_surface(vs: dict, horizon: str, default: float = 0.0) -> float:
    """Schema-robust realised-vol read (new rv_21d / legacy realised_vol.21d_pct).

    The legacy-only read silently defaulted for months, so stops were sized
    for a 20%-vol market during a 42%-vol break. Mirrors strategy_selector.
    """
    ts = vs.get("term_structure") or {}
    v = _safe(ts.get(f"rv_{horizon}"), 0.0)
    if v > 0:
        return v
    legacy = ts.get("realised_vol") or {}
    v = _safe(legacy.get(f"{horizon}_pct"), 0.0)
    return v if v > 0 else default


def _blended_rv_pct() -> float:
    """max(rv_5d, rv_21d) — asymmetric on purpose: a fresh vol spike widens
    stops and shrinks size immediately, while a calm-down must persist into
    the 21d window before risk re-expands."""
    vs = _load("vol_surface.json")
    rv_21d = _rv_from_vol_surface(vs, "21d", default=DEFAULT_RV_PCT)
    rv_5d = _rv_from_vol_surface(vs, "5d", default=rv_21d)
    return max(rv_5d, rv_21d)


def _vol_breaker_multiplier(rv_blend_pct: float) -> float:
    """Volatility-targeting size multiplier in [VOL_BREAKER_FLOOR, 1.0]."""
    if rv_blend_pct <= VOL_TARGET_ANN_PCT or rv_blend_pct <= 0:
        return 1.0
    return max(VOL_BREAKER_FLOOR, VOL_TARGET_ANN_PCT / rv_blend_pct)


def _atr_proxy(ticker: str, price: float) -> float:
    """Approximate ATR via blended realised vol — daily move = vol/√252.

    Phase XXVII: uses max(rv_5d, rv_21d) instead of the (broken) 21d-only
    read, so stops breathe with fresh volatility instead of being sized for
    last month's regime.
    """
    rv_blend = _blended_rv_pct()
    daily_move_pct = rv_blend / math.sqrt(252)
    return price * daily_move_pct / 100.0


def _mean_rev_side(ticker: str, price: float, fallback_direction: str) -> str:
    """
    For mean-reversion, choose side based on distance to short-term MAs.
    Above 20d SMA → fade by going SHORT; below → fade by going LONG.
    Falls back to the stacker direction if technicals are not available.
    """
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="60d", interval="1d", auto_adjust=True)
        if hist is None or hist.empty or len(hist) < 20:
            raise ValueError("insufficient history")
        sma20 = float(hist["Close"].tail(20).mean())
        sma50 = float(hist["Close"].tail(50).mean()) if len(hist) >= 50 else sma20
        # Use the closer MA for the fade signal
        ref = sma20 if abs(price - sma20) >= abs(price - sma50) else sma50
        return "SHORT" if price > ref else "LONG"
    except Exception:
        if fallback_direction == "BUY":
            return "LONG"
        if fallback_direction == "SELL":
            return "SHORT"
        return "LONG"  # default fade-toward-mean assumption


def _execute_strategy(book: dict, strategy: str, selector: dict, marks: dict) -> list[dict]:
    """Open NEW trades dictated by the strategy class.  Returns list of trade dicts."""
    direction = (selector.get("direction") or "HOLD").upper()
    final_size = _safe(selector.get("final_size_pct"))
    if final_size <= 0 or strategy == "CASH":
        return []

    # Phase XXV-E: compound two independent risk scalars on the ALPHA sleeve only.
    #   pmult   — performance-targeter multiplier (pacing vs monthly target)
    #   dd_mult — drawdown-controller multiplier (graduated tier: 1.0→0.10)
    # The hedge sleeve is deliberately exempt from both: it provides crisis
    # protection and must not shrink when the book is under drawdown stress.
    targeter = _load("performance_targeter.json")
    pmult = _safe((targeter.get("risk_multiplier") or {}).get("final"), 1.0)
    dd = _load("drawdown_controller.json")
    dd_mult = _safe(dd.get("sizing_multiplier"), 1.0)

    # Phase XXVII: volatility-targeting circuit breaker on the ALPHA sleeve.
    # Blended realised vol above target proportionally shrinks every new
    # entry, independent of (and compounding with) the selector's regime
    # multipliers. Hedge sleeve stays exempt — it is the crisis protection.
    rv_blend = _blended_rv_pct()
    vol_mult = _vol_breaker_multiplier(rv_blend)

    adjusted_size = min(95.0, final_size * pmult * dd_mult * vol_mult)

    book_equity = _safe_book_equity(book)
    if book_equity <= 0:
        return []

    # Phase XXV: scale the alpha risk-budget down by the hedge sleeve fraction so
    # combined exposure (hedge + alpha) never exceeds 100% of book equity.
    hedge_frac = _hedge_fraction(book, book_equity)
    target_usd = book_equity * adjusted_size / 100.0 * (1.0 - hedge_frac)

    trades_opened: list[dict] = []
    metal_ticker = DEFAULT_TICKER
    metal_price = marks.get(metal_ticker)
    if metal_price is None:
        return []

    atr = _atr_proxy(metal_ticker, metal_price)

    if strategy == "TREND":
        side = "LONG" if direction == "BUY" else "SHORT"
        stop = metal_price - 2.0 * atr if side == "LONG" else metal_price + 2.0 * atr
        target = metal_price * 1.015 if side == "LONG" else metal_price * 0.985
        t = _open_trade(book, metal_ticker, side, target_usd, metal_price,
                        strategy, stop, target, note=f"TREND on conviction tier")
        if t:
            trades_opened.append(t)

    elif strategy == "MEAN_REVERSION":
        # Fade based on extension to moving averages.  Pull short-term technical context.
        side = _mean_rev_side(metal_ticker, metal_price, direction)
        # Tight stop, modest target
        stop = metal_price - 1.5 * atr if side == "LONG" else metal_price + 1.5 * atr
        target = metal_price * 1.008 if side == "LONG" else metal_price * 0.992
        t = _open_trade(book, metal_ticker, side, target_usd * 0.6, metal_price,
                        strategy, stop, target,
                        note=f"MEAN_REV {side} fade vs short-term MA")
        if t:
            trades_opened.append(t)

    elif strategy == "PAIRS":
        # Use first strong cointegration signal
        ce = _load("cointegration_engine.json")
        strong = [
            s for s in ce.get("actionable_signals", [])
            if abs(_safe(s.get("z_score"))) >= 2.0
        ]
        if strong:
            sig = strong[0]
            name = sig.get("name", "")  # e.g. "GC=F/SI=F"
            try:
                leg_a, leg_b = name.split("/")
            except ValueError:
                leg_a, leg_b = metal_ticker, "SI=F"
            # Refresh marks for both legs
            for leg in (leg_a, leg_b):
                if leg not in marks:
                    marks[leg] = _fetch_price(leg) or 0.0
            pa = marks.get(leg_a, 0.0)
            pb = marks.get(leg_b, 0.0)
            if pa > 0 and pb > 0:
                z = _safe(sig.get("z_score"))
                # If z > 0 spread is rich → short leg_a, long leg_b
                side_a = "SHORT" if z > 0 else "LONG"
                side_b = "LONG" if z > 0 else "SHORT"
                leg_notional = target_usd / 2.0
                atr_a = _atr_proxy(leg_a, pa)
                atr_b = _atr_proxy(leg_b, pb)
                ta = _open_trade(book, leg_a, side_a, leg_notional, pa,
                                 strategy,
                                 pa + 2 * atr_a if side_a == "SHORT" else pa - 2 * atr_a,
                                 pa * (0.99 if side_a == "SHORT" else 1.01),
                                 note=f"PAIR leg A z={z:+.2f}")
                tb = _open_trade(book, leg_b, side_b, leg_notional, pb,
                                 strategy,
                                 pb + 2 * atr_b if side_b == "SHORT" else pb - 2 * atr_b,
                                 pb * (0.99 if side_b == "SHORT" else 1.01),
                                 note=f"PAIR leg B z={z:+.2f}")
                if ta:
                    trades_opened.append(ta)
                if tb:
                    trades_opened.append(tb)

    elif strategy == "VOL_SHORT":
        # Paper proxy: open LONG metal at half size with very tight stop —
        # acts like a short-vol overlay (collects time decay if range-bound).
        stop = metal_price - 1.0 * atr
        target = metal_price * 1.005
        t = _open_trade(book, metal_ticker, "LONG", target_usd * 0.5, metal_price,
                        strategy, stop, target,
                        note="VOL_SHORT proxy — long with tight stop")
        if t:
            trades_opened.append(t)

    elif strategy == "TAIL_HEDGE":
        # Paper proxy: open SHORT metal at 30% size as protective overlay
        size = target_usd * 0.30
        stop = metal_price * 1.02
        target = metal_price * 0.97
        t = _open_trade(book, metal_ticker, "SHORT", size, metal_price,
                        strategy, stop, target,
                        note="TAIL_HEDGE protective short overlay")
        if t:
            trades_opened.append(t)

    return trades_opened


# ──────────────────────────────────────────────────────────────────────────────
# Treasury hedge sleeve  (Phase XXV)
# ──────────────────────────────────────────────────────────────────────────────
def _hedge_open_trades(book: dict) -> list[dict]:
    return [t for t in book["open_trades"] if t.get("strategy") == HEDGE_STRATEGY]


def _hedge_notional(book: dict) -> float:
    return sum(_safe(t.get("notional_usd")) for t in _hedge_open_trades(book))


def _hedge_fraction(book: dict, book_equity: float) -> float:
    """Current hedge notional as a fraction of book equity, capped at the ceiling."""
    if book_equity <= 0:
        return 0.0
    return min(_hedge_notional(book) / book_equity, HEDGE_MAX_FRACTION)


def _close_hedge_positions(book: dict, marks: dict[str, float], reason: str) -> int:
    """Flatten every open hedge-sleeve trade for which we have a mark."""
    n = 0
    remaining = []
    for t in book["open_trades"]:
        if t.get("strategy") != HEDGE_STRATEGY:
            remaining.append(t)
            continue
        mark = marks.get(t["ticker"])
        if mark is None:
            remaining.append(t)  # keep until we can mark it
            continue
        _close_trade(book, t, mark, reason)
        n += 1
    book["open_trades"] = remaining
    return n


def _stress_eval_gate() -> tuple[bool, str]:
    """Phase XXV-F: validation gate that blocks LOCAL_ACTIVE when the
    historical stress evaluation shows the overlay is net-negative.

    Returns (allowed: bool, reason: str).

    Decision table:
      OVERLAY_BENEFICIAL  → allowed
      OVERLAY_MIXED       → allowed (note logged; operator should monitor)
      OVERLAY_NEGATIVE    → BLOCKED — overlay degrades the book in crisis
      file absent         → allowed (first-run; eval hasn't been executed yet)
      file unreadable     → allowed (defensive; don't kill the sleeve on I/O noise)
    """
    if not STRESS_EVAL_FILE.exists():
        return True, "stress eval not yet run — allowed by default (run treasury_overlay_stress_eval.py)"
    try:
        ev = json.loads(STRESS_EVAL_FILE.read_text())
        verdict = (ev.get("verdict") or "").upper()
        note = ev.get("note", "")
        delta = ev.get("delta_avg_sharpe", 0.0)
        rescued = ev.get("n_rescued", 0)
        if verdict == "OVERLAY_NEGATIVE":
            return False, (
                f"OVERLAY_NEGATIVE: delta_sharpe={delta:+.3f} "
                f"rescued={rescued} — {note[:120]}"
            )
        return True, f"{verdict}: delta_sharpe={delta:+.3f} rescued={rescued}"
    except Exception as exc:
        return True, f"stress eval unreadable ({exc}) — allowed by default"


def _execute_treasury_hedge(
    book: dict,
    marks: dict[str, float],
    hedge: dict | None = None,
) -> list[dict]:
    """Phase XXV local Treasury hedge sleeve.

    Executes the treasury_hedge_overlay recommendation STRICTLY into the internal
    paper_internal book (local_only — never routed to a broker). Behaviour:

      - Only TREASURY_HEDGE_MODE=LOCAL_ACTIVE executes; any other mode flattens
        an existing sleeve and stands aside.
      - Rebalances ONLY on a transition: a macro regime change (quadrant/tier) or
        a change in the effective target instrument/size — the latter captures a
        Sharia-gate flip (TLT/IEF <-> GLD). A stable regime holds (no churn).
      - Sharia gate: coupon-bearing TLT/IEF only when TREASURY_SHARIA_CLEARED is
        true; otherwise the overlay reroutes the same budget to GLD and tags it
        sub_tag=sharia_fallback_gld.
      - Capped at HEDGE_MAX_FRACTION of book equity.

    Returns the list of newly opened hedge trades (may be empty).
    """
    from scripts.treasury_hedge_overlay import sanitize_hedge_recommendation

    hedge = sanitize_hedge_recommendation(
        hedge if hedge is not None else _load("treasury_hedge.json")
    )
    mode = (os.environ.get("TREASURY_HEDGE_MODE", "SIGNAL_ONLY") or "SIGNAL_ONLY").upper()
    state = book.get("hedge_state") or {}

    # ── E: Durable kill-switch ────────────────────────────────────────────────
    # The halt flag is checked here because the hedge sleeve is local_only and
    # bypasses route_order (which carries its own halt check). When halted we
    # FREEZE — no new positions, no rebalancing — but do NOT close existing
    # hedge positions: in a crisis event the sleeve is likely providing
    # protection and forcibly unwinding it at the halt moment could be harmful.
    if HALT_FLAG.exists():
        return []

    # ── F: Stress-eval validation gate ───────────────────────────────────────
    # Block LOCAL_ACTIVE if the historical overlay evaluation shows the sleeve
    # is net-negative across crisis windows. On OVERLAY_NEGATIVE we also close
    # any open hedge positions (the overlay is known to be harmful).
    if mode == "LOCAL_ACTIVE":
        gate_ok, gate_reason = _stress_eval_gate()
        if not gate_ok:
            _close_hedge_positions(book, marks, "STRESS_EVAL_BLOCKED")
            book["hedge_state"] = {
                "instrument":    None,
                "allocation_pct": 0.0,
                "quadrant":      None,
                "tier":          None,
                "sub_tag":       None,
                "mode":          mode,
                "gate_blocked":  True,
                "gate_reason":   gate_reason,
                "updated_at":    _now_iso(),
            }
            return []

    # Only LOCAL_ACTIVE executes into the internal book. Any other mode (incl.
    # SIGNAL_ONLY and the deprecated IBKR ACTIVE) must not hold a local sleeve.
    if mode != "LOCAL_ACTIVE":
        if _hedge_open_trades(book):
            _close_hedge_positions(book, marks, "HEDGE_MODE_OFF")
        book["hedge_state"] = {
            "instrument": None, "allocation_pct": 0.0, "quadrant": None,
            "tier": None, "sub_tag": None, "mode": mode, "updated_at": _now_iso(),
        }
        return []

    instrument = hedge.get("effective_instrument")
    target_pct = min(_safe(hedge.get("effective_allocation_pct")), HEDGE_MAX_FRACTION * 100.0)
    sub_tag = hedge.get("sub_tag")
    quadrant = hedge.get("regime_quadrant")
    tier = hedge.get("crisis_tier")

    desired = (instrument, round(target_pct, 1))
    current = (state.get("instrument"), round(_safe(state.get("allocation_pct")), 1))
    regime_changed = (quadrant, tier) != (state.get("quadrant"), state.get("tier"))
    target_changed = desired != current
    has_position = bool(_hedge_open_trades(book))

    # Rebalance only on a transition; otherwise hold to prevent churn.
    transition = regime_changed or target_changed
    needs_action = transition or (bool(instrument) and target_pct > 0 and not has_position)
    if not needs_action:
        return []

    # Flatten any existing sleeve before re-establishing (instrument/size may differ).
    _close_hedge_positions(book, marks, f"HEDGE_REBALANCE→{instrument or 'FLAT'}")

    opened: list[dict] = []
    if instrument and target_pct > 0:
        price = marks.get(instrument)
        if price and price > 0:
            book_equity = _safe_book_equity(book)
            if book_equity <= 0:
                return []
            notional = book_equity * target_pct / 100.0
            note = (
                "treasury hedge sleeve (sovereign duration)"
                if not sub_tag
                else "Sharia fallback — physical-gold proxy (GLD)"
            )
            t = _open_trade(
                book, instrument, "LONG", notional, price, HEDGE_STRATEGY,
                stop_price=None, target_price=None, note=note,
                local_only=True, sub_tag=sub_tag,
            )
            if t:
                opened.append(t)

    _, gate_reason = _stress_eval_gate() if mode == "LOCAL_ACTIVE" else (True, None)
    book["hedge_state"] = {
        "instrument":     instrument if opened else None,
        "allocation_pct": target_pct if opened else 0.0,
        "quadrant":       quadrant,
        "tier":           tier,
        "sub_tag":        sub_tag if opened else None,
        "gate_action":    hedge.get("gate_action"),
        "gate_blocked":   False,
        "stress_eval":    gate_reason,
        "mode":           mode,
        "updated_at":     _now_iso(),
    }
    return opened


def _by_strategy_rollup(book: dict) -> dict:
    """Per-strategy P&L separation so attribution can isolate hedge drag from
    core alpha. Keyed by strategy tag; hedge fallback trades carry sub_tag."""
    out: dict[str, dict] = {}

    def _slot(name: str) -> dict:
        return out.setdefault(name, {
            "realized_pl_usd": 0.0, "n_closed": 0,
            "open_notional_usd": 0.0, "n_open": 0,
        })

    for t in book.get("closed_trades", []):
        s = _slot(t.get("strategy", "UNKNOWN"))
        s["realized_pl_usd"] += _safe(t.get("pl_usd"))
        s["n_closed"] += 1
    for t in book.get("open_trades", []):
        s = _slot(t.get("strategy", "UNKNOWN"))
        s["open_notional_usd"] += _safe(t.get("notional_usd"))
        s["n_open"] += 1

    for s in out.values():
        s["realized_pl_usd"] = round(s["realized_pl_usd"], 2)
        s["open_notional_usd"] = round(s["open_notional_usd"], 2)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# NAV history
# ──────────────────────────────────────────────────────────────────────────────
def _append_nav(today: str, nav: float, cash: float, open_pl: float) -> None:
    NAV_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not NAV_CSV.exists()
    # Skip if today's row already exists
    if not new:
        try:
            existing_dates = set()
            with NAV_CSV.open() as f:
                next(f, None)
                for line in f:
                    p = line.strip().split(",")
                    if p:
                        existing_dates.add(p[0])
            if today in existing_dates:
                # update in place
                lines = NAV_CSV.read_text().splitlines()
                header = lines[0]
                out_lines = [header]
                for line in lines[1:]:
                    if line.startswith(today + ","):
                        out_lines.append(f"{today},{nav:.2f},{cash:.2f},{open_pl:.2f}")
                    else:
                        out_lines.append(line)
                NAV_CSV.write_text("\n".join(out_lines) + "\n")
                return
        except Exception:
            pass
    with NAV_CSV.open("a") as f:
        if new:
            f.write("date,nav_usd,cash_usd,open_pl_usd\n")
        f.write(f"{today},{nav:.2f},{cash:.2f},{open_pl:.2f}\n")


def _nav_stats() -> dict:
    if not NAV_CSV.exists():
        return {"n": 0}
    try:
        navs = []
        with NAV_CSV.open() as f:
            next(f, None)
            for line in f:
                p = line.strip().split(",")
                if len(p) >= 2:
                    navs.append((p[0], float(p[1])))
        if len(navs) < 2:
            return {"n": len(navs)}
        start = navs[0][1]
        end = navs[-1][1]
        cum = (end / start - 1.0) * 100 if start > 0 else 0.0
        # MTD
        today = date.today()
        fb = today.replace(day=1)
        while fb.weekday() >= 5:
            fb = date.fromordinal(fb.toordinal() + 1)
        mtd = [n for d, n in navs if d >= fb.isoformat()]
        mtd_ret = (mtd[-1] / mtd[0] - 1.0) * 100 if len(mtd) >= 2 and mtd[0] > 0 else 0.0
        # daily diffs
        diffs = [navs[i][1] / navs[i-1][1] - 1.0 for i in range(1, len(navs)) if navs[i-1][1] > 0]
        avg = sum(diffs) / len(diffs) if diffs else 0.0
        var = sum((x - avg) ** 2 for x in diffs) / len(diffs) if diffs else 0.0
        std = math.sqrt(var)
        sharpe = (avg / std * math.sqrt(252)) if std > 0 else None
        return {
            "n":               len(navs),
            "start_nav":       round(start, 2),
            "latest_nav":      round(end, 2),
            "cum_return_pct":  round(cum, 3),
            "mtd_return_pct":  round(mtd_ret, 3),
            "sharpe_approx":   round(sharpe, 3) if sharpe is not None else None,
            "first_date":      navs[0][0],
            "latest_date":     navs[-1][0],
        }
    except Exception:
        return {"n": 0}


# ──────────────────────────────────────────────────────────────────────────────
# Live market-data ingest / silent MTM refresh
# ──────────────────────────────────────────────────────────────────────────────
INGEST_CORE_UNIVERSE: tuple[str, ...] = (
    "GC=F", "SI=F", "GLD", "TLT", "IEF", "IAU",
)


def collect_ingest_universe(book: dict | None = None) -> set[str]:
    """Active price universe: core symbols + open book legs + hedge instrument."""
    tickers: set[str] = set(INGEST_CORE_UNIVERSE)
    book = book or _load_book()
    for trade in book.get("open_trades") or []:
        sym = trade.get("ticker")
        if sym:
            tickers.add(str(sym))

    from scripts.treasury_hedge_overlay import sanitize_hedge_recommendation

    hedge = sanitize_hedge_recommendation(_load("treasury_hedge.json"))
    for key in ("effective_instrument", "instrument"):
        inst = hedge.get(key)
        if inst:
            tickers.add(str(inst))
    return tickers


def fetch_live_marks(tickers: set[str] | list[str]) -> dict[str, float]:
    """Fetch latest marks via yfinance; skip symbols that time out or fail."""
    marks: dict[str, float] = {}
    for sym in sorted(set(tickers)):
        try:
            px = _fetch_price(sym)
            if px is not None and px > 0:
                marks[sym] = px
        except Exception:
            continue
    return marks


def run_mtm_refresh(marks: dict[str, float] | None = None) -> dict:
    """Silent MTM pass — mark open positions, apply exits, persist book/NAV.

    Does not open new alpha trades or rebalance the treasury hedge sleeve.
    Intended for the API market-data ingestion loop between full pipeline runs.
    """
    book = _load_book()
    universe = collect_ingest_universe(book)

    if marks is None:
        marks = fetch_live_marks(universe)
    else:
        missing = universe - set(marks)
        if missing:
            marks = {**marks, **fetch_live_marks(missing)}

    n_exits = _check_exits(book, marks)
    equity, open_pl = _book_equity(book, marks)

    book["last_mtm"] = _now_iso()
    book["n_mtm_runs"] = int(book.get("n_mtm_runs", 0)) + 1
    book["latest_marks"] = {k: round(v, 4) for k, v in marks.items()}
    _save_book(book)
    _append_nav(_today_iso(), equity, book["cash_usd"], open_pl)

    prior_summary = _load("multi_strategy_trader.json")
    start_cap = _safe(book.get("starting_capital"), STARTING_CAPITAL)
    summary = {
        **prior_summary,
        "schema_version": "1.0",
        "engine": "multi_strategy_trader",
        "generated_at": _now_iso(),
        "mtm_refresh": True,
        "strategy": prior_summary.get("strategy") or book.get("last_strategy"),
        "n_open": len(book["open_trades"]),
        "n_closed_total": len(book["closed_trades"]),
        "n_exits_this_run": n_exits,
        "n_new_trades": 0,
        "hedge_notional_usd": round(_hedge_notional(book), 2),
        "hedge_fraction": round(_hedge_fraction(book, max(equity, 0.0)), 4),
        "by_strategy": _by_strategy_rollup(book),
        "book_equity_usd": round(equity, 2),
        "cash_usd": round(book["cash_usd"], 2),
        "open_pl_usd": round(open_pl, 2),
        "starting_capital": start_cap,
        "lifetime_pl_pct": round(
            (equity / start_cap - 1.0) * 100, 3
        ) if start_cap > 0 else 0.0,
        "nav_stats": _nav_stats(),
        "marks_fetched": len(marks),
        "universe": sorted(universe),
    }
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2, default=str))

    return {
        "status": "OK",
        "book_equity_usd": summary["book_equity_usd"],
        "open_pl_usd": summary["open_pl_usd"],
        "n_exits": n_exits,
        "marks_fetched": len(marks),
        "universe": summary["universe"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────
def run_multi_strategy_trader(dry_run: bool = False) -> dict:
    """Execute hedge sleeve and alpha strategy against the Phase XIV paper book."""
    book = _load_book()
    selector = _load("strategy_selector.json")
    if not selector:
        # Without a selector decision, do nothing destructive
        return {
            "status":   "NO_SELECTOR",
            "strategy": None,
            "n_open":   len(book["open_trades"]),
            "generated_at": _now_iso(),
        }

    strategy = (selector.get("strategy") or "CASH").upper()
    prior_strategy = book.get("last_strategy")

    # Phase XXV: load the treasury hedge recommendation up front so its
    # instrument can be marked and the sleeve executed this run.
    from scripts.treasury_hedge_overlay import sanitize_hedge_recommendation

    hedge_reco = sanitize_hedge_recommendation(_load("treasury_hedge.json"))

    # Collect mark prices for open trades + the metal ticker
    tickers_to_mark = {DEFAULT_TICKER}
    for t in book["open_trades"]:
        tickers_to_mark.add(t["ticker"])
    # Mark the effective hedge instrument when the sleeve is live.
    if (os.environ.get("TREASURY_HEDGE_MODE", "").upper() == "LOCAL_ACTIVE"):
        eff = hedge_reco.get("effective_instrument")
        if eff:
            tickers_to_mark.add(eff)
    if strategy == "PAIRS":
        ce = _load("cointegration_engine.json")
        for s in ce.get("actionable_signals", []) or []:
            name = s.get("name", "")
            if "/" in name:
                leg_a, leg_b = name.split("/")
                tickers_to_mark.add(leg_a)
                tickers_to_mark.add(leg_b)
    marks: dict[str, float] = {}
    for t in tickers_to_mark:
        p = _fetch_price(t)
        if p:
            marks[t] = p

    # Step 1 — check exits on existing positions
    n_exits = _check_exits(book, marks)

    # Step 2 — strategy change closes all priors (hedge sleeve is exempt)
    n_closed_strategy_change = 0
    if prior_strategy and prior_strategy != strategy:
        n_closed_strategy_change = _close_all(book, marks, f"STRATEGY_CHANGE→{strategy}")

    # Step 2.5 — Phase XXV treasury hedge sleeve. Runs BEFORE alpha so the alpha
    # risk-budget can be scaled back by the hedge fraction (total exposure ≤100%).
    hedge_trades: list[dict] = []
    if not dry_run:
        hedge_trades = _execute_treasury_hedge(book, marks, hedge=hedge_reco)

    # Step 3 — open new trades dictated by the strategy
    new_trades: list[dict] = []
    if strategy != "CASH" and not dry_run:
        new_trades = _execute_strategy(book, strategy, selector, marks)

    # Step 4 — mark-to-market & log NAV
    equity, open_pl = _book_equity(book, marks)
    book["last_strategy"] = strategy
    book["last_run"] = _now_iso()
    book["n_runs"] = int(book.get("n_runs", 0)) + 1

    if not dry_run:
        _save_book(book)
        _append_nav(_today_iso(), equity, book["cash_usd"], open_pl)

    stats = _nav_stats()
    rv_blend = _blended_rv_pct()
    summary = {
        "schema_version":   "1.1",
        "engine":           "multi_strategy_trader",
        "generated_at":     _now_iso(),
        "strategy":         strategy,
        "prior_strategy":   prior_strategy,
        "strategy_changed": bool(prior_strategy and prior_strategy != strategy),
        "vol_breaker": {
            "rv_blend_ann_pct":   round(rv_blend, 2),
            "vol_target_ann_pct": VOL_TARGET_ANN_PCT,
            "size_multiplier":    round(_vol_breaker_multiplier(rv_blend), 3),
            "active":             _vol_breaker_multiplier(rv_blend) < 1.0,
        },
        "n_open":           len(book["open_trades"]),
        "n_closed_total":   len(book["closed_trades"]),
        "n_new_trades":     len(new_trades),
        "n_exits_this_run": n_exits,
        "n_closed_on_strategy_change": n_closed_strategy_change,
        "n_hedge_trades":   len(hedge_trades),
        "hedge_state":      book.get("hedge_state"),
        "hedge_notional_usd": round(_hedge_notional(book), 2),
        "hedge_fraction":   round(_hedge_fraction(book, max(equity, 0.0)), 4),
        "by_strategy":      _by_strategy_rollup(book),
        "book_equity_usd":  round(equity, 2),
        "cash_usd":         round(book["cash_usd"], 2),
        "open_pl_usd":      round(open_pl, 2),
        "starting_capital": book["starting_capital"],
        "lifetime_pl_pct":  round(
            (equity / start_cap - 1.0) * 100, 3
        ) if (start_cap := _safe(book.get("starting_capital"), STARTING_CAPITAL)) > 0 else 0.0,
        "nav_stats":        stats,
        "new_trade_ids":    [t["trade_id"] for t in new_trades],
        "dry_run":          dry_run,
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> int:
    """CLI entrypoint for the multi-strategy trader."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Compute but do not persist trades")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    s = run_multi_strategy_trader(dry_run=args.dry_run)
    if args.quiet:
        return 0
    print("=" * 64)
    print(f"MULTI-STRATEGY TRADER  ({s['generated_at']})")
    print("=" * 64)
    print(f"  Strategy        : {s['strategy']}  (was {s.get('prior_strategy')})")
    print(f"  Book equity     : ${s['book_equity_usd']:,.2f}  "
          f"(cash ${s['cash_usd']:,.2f} + open P&L ${s['open_pl_usd']:+,.2f})")
    print(f"  Lifetime return : {s['lifetime_pl_pct']:+.2f}%")
    print(f"  Open trades     : {s['n_open']}  closed lifetime: {s['n_closed_total']}")
    print(f"  This run        : opened={s['n_new_trades']}  "
          f"exits={s['n_exits_this_run']}  "
          f"closed-on-strategy-change={s['n_closed_on_strategy_change']}")
    if s.get("nav_stats", {}).get("n"):
        ns = s["nav_stats"]
        print(f"  NAV history     : {ns['n']} rows  "
              f"MTD={ns.get('mtd_return_pct', 0):+.2f}%  "
              f"cum={ns.get('cum_return_pct', 0):+.2f}%  "
              f"Sharpe≈{ns.get('sharpe_approx')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
