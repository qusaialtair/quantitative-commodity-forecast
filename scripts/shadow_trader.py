#!/usr/bin/env python3
"""
scripts/shadow_trader.py
========================
Paper-trading shadow engine implementing the OUNCE ACCUMULATION mandate.
Reads the Investment Committee's decision_log.json and executes simulated
trades at the live yfinance spot price, tracking inventory in SQLite.

State machine (Option B — explicit transitions only):
  State: METAL  (holding gold oz)
  State: FIAT   (holding USD cash)

  ACCUMULATE    : METAL/FIAT → METAL  (deploy all cash into oz)
  HOLD_METAL    : no transition       (maintain current state)
  STRATEGIC_EXIT: METAL → FIAT        (liquidate all oz → cash)
                  FIAT  → no action   (already protected)
  RE_ENTER      : FIAT  → METAL       (deploy all cash into oz)
                  METAL → no action   (already in metal)

A passive or ambiguous signal (HOLD_METAL) NEVER deploys capital.
Re-entering requires an explicit ACCUMULATE or RE_ENTER command.

Portfolio model:
  Starting capital : $100,000 USD (100% FIAT initially)
  Position sizing  : risk_manager.evaluate() — target vol 15%, VaR_95 hard stop 2.5%
  Slippage         : 5 bps per side
  Commission       : $2.50 flat per transaction (simulated exchange fee)

Usage:
  python3 scripts/shadow_trader.py --transact [--ticker GC=F]
  python3 scripts/shadow_trader.py --mark     [--ticker GC=F]
  python3 scripts/shadow_trader.py --status
  python3 scripts/shadow_trader.py --report
  python3 scripts/shadow_trader.py --install-launchd
"""

from __future__ import annotations
import argparse, json, math, sqlite3, subprocess, sys
from contextlib import contextmanager
from datetime import datetime, date, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import pandas as pd
import yfinance as yf

from scripts import risk_manager as _risk_manager

# ── Constants ─────────────────────────────────────────────────────────────────────

STARTING_CAPITAL = 100_000.0
SLIPPAGE_BPS     = 5.0
COMMISSION_USD   = 2.50
DEFAULT_TICKER   = "GC=F"

DB_PATH       = ROOT / "data" / "shadow_book.db"
DECISION_LOG  = ROOT / "data" / "decision_log.json"

# State machine action sets
_BUY_ACTIONS  = {"ACCUMULATE", "RE_ENTER"}
_SELL_ACTIONS = {"STRATEGIC_EXIT"}
# HOLD_METAL → no transition


# ── Database schema ───────────────────────────────────────────────────────────────

_SCHEMA = """
-- Single-row ledger: always reflects the current portfolio inventory.
CREATE TABLE IF NOT EXISTS portfolio_state (
    id              INTEGER  PRIMARY KEY DEFAULT 1,
    as_of_date      TEXT     NOT NULL,
    ticker          TEXT     NOT NULL DEFAULT 'GC=F',
    gold_oz         REAL     NOT NULL DEFAULT 0.0,
    cash_usd        REAL     NOT NULL DEFAULT 100000.0,
    portfolio_state TEXT     NOT NULL DEFAULT 'FIAT',   -- 'METAL' | 'FIAT'
    last_spot       REAL,
    portfolio_value REAL,
    updated_at      TEXT     NOT NULL
);

-- Full action log with before/after inventory snapshots and LLM thesis.
CREATE TABLE IF NOT EXISTS actions (
    id                  INTEGER  PRIMARY KEY AUTOINCREMENT,
    action_date         TEXT     NOT NULL,
    ticker              TEXT     NOT NULL DEFAULT 'GC=F',
    action_type         TEXT     NOT NULL,

    gold_oz_before      REAL     NOT NULL,
    cash_usd_before     REAL     NOT NULL,

    spot_price          REAL     NOT NULL,
    fill_price          REAL     NOT NULL,
    slippage_bps        REAL     NOT NULL DEFAULT 5.0,
    fee_usd             REAL     NOT NULL DEFAULT 2.50,

    oz_transacted       REAL     NOT NULL,
    cash_impact_usd     REAL     NOT NULL,

    gold_oz_after       REAL     NOT NULL,
    cash_usd_after      REAL     NOT NULL,

    action_triggered    TEXT     NOT NULL,
    cio_reasoning       TEXT,
    quant_conviction    INTEGER,
    macro_conviction    INTEGER,
    quant_thesis        TEXT,
    macro_thesis        TEXT,

    hmm_state           TEXT,
    hmm_veto_active     INTEGER  NOT NULL DEFAULT 0,
    oracle_score        REAL,
    real_yield          REAL,
    copper_gold_z       REAL,
    cot_gold_z          REAL,
    dxy_current         REAL,
    vix_current         REAL,

    created_at          TEXT     NOT NULL
);

-- Audit log for each STRATEGIC_EXIT → RE_ENTER cycle.
-- Primary alpha measurement: did we accumulate more oz per cycle?
CREATE TABLE IF NOT EXISTS strategic_cycles (
    id                  INTEGER  PRIMARY KEY AUTOINCREMENT,

    exit_action_id      INTEGER  NOT NULL REFERENCES actions(id),
    exit_date           TEXT     NOT NULL,
    exit_spot           REAL     NOT NULL,
    exit_oz_sold        REAL     NOT NULL,
    exit_proceeds_usd   REAL     NOT NULL,

    reenter_action_id   INTEGER  REFERENCES actions(id),
    reenter_date        TEXT,
    reenter_spot        REAL,
    reenter_oz_bought   REAL,

    oz_delta            REAL,
    reentry_discount    REAL,
    days_in_fiat        INTEGER,

    status              TEXT     NOT NULL DEFAULT 'OPEN',
    created_at          TEXT     NOT NULL,
    updated_at          TEXT     NOT NULL
);

-- Daily mark-to-market equity curve.
CREATE TABLE IF NOT EXISTS daily_marks (
    mark_date       TEXT     NOT NULL,
    ticker          TEXT     NOT NULL,
    spot_price      REAL     NOT NULL,
    gold_oz         REAL     NOT NULL,
    cash_usd        REAL     NOT NULL,
    portfolio_value REAL     NOT NULL,
    created_at      TEXT     NOT NULL,
    PRIMARY KEY (mark_date, ticker)
);
"""


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    # Migrate actions table — add risk columns if not present (silent on existing)
    _risk_cols = [
        ("risk_target_weight", "REAL"),
        ("risk_var_95",        "REAL"),
        ("risk_var_override",  "INTEGER"),
        ("risk_deploy_usd",    "REAL"),
    ]
    for col, dtype in _risk_cols:
        try:
            conn.execute(f"ALTER TABLE actions ADD COLUMN {col} {dtype}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
    # Migrate daily_marks — rebuild if old schema (trade_id column present)
    _dm_cols = {r[1] for r in conn.execute("PRAGMA table_info(daily_marks)").fetchall()}
    if "trade_id" in _dm_cols or "gold_oz" not in _dm_cols or "cash_usd" not in _dm_cols:
        conn.executescript("""
            DROP TABLE IF EXISTS daily_marks;
            CREATE TABLE daily_marks (
                mark_date       TEXT     NOT NULL,
                ticker          TEXT     NOT NULL,
                spot_price      REAL     NOT NULL,
                gold_oz         REAL     NOT NULL,
                cash_usd        REAL     NOT NULL,
                portfolio_value REAL     NOT NULL,
                created_at      TEXT     NOT NULL,
                PRIMARY KEY (mark_date, ticker)
            );
        """)
    # Seed portfolio_state if empty
    existing = conn.execute("SELECT COUNT(*) FROM portfolio_state").fetchone()[0]
    if existing == 0:
        conn.execute("""
            INSERT INTO portfolio_state
                (id, as_of_date, ticker, gold_oz, cash_usd,
                 portfolio_state, portfolio_value, updated_at)
            VALUES (1, ?, ?, 0.0, ?, 'FIAT', ?, ?)
        """, (
            date.today().isoformat(),
            DEFAULT_TICKER,
            STARTING_CAPITAL,
            STARTING_CAPITAL,
            _utcnow(),
        ))
    conn.commit()


@contextmanager
def _get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    _init_db(conn)
    try:
        yield conn
    finally:
        conn.close()


# ── Helpers ───────────────────────────────────────────────────────────────────────

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fetch_spot(ticker: str) -> float:
    raw = yf.download(ticker, period="2d", interval="1d",
                      progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    cl = raw["Close"].dropna()
    if cl.empty:
        raise RuntimeError(f"No price data returned for {ticker}")
    return float(cl.iloc[-1])


def _load_latest_decision(ticker: str) -> dict | None:
    if not DECISION_LOG.exists():
        return None
    try:
        with open(DECISION_LOG) as f:
            log = json.load(f)
        entries = [d for d in log if d.get("ticker") == ticker]
        return entries[-1] if entries else None
    except Exception:
        return None


def _get_state(conn: sqlite3.Connection) -> sqlite3.Row:
    return conn.execute("SELECT * FROM portfolio_state WHERE id = 1").fetchone()


def _upsert_state(
    conn:      sqlite3.Connection,
    ticker:    str,
    gold_oz:   float,
    cash_usd:  float,
    spot:      float,
    pf_state:  str,
) -> None:
    value = gold_oz * spot + cash_usd
    conn.execute("""
        INSERT INTO portfolio_state
            (id, as_of_date, ticker, gold_oz, cash_usd,
             portfolio_state, last_spot, portfolio_value, updated_at)
        VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            as_of_date      = excluded.as_of_date,
            ticker          = excluded.ticker,
            gold_oz         = excluded.gold_oz,
            cash_usd        = excluded.cash_usd,
            portfolio_state = excluded.portfolio_state,
            last_spot       = excluded.last_spot,
            portfolio_value = excluded.portfolio_value,
            updated_at      = excluded.updated_at
    """, (
        date.today().isoformat(), ticker,
        round(gold_oz, 6), round(cash_usd, 2),
        pf_state,
        round(spot, 2), round(value, 2),
        _utcnow(),
    ))
    conn.commit()


def _log_action(
    conn:         sqlite3.Connection,
    ticker:       str,
    action_type:  str,
    oz_before:    float,
    cash_before:  float,
    spot:         float,
    fill:         float,
    oz_delta:     float,
    cash_delta:   float,
    oz_after:     float,
    cash_after:   float,
    decision:     dict,
    risk_fields:  dict | None = None,
) -> int:
    rf = risk_fields or {}
    cur = conn.execute("""
        INSERT INTO actions (
            action_date, ticker, action_type,
            gold_oz_before, cash_usd_before,
            spot_price, fill_price, slippage_bps, fee_usd,
            oz_transacted, cash_impact_usd,
            gold_oz_after, cash_usd_after,
            action_triggered, cio_reasoning,
            quant_conviction, macro_conviction, quant_thesis, macro_thesis,
            hmm_state, hmm_veto_active, oracle_score,
            real_yield, copper_gold_z, cot_gold_z, dxy_current, vix_current,
            risk_target_weight, risk_var_95, risk_var_override, risk_deploy_usd,
            created_at
        ) VALUES (
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
        )
    """, (
        date.today().isoformat(), ticker, action_type,
        round(oz_before, 6), round(cash_before, 2),
        round(spot, 2), round(fill, 4), SLIPPAGE_BPS, COMMISSION_USD,
        round(oz_delta, 6), round(cash_delta, 2),
        round(oz_after, 6), round(cash_after, 2),
        decision.get("action_taken", ""),
        decision.get("original_reasoning", ""),
        decision.get("quant_conviction"),
        decision.get("macro_conviction"),
        decision.get("quant_thesis"),
        decision.get("macro_thesis"),
        decision.get("hmm_state"),
        int(decision.get("hmm_veto_active", 0)),
        decision.get("oracle_score"),
        decision.get("real_yield"),
        decision.get("copper_gold_z"),
        decision.get("cot_gold_z"),
        decision.get("dxy_current"),
        decision.get("vix_current"),
        rf.get("risk_target_weight"),
        rf.get("risk_var_95"),
        rf.get("risk_var_override"),
        rf.get("risk_deploy_usd"),
        _utcnow(),
    ))
    conn.commit()
    return cur.lastrowid


# ── Strategic cycle tracking ──────────────────────────────────────────────────────

def _open_cycle(conn: sqlite3.Connection, action_id: int, oz_sold: float,
                proceeds: float, spot: float) -> None:
    conn.execute("""
        INSERT INTO strategic_cycles
            (exit_action_id, exit_date, exit_spot, exit_oz_sold,
             exit_proceeds_usd, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, 'OPEN', ?, ?)
    """, (
        action_id, date.today().isoformat(),
        round(spot, 2), round(oz_sold, 6),
        round(proceeds, 2),
        _utcnow(), _utcnow(),
    ))
    conn.commit()


def _close_cycle(conn: sqlite3.Connection, action_id: int,
                 oz_bought: float, spot: float) -> None:
    open_cyc = conn.execute(
        "SELECT * FROM strategic_cycles WHERE status = 'OPEN' "
        "ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if not open_cyc:
        return

    exit_spot   = float(open_cyc["exit_spot"])
    oz_sold     = float(open_cyc["exit_oz_sold"])
    exit_date   = open_cyc["exit_date"]
    days_fiat   = (date.today() - date.fromisoformat(exit_date)).days
    oz_delta    = round(oz_bought - oz_sold, 6)
    discount    = round((exit_spot - spot) / exit_spot * 100, 4) if exit_spot > 0 else 0.0

    conn.execute("""
        UPDATE strategic_cycles SET
            reenter_action_id = ?,
            reenter_date      = ?,
            reenter_spot      = ?,
            reenter_oz_bought = ?,
            oz_delta          = ?,
            reentry_discount  = ?,
            days_in_fiat      = ?,
            status            = 'CLOSED',
            updated_at        = ?
        WHERE id = ?
    """, (
        action_id, date.today().isoformat(),
        round(spot, 2), round(oz_bought, 6),
        oz_delta, discount, days_fiat,
        _utcnow(), open_cyc["id"],
    ))
    conn.commit()

    sign = "+" if oz_delta >= 0 else ""
    result = "ACCUMULATED" if oz_delta > 0 else "LOST OZ"
    print(f"  Cycle closed: sold {oz_sold:.4f} oz @ ${exit_spot:,.2f} → "
          f"bought {oz_bought:.4f} oz @ ${spot:,.2f}  "
          f"oz_delta {sign}{oz_delta:.4f} [{result}]  "
          f"discount {discount:+.2f}%  days_in_fiat {days_fiat}d")


# ── Commands ──────────────────────────────────────────────────────────────────────

def cmd_transact(ticker: str) -> None:
    """Execute today's CIO decision as a simulated trade."""
    decision = _load_latest_decision(ticker)
    if not decision:
        print(f"  No CIO decision found for {ticker} in decision_log.json")
        return

    action   = decision.get("action_taken", "HOLD_METAL")
    dec_date = decision.get("date", "")

    print(f"\n  Decision  : {action}")
    print(f"  Date      : {dec_date}  ({decision.get('timestamp', '')})")

    with _get_conn() as conn:
        # Idempotency: skip if this decision date was already processed
        last_action = conn.execute(
            "SELECT action_date FROM actions WHERE ticker = ? "
            "ORDER BY created_at DESC LIMIT 1", (ticker,)
        ).fetchone()
        if last_action and last_action["action_date"] >= dec_date:
            print(f"  Already processed {dec_date} — skipping.")
            return

        state    = _get_state(conn)
        pf_state = state["portfolio_state"]   # 'METAL' | 'FIAT'
        gold_oz  = float(state["gold_oz"])
        cash_usd = float(state["cash_usd"])

        try:
            spot = _fetch_spot(ticker)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            return
        print(f"  Live spot : ${spot:,.2f}  |  State: {pf_state}  "
              f"|  oz: {gold_oz:.4f}  cash: ${cash_usd:,.0f}")

        # ── State machine ────────────────────────────────────────────────────────

        if action in _BUY_ACTIONS:
            if cash_usd < 1.0:
                print(f"  {action} — no cash available to deploy (already fully in metal).")
                return

            # ── Risk Manager: volatility-adjusted sizing + VaR gate ─────────────
            rd = _risk_manager.evaluate(
                action=action,
                quant_conviction=int(decision.get("quant_conviction") or 5),
                macro_conviction=int(decision.get("macro_conviction") or 5),
                hmm_state=decision.get("hmm_state") or "RANGING",
                hmm_veto=bool(decision.get("hmm_veto_active", False)),
                portfolio={
                    "cash_usd":        cash_usd,
                    "gold_oz":         gold_oz,
                    "portfolio_value": gold_oz * spot + cash_usd,
                },
                ticker=ticker,
                spot_price=spot,
                commission=COMMISSION_USD,
                slippage_bps=SLIPPAGE_BPS,
            )

            print(f"  Risk: weight={rd.target_weight:.2%}  "
                  f"deploy=${rd.deploy_usd:,.0f}  "
                  f"VaR_95={rd.var_95_daily:.2%}  "
                  f"vol_21d={rd.realized_vol_21d_annual:.2%}  "
                  f"override={rd.var_override}")

            if rd.approved_action == "HOLD_METAL":
                first_note = rd.risk_notes.split("  |  ")[0].strip()
                print(f"  Risk manager downgraded to HOLD_METAL: {first_note}")
                value = gold_oz * spot + cash_usd
                print(f"  HOLD_METAL — no transition.  portfolio value ${value:,.0f}")
                return

            fill       = round(spot * (1.0 + SLIPPAGE_BPS / 10_000), 4)
            oz_bought  = rd.oz_to_transact
            cash_spent = rd.deploy_usd

            oz_after   = gold_oz + oz_bought
            cash_after = round(cash_usd - cash_spent, 2)

            risk_fields = {
                "risk_target_weight": rd.target_weight,
                "risk_var_95":        rd.var_95_daily,
                "risk_var_override":  int(rd.var_override),
                "risk_deploy_usd":    rd.deploy_usd,
            }

            action_id = _log_action(
                conn, ticker, action,
                gold_oz, cash_usd, spot, fill,
                oz_bought, -cash_spent, oz_after, cash_after, decision,
                risk_fields,
            )
            _upsert_state(conn, ticker, oz_after, cash_after, spot, "METAL")

            if action == "RE_ENTER":
                _close_cycle(conn, action_id, oz_bought, spot)

            print(f"  {action}  {oz_bought:.4f} oz @ ${fill:,.2f}  "
                  f"(${cash_spent:,.0f} deployed  fee ${COMMISSION_USD:.2f})")
            cash_msg = f"${cash_after:,.0f} cash reserved" if cash_after > 1.0 else "$0 cash"
            print(f"  Portfolio → {oz_after:.4f} oz metal  {cash_msg}")

        elif action in _SELL_ACTIONS:
            if gold_oz < 0.0001:
                print(f"  STRATEGIC_EXIT — already in FIAT (0 oz) — no action.")
                return
            slip_mult  = 1.0 - SLIPPAGE_BPS / 10_000
            fill       = round(spot * slip_mult, 4)
            proceeds   = gold_oz * fill - COMMISSION_USD

            oz_after   = 0.0
            cash_after = proceeds

            action_id = _log_action(
                conn, ticker, action,
                gold_oz, cash_usd, spot, fill,
                -gold_oz, proceeds, oz_after, cash_after, decision,
            )
            _upsert_state(conn, ticker, oz_after, cash_after, spot, "FIAT")
            _open_cycle(conn, action_id, gold_oz, proceeds, spot)

            print(f"  STRATEGIC_EXIT  sold {gold_oz:.4f} oz @ ${fill:,.2f}  "
                  f"proceeds ${proceeds:,.0f}  (fee ${COMMISSION_USD:.2f})")
            print(f"  Portfolio → 0 oz metal  ${cash_after:,.0f} cash")

        else:  # HOLD_METAL
            value = gold_oz * spot + cash_usd
            print(f"  HOLD_METAL — no transition.  "
                  f"portfolio value ${value:,.0f}  (state: {pf_state})")


def cmd_mark(ticker: str) -> None:
    """Mark portfolio to market. One entry per day per ticker."""
    today_str = date.today().isoformat()
    with _get_conn() as conn:
        if conn.execute(
            "SELECT mark_date FROM daily_marks WHERE mark_date = ? AND ticker = ?",
            (today_str, ticker),
        ).fetchone():
            print(f"  Already marked {ticker} for {today_str}.")
            return

        state    = _get_state(conn)
        gold_oz  = float(state["gold_oz"])
        cash_usd = float(state["cash_usd"])
        pf_state = state["portfolio_state"]

        try:
            spot = _fetch_spot(ticker)
        except Exception as exc:
            print(f"  ERROR fetching spot: {exc}")
            return

        value = gold_oz * spot + cash_usd

        conn.execute("""
            INSERT OR REPLACE INTO daily_marks
                (mark_date, ticker, spot_price, gold_oz, cash_usd, portfolio_value, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            today_str, ticker,
            round(spot, 2), round(gold_oz, 6),
            round(cash_usd, 2), round(value, 2),
            _utcnow(),
        ))
        # Keep portfolio_state last_spot fresh
        conn.execute(
            "UPDATE portfolio_state SET last_spot=?, portfolio_value=?, updated_at=? WHERE id=1",
            (round(spot, 2), round(value, 2), _utcnow()),
        )
        conn.commit()

        if pf_state == "METAL":
            cost_basis = conn.execute(
                "SELECT fill_price FROM actions WHERE ticker=? AND action_type IN "
                "('ACCUMULATE','RE_ENTER') ORDER BY created_at DESC LIMIT 1", (ticker,)
            ).fetchone()
            if cost_basis:
                cb    = float(cost_basis["fill_price"])
                unrealised = (spot - cb) * gold_oz - COMMISSION_USD
                sign  = "+" if unrealised >= 0 else ""
                print(f"  Marked {ticker} {today_str}  spot ${spot:,.2f}  "
                      f"oz {gold_oz:.4f}  unrealised {sign}${unrealised:,.0f}  "
                      f"portfolio ${value:,.0f}")
                return
        print(f"  Marked {ticker} {today_str}  spot ${spot:,.2f}  "
              f"portfolio ${value:,.0f}  [{pf_state}]")


def cmd_status() -> None:
    """Print current portfolio state and recent actions."""
    W = 68
    print("\n" + "━" * W)
    print("  SHADOW PORTFOLIO — Status")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("━" * W)

    with _get_conn() as conn:
        state    = _get_state(conn)
        gold_oz  = float(state["gold_oz"])
        cash_usd = float(state["cash_usd"])
        pf_state = state["portfolio_state"]

        try:
            spot  = _fetch_spot(state["ticker"])
            value = gold_oz * spot + cash_usd
        except Exception:
            spot  = float(state["last_spot"] or 0)
            value = float(state["portfolio_value"] or STARTING_CAPITAL)

        pnl      = value - STARTING_CAPITAL
        pnl_pct  = pnl / STARTING_CAPITAL * 100
        sign     = "+" if pnl >= 0 else ""

        print(f"  State            : {pf_state}")
        print(f"  Gold held        : {gold_oz:.4f} oz")
        print(f"  Cash (USD)       : ${cash_usd:,.0f}")
        print(f"  Live spot        : ${spot:,.2f}")
        print(f"  Portfolio value  : ${value:,.0f}  ({sign}${pnl:,.0f} / {sign}{pnl_pct:.2f}%)")

        # Strategic cycles
        cycles = conn.execute(
            "SELECT COUNT(*) as n, status FROM strategic_cycles GROUP BY status"
        ).fetchall()
        for c in cycles:
            print(f"  Strategic cycles : {c['n']} {c['status']}")

        open_cyc = conn.execute(
            "SELECT * FROM strategic_cycles WHERE status = 'OPEN' LIMIT 1"
        ).fetchone()
        if open_cyc:
            days = (date.today() - date.fromisoformat(open_cyc["exit_date"])).days
            print(f"  Open exit        : sold {open_cyc['exit_oz_sold']:.4f} oz @ "
                  f"${open_cyc['exit_spot']:,.2f}  ({days}d in fiat)")

        # Recent actions
        recent = conn.execute(
            "SELECT action_date, action_type, oz_transacted, fill_price "
            "FROM actions ORDER BY created_at DESC LIMIT 5"
        ).fetchall()
        if recent:
            print(f"\n  {'Date':>12}  {'Action':>16}  {'Oz':>10}  {'Fill':>10}")
            print("  " + "─" * (W - 2))
            for r in recent:
                sign_oz = "+" if float(r["oz_transacted"] or 0) >= 0 else ""
                print(f"  {r['action_date']:>12}  {r['action_type']:>16}  "
                      f"{sign_oz}{float(r['oz_transacted'] or 0):>9.4f}  "
                      f"${float(r['fill_price'] or 0):>9,.2f}")

    print("━" * W + "\n")


def cmd_report(ticker: str | None = None) -> None:
    """Compute and print a performance snapshot."""
    W = 68
    with _get_conn() as conn:
        q_ticker = ticker or DEFAULT_TICKER

        all_actions = conn.execute(
            "SELECT * FROM actions WHERE ticker = ? ORDER BY created_at",
            (q_ticker,),
        ).fetchall()

        cycles = conn.execute(
            "SELECT * FROM strategic_cycles ORDER BY created_at"
        ).fetchall()

        marks = conn.execute(
            "SELECT mark_date, portfolio_value FROM daily_marks "
            "WHERE ticker = ? ORDER BY mark_date", (q_ticker,)
        ).fetchall()

        state    = _get_state(conn)
        gold_oz  = float(state["gold_oz"])
        cash_usd = float(state["cash_usd"])

        try:
            spot  = _fetch_spot(q_ticker)
        except Exception:
            spot  = float(state["last_spot"] or 0)

        current_value = gold_oz * spot + cash_usd
        total_pnl     = current_value - STARTING_CAPITAL
        total_pnl_pct = total_pnl / STARTING_CAPITAL * 100

        # Closed cycles stats
        closed_cycles = [c for c in cycles if c["status"] == "CLOSED"]
        good_cycles   = [c for c in closed_cycles
                         if c["oz_delta"] is not None and float(c["oz_delta"]) > 0]
        avg_oz_delta  = (sum(float(c["oz_delta"]) for c in closed_cycles) / len(closed_cycles)
                         if closed_cycles else float("nan"))
        avg_discount  = (sum(float(c["reentry_discount"] or 0) for c in closed_cycles)
                         / len(closed_cycles) if closed_cycles else float("nan"))
        exit_quality  = (len(good_cycles) / len(closed_cycles) * 100
                         if closed_cycles else float("nan"))

        # Sharpe from daily marks equity curve
        sharpe = float("nan")
        max_dd = float("nan")
        if len(marks) >= 5:
            vals = [float(m["portfolio_value"]) for m in marks]
            import numpy as np
            vals_arr = np.array(vals)
            rets     = np.diff(vals_arr) / vals_arr[:-1]
            if rets.std() > 0:
                sharpe = float(rets.mean() / rets.std() * 252 ** 0.5)
            peak   = np.maximum.accumulate(vals_arr)
            max_dd = float(((vals_arr - peak) / peak).min() * 100)

        # Agent attribution from actions
        def _avg_field(field, action_types):
            vals = [a[field] for a in all_actions
                    if a["action_type"] in action_types and a[field] is not None]
            return sum(vals) / len(vals) if vals else float("nan")

        avg_q_buy  = _avg_field("quant_conviction", _BUY_ACTIONS)
        avg_q_sell = _avg_field("quant_conviction", _SELL_ACTIONS)
        avg_m_buy  = _avg_field("macro_conviction", _BUY_ACTIONS)
        avg_m_sell = _avg_field("macro_conviction", _SELL_ACTIONS)

        sign = "+" if total_pnl >= 0 else ""
        print("\n" + "━" * W)
        print("  SHADOW PORTFOLIO — Performance Report")
        print(f"  {date.today().isoformat()}  |  {q_ticker}  "
              f"|  Starting capital: ${STARTING_CAPITAL:,.0f}")
        print("━" * W)
        print(f"  Current portfolio value    ${current_value:>10,.0f}")
        print(f"  Total PnL                  {sign}${total_pnl:>9,.0f}  ({sign}{total_pnl_pct:.2f}%)")
        print(f"  Total actions logged       {len(all_actions):>10}")
        print(f"  Sharpe (annualised)        {sharpe:>10.2f}" if not math.isnan(sharpe)
              else "  Sharpe (annualised)        N/A")
        print(f"  Max drawdown               {max_dd:>+9.1f}%" if not math.isnan(max_dd)
              else "  Max drawdown               N/A")

        print("\n  ── Strategic Cycle Scorecard ──────────────────────────────")
        print(f"  Total cycles               {len(cycles):>10}")
        print(f"  Closed cycles              {len(closed_cycles):>10}")
        print(f"  Open cycles (in fiat)      {len(cycles) - len(closed_cycles):>10}")
        if closed_cycles:
            print(f"  Exit quality (oz+)         {exit_quality:>9.1f}%")
            sign_d = "+" if avg_oz_delta >= 0 else ""
            print(f"  Avg oz delta / cycle       {sign_d}{avg_oz_delta:>8.4f} oz")
            sign_disc = "+" if avg_discount >= 0 else ""
            print(f"  Avg reentry discount       {sign_disc}{avg_discount:>8.2f}%")

        print("\n  ── Agent Attribution ──────────────────────────────────────")
        print(f"  Quant conv. on BUY/SELL    {avg_q_buy:>+8.1f} / {avg_q_sell:>+.1f}"
              if not math.isnan(avg_q_buy) else "  Quant conviction          N/A")
        print(f"  Macro conv. on BUY/SELL    {avg_m_buy:>+8.1f} / {avg_m_sell:>+.1f}"
              if not math.isnan(avg_m_buy) else "  Macro conviction          N/A")

        if closed_cycles:
            print("\n  ── Cycle Detail ───────────────────────────────────────────")
            print(f"  {'Exit':>12}  {'Re-enter':>12}  {'ExitP':>8}  "
                  f"{'ReentP':>8}  {'Disc%':>7}  {'oz_delta':>9}")
            for c in closed_cycles[-8:]:
                s   = "+" if (c["oz_delta"] or 0) >= 0 else ""
                d   = f"{float(c['reentry_discount'] or 0):+.2f}%"
                print(f"  {c['exit_date']:>12}  {c['reenter_date']:>12}  "
                      f"${float(c['exit_spot']):>7,.0f}  "
                      f"${float(c['reenter_spot'] or 0):>7,.0f}  "
                      f"{d:>7}  {s}{float(c['oz_delta'] or 0):>8.4f}")

        print("━" * W + "\n")


# ── launchd installer ─────────────────────────────────────────────────────────────

def install_launchd() -> None:
    """Schedule --transact + --mark at 22:45 UTC (after CIO runs at 22:30 UTC)."""
    python  = sys.executable
    script  = str(Path(__file__).resolve())
    workdir = str(ROOT)
    label   = "com.metals.shadow_trader"
    plist   = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    logfile = "/tmp/metals_shadow_trader.log"

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>             <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/sh</string>
        <string>-c</string>
        <string>{python} {script} --transact &amp;&amp; {python} {script} --mark</string>
    </array>
    <key>WorkingDirectory</key>  <string>{workdir}</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>    <integer>22</integer>
        <key>Minute</key>  <integer>45</integer>
    </dict>
    <key>StandardOutPath</key>   <string>{logfile}</string>
    <key>StandardErrorPath</key> <string>{logfile}</string>
    <key>RunAtLoad</key>         <false/>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
"""
    plist.parent.mkdir(parents=True, exist_ok=True)
    plist.write_text(xml)
    subprocess.run(["launchctl", "unload", str(plist)], capture_output=True)
    result = subprocess.run(["launchctl", "load", str(plist)],
                            capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  launchd job installed: {label}  (22:45 UTC daily)")
        print(f"  Plist : {plist}")
        print(f"  Logs  : {logfile}")
    else:
        print(f"  launchctl load failed: {result.stderr}")


# ── CLI ───────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Shadow trader — ounce accumulation paper portfolio")
    parser.add_argument("--transact",        action="store_true",
        help="Execute today's CIO decision as a simulated trade")
    parser.add_argument("--mark",            action="store_true",
        help="Mark portfolio to market (one entry per day)")
    parser.add_argument("--status",          action="store_true",
        help="Print current portfolio state and recent actions")
    parser.add_argument("--report",          action="store_true",
        help="Compute and print performance + cycle scorecard")
    parser.add_argument("--install-launchd", action="store_true",
        help="Schedule --transact + --mark at 22:45 UTC daily")
    parser.add_argument("--ticker", default=DEFAULT_TICKER,
        help=f"Metal ticker (default: {DEFAULT_TICKER})")
    args = parser.parse_args()

    if args.install_launchd:
        install_launchd()
        return

    if not any([args.transact, args.mark, args.status, args.report]):
        parser.print_help()
        return

    if args.transact:
        print(f"\n  -- TRANSACT ({args.ticker}) --")
        cmd_transact(args.ticker)

    if args.mark:
        print(f"\n  -- MARK TO MARKET ({args.ticker}) --")
        cmd_mark(args.ticker)

    if args.status:
        cmd_status()

    if args.report:
        cmd_report(args.ticker)


if __name__ == "__main__":
    main()
