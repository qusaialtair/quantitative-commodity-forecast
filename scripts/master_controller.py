#!/usr/bin/env python3
"""
scripts/master_controller.py
============================
Daily pipeline orchestrator for the gold trading AI system.

Execution order (strictly sequential):
    Stage 1  alt_data_harvester  --update           [HARD ABORT gate]
    Stage 2  regime_detector     --fit GC=F          [soft — cached fallback]
    Stage 3  metal_logic         evaluate_metal_swing [soft — falls back HOLD_METAL]
    Stage 4  risk_manager        evaluate()  (audit)  [soft — logs sizing preview]
    Stage 5  shadow_trader       --transact --mark     [soft — paper trade + MTM]

Abort logic:
    Stage 1 failure → immediate abort, writes pipeline_state.json with status=ABORTED
    Stages 2-5 failure → log ERROR, continue to next stage with degraded state

Outputs:
    data/logs/pipeline_YYYY-MM-DD.log  — structured daily audit log (UTC)
    data/pipeline_state.json           — atomic frontend snapshot (renamed from .tmp)

Usage:
    python3 scripts/master_controller.py
    python3 scripts/master_controller.py --ticker GC=F
    python3 scripts/master_controller.py --dry-run   # log all stages, skip subprocess/LLM
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import pandas as pd
import yfinance as yf

# ── Paths ──────────────────────────────────────────────────────────────────────

LOG_DIR         = ROOT / "data" / "logs"
PIPELINE_STATE  = ROOT / "data" / "pipeline_state.json"
DECISION_LOG    = ROOT / "data" / "decision_log.json"
CURRENT_REGIME  = ROOT / "data" / "current_regime.json"
ALT_DATA_CSV    = ROOT / "data" / "alt_data.csv"
SHADOW_DB       = ROOT / "data" / "shadow_book.db"

DEFAULT_TICKER   = "GC=F"
STARTING_CAPITAL = 100_000.0

_LOG_FMT  = "%(asctime)s UTC [%(name)-12s] %(levelname)-7s %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


# ── Logging setup ──────────────────────────────────────────────────────────────

def _setup_logging(run_date: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"pipeline_{run_date}.log"

    # Remove stale handlers from any previous call in the same process
    root = logging.getLogger()
    root.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format=_LOG_FMT,
        datefmt=_DATE_FMT,
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Suppress noisy third-party loggers
    for noisy in ("yfinance", "urllib3", "peewee", "hmmlearn", "numba",
                  "matplotlib", "PIL", "httpx", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    log = logging.getLogger("PIPELINE")
    log.info("=" * 64)
    log.info(f"Run date  : {run_date}")
    log.info(f"Log file  : {log_path}")
    return log


# ── Subprocess helper ──────────────────────────────────────────────────────────

def _run_subprocess(
    cmd: list[str],
    stage_log: logging.Logger,
    timeout: int = 300,
) -> tuple[int, str]:
    """
    Execute `cmd`, stream lines to stage_log, return (returncode, combined_output).
    Never raises — all errors are caught and returned as (1, error_string).
    """
    stage_log.info("CMD: " + " ".join(cmd))
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            cwd=str(ROOT), timeout=timeout,
        )
        elapsed = time.time() - t0
        combined = (result.stdout + "\n" + result.stderr).strip()
        for line in combined.splitlines():
            if line.strip():
                stage_log.info("  | " + line)
        stage_log.info(f"Exit {result.returncode}  ({elapsed:.1f}s)")
        return result.returncode, combined
    except subprocess.TimeoutExpired:
        stage_log.error(f"Subprocess timed out after {timeout}s")
        return 1, f"TIMEOUT after {timeout}s"
    except Exception as exc:
        stage_log.error(f"Subprocess error: {exc}")
        return 1, str(exc)


# ── Stage result dict ──────────────────────────────────────────────────────────

def _sr(status: str, duration: float, note: str = "") -> dict:
    return {"status": status, "duration_s": round(duration, 1), "note": note}


# ── Data helpers (shared by stages and _build_pipeline_state) ─────────────────

def _read_shadow_portfolio() -> dict:
    """Return portfolio_state row from shadow_book.db as a plain dict."""
    try:
        conn = sqlite3.connect(SHADOW_DB)
        conn.row_factory = sqlite3.Row
        row  = conn.execute(
            "SELECT * FROM portfolio_state WHERE id = 1"
        ).fetchone()
        conn.close()
        return dict(row) if row else {}
    except Exception:
        return {}


def _read_latest_decision(ticker: str) -> dict:
    """Return the most recent decision_log.json entry for ticker."""
    try:
        entries = [
            d for d in json.loads(DECISION_LOG.read_text())
            if d.get("ticker") == ticker
        ]
        return entries[-1] if entries else {}
    except Exception:
        return {}


def _read_regime(ticker: str) -> dict:
    """Return current_regime.json entry for ticker."""
    try:
        return json.loads(CURRENT_REGIME.read_text()).get(ticker, {})
    except Exception:
        return {}


def _fetch_price_and_mas(ticker: str) -> tuple[float, dict]:
    """Download ~1 year of daily closes, return (last_close, {sma20, sma50, sma200})."""
    raw = yf.download(ticker, period="1y", interval="1d",
                      progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    cl = raw["Close"].dropna()
    cur_p = float(cl.iloc[-1])
    return cur_p, {
        "sma20":  float(cl.rolling(20).mean().iloc[-1])  if len(cl) >= 20  else cur_p,
        "sma50":  float(cl.rolling(50).mean().iloc[-1])  if len(cl) >= 50  else cur_p,
        "sma200": float(cl.rolling(200).mean().iloc[-1]) if len(cl) >= 200 else cur_p,
    }


def _load_alt_row() -> dict:
    """Return last row of alt_data.csv as a plain dict, stripping NaN."""
    try:
        df  = pd.read_csv(ALT_DATA_CSV, index_col=0, parse_dates=True)
        row = df.dropna(how="all").iloc[-1].to_dict()
        return {k: (None if pd.isna(v) else v) for k, v in row.items()}
    except Exception:
        return {}


def _load_oracle_score(ticker: str) -> float | None:
    try:
        from scripts.oracle_engine import get_latest_scores
        return get_latest_scores([ticker]).get(ticker)
    except Exception:
        return None


def _load_portfolio_entry(ticker: str) -> dict:
    try:
        data = json.loads((ROOT / "data" / "portfolio.json").read_text())
        if isinstance(data, list):
            for entry in data:
                if entry.get("ticker") == ticker:
                    return entry
        return {}
    except Exception:
        return {}


def _refresh_oracle(ticker: str, slog: logging.Logger) -> None:
    """Re-query Perplexity via oracle_engine and write a fresh row to oracle_history.csv."""
    try:
        from scripts.oracle_engine import run_oracle
        scores = run_oracle([ticker])
        score  = scores.get(ticker)
        if score is not None:
            slog.info(f"Oracle refreshed — {ticker} score={score:.4f}")
        else:
            slog.warning("Oracle query returned None — stale CSV score will be used")
    except Exception as exc:
        slog.warning(f"Oracle refresh failed (stale score will be used): {exc}")


def _load_lstm_predictions(cur_p: float) -> tuple[float, float, float]:
    """
    Run GoldLSTM-v1 inference and return (pred_5d, pred_21d, pred_252d).

    The LSTM is a 1-step-ahead model; its adjusted_price is used as the
    t+5d directional proxy.  t+21d and t+252d remain at cur_p (model
    limitation — they signal no additional directional information).
    Falls back to (cur_p, cur_p, cur_p) on any error.
    """
    try:
        from models.lstm_predictor import predict_next
        result = predict_next()
        if "error" in result:
            return cur_p, cur_p, cur_p
        pred_5d = float(result.get("adjusted_price", cur_p))
        return pred_5d, cur_p, cur_p
    except Exception:
        return cur_p, cur_p, cur_p


# ── Stage 1: Harvester (HARD ABORT) ───────────────────────────────────────────

def stage_harvester(dry_run: bool) -> dict:
    t0   = time.time()
    slog = logging.getLogger("HARVESTER")
    slog.info("Starting alt_data_harvester --update")

    if dry_run:
        slog.info("DRY-RUN: skipping subprocess")
        return _sr("DRY_RUN", 0)

    rc, out = _run_subprocess(
        [sys.executable, str(ROOT / "scripts" / "alt_data_harvester.py"), "--update"],
        slog, timeout=180,
    )
    elapsed = time.time() - t0

    if rc != 0:
        slog.error(f"HARVESTER FAILED — pipeline will ABORT (exit {rc})")
        return _sr("ABORTED", elapsed, f"exit code {rc}: {out[:300]}")

    slog.info(f"OK ({elapsed:.1f}s) — alt_data.csv refreshed")
    return _sr("OK", elapsed)


# ── Stage 2: Regime Detector ───────────────────────────────────────────────────

def stage_regime(ticker: str, dry_run: bool) -> dict:
    t0   = time.time()
    slog = logging.getLogger("REGIME   ")
    slog.info(f"Starting regime_detector --fit {ticker}")

    if dry_run:
        slog.info("DRY-RUN: skipping subprocess")
        return _sr("DRY_RUN", 0)

    rc, _ = _run_subprocess(
        [sys.executable, str(ROOT / "scripts" / "regime_detector.py"),
         "--fit", ticker],
        slog, timeout=180,
    )
    elapsed = time.time() - t0

    if rc != 0:
        slog.warning(f"Non-zero exit {rc} — will use cached regime")
        return _sr("FAILED", elapsed, f"exit code {rc}")

    rj    = _read_regime(ticker)
    state = rj.get("state_label", "UNKNOWN")
    probs = rj.get("probabilities", {})
    veto  = rj.get("regime_veto", False)
    slog.info(
        f"OK — state={state}  "
        f"p_bull={probs.get('BULLISH', 0):.2f}  "
        f"p_bear={probs.get('BEARISH', 0):.2f}  "
        f"veto={veto}  ({elapsed:.1f}s)"
    )
    return _sr("OK", elapsed, f"state={state}")


# ── Stage 3: Investment Committee (metal_logic) ────────────────────────────────

def stage_metal_logic(ticker: str, dry_run: bool) -> dict:
    t0   = time.time()
    slog = logging.getLogger("COMMITTEE")
    slog.info(f"Starting Investment Committee for {ticker}")

    if dry_run:
        slog.info("DRY-RUN: skipping LLM calls")
        return _sr("DRY_RUN", 0)

    try:
        from scripts.metal_logic import evaluate_metal_swing

        cur_p, mas = _fetch_price_and_mas(ticker)
        slog.info(
            f"Spot ${cur_p:,.2f}  "
            f"SMA20={mas['sma20']:,.0f}  SMA50={mas['sma50']:,.0f}"
        )

        # DXY and VIX — fetch 10d for momentum calculations
        dxy_raw = yf.download(["DX-Y.NYB", "^VIX"], period="10d",
                               interval="1d", progress=False, auto_adjust=True)
        if isinstance(dxy_raw.columns, pd.MultiIndex):
            dxy_raw = dxy_raw["Close"]

        def _pct(s, n):
            s = s.dropna()
            if len(s) >= n + 1:
                return float((s.iloc[-1] - s.iloc[-(n+1)]) / s.iloc[-(n+1)] * 100)
            return 0.0

        def _pts(s, n):
            s = s.dropna()
            return float(s.iloc[-1] - s.iloc[-(n+1)]) if len(s) >= n + 1 else 0.0

        dxy_s = dxy_raw["DX-Y.NYB"] if "DX-Y.NYB" in dxy_raw.columns else pd.Series(dtype=float)
        vix_s = dxy_raw["^VIX"]     if "^VIX"     in dxy_raw.columns else pd.Series(dtype=float)
        dxy_cur = float(dxy_s.dropna().iloc[-1]) if len(dxy_s.dropna()) else 0.0
        vix_cur = float(vix_s.dropna().iloc[-1]) if len(vix_s.dropna()) else 0.0

        alt = _load_alt_row()
        macro_data = {
            "dxy_current":    dxy_cur,
            "dxy_1d_pct":     _pct(dxy_s, 1),
            "dxy_5d_pct":     _pct(dxy_s, 5),
            "vix_current":    vix_cur,
            "vix_1d":         _pts(vix_s, 1),
            "vix_5d":         _pts(vix_s, 5),
            "real_yield":     float(alt.get("real_yield_10y",           0.0) or 0.0),
            "cot_gold_raw":   float(alt.get("cot_gold_mm_net_raw",      0.0) or 0.0),
            "cot_gold_z":     float(alt.get("cot_gold_mm_net_zscore",   0.0) or 0.0),
            "cot_silver_raw": float(alt.get("cot_silver_mm_net_raw",    0.0) or 0.0),
            "cot_silver_z":   float(alt.get("cot_silver_mm_net_zscore", 0.0) or 0.0),
        }
        cg_z = alt.get("copper_gold_ratio_zscore")
        if cg_z is not None:
            cg_z = float(cg_z)

        # Refresh Perplexity oracle score before reading it
        _refresh_oracle(ticker, slog)
        o_scr = _load_oracle_score(ticker)
        port  = _load_portfolio_entry(ticker)

        # LSTM predictions (t+5d proxy from GoldLSTM-v1; t+21d/252d at spot)
        pred_5d, pred_21d, pred_252d = _load_lstm_predictions(cur_p)
        slog.info(
            f"LSTM t+5d={pred_5d:,.2f} ({(pred_5d / cur_p - 1) * 100:+.2f}%)"
            if pred_5d != cur_p else "LSTM unavailable — spot fallback used"
        )

        matrix = evaluate_metal_swing(
            ticker, cur_p, port,
            pred_5d, pred_21d, pred_252d,
            moving_averages   = mas,
            macro_data        = macro_data,
            copper_gold_z     = cg_z,
            live_oracle_score = o_scr,
        )

        action  = matrix.get("Action", "HOLD_METAL")
        veto    = matrix.get("veto", False)
        elapsed = time.time() - t0
        slog.info(
            f"OK — Action={action}  veto={veto}  "
            f"oracle={o_scr}  ({elapsed:.1f}s)"
        )
        return _sr("OK", elapsed, f"action={action}")

    except Exception as exc:
        elapsed = time.time() - t0
        slog.error(f"FAILED: {exc}")
        return _sr("FAILED", elapsed, str(exc)[:200])


# ── Stage 4: Risk Manager (audit preview — shadow_trader calls it internally) ──

def stage_risk(ticker: str, dry_run: bool) -> dict:
    t0   = time.time()
    slog = logging.getLogger("RISK     ")
    slog.info("Starting risk_manager (audit preview)")

    if dry_run:
        slog.info("DRY-RUN: skipping")
        return _sr("DRY_RUN", 0)

    try:
        from scripts.risk_manager import evaluate as rm_evaluate

        decision = _read_latest_decision(ticker)
        action   = decision.get("action_taken", "HOLD_METAL")

        shadow = _read_shadow_portfolio()
        cur_p  = float(shadow.get("last_spot") or 0.0)
        if cur_p <= 0.0:
            raw = yf.download(ticker, period="2d", interval="1d",
                              progress=False, auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            cur_p = float(raw["Close"].dropna().iloc[-1])

        rj       = _read_regime(ticker)
        portfolio = {
            "cash_usd":        float(shadow.get("cash_usd",         STARTING_CAPITAL)),
            "gold_oz":         float(shadow.get("gold_oz",          0.0)),
            "portfolio_value": float(shadow.get("portfolio_value",  STARTING_CAPITAL)),
        }

        rd = rm_evaluate(
            action           = action,
            quant_conviction = int(decision.get("quant_conviction") or 5),
            macro_conviction = int(decision.get("macro_conviction") or 5),
            hmm_state        = rj.get("state_label", "VOLATILE"),
            hmm_veto         = bool(rj.get("regime_veto", False)),
            portfolio        = portfolio,
            ticker           = ticker,
            spot_price       = cur_p,
        )

        elapsed = time.time() - t0
        slog.info(
            f"OK — action={rd.approved_action}  weight={rd.target_weight:.2%}  "
            f"deploy=${rd.deploy_usd:,.0f}  VaR_95={rd.var_95_daily:.2%}  "
            f"vol_21d={rd.realized_vol_21d_annual:.2%}  "
            f"override={rd.var_override}  ({elapsed:.1f}s)"
        )
        return _sr(
            "OK", elapsed,
            f"action={rd.approved_action} weight={rd.target_weight:.3f}",
        )

    except Exception as exc:
        elapsed = time.time() - t0
        slog.error(f"FAILED: {exc}")
        return _sr("FAILED", elapsed, str(exc)[:200])


# ── Stage 5: Shadow Trader ─────────────────────────────────────────────────────

def stage_shadow(ticker: str, dry_run: bool) -> dict:
    t0   = time.time()
    slog = logging.getLogger("SHADOW   ")
    slog.info("Starting shadow_trader --transact --mark")

    if dry_run:
        slog.info("DRY-RUN: skipping subprocess")
        return _sr("DRY_RUN", 0)

    rc, _ = _run_subprocess(
        [sys.executable, str(ROOT / "scripts" / "shadow_trader.py"),
         "--transact", "--mark", "--ticker", ticker],
        slog, timeout=60,
    )
    elapsed = time.time() - t0

    if rc != 0:
        slog.warning(f"Non-zero exit {rc} — trade may not have executed")
        return _sr("FAILED", elapsed, f"exit code {rc}")

    slog.info(f"OK ({elapsed:.1f}s)")
    return _sr("OK", elapsed)


# ── Stage 6: Position Manager ─────────────────────────────────────────────

def stage_position_manager(ticker: str, dry_run: bool) -> dict:
    t0   = time.time()
    slog = logging.getLogger("POS_MGR  ")
    slog.info(f"Starting position manager for {ticker}")

    if dry_run:
        slog.info("DRY-RUN: skipping")
        return _sr("DRY_RUN", 0)

    try:
        from scripts.position_manager import run_position_manager
        shadow = _read_shadow_portfolio()
        gold_oz = float(shadow.get("gold_oz", 0.0))

        entry_price = float(shadow.get("avg_entry") or 0.0)
        holding_days = 0
        if gold_oz > 0 and entry_price > 0:
            holding_days = 5  # default audit window for open positions

        deploy_usd = float(shadow.get("cash_usd", 25000.0))
        result = run_position_manager(
            ticker=ticker,
            deploy_usd=deploy_usd,
            direction="long",
            entry_price=entry_price if entry_price > 0 else None,
            holding_days=holding_days,
        )

        elapsed = time.time() - t0
        entry_q = result.get("entry_signal", {}).get("score", 0)
        atr_val = result.get("atr", 0)
        slog.info(
            f"OK — ATR=${atr_val:,.2f}  entry_quality={entry_q}/100  ({elapsed:.1f}s)"
        )
        return _sr("OK", elapsed, f"entry_quality={entry_q} ATR=${atr_val:.2f}")

    except Exception as exc:
        elapsed = time.time() - t0
        slog.error(f"FAILED: {exc}")
        return _sr("FAILED", elapsed, str(exc)[:200])


# ── Stage 7: Model Performance Tracker ───────────────────────────────────

def stage_performance_tracker(dry_run: bool) -> dict:
    t0   = time.time()
    slog = logging.getLogger("TRACKER  ")
    slog.info("Starting model performance tracker")

    if dry_run:
        slog.info("DRY-RUN: skipping")
        return _sr("DRY_RUN", 0)

    try:
        from scripts.model_performance_tracker import run as tracker_run
        payload = tracker_run(lookback_days=30, dry_run=False)
        elapsed = time.time() - t0

        weights = payload.get("weights", {})
        w_strs = [f"{m}={w:.2f}" for m, w in weights.items()]
        slog.info(f"OK — weights: {', '.join(w_strs)}  ({elapsed:.1f}s)")
        return _sr("OK", elapsed, f"weights updated: {len(weights)} models")

    except Exception as exc:
        elapsed = time.time() - t0
        slog.error(f"FAILED: {exc}")
        return _sr("FAILED", elapsed, str(exc)[:200])


def stage_treasury_hedge(dry_run: bool) -> dict:
    """Phase XXV: Treasury hedge overlay recommendation.

    Reads macro_regime + crisis_detector, outputs hedge sizing for TLT/IEF.
    Default mode SIGNAL_ONLY (recommend only; no order routed).
    """
    t0   = time.time()
    slog = logging.getLogger("HEDGE    ")
    mode = os.environ.get("TREASURY_HEDGE_MODE", "SIGNAL_ONLY").upper()
    slog.info(f"Starting treasury hedge overlay  (mode={mode})")

    if dry_run:
        slog.info("DRY-RUN: skipping")
        return _sr("DRY_RUN", 0)

    try:
        from scripts.treasury_hedge_overlay import run as hedge_run
        result = hedge_run(write=True)
        elapsed = time.time() - t0
        instr = result.get("instrument") or "—"
        pct = result.get("allocation_pct", 0.0)
        slog.info(f"OK — recommend {instr} {pct:.1f}%  ({elapsed:.1f}s)")
        return _sr("OK", elapsed,
                   f"{instr} {pct:.1f}% · {result.get('regime_quadrant')}/{result.get('crisis_tier')}")
    except Exception as exc:
        elapsed = time.time() - t0
        slog.error(f"FAILED: {exc}")
        return _sr("FAILED", elapsed, str(exc)[:200])


def stage_reconciler(dry_run: bool) -> dict:
    """Phase XXIV: diff internal book vs IBKR broker positions.

    Soft-fail. Writes data/position_reconciliation.json. In dry_run, the
    reconciler runs with use_ibkr=False (shadow-only sanity check).

    NOTE [IBKR pivot]: the IBKR diff path is DEPRECATED. With EXECUTION_MODE
    permanently pinned to paper_internal, use_ibkr resolves to False and this
    stage degrades to a shadow-only sanity check against the internal book.
    No broker connection is attempted.
    """
    t0   = time.time()
    slog = logging.getLogger("RECONCILE")
    # Pinned to paper_internal post-IBKR-pivot, so use_ibkr is always False.
    use_ibkr = (os.environ.get("EXECUTION_MODE", "paper_internal") != "paper_internal")
    slog.info(f"Starting reconciler  (use_ibkr={use_ibkr})")

    if dry_run:
        slog.info("DRY-RUN: skipping")
        return _sr("DRY_RUN", 0)

    try:
        from scripts.position_reconciler import run_reconciler
        result = run_reconciler(use_ibkr=use_ibkr)
        elapsed = time.time() - t0

        status = result.get("status", "OK")
        n_drift = result.get("diff", {}).get("n_drift_total", 0)
        if status == "OK":
            slog.info(f"OK — no drift  ({elapsed:.1f}s)")
            return _sr("OK", elapsed, f"no drift; broker_mode={result.get('ibkr_mode')}")
        slog.warning(f"{status} — {n_drift} drift(s)  ({elapsed:.1f}s)")
        return _sr("OK", elapsed, f"{status}: {n_drift} drift(s)")

    except Exception as exc:
        elapsed = time.time() - t0
        slog.error(f"FAILED: {exc}")
        return _sr("FAILED", elapsed, str(exc)[:200])


# ── pipeline_state.json builder ────────────────────────────────────────────────

def _build_pipeline_state(
    run_date:        str,
    ticker:          str,
    stages:          dict,
    pipeline_status: str,
    abort_reason:    str | None,
) -> dict:
    """
    Compile all live state into the frontend snapshot JSON.
    Reads from shadow_book.db, current_regime.json, and decision_log.json.
    Falls back gracefully on any read error.
    """

    # ── Portfolio ──────────────────────────────────────────────────────────────
    shadow    = _read_shadow_portfolio()
    gold_oz   = float(shadow.get("gold_oz",         0.0))
    cash_usd  = float(shadow.get("cash_usd",         STARTING_CAPITAL))
    pf_value  = float(shadow.get("portfolio_value",  STARTING_CAPITAL))
    last_spot = float(shadow.get("last_spot",        0.0) or 0.0)
    pf_state  = str(shadow.get("portfolio_state",   "FIAT"))
    unreal    = pf_value - STARTING_CAPITAL

    portfolio = {
        "gold_oz":         round(gold_oz,  6),
        "cash_usd":        round(cash_usd, 2),
        "portfolio_value": round(pf_value, 2),
        "last_spot":       round(last_spot, 2),
        "state":           pf_state,
        "unrealised_pnl":  round(unreal, 2),
        "unrealised_pct":  round(unreal / STARTING_CAPITAL * 100, 4),
        "starting_capital": STARTING_CAPITAL,
    }

    # ── Regime ─────────────────────────────────────────────────────────────────
    regime = {
        "hmm_state": "UNKNOWN", "hmm_veto_active": False,
        "p_bullish": 0.0, "p_volatile": 0.0, "p_bearish": 0.0,
        "fitted_at": "",
    }
    rj = _read_regime(ticker)
    if rj:
        probs = rj.get("probabilities", {})
        regime = {
            "hmm_state":       rj.get("state_label", "UNKNOWN"),
            "hmm_veto_active": bool(rj.get("regime_veto", False)),
            "p_bullish":       round(probs.get("BULLISH", 0.0), 4),
            "p_volatile":      round(probs.get("VOLATILE", probs.get("RANGING", 0.0)), 4),
            "p_bearish":       round(probs.get("BEARISH", 0.0), 4),
            "fitted_at":       rj.get("fitted_at", ""),
        }

    # ── Committee ──────────────────────────────────────────────────────────────
    committee = {
        "action_taken": "HOLD_METAL", "quant_conviction": None,
        "macro_conviction": None, "quant_thesis": "",
        "macro_thesis": "", "cio_reasoning": "",
        "oracle_score": None, "veto_active": False, "decision_date": "",
    }
    dec = _read_latest_decision(ticker)
    if dec:
        committee = {
            "action_taken":    dec.get("action_taken",       "HOLD_METAL"),
            "quant_conviction":dec.get("quant_conviction"),
            "macro_conviction":dec.get("macro_conviction"),
            "quant_thesis":    dec.get("quant_thesis",       ""),
            "macro_thesis":    dec.get("macro_thesis",       ""),
            "cio_reasoning":   dec.get("original_reasoning", ""),
            "oracle_score":    dec.get("oracle_score"),
            "veto_active":     bool(
                dec.get("veto", False) or dec.get("hmm_veto_active", False)
            ),
            "decision_date":   dec.get("date", ""),
        }

    # ── Risk (read from shadow_book.db actions — written by shadow_trader) ─────
    risk = {
        "approved_action":  committee["action_taken"],
        "target_weight":    None,
        "deploy_usd":       None,
        "var_95_daily":     None,
        "realized_vol_21d": None,
        "var_override":     False,
    }
    try:
        conn = sqlite3.connect(SHADOW_DB)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT risk_target_weight, risk_var_95, risk_var_override, risk_deploy_usd "
            "FROM actions WHERE ticker = ? ORDER BY created_at DESC LIMIT 1",
            (ticker,),
        ).fetchone()
        conn.close()
        if row and row["risk_target_weight"] is not None:
            risk = {
                "approved_action":  committee["action_taken"],
                "target_weight":    row["risk_target_weight"],
                "deploy_usd":       row["risk_deploy_usd"],
                "var_95_daily":     row["risk_var_95"],
                "realized_vol_21d": None,
                "var_override":     bool(row["risk_var_override"]),
            }
    except Exception:
        pass

    # ── Position management snapshot ─────────────────────────────────────────
    position_mgmt = {}
    try:
        pm_path = ROOT / "data" / "position_management.json"
        if pm_path.exists():
            position_mgmt = json.loads(pm_path.read_text())
    except Exception:
        pass

    # ── Ensemble weights snapshot ─────────────────────────────────────────
    ensemble_info = {}
    try:
        ew_path = ROOT / "data" / "ensemble_weights.json"
        if ew_path.exists():
            ensemble_info = json.loads(ew_path.read_text())
    except Exception:
        pass

    # ── Institutional risk snapshot (tail risk + factor attribution) ───────
    institutional_risk = {}
    try:
        tre_path = ROOT / "data" / "tail_risk_engine.json"
        if tre_path.exists():
            tre = json.loads(tre_path.read_text())
            evt = tre.get("tail_risk", {}).get("methods", {}).get("evt_pot", {})
            fac = tre.get("factor_attribution", {})
            institutional_risk = {
                "evt_cvar_99_pct":         evt.get("cvar_990"),
                "evt_cvar_995_pct":        evt.get("cvar_995"),
                "tail_fatness_premium_pct":tre.get("tail_risk", {}).get("tail_fatness_premium_pct"),
                "factor_r_squared":        fac.get("r_squared"),
                "alpha_annualised_pct":    fac.get("alpha_annualised_pct"),
                "alpha_t_stat":            fac.get("alpha_t_stat"),
                "information_ratio":       fac.get("information_ratio"),
                "sharpe_ratio":            tre.get("performance", {}).get("sharpe_ratio"),
                "sortino_ratio":           tre.get("performance", {}).get("sortino_ratio"),
                "excess_kurtosis":         tre.get("performance", {}).get("excess_kurtosis"),
                "generated_at":            tre.get("generated_at"),
            }
    except Exception:
        pass

    # ── Drawdown tier snapshot ─────────────────────────────────────────────
    drawdown_tier = {}
    try:
        dd_path = ROOT / "data" / "drawdown_controller.json"
        if dd_path.exists():
            dd = json.loads(dd_path.read_text())
            drawdown_tier = {
                "tier_name":          dd.get("tier_name"),
                "current_dd_pct":     dd.get("current_dd_pct"),
                "sizing_multiplier":  dd.get("sizing_multiplier"),
                "action":             dd.get("action"),
                "worst_dd_pct":       dd.get("worst_dd_pct"),
                "recovery_pct":       dd.get("recovery_pct"),
            }
    except Exception:
        pass

    # ── Cointegration & mean-reversion snapshot ────────────────────────────
    cointegration = {}
    try:
        ce_path = ROOT / "data" / "cointegration_engine.json"
        if ce_path.exists():
            ce = json.loads(ce_path.read_text())
            cointegration = {
                "n_pairs":             ce.get("n_pairs"),
                "n_cointegrated_5pct": ce.get("n_cointegrated_5pct"),
                "n_actionable":        ce.get("n_actionable"),
                "actionable_signals":  ce.get("actionable_signals", []),
                "generated_at":        ce.get("generated_at"),
            }
    except Exception:
        pass

    # ── Phase XIII snapshots ───────────────────────────────────────────────
    economic_calendar = {}
    try:
        ec_path = ROOT / "data" / "economic_calendar.json"
        if ec_path.exists():
            ec = json.loads(ec_path.read_text())
            economic_calendar = {
                "position_guard":  ec.get("position_guard"),
                "guard_severity":  ec.get("guard_severity"),
                "blocked_today":   ec.get("blocked_today"),
                "next_event":      ec.get("next_event"),
                "n_blackout":      ec.get("n_blackout_active"),
                "events_window":   ec.get("events", [])[:8],
                "generated_at":    ec.get("generated_at"),
            }
    except Exception:
        pass

    earnings_calendar = {}
    try:
        ear_path = ROOT / "data" / "earnings_calendar.json"
        if ear_path.exists():
            ear = json.loads(ear_path.read_text())
            earnings_calendar = {
                "n_blocked":         ear.get("n_blocked"),
                "blackout_tickers":  ear.get("blackout_tickers", []),
                "filtered_universe": ear.get("filtered_universe", []),
                "next_5":            ear.get("next_5_earnings", [])[:5],
                "generated_at":      ear.get("generated_at"),
            }
    except Exception:
        pass

    data_quality = {}
    try:
        dq_path = ROOT / "data" / "data_quality.json"
        if dq_path.exists():
            dq = json.loads(dq_path.read_text())
            data_quality = {
                "overall_status":  dq.get("overall_status"),
                "n_checks":        dq.get("n_checks"),
                "n_failures":      dq.get("n_failures"),
                "by_severity":     dq.get("by_severity"),
                "failed_checks":   [c.get("check") for c in dq.get("failures", [])],
                "generated_at":    dq.get("generated_at"),
            }
    except Exception:
        pass

    pairs_trader = {}
    try:
        pt_path = ROOT / "data" / "pairs_trader.json"
        if pt_path.exists():
            pt = json.loads(pt_path.read_text())
            pairs_trader = {
                "n_signals":      pt.get("n_signals"),
                "n_trades_built": pt.get("n_trades_built"),
                "n_open":         pt.get("n_open"),
                "max_pair_notional_usd":pt.get("max_pair_notional_usd"),
                "trades":         pt.get("trades", []),
                "generated_at":   pt.get("generated_at"),
            }
    except Exception:
        pass

    pnl_tracker = {}
    try:
        pn_path = ROOT / "data" / "pnl_tracker.json"
        if pn_path.exists():
            pn = json.loads(pn_path.read_text())
            pnl_tracker = {
                "latest_nav_usd":   pn.get("latest_nav_usd"),
                "day_pnl_usd":      pn.get("day_pnl_usd"),
                "day_pnl_pct":      pn.get("day_pnl_pct"),
                "cumulative_return_pct":pn.get("cumulative_return_pct"),
                "n_history":        pn.get("n_history"),
                "windows":          pn.get("windows", {}),
                "generated_at":     pn.get("generated_at"),
            }
    except Exception:
        pass

    # ── Phase XIV snapshots ────────────────────────────────────────────────
    alpha_stacker = {}
    try:
        ap_path = ROOT / "data" / "alpha_stacker.json"
        if ap_path.exists():
            ap = json.loads(ap_path.read_text())
            alpha_stacker = {
                "decision":          ap.get("decision", {}),
                "stack_summary":     ap.get("stack", {}),
                "by_family":         ap.get("by_family", {}),
                "top_drivers":       ap.get("top_drivers", [])[:5],
                "top_detractors":    ap.get("top_detractors", [])[:5],
                "n_signals":         (ap.get("stack") or {}).get("n_signals", 0),
                "risk_flags":        ap.get("risk_flags", []),
                "n_risk_flags":      ap.get("n_risk_flags", 0),
                "generated_at":      ap.get("generated_at"),
            }
    except Exception:
        pass

    strategy_selector = {}
    try:
        ss_path = ROOT / "data" / "strategy_selector.json"
        if ss_path.exists():
            ss = json.loads(ss_path.read_text())
            strategy_selector = {
                "strategy":            ss.get("strategy"),
                "strategy_description":ss.get("strategy_description"),
                "direction":           ss.get("direction"),
                "final_size_pct":      ss.get("final_size_pct"),
                "size_stack":          ss.get("size_stack", {}),
                "regime_context":      ss.get("regime_context", {}),
                "alpha_stacker_brief": ss.get("alpha_stacker", {}),
                "reasoning":           ss.get("reasoning", []),
                "actionable_pairs_count": ss.get("actionable_pairs_count"),
                "strong_pairs_count":  ss.get("strong_pairs_count"),
                "generated_at":        ss.get("generated_at"),
            }
    except Exception:
        pass

    performance_targeter = {}
    try:
        ptg_path = ROOT / "data" / "performance_targeter.json"
        if ptg_path.exists():
            ptg = json.loads(ptg_path.read_text())
            performance_targeter = {
                "target":          ptg.get("target", {}),
                "progress":        ptg.get("progress", {}),
                "risk_multiplier": ptg.get("risk_multiplier", {}),
                "context":         ptg.get("context", {}),
                "generated_at":    ptg.get("generated_at"),
            }
    except Exception:
        pass

    multi_strategy_book = {}
    try:
        mst_path = ROOT / "data" / "multi_strategy_trader.json"
        if mst_path.exists():
            mst = json.loads(mst_path.read_text())
            multi_strategy_book = {
                "strategy":         mst.get("strategy"),
                "prior_strategy":   mst.get("prior_strategy"),
                "strategy_changed": mst.get("strategy_changed"),
                "book_equity_usd":  mst.get("book_equity_usd"),
                "cash_usd":         mst.get("cash_usd"),
                "open_pl_usd":      mst.get("open_pl_usd"),
                "lifetime_pl_pct":  mst.get("lifetime_pl_pct"),
                "n_open":           mst.get("n_open"),
                "n_closed_total":   mst.get("n_closed_total"),
                "n_new_trades":     mst.get("n_new_trades"),
                "n_exits_this_run": mst.get("n_exits_this_run"),
                "nav_stats":        mst.get("nav_stats", {}),
                "generated_at":     mst.get("generated_at"),
            }
    except Exception:
        pass

    crisis_detector = {}
    try:
        cd_path = ROOT / "data" / "crisis_detector.json"
        if cd_path.exists():
            cd = json.loads(cd_path.read_text())
            crisis_detector = {
                "score":          cd.get("score"),
                "tier":           cd.get("tier"),
                "price_score":    cd.get("price_score"),
                "engine_bump":    cd.get("engine_bump"),
                "components":     cd.get("components", {}),
                "engine_bumps_applied": cd.get("engine_bumps_applied", []),
                "guidance":       cd.get("guidance"),
                "generated_at":   cd.get("generated_at"),
            }
    except Exception:
        pass

    stress_backtest = {}
    try:
        st_path = ROOT / "data" / "stress_backtest.json"
        if st_path.exists():
            st = json.loads(st_path.read_text())
            stress_backtest = {
                "ticker":          st.get("ticker"),
                "history_range":   st.get("history_range"),
                "n_windows":       st.get("n_windows"),
                "n_valid":         st.get("n_valid"),
                "aggregate":       st.get("aggregate", {}),
                "window_summary":  [
                    {
                        "label":   w.get("label"),
                        "verdict": w.get("verdict"),
                        "sharpe":  w.get("sharpe"),
                        "annualised_pct": w.get("annualised_pct"),
                        "max_drawdown_pct": w.get("max_drawdown_pct"),
                    }
                    for w in st.get("windows", [])
                ],
                "generated_at":    st.get("generated_at"),
            }
    except Exception:
        pass

    multi_asset_backtest = {}
    try:
        ma_path = ROOT / "data" / "multi_asset_backtest.json"
        if ma_path.exists():
            ma = json.loads(ma_path.read_text())
            multi_asset_backtest = {
                "lookback_days":         ma.get("lookback_days"),
                "halal_tickers_used":    ma.get("halal_tickers_used", []),
                "weights":               ma.get("weights", {}),
                "books":                 ma.get("books", {}),
                "benchmarks":            ma.get("benchmarks", {}),
                "delta_vs_spy":          ma.get("delta_vs_spy", {}),
                "verdict":               ma.get("verdict"),
                "note":                  ma.get("note"),
                "generated_at":          ma.get("generated_at"),
            }
    except Exception:
        pass

    # ── Phase XXIII / XXVI: ML conviction stack ─────────────────────────────
    conviction_weights = {}
    try:
        cw_path = ROOT / "data" / "conviction_weights.json"
        if cw_path.exists():
            cw = json.loads(cw_path.read_text())
            conviction_weights = {
                "weights":          cw.get("weights", {}),
                "top_component":    max(cw.get("weights", {}).items(),
                                        key=lambda kv: kv[1], default=(None, 0))[0],
                "generated_at":     cw.get("generated_at"),
            }
    except Exception:
        pass

    ml_conviction = {}
    try:
        mc_path = ROOT / "data" / "ml_conviction_poc.json"
        if mc_path.exists():
            mc = json.loads(mc_path.read_text())
            cp = mc.get("comparison", {})
            ml_conviction = {
                "oos_ic_ml":       cp.get("oos_ic_ml"),
                "sharpe_ml":       cp.get("sharpe_ml"),
                "sharpe_rule":     cp.get("sharpe_rule"),
                "ic_pass":         cp.get("ic_pass"),
                "sharpe_pass":     cp.get("sharpe_pass"),
                "gate_passed":     cp.get("gate_passed"),
                "generated_at":    mc.get("generated_at"),
            }
    except Exception:
        pass

    ml_walk_forward = {}
    try:
        wf_path = ROOT / "data" / "ml_walk_forward.json"
        if wf_path.exists():
            wf = json.loads(wf_path.read_text())
            ml_walk_forward = {
                "median_ann_pct":     wf.get("ml", {}).get("median_ann_pct"),
                "positive_share_pct": wf.get("ml", {}).get("positive_share_pct"),
                "avg_sharpe":         wf.get("ml", {}).get("avg_sharpe"),
                "gate_passed":        wf.get("gate_passed"),
                "verdict":            wf.get("verdict"),
                "generated_at":       wf.get("generated_at"),
            }
    except Exception:
        pass

    # ── Phase XXV-b: Treasury overlay stress eval ───────────────────────────
    multi_asset_stress_backtest = {}
    try:
        mast_path = ROOT / "data" / "multi_asset_stress_backtest.json"
        if mast_path.exists():
            mast = json.loads(mast_path.read_text())
            multi_asset_stress_backtest = {
                "n_windows":    mast.get("n_windows"),
                "verdict":      mast.get("verdict"),
                "note":         mast.get("note"),
                "generated_at": mast.get("generated_at"),
            }
    except Exception:
        pass

    treasury_overlay_stress = {}
    try:
        tos_path = ROOT / "data" / "treasury_overlay_stress_eval.json"
        if tos_path.exists():
            tos = json.loads(tos_path.read_text())
            treasury_overlay_stress = {
                "verdict":           tos.get("verdict"),
                "delta_avg_sharpe":  tos.get("delta_avg_sharpe"),
                "n_rescued":         tos.get("n_rescued"),
                "n_regressed":       tos.get("n_regressed"),
                "note":              tos.get("note"),
                "generated_at":      tos.get("generated_at"),
            }
    except Exception:
        pass

    strategy_backtest = {}
    try:
        bt_path = ROOT / "data" / "strategy_backtest.json"
        if bt_path.exists():
            bt = json.loads(bt_path.read_text())
            strategy_backtest = {
                "ticker":           bt.get("ticker"),
                "lookback_days":    bt.get("lookback_days"),
                "n_simulated":      bt.get("n_simulated"),
                "first_date":       bt.get("first_date"),
                "last_date":        bt.get("last_date"),
                "target":           bt.get("target", {}),
                "performance":      bt.get("performance", {}),
                "monthly":          {
                    k: v for k, v in (bt.get("monthly") or {}).items()
                    if k != "returns"   # full monthly series stays in the raw file
                },
                "monthly_returns_brief": (bt.get("monthly") or {}).get("returns", [])[-12:],
                "by_strategy":      bt.get("by_strategy", []),
                "achievability_verdict": bt.get("achievability_verdict"),
                "achievability_note":    bt.get("achievability_note"),
                "generated_at":     bt.get("generated_at"),
            }
    except Exception:
        pass

    # ── Phase XII snapshots ────────────────────────────────────────────────
    system_health = {}
    try:
        sh_path = ROOT / "data" / "system_health.json"
        if sh_path.exists():
            sh = json.loads(sh_path.read_text())
            system_health = {
                "overall_status":  sh.get("overall_status"),
                "n_flags_total":   sh.get("n_flags_total"),
                "by_severity":     sh.get("by_severity"),
                "n_critical":      sh.get("by_severity", {}).get("CRITICAL", 0),
                "flags_brief":     [
                    {"severity": f.get("severity"), "kind": f.get("kind")}
                    for f in sh.get("flags", [])[:10]
                ],
                "generated_at":    sh.get("generated_at"),
            }
    except Exception:
        pass

    trade_basket = {}
    try:
        tb_path = ROOT / "data" / "trade_basket.json"
        if tb_path.exists():
            tb = json.loads(tb_path.read_text())
            trade_basket = {
                "n_long":         tb.get("n_long"),
                "n_short":        tb.get("n_short"),
                "long_basket":    tb.get("long_basket", []),
                "short_basket":   tb.get("short_basket", []),
                "cash_pct":       tb.get("cash_pct"),
                "gross_cap":      tb.get("gross_exposure_cap_pct"),
                "generated_at":   tb.get("generated_at"),
            }
    except Exception:
        pass

    position_reconciliation = {}
    try:
        pr_path = ROOT / "data" / "position_reconciliation.json"
        if pr_path.exists():
            pr = json.loads(pr_path.read_text())
            position_reconciliation = {
                "status":          pr.get("status"),
                "ibkr_mode":       pr.get("ibkr_mode"),
                "n_drift_total":   pr.get("diff", {}).get("n_drift_total"),
                "n_broker":        pr.get("diff", {}).get("n_broker"),
                "n_shadow":        pr.get("diff", {}).get("n_shadow"),
                "generated_at":    pr.get("generated_at"),
            }
    except Exception:
        pass

    slippage_report = {}
    try:
        slip_path = ROOT / "data" / "slippage_report.json"
        if slip_path.exists():
            slp = json.loads(slip_path.read_text())
            slippage_report = {
                "n_paired":            slp.get("n_paired"),
                "fill_match_rate_pct": slp.get("fill_match_rate_pct"),
                "avg_slippage_bps":    slp.get("all_time", {}).get("avg_slippage_bps"),
                "rolling_30d_avg":     slp.get("rolling_30d", {}).get("avg_slippage_bps"),
                "warning":             slp.get("warning"),
                "generated_at":        slp.get("generated_at"),
            }
    except Exception:
        pass

    operator_runbook = {}
    try:
        run_path = ROOT / "data" / "operator_runbook.json"
        if run_path.exists():
            rb = json.loads(run_path.read_text())
            operator_runbook = {
                "run_date":         rb.get("run_date"),
                "checklist_ok":     rb.get("n_checklist_ok"),
                "checklist_total":  rb.get("n_checklist"),
                "n_risk_flags":     len(rb.get("risk_flags", [])),
                "metals_action":    rb.get("metals_action"),
                "markdown_path":    rb.get("markdown_path"),
                "generated_at":     rb.get("generated_at"),
            }
    except Exception:
        pass

    # ── Trade idea snapshot ────────────────────────────────────────────────
    trade_idea = {}
    try:
        ti_path = ROOT / "data" / "trade_idea.json"
        if ti_path.exists():
            ti = json.loads(ti_path.read_text())
            trade_idea = {
                "trade_card":      ti.get("trade_card", {}),
                "n_risk_flags":    ti.get("n_risk_flags"),
                "risk_flags":      ti.get("risk_flags", []),
                "ibkr_ready":      ti.get("ibkr_ready"),
                "reasoning_summary": {
                    "side_vote_count": len(ti.get("reasoning", {}).get("side_vote", [])),
                    "size_stack_count":len(ti.get("reasoning", {}).get("size_stack", [])),
                },
                "generated_at":    ti.get("generated_at"),
            }
    except Exception:
        pass

    # ── Halal universe snapshot ────────────────────────────────────────────
    halal_universe = {}
    try:
        hu_path = ROOT / "data" / "halal_universe.json"
        if hu_path.exists():
            hu = json.loads(hu_path.read_text())
            halal_universe = {
                "n_candidates":      hu.get("n_candidates"),
                "n_passing":         hu.get("n_passing"),
                "n_rejected":        hu.get("n_rejected"),
                "tickers":           hu.get("tickers", []),
                "rejection_summary": hu.get("rejection_summary", {}),
                "generated_at":      hu.get("generated_at"),
            }
    except Exception:
        pass

    # ── Metals rebalancer snapshot ─────────────────────────────────────────
    metals_rebalancer = {}
    try:
        mr_path = ROOT / "data" / "metals_rebalancer.json"
        if mr_path.exists():
            mr = json.loads(mr_path.read_text())
            metals_rebalancer = {
                "candidate_action":  mr.get("candidate_action"),
                "candidate_target":  mr.get("candidate_target"),
                "candidate_reasons": mr.get("candidate_reasons"),
                "sell_trigger_fired":mr.get("sell_trigger", {}).get("fired"),
                "buy_trigger_fired": mr.get("buy_trigger", {}).get("fired"),
                "n_open_trades":     mr.get("n_open_trades"),
                "n_closed_this_run": mr.get("n_closed_this_run"),
                "long_run_stats":    mr.get("long_run_stats"),
                "generated_at":      mr.get("generated_at"),
            }
    except Exception:
        pass

    # ── Continuous trainer snapshot ────────────────────────────────────────
    continuous_trainer = {}
    try:
        ct_path = ROOT / "data" / "continuous_trainer.json"
        if ct_path.exists():
            ct = json.loads(ct_path.read_text())
            continuous_trainer = {
                "n_tasks":      ct.get("n_tasks"),
                "n_run":        ct.get("n_run"),
                "n_successful": ct.get("n_successful"),
                "n_failed":     ct.get("n_failed"),
                "promotion":    ct.get("promotion"),
                "generated_at": ct.get("generated_at"),
            }
    except Exception:
        pass

    # ── MRM Champion / Challenger snapshot ──────────────────────────────────
    mrm = {}
    try:
        mrm_path = ROOT / "data" / "mrm_champion.json"
        if mrm_path.exists():
            m = json.loads(mrm_path.read_text())
            mrm = {
                "current_champion": m.get("current_champion"),
                "new_champion":     m.get("new_champion"),
                "decision":         m.get("decision"),
                "score_delta":      m.get("score_delta"),
                "rankings":         m.get("rankings", []),
                "generated_at":     m.get("generated_at"),
            }
    except Exception:
        pass

    # ── Strategy sandbox snapshot ───────────────────────────────────────────
    strategy_sandbox = {}
    try:
        ss_path = ROOT / "data" / "strategy_sandbox.json"
        if ss_path.exists():
            ss = json.loads(ss_path.read_text())
            strategy_sandbox = {
                "best_strategy":            ss.get("best_strategy"),
                "ranked_by_total_sharpe":   ss.get("ranked_by_total_sharpe"),
                "n_strategies":             len(ss.get("strategies", [])),
                "total_sharpe_per_strategy":{
                    k: v.get("total", {}).get("sharpe")
                    for k, v in ss.get("per_strategy", {}).items()
                },
                "generated_at":             ss.get("generated_at"),
            }
    except Exception:
        pass

    # ── Tear sheet snapshot ─────────────────────────────────────────────────
    tear_sheet_snapshot = {}
    try:
        pf_path = ROOT / "data" / "form_pf_lite.json"
        if pf_path.exists():
            pf = json.loads(pf_path.read_text())
            tear_sheet_snapshot = {
                "tear_sheet_path": str(ROOT / "data" / "tear_sheet.md"),
                "form_pf":         pf,
            }
    except Exception:
        pass

    # ── DeepSeek explainer last turn ───────────────────────────────────────
    deepseek_last = {}
    try:
        ds_path = ROOT / "data" / "deepseek_last_turn.json"
        if ds_path.exists():
            ds = json.loads(ds_path.read_text())
            deepseek_last = {
                "ts":              ds.get("ts"),
                "kind":            ds.get("kind", "qa"),
                "question":        ds.get("question"),
                "model":           ds.get("model"),
                "total_tokens":    ds.get("total_tokens"),
                "answer_preview":  (ds.get("answer", "") or "")[:300],
                "dossier_engines": len(ds.get("dossier_keys", [])),
            }
    except Exception:
        pass

    # ── IBKR adapter status ─────────────────────────────────────────────────
    ibkr = {}
    try:
        from scripts.ibkr_adapter import _today_order_count, _load_halal_universe
        ibkr = {
            "dry_run_default":      True,
            "halal_universe_size":  len(_load_halal_universe()),
            "today_order_count":    _today_order_count(),
            "audit_log_path":       str(ROOT / "data" / "ibkr_audit.jsonl"),
        }
    except Exception:
        pass

    # ── Alert router snapshot ───────────────────────────────────────────────
    alert_router = {}
    try:
        ar_path = ROOT / "data" / "alert_router.json"
        if ar_path.exists():
            ar = json.loads(ar_path.read_text())
            alert_router = {
                "n_alerts":     ar.get("n_alerts", 0),
                "alerts":       ar.get("alerts", []),
                "generated_at": ar.get("generated_at"),
            }
    except Exception:
        pass

    # ── Audit trail snapshot ───────────────────────────────────────────────
    audit_trail = {}
    try:
        at_path = ROOT / "data" / "audit_trail_status.json"
        if at_path.exists():
            at = json.loads(at_path.read_text())
            audit_trail = {
                "chain_valid":     at.get("chain_valid"),
                "n_total":         at.get("n_total"),
                "first_break_id":  at.get("first_break_id"),
                "counts_by_type":  at.get("counts_by_type", {}),
                "last_hash":       at.get("last_hash"),
                "generated_at":    at.get("generated_at"),
            }
    except Exception:
        pass

    # ── DR backup snapshot ─────────────────────────────────────────────────
    dr_backup = {}
    try:
        dr_path = ROOT / "data" / "dr_backup.json"
        if dr_path.exists():
            dr = json.loads(dr_path.read_text())
            snap = dr.get("snapshot", {})
            dr_backup = {
                "success":          snap.get("success"),
                "archive_path":     snap.get("archive_path"),
                "size_mb":          snap.get("size_mb"),
                "encrypted":        snap.get("encrypted"),
                "n_snapshots":      dr.get("n_snapshots"),
                "verification_valid":dr.get("verification", {}).get("valid"),
                "generated_at":     dr.get("generated_at"),
            }
    except Exception:
        pass

    # ── Latency profile snapshot ───────────────────────────────────────────
    latency_profile = {}
    try:
        lp_path = ROOT / "data" / "latency_profile.json"
        if lp_path.exists():
            lp = json.loads(lp_path.read_text())
            latency_profile = {
                "current_run_total_s":lp.get("current_run_total_s"),
                "slowest_stage":      lp.get("slowest_stage"),
                "slowest_duration_s": lp.get("slowest_duration_s"),
                "stage_durations_s":  lp.get("stage_durations_s", {}),
                "n_recommendations":  lp.get("n_recommendations"),
                "recommendations":    lp.get("recommendations", []),
                "generated_at":       lp.get("generated_at"),
            }
    except Exception:
        pass

    # ── Brinson attribution snapshot ───────────────────────────────────────
    brinson = {}
    try:
        br_path = ROOT / "data" / "brinson_attribution.json"
        if br_path.exists():
            br = json.loads(br_path.read_text())
            brinson = {
                "portfolio_return_pct":  br.get("portfolio_return_pct"),
                "benchmark_return_pct":  br.get("benchmark_return_pct"),
                "excess_return_pct":     br.get("excess_return_pct"),
                "allocation_effect_pct": br.get("allocation_effect_pct"),
                "selection_effect_pct":  br.get("selection_effect_pct"),
                "interaction_pct":       br.get("interaction_pct"),
                "portfolio_weights":     br.get("portfolio_weights", {}),
                "generated_at":          br.get("generated_at"),
            }
    except Exception:
        pass

    # ── Fama-French snapshot ───────────────────────────────────────────────
    fama_french = {}
    try:
        ff_path = ROOT / "data" / "fama_french.json"
        if ff_path.exists():
            ff = json.loads(ff_path.read_text())
            fama_french = {
                "alpha_annualised_pct":    ff.get("alpha_annualised_pct"),
                "alpha_t_stat":            ff.get("alpha_t_stat"),
                "alpha_significant":       ff.get("alpha_significant"),
                "r_squared":               ff.get("r_squared"),
                "information_ratio":       ff.get("information_ratio"),
                "dominant_factor":         ff.get("dominant_factor"),
                "n_significant_factors":   ff.get("n_significant_factors"),
                "factor_summary":          ff.get("factor_summary", []),
                "residual_vol_ann_pct":    ff.get("residual_vol_ann_pct"),
                "generated_at":            ff.get("generated_at"),
            }
    except Exception:
        pass

    # ── IC/IR tracker snapshot ─────────────────────────────────────────────
    ic_ir = {}
    try:
        ii_path = ROOT / "data" / "ic_ir_tracker.json"
        if ii_path.exists():
            ii = json.loads(ii_path.read_text())
            ic_ir = {
                "windows_days":         ii.get("windows_days"),
                "ranked_by_ir":         ii.get("ranked_by_ir", []),
                "ranked_by_ic":         ii.get("ranked_by_ic", []),
                "deployable_signals":   ii.get("deployable_signals", []),
                "n_deployable":         ii.get("n_deployable"),
                "per_signal_summary": {
                    k: {
                        "ir_63d":    v.get("ir_63d"),
                        "ic_63d":    v.get("ic_63d"),
                        "deployable":v.get("deployable"),
                    } for k, v in ii.get("per_signal", {}).items()
                },
                "generated_at":         ii.get("generated_at"),
            }
    except Exception:
        pass

    # ── Decision quality snapshot ──────────────────────────────────────────
    decision_quality = {}
    try:
        dq_path = ROOT / "data" / "decision_quality.json"
        if dq_path.exists():
            dq = json.loads(dq_path.read_text())
            decision_quality = {
                "realised_positive_rate": dq.get("realised_positive_rate"),
                "best_signal":            dq.get("best_signal"),
                "best_brier":             dq.get("best_brier"),
                "best_skill":             dq.get("best_skill"),
                "ranked_by_skill":        dq.get("ranked_by_skill", []),
                "per_signal_brief": {
                    k: {
                        "brier":       v.get("brier"),
                        "skill_brier": v.get("skill_brier"),
                        "ece":         v.get("ece"),
                        "reliability": v.get("reliability"),
                    } for k, v in dq.get("per_signal", {}).items()
                },
                "generated_at":           dq.get("generated_at"),
            }
    except Exception:
        pass

    # ── Conformal intervals snapshot ───────────────────────────────────────
    conformal = {}
    try:
        co_path = ROOT / "data" / "conformal_intervals.json"
        if co_path.exists():
            co = json.loads(co_path.read_text())
            live = co.get("live_intervals", {})
            intervals = co.get("intervals", {})
            conformal = {
                "horizon":              co.get("horizon"),
                "latest_forecast_pct":  co.get("latest_forecast_pct"),
                "alpha_05_width_pct":   intervals.get("alpha_05", {}).get("interval_width_pct"),
                "alpha_05_coverage":    intervals.get("alpha_05", {}).get("empirical_coverage"),
                "alpha_05_valid":       intervals.get("alpha_05", {}).get("valid"),
                "alpha_10_width_pct":   intervals.get("alpha_10", {}).get("interval_width_pct"),
                "alpha_10_coverage":    intervals.get("alpha_10", {}).get("empirical_coverage"),
                "alpha_10_valid":       intervals.get("alpha_10", {}).get("valid"),
                "live_05_lower_pct":    live.get("alpha_05", {}).get("lower_pct"),
                "live_05_upper_pct":    live.get("alpha_05", {}).get("upper_pct"),
                "live_10_lower_pct":    live.get("alpha_10", {}).get("lower_pct"),
                "live_10_upper_pct":    live.get("alpha_10", {}).get("upper_pct"),
                "model_r2_train":       co.get("model_r2", {}).get("train"),
                "model_r2_test":        co.get("model_r2", {}).get("test"),
                "generated_at":         co.get("generated_at"),
            }
    except Exception:
        pass

    # ── RL sizing snapshot ─────────────────────────────────────────────────
    rl_sizing = {}
    try:
        rl_path = ROOT / "data" / "rl_sizing.json"
        if rl_path.exists():
            rl = json.loads(rl_path.read_text())
            rl_sizing = {
                "n_train":              rl.get("n_train"),
                "n_test":               rl.get("n_test"),
                "rl_train_sharpe":      rl.get("rl_train_sharpe"),
                "rl_test_sharpe":       rl.get("rl_test_sharpe"),
                "baseline_test_sharpe": rl.get("baseline_test_sharpe"),
                "test_lift_sharpe":     rl.get("test_lift_sharpe"),
                "avg_action_test":      rl.get("avg_action_test"),
                "n_visited_states":     rl.get("n_visited_states"),
                "generated_at":         rl.get("generated_at"),
            }
    except Exception:
        pass

    # ── Ensemble stacking snapshot ─────────────────────────────────────────
    ensemble_stacking = {}
    try:
        es_path = ROOT / "data" / "ensemble_stacking.json"
        if es_path.exists():
            es = json.loads(es_path.read_text())
            ensemble_stacking = {
                "n_obs":            es.get("n_obs"),
                "label_horizon":    es.get("label_horizon"),
                "folds":            es.get("folds"),
                "base_metrics":     es.get("base_metrics", {}),
                "meta_metrics":     es.get("meta_metrics", {}),
                "stacking_lift":    es.get("stacking_lift", {}),
                "best_single_base": es.get("best_single_base", {}),
                "generated_at":     es.get("generated_at"),
            }
    except Exception:
        pass

    # ── Purged K-fold snapshot ─────────────────────────────────────────────
    purged_kfold = {}
    try:
        pk_path = ROOT / "data" / "purged_kfold.json"
        if pk_path.exists():
            pk = json.loads(pk_path.read_text())
            s = pk.get("summary", {})
            purged_kfold = {
                "n_splits":             pk.get("n_splits"),
                "mean_sharpe":          s.get("mean_sharpe"),
                "std_sharpe":           s.get("std_sharpe"),
                "stability_ratio":      s.get("stability_ratio"),
                "min_sharpe":           s.get("min_sharpe"),
                "max_sharpe":           s.get("max_sharpe"),
                "n_positive_folds":     s.get("n_positive_folds"),
                "worst_max_dd_pct":     s.get("worst_max_drawdown_pct"),
                "avg_max_dd_pct":       s.get("avg_max_drawdown_pct"),
                "generated_at":         pk.get("generated_at"),
            }
    except Exception:
        pass

    # ── Bayesian HPO snapshot (read-only; HPO runs on demand) ──────────────
    bayesian_hpo = {}
    try:
        bh_path = ROOT / "data" / "bayesian_hpo.json"
        if bh_path.exists():
            bh = json.loads(bh_path.read_text())
            bayesian_hpo = {
                "n_trials":         bh.get("n_trials"),
                "oos_days":         bh.get("oos_days"),
                "baseline_sharpe":  bh.get("baseline_sharpe"),
                "best_sharpe":      bh.get("best_sharpe"),
                "improvement":      bh.get("improvement"),
                "best_params":      bh.get("best_params", {}),
                "best_dd_pct":      bh.get("best_metrics", {}).get("max_drawdown_pct"),
                "best_ret_pct":     bh.get("best_metrics", {}).get("ann_return_pct"),
                "generated_at":     bh.get("generated_at"),
            }
    except Exception:
        pass

    # ── Options pricer snapshot ────────────────────────────────────────────
    options_pricer = {}
    try:
        op_path = ROOT / "data" / "options_pricer.json"
        if op_path.exists():
            op = json.loads(op_path.read_text())
            options_pricer = {
                "ticker":          op.get("ticker"),
                "spot":            op.get("spot"),
                "sigma":           op.get("sigma"),
                "tenor_days":      op.get("tenor_days"),
                "atm_call_price":  op.get("atm_call", {}).get("price"),
                "atm_put_price":   op.get("atm_put", {}).get("price"),
                "atm_delta_call":  op.get("atm_call", {}).get("delta"),
                "atm_gamma":       op.get("atm_call", {}).get("gamma"),
                "atm_vega":        op.get("atm_call", {}).get("vega"),
                "parity_residual": op.get("parity_residual"),
                "generated_at":    op.get("generated_at"),
            }
    except Exception:
        pass

    # ── Tail hedge snapshot ────────────────────────────────────────────────
    tail_hedge = {}
    try:
        th_path = ROOT / "data" / "tail_hedge.json"
        if th_path.exists():
            th = json.loads(th_path.read_text())
            tail_hedge = {
                "strike":                  th.get("strike"),
                "put_price_per_share":     th.get("put_price_per_share"),
                "contracts_needed":        th.get("contracts_needed"),
                "annual_premium_usd":      th.get("annual_premium_usd"),
                "annual_drag_pct":         th.get("annual_drag_pct"),
                "residual_cvar_pct":       th.get("residual_cvar_pct"),
                "constraint_binding":      th.get("constraint_binding"),
                "current_cvar_daily_pct":  th.get("current_cvar_daily_pct"),
                "target_cvar_pct":         th.get("target_cvar_pct"),
                "generated_at":            th.get("generated_at"),
            }
    except Exception:
        pass

    # ── Carry analyzer snapshot ────────────────────────────────────────────
    carry = {}
    try:
        ca_path = ROOT / "data" / "carry_analyzer.json"
        if ca_path.exists():
            ca = json.loads(ca_path.read_text())
            carry = {
                "price_source":      ca.get("price_source"),
                "price_now":         ca.get("price_now"),
                "fair_carry_pct":    ca.get("carry", {}).get("fair_pct"),
                "carry_burden":      ca.get("carry", {}).get("burden"),
                "dynamic_usd_rate":  ca.get("carry", {}).get("dynamic_usd_rate"),
                "real_yield_used":   ca.get("carry", {}).get("real_yield_used"),
                "excess_vs_carry_21d": ca.get("excess_vs_carry", {}).get("21d_pct"),
                "excess_vs_carry_63d": ca.get("excess_vs_carry", {}).get("63d_pct"),
                "ann_realised_21d":  ca.get("spot_returns", {}).get("ann_realised_21d_pct"),
                "generated_at":      ca.get("generated_at"),
            }
    except Exception:
        pass

    # ── Term structure snapshot ────────────────────────────────────────────
    term_structure = {}
    try:
        ts_path = ROOT / "data" / "term_structure.json"
        if ts_path.exists():
            ts = json.loads(ts_path.read_text())
            term_structure = {
                "n_contracts":        ts.get("n_contracts"),
                "front_symbol":       ts.get("front_symbol"),
                "front_price":        ts.get("front_price"),
                "back_symbol":        ts.get("back_symbol"),
                "back_price":         ts.get("back_price"),
                "overall_slope_pct":  ts.get("overall_slope_pct"),
                "curve_shape":        ts.get("curve_shape"),
                "roll_yield_pct":     ts.get("roll_yield_pct"),
                "curve_r_squared":    ts.get("curve_r_squared"),
                "stress_flag":        ts.get("stress_flag"),
                "generated_at":       ts.get("generated_at"),
            }
    except Exception:
        pass

    # ── Macro nowcast snapshot ─────────────────────────────────────────────
    macro_nowcast = {}
    try:
        mn_path = ROOT / "data" / "macro_nowcast.json"
        if mn_path.exists():
            mn = json.loads(mn_path.read_text())
            macro_nowcast = {
                "composite_score": mn.get("composite_score"),
                "regime":          mn.get("regime"),
                "n_active":        mn.get("n_active"),
                "n_components":    mn.get("n_components"),
                "components":      mn.get("components", {}),
                "top_drivers":     mn.get("top_drivers", []),
                "generated_at":    mn.get("generated_at"),
            }
    except Exception:
        pass

    # ── ETF flow snapshot ──────────────────────────────────────────────────
    etf_flows = {}
    try:
        ef_path = ROOT / "data" / "etf_flows.json"
        if ef_path.exists():
            ef = json.loads(ef_path.read_text())
            etf_flows = {
                "headline":          ef.get("headline"),
                "n_inflows":         ef.get("n_inflows"),
                "n_outflows":        ef.get("n_outflows"),
                "divergent":         ef.get("divergent"),
                "gold_7d_usd":       ef.get("gold_bucket", {}).get("flow_7d_usd"),
                "gold_21d_usd":      ef.get("gold_bucket", {}).get("flow_21d_usd"),
                "gold_bucket_regime":ef.get("gold_bucket", {}).get("bucket_regime"),
                "silver_7d_usd":     ef.get("silver_bucket", {}).get("flow_7d_usd"),
                "silver_21d_usd":    ef.get("silver_bucket", {}).get("flow_21d_usd"),
                "silver_bucket_regime":ef.get("silver_bucket", {}).get("bucket_regime"),
                "generated_at":      ef.get("generated_at"),
            }
    except Exception:
        pass

    # ── Geopolitical event snapshot ────────────────────────────────────────
    geopolitical = {}
    try:
        ge_path = ROOT / "data" / "geopolitical_events.json"
        if ge_path.exists():
            ge = json.loads(ge_path.read_text())
            geopolitical = {
                "regime":           ge.get("regime"),
                "current_score":    ge.get("current_score"),
                "priority":         ge.get("priority"),
                "delta_dod":        ge.get("delta_dod"),
                "event_flag":       ge.get("event_flag"),
                "regime_shift":     ge.get("regime_shift"),
                "z_score":          ge.get("z_score"),
                "n_spikes_history": len(ge.get("spike_history", [])),
                "generated_at":     ge.get("generated_at"),
            }
    except Exception:
        pass

    # ── Central bank speech snapshot ───────────────────────────────────────
    cb_speech = {}
    try:
        cb_path = ROOT / "data" / "cb_speech.json"
        if cb_path.exists():
            cb = json.loads(cb_path.read_text())
            series = cb.get("series", {})
            cb_speech = {
                "n_obs":              cb.get("n_obs_history"),
                "fed_regime":         cb.get("fed_regime"),
                "fed_latest":         cb.get("fed_latest"),
                "fed_z":              cb.get("fed_z"),
                "regime_shift":       cb.get("regime_shift_detected"),
                "pplx_geo_risk_latest":  series.get("pplx_geo_risk", {}).get("latest"),
                "pplx_phys_demand_latest": series.get("pplx_phys_demand", {}).get("latest"),
                "pplx_macro_latest":  series.get("pplx_macro", {}).get("latest"),
                "generated_at":       cb.get("generated_at"),
            }
    except Exception:
        pass

    # ── News sentiment snapshot ────────────────────────────────────────────
    news_sentiment = {}
    try:
        ns_path = ROOT / "data" / "news_sentiment.json"
        if ns_path.exists():
            ns = json.loads(ns_path.read_text())
            agg = ns.get("aggregate", {})
            news_sentiment = {
                "n_obs":               ns.get("n_total_obs"),
                "n_tickers":           ns.get("n_tickers"),
                "avg_sentiment":       agg.get("avg_sentiment"),
                "consensus_regime":    agg.get("consensus_regime"),
                "dispersion":          agg.get("dispersion"),
                "divergent":           agg.get("divergent"),
                "top_movers":          agg.get("top_movers", []),
                "per_ticker_brief":    {
                    t: {
                        "latest":   v.get("latest"),
                        "regime":   v.get("regime"),
                        "momentum": v.get("momentum"),
                    } for t, v in ns.get("per_ticker", {}).items()
                },
                "generated_at":        ns.get("generated_at"),
            }
    except Exception:
        pass

    # ── Capacity snapshot ──────────────────────────────────────────────────
    capacity = {}
    try:
        cap_path = ROOT / "data" / "capacity_analyzer.json"
        if cap_path.exists():
            cap = json.loads(cap_path.read_text())
            phys = cap.get("thresholds_physical", {})
            paper = cap.get("thresholds_paper", {})
            capacity = {
                "ticker":               cap.get("ticker"),
                "expected_alpha_pct":   cap.get("expected_alpha_pct"),
                "alpha_source":         cap.get("alpha_source"),
                "adv_usd":              cap.get("adv_usd"),
                "deploy_pct_capital":   cap.get("deploy_pct_capital"),
                "turnover_per_year":    cap.get("turnover_per_year"),
                "physical_cap_25pct":   phys.get("decay_25pct", {}).get("aum_cap_usd"),
                "physical_cap_50pct":   phys.get("decay_50pct", {}).get("aum_cap_usd"),
                "physical_cap_breakeven":phys.get("decay_100pct", {}).get("aum_cap_usd"),
                "paper_cap_25pct":      paper.get("decay_25pct", {}).get("aum_cap_usd"),
                "paper_cap_50pct":      paper.get("decay_50pct", {}).get("aum_cap_usd"),
                "paper_cap_breakeven":  paper.get("decay_100pct", {}).get("aum_cap_usd"),
                "generated_at":         cap.get("generated_at"),
            }
    except Exception:
        pass

    # ── Stop-loss snapshot ─────────────────────────────────────────────────
    stop_loss = {}
    try:
        sl_path = ROOT / "data" / "stop_loss_optimizer.json"
        if sl_path.exists():
            sl = json.loads(sl_path.read_text())
            stop_loss = {
                "current_price":       sl.get("current_price"),
                "atr_14":              sl.get("atr_14"),
                "vol_regime":          sl.get("vol_regime"),
                "recommended_method":  sl.get("final_recommendation"),
                "stop_price":          sl.get("final_stop_price"),
                "stop_distance_pct":   sl.get("final_stop_distance_pct"),
                "best_by_expectancy":  sl.get("best_by_expectancy"),
                "best_by_profit_factor": sl.get("best_by_profit_factor"),
                "backtest_summary":    {
                    m: {
                        "win_rate_pct":   b.get("win_rate_pct"),
                        "profit_factor":  b.get("profit_factor"),
                        "expectancy_pct": b.get("expectancy_pct"),
                    } for m, b in sl.get("backtest", {}).items()
                },
                "generated_at":        sl.get("generated_at"),
            }
    except Exception:
        pass

    # ── Adverse selection snapshot ─────────────────────────────────────────
    adverse_selection = {}
    try:
        as_path = ROOT / "data" / "adverse_selection.json"
        if as_path.exists():
            asv = json.loads(as_path.read_text())
            adverse_selection = {
                "ticker":           asv.get("ticker"),
                "n_bars":           asv.get("n_hourly_bars"),
                "worst_hours":      asv.get("worst_hours", [])[:3],
                "best_hours":       asv.get("best_hours", [])[:3],
                "session_summary":  asv.get("session_summary", {}),
                "recommendation":   asv.get("recommendation"),
                "generated_at":     asv.get("generated_at"),
            }
    except Exception:
        pass

    # ── Smart order router snapshot ────────────────────────────────────────
    smart_order_router = {}
    try:
        sor_path = ROOT / "data" / "smart_order_router.json"
        if sor_path.exists():
            sor = json.loads(sor_path.read_text())
            rc = sor.get("recommended_cost", {})
            smart_order_router = {
                "ticker":           sor.get("ticker"),
                "notional_usd":     sor.get("notional_usd"),
                "recommended_algo": sor.get("recommended_algo"),
                "horizon_minutes":  sor.get("horizon_minutes"),
                "n_slices":         sor.get("n_slices"),
                "participation_60min": sor.get("participation_60min"),
                "total_cost_bps":   rc.get("total_oneway_bps"),
                "impact_bps":       rc.get("impact_bps"),
                "spread_bps":       rc.get("spread_bps"),
                "physical_bps":     rc.get("physical_bps"),
                "cheapest_algo":    sor.get("cheapest_algo"),
                "savings_vs_cheapest_bps": sor.get("savings_vs_cheapest_bps"),
                "generated_at":     sor.get("generated_at"),
            }
    except Exception:
        pass

    # ── Bayesian Model Averaging snapshot ──────────────────────────────────
    bma = {}
    try:
        bma_path = ROOT / "data" / "bma_weights.json"
        if bma_path.exists():
            b = json.loads(bma_path.read_text())
            bt = b.get("backtest", {})
            bma = {
                "top_source":       b.get("top_source"),
                "bma_weights":      b.get("weights", {}).get("bma", {}),
                "ir_weights":       b.get("weights", {}).get("ir", {}),
                "bma_sharpe":       bt.get("bma", {}).get("sharpe"),
                "bma_return_pct":   bt.get("bma", {}).get("ann_return_pct"),
                "bma_vol_pct":      bt.get("bma", {}).get("ann_vol_pct"),
                "bma_dd_pct":       bt.get("bma", {}).get("max_drawdown_pct"),
                "ir_sharpe":        bt.get("ir", {}).get("sharpe"),
                "eq_sharpe":        bt.get("equal", {}).get("sharpe"),
                "per_source":       b.get("per_source", []),
                "bma_window_days":  b.get("bma_window_days"),
                "generated_at":     b.get("generated_at"),
            }
    except Exception:
        pass

    # ── Macro regime snapshot ──────────────────────────────────────────────
    macro_regime = {}
    try:
        mr_path = ROOT / "data" / "macro_regime.json"
        if mr_path.exists():
            mr = json.loads(mr_path.read_text())
            macro_regime = {
                "quadrant":         mr.get("quadrant"),
                "confidence":       mr.get("confidence"),
                "growth_score":     mr.get("growth_score"),
                "inflation_score":  mr.get("inflation_score"),
                "asset_tilts":      mr.get("asset_tilts", {}),
                "description":      mr.get("description"),
                "spy_21d_mom_pct":  mr.get("growth_components", {}).get("spy_21d_mom_pct"),
                "dxy_21d_mom_pct":  mr.get("growth_components", {}).get("dxy_21d_mom_pct"),
                "gold_21d_mom_pct": mr.get("inflation_components", {}).get("gold_21d_mom_pct"),
                "generated_at":     mr.get("generated_at"),
            }
    except Exception:
        pass

    # ── Structural breaks snapshot ─────────────────────────────────────────
    structural_breaks = {}
    try:
        sb_path = ROOT / "data" / "structural_breaks.json"
        if sb_path.exists():
            sb = json.loads(sb_path.read_text())
            structural_breaks = {
                "cusum_break":        sb.get("summary", {}).get("cusum_break"),
                "cusum_stat":         sb.get("cusum", {}).get("test_stat"),
                "n_mean_breaks":      sb.get("summary", {}).get("n_mean_breaks"),
                "n_variance_breaks":  sb.get("summary", {}).get("n_variance_breaks"),
                "most_recent_break":  sb.get("summary", {}).get("most_recent_break"),
                "days_since_last_break": sb.get("summary", {}).get("days_since_last_break"),
                "recent_var_breaks":  sb.get("variance_breaks", [])[-3:],
                "generated_at":       sb.get("generated_at"),
            }
    except Exception:
        pass

    # ── DCC-GARCH dynamic correlations snapshot ────────────────────────────
    dcc_garch = {}
    try:
        dcc_path = ROOT / "data" / "dcc_garch.json"
        if dcc_path.exists():
            dcc = json.loads(dcc_path.read_text())
            p = dcc.get("dcc_params", {})
            # Top 3 most stressed pairs by |z|
            pairs = sorted(
                dcc.get("pairs", []),
                key=lambda x: abs(x.get("z_score", 0)),
                reverse=True,
            )[:3]
            dcc_garch = {
                "tickers":              dcc.get("tickers", []),
                "dcc_a":                p.get("a"),
                "dcc_b":                p.get("b"),
                "dcc_a_plus_b":         p.get("a_plus_b"),
                "dcc_log_likelihood":   p.get("log_likelihood"),
                "avg_corr_now":         dcc.get("avg_pairwise_corr_now"),
                "avg_corr_long_run":    dcc.get("avg_pairwise_corr_long_run"),
                "n_stressed_pairs":     dcc.get("n_stressed", 0),
                "stressed_pairs":       dcc.get("stressed_pairs", []),
                "top_movers":           [
                    {
                        "pair": p_["pair"],
                        "current": p_.get("current_corr"),
                        "mean": p_.get("mean_corr"),
                        "z": p_.get("z_score"),
                    } for p_ in pairs
                ],
                "current_correlation":  dcc.get("current_correlation", {}),
                "generated_at":         dcc.get("generated_at"),
            }
    except Exception:
        pass

    # ── Vol target & risk budget snapshot ──────────────────────────────────
    vol_target = {}
    try:
        vt_path = ROOT / "data" / "vol_target_budget.json"
        if vt_path.exists():
            vt = json.loads(vt_path.read_text())
            vol_target = {
                "target_vol_pct":   vt.get("target_vol_pct"),
                "current_vol_pct":  vt.get("current_vol_pct"),
                "leverage_raw":     vt.get("leverage_raw"),
                "leverage_capped":  vt.get("leverage_capped"),
                "leverage_action":  vt.get("guidance", {}).get("leverage_action"),
                "deploy_pct_capital":vt.get("guidance", {}).get("deploy_pct_of_capital"),
                "ir_weights":       vt.get("ir_weighted", {}).get("weights", {}),
                "ir_blend_vol_pct": vt.get("ir_weighted", {}).get("blend_vol_pct"),
                "er_weights":       vt.get("equal_risk", {}).get("weights", {}),
                "er_blend_vol_pct": vt.get("equal_risk", {}).get("blend_vol_pct"),
                "n_sources":        vt.get("n_sources"),
                "generated_at":     vt.get("generated_at"),
            }
    except Exception:
        pass

    # ── Mean-CVaR snapshot ─────────────────────────────────────────────────
    mean_cvar_snapshot = {}
    try:
        mc_path = ROOT / "data" / "mean_cvar.json"
        if mc_path.exists():
            mc = json.loads(mc_path.read_text())
            m = mc.get("metrics", {})
            min_m = m.get("min_cvar", {})
            mean_m = m.get("mean_cvar", {})
            mean_cvar_snapshot = {
                "tickers":              mc.get("tickers", []),
                "alpha":                mc.get("alpha"),
                "min_cvar_weights":     min_m.get("weights", {}),
                "mean_cvar_weights":    mean_m.get("weights", {}),
                "min_cvar_sharpe":      min_m.get("sharpe"),
                "min_cvar_vol_pct":     min_m.get("ann_vol_pct"),
                "min_cvar_dd_pct":      min_m.get("max_drawdown_pct"),
                "min_cvar_cvar_pct":    min_m.get("cvar_pct"),
                "mean_cvar_sharpe":     mean_m.get("sharpe"),
                "mean_cvar_vol_pct":    mean_m.get("ann_vol_pct"),
                "mean_cvar_dd_pct":     mean_m.get("max_drawdown_pct"),
                "mean_cvar_cvar_pct":   mean_m.get("cvar_pct"),
                "target_ann_pct":       mc.get("avg_asset_target_ann_pct"),
                "generated_at":         mc.get("generated_at"),
            }
    except Exception:
        pass

    # ── Black-Litterman snapshot ───────────────────────────────────────────
    black_litterman = {}
    try:
        bl_path = ROOT / "data" / "black_litterman.json"
        if bl_path.exists():
            bl = json.loads(bl_path.read_text())
            m = bl.get("metrics", {})
            bl_m = m.get("black_litterman", {})
            mkt_m = m.get("market_weights", {})
            eq_m = m.get("equal_weight", {})
            black_litterman = {
                "tickers":          bl.get("tickers", []),
                "n_views":          bl.get("n_views"),
                "view_descriptions":bl.get("view_descriptions", []),
                "weights_bl":       bl_m.get("weights", {}),
                "weights_market":   mkt_m.get("weights", {}),
                "weights_equal":    eq_m.get("weights", {}),
                "bl_sharpe":        bl_m.get("sharpe"),
                "bl_vol_pct":       bl_m.get("ann_vol_pct"),
                "bl_max_dd_pct":    bl_m.get("max_drawdown_pct"),
                "bl_return_pct":    bl_m.get("ann_return_pct"),
                "mkt_sharpe":       mkt_m.get("sharpe"),
                "asset_table":      bl.get("asset_table", []),
                "generated_at":     bl.get("generated_at"),
            }
    except Exception:
        pass

    # ── HRP allocator snapshot ─────────────────────────────────────────────
    hrp_allocation = {}
    try:
        hrp_path = ROOT / "data" / "hrp_allocator.json"
        if hrp_path.exists():
            hrp = json.loads(hrp_path.read_text())
            m = hrp.get("metrics", {})
            hrp_allocation = {
                "tickers":           hrp.get("tickers", []),
                "weights_hrp":       m.get("hrp", {}).get("weights", {}),
                "weights_equal":     m.get("equal_weight", {}).get("weights", {}),
                "weights_inv_vol":   m.get("inverse_vol", {}).get("weights", {}),
                "hrp_sharpe":        m.get("hrp", {}).get("sharpe"),
                "hrp_vol_pct":       m.get("hrp", {}).get("ann_vol_pct"),
                "hrp_max_dd_pct":    m.get("hrp", {}).get("max_drawdown_pct"),
                "hrp_div_ratio":     m.get("hrp", {}).get("diversification_ratio"),
                "hrp_eff_n_bets":    m.get("hrp", {}).get("effective_n_bets"),
                "eq_sharpe":         m.get("equal_weight", {}).get("sharpe"),
                "iv_sharpe":         m.get("inverse_vol", {}).get("sharpe"),
                "generated_at":      hrp.get("generated_at"),
            }
    except Exception:
        pass

    # ── Signal decay snapshot ──────────────────────────────────────────────
    signal_decay = {}
    try:
        sd_path = ROOT / "data" / "signal_decay.json"
        if sd_path.exists():
            sd = json.loads(sd_path.read_text())
            ranked_ic = sd.get("ranked_by_ic", [])
            top_sig = ranked_ic[0] if ranked_ic else None
            top_data = sd.get("signals", {}).get(top_sig, {}) if top_sig else {}
            signal_decay = {
                "n_signals":          sd.get("n_signals"),
                "top_signal":         top_sig,
                "top_signal_ic":      top_data.get("best_horizon_ic"),
                "top_signal_t":       top_data.get("best_horizon_t"),
                "top_signal_half_life": top_data.get("half_life_days"),
                "top_signal_rebalance":top_data.get("rebalance_days"),
                "decaying_signals":   sd.get("decaying_signals", []),
                "strengthening_signals": sd.get("strengthening_signals", []),
                "ranked_by_ic":       ranked_ic,
                "ranked_by_half_life":sd.get("ranked_by_half_life", []),
                "generated_at":       sd.get("generated_at"),
            }
    except Exception:
        pass

    # ── Volatility surface snapshot ────────────────────────────────────────
    vol_surface = {}
    try:
        vs_path = ROOT / "data" / "vol_surface.json"
        if vs_path.exists():
            vs = json.loads(vs_path.read_text())
            vol_surface = {
                "vol_regime":       vs.get("vol_regime"),
                "phase":            vs.get("phase"),
                "curve_shape":      vs.get("curve_shape"),
                "vol_21d_pct":      vs.get("term_structure", {}).get("rv_21d"),
                "vol_21d_pctile":   vs.get("vol_21d_pctile"),
                "vol_5d_pct":       vs.get("term_structure", {}).get("rv_5d"),
                "vol_252d_pct":     vs.get("term_structure", {}).get("rv_252d"),
                "vol_of_vol":       vs.get("vol_of_vol"),
                "kelly_multiplier": vs.get("actions", {}).get("kelly_fraction_multiplier"),
                "stop_atr_mult":    vs.get("actions", {}).get("stop_atr_multiplier"),
                "generated_at":     vs.get("generated_at"),
            }
    except Exception:
        pass

    # ── Alpha attribution snapshot ─────────────────────────────────────────
    alpha_attribution = {}
    try:
        aa_path = ROOT / "data" / "alpha_attribution.json"
        if aa_path.exists():
            aa = json.loads(aa_path.read_text())
            ranked = aa.get("ranked_by_sharpe", [])
            top_src = ranked[0] if ranked else None
            top_full = aa.get("full_history", {}).get(top_src, {}) if top_src else {}
            top_ir = aa.get("information_ratios", {}).get(top_src, {}) if top_src else {}
            comb = aa.get("combined", {})
            es = comb.get("equal_weight_summary", {})
            alpha_attribution = {
                "n_sources":            len(aa.get("sources", [])),
                "top_source":           top_src,
                "top_source_sharpe":    top_full.get("sharpe"),
                "top_source_ir":        top_ir.get("information_ratio"),
                "top_source_active_pct":top_ir.get("active_return_pct"),
                "combined_sharpe":      es.get("sharpe"),
                "combined_return_pct":  es.get("ann_return_pct"),
                "combined_vol_pct":     es.get("ann_vol_pct"),
                "diversification_ratio":comb.get("diversification_ratio"),
                "ranked_by_sharpe":     ranked,
                "ranked_by_ir":         aa.get("ranked_by_information_ratio", []),
                "generated_at":         aa.get("generated_at"),
            }
    except Exception:
        pass

    # ── Transaction cost analysis snapshot ─────────────────────────────────
    tca_snapshot = {}
    try:
        tca_path = ROOT / "data" / "transaction_cost_model.json"
        if tca_path.exists():
            tca = json.loads(tca_path.read_text())
            agg = tca.get("aggregate", {})
            tca_snapshot = {
                "vol_regime":          tca.get("vol_regime"),
                "vol_multiplier":      tca.get("vol_multiplier"),
                "avg_oneway_cost_bps": agg.get("avg_oneway_cost_bps"),
                "min_oneway_cost_bps": agg.get("min_oneway_cost_bps"),
                "max_oneway_cost_bps": agg.get("max_oneway_cost_bps"),
                "n_trades":            agg.get("n_trades"),
                "metals_summary":      [
                    {"ticker": m.get("ticker"),
                     "oneway_bps": m.get("total_oneway_bps"),
                     "physical_bps": m.get("physical_premium_bps")}
                    for m in tca.get("metals", []) if "error" not in m
                ],
                "generated_at":        tca.get("generated_at"),
            }
    except Exception:
        pass

    return {
        "run_date":        run_date,
        "run_timestamp":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pipeline_status": pipeline_status,
        "abort_reason":    abort_reason,
        "ticker":          ticker,
        "stages":          stages,
        "portfolio":       portfolio,
        "regime":          regime,
        "risk":            risk,
        "committee":       committee,
        "position_mgmt":   position_mgmt,
        "ensemble_info":   ensemble_info,
        "institutional_risk": institutional_risk,
        "drawdown_tier":   drawdown_tier,
        "cointegration":   cointegration,
        "tca":             tca_snapshot,
        "alpha_attribution": alpha_attribution,
        "vol_surface":       vol_surface,
        "signal_decay":      signal_decay,
        "hrp_allocation":    hrp_allocation,
        "black_litterman":   black_litterman,
        "mean_cvar":         mean_cvar_snapshot,
        "vol_target":        vol_target,
        "dcc_garch":         dcc_garch,
        "structural_breaks": structural_breaks,
        "macro_regime":      macro_regime,
        "bma":               bma,
        "smart_order_router":smart_order_router,
        "adverse_selection": adverse_selection,
        "stop_loss":         stop_loss,
        "capacity":          capacity,
        "news_sentiment":    news_sentiment,
        "cb_speech":         cb_speech,
        "geopolitical":      geopolitical,
        "etf_flows":         etf_flows,
        "macro_nowcast":     macro_nowcast,
        "options_pricer":    options_pricer,
        "tail_hedge":        tail_hedge,
        "carry":             carry,
        "term_structure":    term_structure,
        "bayesian_hpo":      bayesian_hpo,
        "purged_kfold":      purged_kfold,
        "ensemble_stacking": ensemble_stacking,
        "rl_sizing":         rl_sizing,
        "conformal":         conformal,
        "brinson":           brinson,
        "fama_french":       fama_french,
        "ic_ir":             ic_ir,
        "decision_quality":  decision_quality,
        "ibkr":              ibkr,
        "alert_router":      alert_router,
        "audit_trail":       audit_trail,
        "dr_backup":         dr_backup,
        "latency_profile":   latency_profile,
        "mrm":               mrm,
        "strategy_sandbox":  strategy_sandbox,
        "tear_sheet":        tear_sheet_snapshot,
        "deepseek_last":     deepseek_last,
        "trade_idea":        trade_idea,
        "halal_universe":    halal_universe,
        "metals_rebalancer": metals_rebalancer,
        "continuous_trainer":continuous_trainer,
        "system_health":     system_health,
        "trade_basket":      trade_basket,
        "position_reconciliation": position_reconciliation,
        "slippage_report":   slippage_report,
        "operator_runbook":  operator_runbook,
        "economic_calendar": economic_calendar,
        "earnings_calendar": earnings_calendar,
        "data_quality":      data_quality,
        "pairs_trader":      pairs_trader,
        "pnl_tracker":       pnl_tracker,
        "alpha_stacker":         alpha_stacker,
        "strategy_selector":     strategy_selector,
        "performance_targeter":  performance_targeter,
        "multi_strategy_book":   multi_strategy_book,
        "strategy_backtest":     strategy_backtest,
        "multi_asset_backtest":       multi_asset_backtest,
        "stress_backtest":            stress_backtest,
        "crisis_detector":            crisis_detector,
        "conviction_weights":         conviction_weights,
        "ml_conviction":              ml_conviction,
        "ml_walk_forward":            ml_walk_forward,
        "multi_asset_stress_backtest":multi_asset_stress_backtest,
        "treasury_overlay_stress":    treasury_overlay_stress,
    }


def _write_atomic(state: dict) -> None:
    """Write pipeline_state.json via tempfile → rename (atomic on POSIX)."""
    PIPELINE_STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = PIPELINE_STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.rename(PIPELINE_STATE)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gold trading AI — daily pipeline orchestrator")
    parser.add_argument("--ticker",  default=DEFAULT_TICKER,
        help=f"Primary metal ticker (default: {DEFAULT_TICKER})")
    parser.add_argument("--dry-run", action="store_true",
        help="Log all stages but skip subprocess calls and LLM calls")
    args     = parser.parse_args()
    ticker   = args.ticker
    dry_run  = args.dry_run
    run_date = date.today().isoformat()

    log    = _setup_logging(run_date)
    stages: dict = {}

    if dry_run:
        log.info("DRY-RUN mode active — no subprocesses, no LLM calls")

    # ── Stage 1: Harvester (hard abort gate) ──────────────────────────────────
    log.info("─" * 32 + " Stage 1: Harvester " + "─" * 12)
    stages["harvester"] = stage_harvester(dry_run)

    if stages["harvester"]["status"] == "ABORTED":
        abort_reason = stages["harvester"]["note"]
        log.error(
            "Pipeline ABORTED — harvester failure prevents safe trading on stale data."
        )
        aborted_state = _build_pipeline_state(
            run_date, ticker, stages, "ABORTED", abort_reason
        )
        _write_atomic(aborted_state)
        try:
            from scripts.telegram_notifier import send_urgent
            send_urgent(
                "Pipeline — Harvester",
                abort_reason,
                context="alt_data_harvester --update failed; pipeline did not trade",
            )
        except Exception:
            pass
        sys.exit(1)

    # ── Stage 2: Regime Detector ───────────────────────────────────────────────
    log.info("─" * 32 + " Stage 2: Regime " + "─" * 15)
    stages["regime"] = stage_regime(ticker, dry_run)

    # ── Stage 3: Investment Committee ─────────────────────────────────────────
    log.info("─" * 32 + " Stage 3: Committee " + "─" * 12)
    stages["metal_logic"] = stage_metal_logic(ticker, dry_run)

    # ── Stage 4: Risk Manager (audit preview) ─────────────────────────────────
    log.info("─" * 32 + " Stage 4: Risk " + "─" * 17)
    stages["risk"] = stage_risk(ticker, dry_run)

    # ── Stage 5: Shadow Trader ────────────────────────────────────────────────
    log.info("─" * 32 + " Stage 5: Shadow " + "─" * 15)
    stages["shadow"] = stage_shadow(ticker, dry_run)

    # ── Stage 6: Position Manager ────────────────────────────────────────────
    log.info("─" * 32 + " Stage 6: Position Mgr " + "─" * 9)
    stages["position_mgr"] = stage_position_manager(ticker, dry_run)

    # ── Stage 7: Model Performance Tracker ────────────────────────────────
    log.info("─" * 32 + " Stage 7: Perf Tracker " + "─" * 9)
    stages["perf_tracker"] = stage_performance_tracker(dry_run)

    # ── Stage 8: Position Reconciler (Phase XXIV) ─────────────────────────
    log.info("─" * 32 + " Stage 8: Reconciler " + "─" * 11)
    stages["reconciler"] = stage_reconciler(dry_run)

    # ── Stage 9: Treasury Hedge Overlay (Phase XXV) ───────────────────────
    log.info("─" * 32 + " Stage 9: Treas Hedge " + "─" * 10)
    stages["treasury_hedge"] = stage_treasury_hedge(dry_run)

    # ── Write pipeline_state.json ─────────────────────────────────────────────
    failed  = [k for k, v in stages.items() if v["status"] == "FAILED"]
    status  = "PARTIAL" if failed else "SUCCESS"
    state   = _build_pipeline_state(run_date, ticker, stages, status, None)
    _write_atomic(state)

    total_s = sum(v["duration_s"] for v in stages.values())
    log.info("=" * 64)
    log.info(
        f"Pipeline {status}  |  "
        f"{len(stages) - len(failed)}/{len(stages)} stages OK  |  "
        f"total {total_s:.1f}s"
    )
    if failed:
        log.warning(f"Degraded stages : {failed}")
    log.info(f"State written   : {PIPELINE_STATE}")

    # Surface regime + action for quick terminal read
    ps = state.get("portfolio", {})
    rg = state.get("regime",    {})
    cm = state.get("committee", {})
    log.info(
        f"Regime={rg.get('hmm_state')}  "
        f"Action={cm.get('action_taken')}  "
        f"pf_value=${ps.get('portfolio_value', 0):,.0f}  "
        f"oz={ps.get('gold_oz', 0):.4f}  "
        f"cash=${ps.get('cash_usd', 0):,.0f}"
    )

    # ── Cross-Asset Correlation Monitor ──────────────────────────────────────
    if not dry_run:
        try:
            from scripts.correlation_monitor import generate_report
            corr_report = generate_report()
            if corr_report.get("regime_signal") == "STRUCTURAL_BREAK":
                log.warning(f"STRUCTURAL BREAK detected — "
                            f"{len(corr_report.get('anomalies', []))} correlation anomalies")
        except Exception as _cm_exc:
            log.warning(f"Correlation monitor failed (non-fatal): {_cm_exc}")

    # ── Monte Carlo Simulation ───────────────────────────────────────────────
    if not dry_run:
        try:
            from scripts.monte_carlo import run_simulation
            mc = run_simulation(ticker=ticker, horizon=21)
            prob_pos = mc.get("probabilities", {}).get("positive_return", 0)
            cvar95 = mc.get("risk", {}).get("cvar_95_pct", 0)
            log.info(f"Monte Carlo: P(+)={prob_pos:.1%}  CVaR95={cvar95:+.2f}%")
        except Exception as _mc_exc:
            log.warning(f"Monte Carlo failed (non-fatal): {_mc_exc}")

    # ── Kelly Criterion Sizing ───────────────────────────────────────────────
    if not dry_run:
        try:
            from scripts.kelly_sizing import run_kelly
            shadow = _read_shadow_portfolio()
            pf_val = float(shadow.get("portfolio_value", STARTING_CAPITAL))
            kelly_result = run_kelly(portfolio_value=pf_val, fraction=0.25)
            deploy = kelly_result.get("sizing", {}).get("deploy_usd", 0)
            log.info(f"Kelly sizing: deploy=${deploy:,.0f}")
        except Exception as _ks_exc:
            log.warning(f"Kelly sizing failed (non-fatal): {_ks_exc}")

    # ── Multi-Timeframe Confluence ───────────────────────────────────────────
    if not dry_run:
        try:
            from scripts.mtf_confluence import compute_confluence
            mtf = compute_confluence(ticker=ticker)
            level = mtf.get("confluence", {}).get("level", "UNKNOWN")
            score = mtf.get("confluence", {}).get("score", 0)
            log.info(f"MTF confluence: {level} ({score:+d}/100)")
        except Exception as _mtf_exc:
            log.warning(f"MTF confluence failed (non-fatal): {_mtf_exc}")

    # ── Tail Risk + Factor Attribution Engine ────────────────────────────────
    # EVT peaks-over-threshold CVaR + 5-factor return decomposition.
    if not dry_run:
        try:
            from scripts.tail_risk_engine import run_tail_risk_engine
            tre = run_tail_risk_engine(ticker=ticker, lookback=1260)
            cvar99 = tre.get("tail_risk", {}).get("methods", {}).get("evt_pot", {}).get("cvar_990", 0)
            premium = tre.get("tail_risk", {}).get("tail_fatness_premium_pct", 0)
            r_sq = tre.get("factor_attribution", {}).get("r_squared", 0)
            ir = tre.get("factor_attribution", {}).get("information_ratio", 0)
            alpha = tre.get("factor_attribution", {}).get("alpha_annualised_pct", 0)
            log.info(f"Tail risk: EVT CVaR99={cvar99:.2f}% (+{premium:.0f}% vs Gaussian)  "
                     f"R2={r_sq:.2f}  alpha={alpha:+.1f}%  IR={ir:+.2f}")
        except Exception as _tr_exc:
            log.warning(f"Tail risk engine failed (non-fatal): {_tr_exc}")

    # ── Stress Test (historical crisis scenarios) ─────────────────────────────
    if not dry_run:
        try:
            from scripts.stress_tester import run_stress_test
            shadow = _read_shadow_portfolio()
            pf_val = float(shadow.get("portfolio_value", STARTING_CAPITAL))
            st = run_stress_test(portfolio_value=pf_val, gold_weight=1.0)
            agg = st.get("aggregate", {})
            log.info(f"Stress test: worst-crisis return {agg.get('worst_crisis_return_pct', 0):+.1f}%  "
                     f"({agg.get('worst_crisis_scenario', 'n/a')})")
        except Exception as _st_exc:
            log.warning(f"Stress tester failed (non-fatal): {_st_exc}")

    # ── Drawdown Recovery Controller ─────────────────────────────────────────
    if not dry_run:
        try:
            from scripts.drawdown_controller import run_drawdown_controller
            shadow = _read_shadow_portfolio()
            pf_val = float(shadow.get("portfolio_value", STARTING_CAPITAL))
            peak = max(pf_val, float(shadow.get("starting_capital", pf_val)))
            ddc = run_drawdown_controller(current_value=pf_val, peak_value=peak)
            log.info(f"Drawdown tier: {ddc.get('tier_name')}  "
                     f"(DD={ddc.get('current_dd_pct', 0):+.2f}%, "
                     f"sizing={ddc.get('sizing_multiplier', 1.0):.0%})")
        except Exception as _dc_exc:
            log.warning(f"Drawdown controller failed (non-fatal): {_dc_exc}")

    # ── Cointegration & Mean-Reversion Engine ────────────────────────────────
    # Engle-Granger on 5 cross-asset pairs; OU half-life; z-score signals.
    if not dry_run:
        try:
            from scripts.cointegration_engine import run_cointegration_engine
            ce = run_cointegration_engine(lookback=1260)
            log.info(f"Cointegration: {ce.get('n_cointegrated_5pct', 0)}/{ce.get('n_pairs', 0)} pairs "
                     f"cointegrated, {ce.get('n_actionable', 0)} actionable signals")
            for sig in ce.get("actionable_signals", []):
                log.info(f"  {sig['signal']:<14s} {sig['name']:<14s} "
                         f"z={sig['z_score']:+.2f}  ½-life={sig['half_life_days']:.0f}d")
        except Exception as _ce_exc:
            log.warning(f"Cointegration engine failed (non-fatal): {_ce_exc}")

    # ── Halal Universe Screener (Phase XI Stage 58) ──────────────────────────
    # Runs only weekly — cache TTL handles refresh.
    if not dry_run:
        try:
            from scripts.halal_screener import run_halal_screener
            # Skip if file is fresh (< 7d old)
            hu_path = ROOT / "data" / "halal_universe.json"
            need_refresh = True
            if hu_path.exists():
                age_days = (
                    datetime.now(timezone.utc).timestamp() - hu_path.stat().st_mtime
                ) / 86400
                need_refresh = age_days > 7
            if need_refresh:
                hu = run_halal_screener()
                log.info(
                    f"Halal screener refreshed: {hu.get('n_passing', 0)}/{hu.get('n_candidates', 0)} passing"
                )
            else:
                log.info("Halal screener cache fresh (< 7d) — skipped")
        except Exception as _hu_exc:
            log.warning(f"Halal screener failed (non-fatal): {_hu_exc}")

    # ── Metals Rebalancer (Phase XI Stage 59) ────────────────────────────────
    if not dry_run:
        try:
            from scripts.metals_rebalancer import run_rebalancer
            mr = run_rebalancer()
            log.info(
                f"Metals rebalancer: action={mr.get('candidate_action')}  "
                f"open_trades={mr.get('n_open_trades', 0)}  "
                f"closed={mr.get('n_closed_this_run', 0)}"
            )
        except Exception as _mr_exc:
            log.warning(f"Metals rebalancer failed (non-fatal): {_mr_exc}")

    # ── Continuous Training Orchestrator (Phase XI Stage 60) ─────────────────
    # Daily scheduler. Runs only due tasks; respects each engine's interval.
    if not dry_run:
        try:
            from scripts.continuous_trainer import run_orchestrator
            ct = run_orchestrator()
            log.info(
                f"Continuous training: ran {ct.get('n_run', 0)}/{ct.get('n_tasks', 0)}  "
                f"success {ct.get('n_successful', 0)}  fail {ct.get('n_failed', 0)}  "
                f"skipped {ct.get('n_skipped', 0)}"
            )
            if ct.get("promotion"):
                p = ct["promotion"]
                log.info(
                    f"⭐ CHAMPION PROMOTED: {p.get('previous_champion')} → "
                    f"{p.get('new_champion')}  Δ={p.get('score_delta', 0):+.4f}"
                )
        except Exception as _ct_exc:
            log.warning(f"Continuous training failed (non-fatal): {_ct_exc}")

    # ── Economic Calendar (Phase XIII Stage 67) ──────────────────────────────
    if not dry_run:
        try:
            from scripts.economic_calendar import run_economic_calendar
            ec = run_economic_calendar()
            ne = ec.get("next_event") or {}
            log.info(
                f"Economic calendar: guard={ec.get('position_guard')}  "
                f"next={ne.get('kind', 'n/a')} T+{ne.get('days_until', '?')}d  "
                f"blocked_today={ec.get('blocked_today', False)}"
            )
        except Exception as _ec_exc:
            log.warning(f"Economic calendar failed (non-fatal): {_ec_exc}")

    # ── Earnings Calendar (Phase XIII Stage 68) ──────────────────────────────
    if not dry_run:
        try:
            from scripts.earnings_calendar import run_earnings_calendar
            er = run_earnings_calendar()
            log.info(
                f"Earnings calendar: checked {er.get('tickers_checked', 0)}  "
                f"blocked {er.get('n_blocked', 0)} "
                f"({', '.join(er.get('blackout_tickers', [])) or 'none'})"
            )
        except Exception as _er_exc:
            log.warning(f"Earnings calendar failed (non-fatal): {_er_exc}")

    # ── Data Quality Monitor (Phase XIII Stage 69) ───────────────────────────
    if not dry_run:
        try:
            from scripts.data_quality import run_data_quality
            dq = run_data_quality()
            log.info(
                f"Data quality: {dq.get('overall_status')}  "
                f"checks={dq.get('n_checks', 0)}  failures={dq.get('n_failures', 0)}"
            )
        except Exception as _dq_exc:
            log.warning(f"Data quality failed (non-fatal): {_dq_exc}")

    # ── Pairs Trader (Phase XIII Stage 70) ───────────────────────────────────
    if not dry_run:
        try:
            from scripts.pairs_trader import run_pairs_trader
            pt = run_pairs_trader()
            log.info(
                f"Pairs trader: signals {pt.get('n_signals', 0)}  "
                f"trades_built {pt.get('n_trades_built', 0)}  open {pt.get('n_open', 0)}"
            )
        except Exception as _pt_exc:
            log.warning(f"Pairs trader failed (non-fatal): {_pt_exc}")

    # ── Daily P&L Tracker (Phase XIII Stage 71) ──────────────────────────────
    if not dry_run:
        try:
            from scripts.pnl_tracker import run_pnl_tracker
            pn = run_pnl_tracker()
            log.info(
                f"P&L tracker: NAV ${pn.get('latest_nav_usd', 0):,.2f}  "
                f"day {pn.get('day_pnl_pct', 0):+.3f}%  "
                f"cum {pn.get('cumulative_return_pct', 0):+.2f}%  "
                f"history={pn.get('n_history', 0)} rows"
            )
        except Exception as _pn_exc:
            log.warning(f"P&L tracker failed (non-fatal): {_pn_exc}")

    # ── Trade Basket (Phase XII Stage 63) ────────────────────────────────────
    if not dry_run:
        try:
            from scripts.trade_basket import run_trade_basket
            tb = run_trade_basket()
            log.info(
                f"Trade basket: long {tb.get('n_long', 0)}  short {tb.get('n_short', 0)}  "
                f"cash {tb.get('cash_pct', 0):.1f}%"
            )
        except Exception as _tb_exc:
            log.warning(f"Trade basket failed (non-fatal): {_tb_exc}")

    # ── Trade Idea Generator (Phase XI Stage 57) ─────────────────────────────
    # MUST run AFTER all other engines so it consumes their outputs.
    if not dry_run:
        try:
            from scripts.trade_idea_generator import run_trade_idea
            shadow = _read_shadow_portfolio()
            pf_val = float(shadow.get("portfolio_value", STARTING_CAPITAL))
            ti = run_trade_idea(portfolio_usd=pf_val)
            tc = ti.get("trade_card", {})
            log.info(
                f"TRADE IDEA: {tc.get('side')} {tc.get('ticker')}  "
                f"size {tc.get('size_pct', 0):.2f}%  "
                f"conviction {tc.get('conviction')}  "
                f"IBKR-ready={ti.get('ibkr_ready', False)}  "
                f"flags={ti.get('n_risk_flags', 0)}"
            )
        except Exception as _ti_exc:
            log.warning(f"Trade idea generator failed (non-fatal): {_ti_exc}")

    # ── MRM Champion (Phase X Stage 52) ──────────────────────────────────────
    if not dry_run:
        try:
            from scripts.mrm_champion import run_mrm
            mr = run_mrm()
            log.info(
                f"MRM: champion={mr.get('current_champion')}  "
                f"decision={mr.get('decision')}  "
                f"delta={mr.get('score_delta', 0):+.4f}"
            )
        except Exception as _mrm_exc:
            log.warning(f"MRM failed (non-fatal): {_mrm_exc}")

    # ── Strategy Sandbox (Phase X Stage 53) ──────────────────────────────────
    if not dry_run:
        try:
            from scripts.strategy_sandbox import run_sandbox
            ss = run_sandbox()
            log.info(
                f"Strategy sandbox: best={ss.get('best_strategy')}  "
                f"strategies={ss.get('strategies')}"
            )
        except Exception as _ss_exc:
            log.warning(f"Strategy sandbox failed (non-fatal): {_ss_exc}")

    # ── Tear Sheet + Form PF Lite (Phase X Stages 54-55) ─────────────────────
    if not dry_run:
        try:
            from scripts.tear_sheet import run_tear_sheet
            ts = run_tear_sheet()
            log.info(
                f"Tear sheet: {ts.get('n_sections', 0)} sections → "
                f"{ts.get('tear_sheet_path')}"
            )
        except Exception as _ts_exc:
            log.warning(f"Tear sheet failed (non-fatal): {_ts_exc}")

    # ── DeepSeek Briefing (Phase X Stage 56 — north-star UI surface) ─────────
    # Generates today's executive briefing. The interactive Q&A path runs
    # on-demand from the CLI or UI.
    if not dry_run:
        try:
            from scripts.deepseek_explainer import briefing
            br = briefing()
            log.info(
                f"DeepSeek briefing: model={br.get('model')}  "
                f"tokens={br.get('total_tokens', 0)}  "
                f"engines={br.get('dossier_keys') and len(br['dossier_keys'])}"
            )
        except Exception as _ds_exc:
            log.warning(f"DeepSeek briefing failed (non-fatal): {_ds_exc}")

    # ── Alert Router (Phase IX Stage 48) ─────────────────────────────────────
    if not dry_run:
        try:
            from scripts.alert_router import run_alert_router
            ar = run_alert_router(send=True)
            log.info(f"Alerts dispatched: {ar.get('n_alerts', 0)}")
        except Exception as _ar_exc:
            log.warning(f"Alert router failed (non-fatal): {_ar_exc}")

    # ── Audit Trail (Phase IX Stage 49) ──────────────────────────────────────
    if not dry_run:
        try:
            from scripts.audit_trail import record_pipeline_snapshot, status_snapshot
            recorded = record_pipeline_snapshot()
            snap = status_snapshot()
            log.info(
                f"Audit: recorded={len(recorded)}  total={snap.get('n_total')}  "
                f"chain_valid={snap.get('chain_valid')}"
            )
        except Exception as _at_exc:
            log.warning(f"Audit trail failed (non-fatal): {_at_exc}")

    # ── DR Snapshot (Phase IX Stage 50) ──────────────────────────────────────
    if not dry_run:
        try:
            from scripts.dr_backup import run_dr_backup
            dr = run_dr_backup()
            snap = dr.get("snapshot", {})
            log.info(
                f"DR backup: {snap.get('size_mb', 0)} MB  "
                f"encrypted={snap.get('encrypted', False)}  "
                f"total snapshots={dr.get('n_snapshots', 0)}"
            )
        except Exception as _dr_exc:
            log.warning(f"DR backup failed (non-fatal): {_dr_exc}")

    # ── Latency Profile (Phase IX Stage 51) ──────────────────────────────────
    if not dry_run:
        try:
            from scripts.latency_profiler import run_latency_profiler
            lp = run_latency_profiler()
            log.info(
                f"Latency: total {lp.get('current_run_total_s', 0)}s  "
                f"slowest {lp.get('slowest_stage')} ({lp.get('slowest_duration_s', 0)}s)  "
                f"recommendations={lp.get('n_recommendations', 0)}"
            )
        except Exception as _lp_exc:
            log.warning(f"Latency profiler failed (non-fatal): {_lp_exc}")

    # ── Brinson Attribution (Phase VIII Stage 43) ─────────────────────────────
    if not dry_run:
        try:
            from scripts.brinson_attribution import run_brinson_attribution
            br = run_brinson_attribution()
            log.info(
                f"Brinson: portfolio={br.get('portfolio_return_pct', 0):+.2f}%  "
                f"benchmark={br.get('benchmark_return_pct', 0):+.2f}%  "
                f"excess={br.get('excess_return_pct', 0):+.2f}%  "
                f"alloc={br.get('allocation_effect_pct', 0):+.2f}% / "
                f"sel={br.get('selection_effect_pct', 0):+.2f}%"
            )
        except Exception as _br_exc:
            log.warning(f"Brinson failed (non-fatal): {_br_exc}")

    # ── Fama-French (Phase VIII Stage 44) ─────────────────────────────────────
    if not dry_run:
        try:
            from scripts.fama_french import run_fama_french
            ff = run_fama_french()
            log.info(
                f"Fama-French: α={ff.get('alpha_annualised_pct', 0):+.2f}%/y "
                f"t={ff.get('alpha_t_stat', 0):+.2f}  "
                f"R²={ff.get('r_squared', 0):.3f}  "
                f"IR={ff.get('information_ratio', 0):+.2f}  "
                f"dominant={ff.get('dominant_factor')}"
            )
        except Exception as _ff_exc:
            log.warning(f"Fama-French failed (non-fatal): {_ff_exc}")

    # ── IC/IR Tracker (Phase VIII Stage 45) ───────────────────────────────────
    if not dry_run:
        try:
            from scripts.ic_ir_tracker import run_ic_ir_tracker
            ii = run_ic_ir_tracker()
            log.info(
                f"IC/IR: top={ii.get('ranked_by_ir', ['n/a'])[0]}  "
                f"deployable={ii.get('n_deployable', 0)}/{len(ii.get('per_signal', {}))}  "
                f"signals: {', '.join(ii.get('deployable_signals', [])) or 'none'}"
            )
        except Exception as _ii_exc:
            log.warning(f"IC/IR tracker failed (non-fatal): {_ii_exc}")

    # ── Decision Quality (Phase VIII Stage 46) ────────────────────────────────
    if not dry_run:
        try:
            from scripts.decision_quality import run_decision_quality
            dq = run_decision_quality()
            log.info(
                f"Decision quality: best={dq.get('best_signal')}  "
                f"Brier={dq.get('best_brier', 0):.4f}  "
                f"skill={dq.get('best_skill', 0):+.4f}  "
                f"realised_pos={dq.get('realised_positive_rate', 0):.3f}"
            )
        except Exception as _dq_exc:
            log.warning(f"Decision quality failed (non-fatal): {_dq_exc}")

    # ── Conformal Intervals (Phase VII Stage 42) ──────────────────────────────
    # Split-conformal Ridge regression; distribution-free prediction intervals.
    if not dry_run:
        try:
            from scripts.conformal_intervals import run_conformal_intervals
            co = run_conformal_intervals()
            intervals = co.get("intervals", {})
            a05 = intervals.get("alpha_05", {})
            log.info(
                f"Conformal: forecast={co.get('latest_forecast_pct', 0):+.2f}%  "
                f"95% width={a05.get('interval_width_pct', 0):.2f}%  "
                f"emp_cov={a05.get('empirical_coverage', 0):.3f}  "
                f"valid={a05.get('valid', False)}"
            )
        except Exception as _co_exc:
            log.warning(f"Conformal failed (non-fatal): {_co_exc}")

    # ── RL Sizing Agent (Phase VII Stage 41) ──────────────────────────────────
    # Tabular Q-learning sizer over (signal × vol × trend) state; train/test split.
    if not dry_run:
        try:
            from scripts.rl_sizing_agent import run_rl_sizing
            rl = run_rl_sizing()
            log.info(
                f"RL sizing: test Sharpe={rl.get('rl_test_sharpe', 0):+.3f} "
                f"vs baseline {rl.get('baseline_test_sharpe', 0):+.3f}  "
                f"lift={rl.get('test_lift_sharpe', 0):+.3f}  "
                f"avg_action={rl.get('avg_action_test', 0):.2f}"
            )
        except Exception as _rl_exc:
            log.warning(f"RL sizing failed (non-fatal): {_rl_exc}")

    # ── Ensemble Stacking (Phase VII Stage 40) ────────────────────────────────
    # 3 base learners (logit / RF / GB) on diverse features, meta = logistic.
    if not dry_run:
        try:
            from scripts.ensemble_stacking import run_ensemble_stacking
            es = run_ensemble_stacking()
            meta = es.get("meta_metrics", {})
            best = es.get("best_single_base", {})
            log.info(
                f"Stacking: meta acc={meta.get('accuracy', 0):.4f} auc={meta.get('auc', 0):.4f}  "
                f"best base acc={best.get('accuracy', 0):.4f} auc={best.get('auc', 0):.4f}  "
                f"lift auc={es.get('stacking_lift', {}).get('auc', 0):+.4f}"
            )
        except Exception as _es_exc:
            log.warning(f"Ensemble stacking failed (non-fatal): {_es_exc}")

    # ── Purged K-Fold CV (Phase VII Stage 39) ─────────────────────────────────
    # López de Prado purged + embargoed K-fold; per-fold Sharpe stability check.
    if not dry_run:
        try:
            from scripts.purged_kfold import run_purged_kfold
            pk = run_purged_kfold()
            s = pk.get("summary", {})
            log.info(
                f"Purged K-fold: mean Sharpe={s.get('mean_sharpe', 0):+.3f}±"
                f"{s.get('std_sharpe', 0):.3f}  "
                f"stability={s.get('stability_ratio', 0)}  "
                f"positive folds={s.get('n_positive_folds', 0)}/{pk.get('n_splits', 0)}"
            )
        except Exception as _pk_exc:
            log.warning(f"Purged K-fold failed (non-fatal): {_pk_exc}")

    # ── Options Pricer (Phase VI Stage 34) ────────────────────────────────────
    if not dry_run:
        try:
            from scripts.options_pricer import run_options_pricer
            op = run_options_pricer()
            log.info(
                f"Options: spot=${op.get('spot', 0):,.2f}  "
                f"σ={op.get('sigma', 0):.2%}  "
                f"ATM call ${op.get('atm_call', {}).get('price', 0):.2f}  "
                f"ATM put ${op.get('atm_put', {}).get('price', 0):.2f}  "
                f"γ={op.get('atm_call', {}).get('gamma', 0):.4f}"
            )
        except Exception as _op_exc:
            log.warning(f"Options pricer failed (non-fatal): {_op_exc}")

    # ── Tail Hedge Overlay (Phase VI Stage 35) ────────────────────────────────
    if not dry_run:
        try:
            from scripts.tail_hedge_overlay import run_tail_hedge
            shadow = _read_shadow_portfolio()
            pf_val = float(shadow.get("portfolio_value", STARTING_CAPITAL))
            th = run_tail_hedge(notional_usd=pf_val)
            log.info(
                f"Tail hedge: contracts={th.get('contracts_needed')}  "
                f"drag={th.get('annual_drag_pct', 0):.3f}%  "
                f"residual CVaR={th.get('residual_cvar_pct', 0):.2f}%  "
                f"binding={th.get('constraint_binding', False)}"
            )
        except Exception as _th_exc:
            log.warning(f"Tail hedge failed (non-fatal): {_th_exc}")

    # ── Carry Analyzer (Phase VI Stage 36) ────────────────────────────────────
    if not dry_run:
        try:
            from scripts.carry_analyzer import run_carry_analyzer
            ca = run_carry_analyzer()
            log.info(
                f"Carry: fair={ca.get('carry', {}).get('fair_pct', 0):+.2f}%  "
                f"burden={ca.get('carry', {}).get('burden')}  "
                f"21d excess={ca.get('excess_vs_carry', {}).get('21d_pct', 0):+.2f}%"
            )
        except Exception as _ca_exc:
            log.warning(f"Carry analyzer failed (non-fatal): {_ca_exc}")

    # ── Term Structure (Phase VI Stage 37) ────────────────────────────────────
    if not dry_run:
        try:
            from scripts.term_structure import run_term_structure
            ts = run_term_structure()
            log.info(
                f"Term structure: {ts.get('curve_shape')}  "
                f"slope={ts.get('overall_slope_pct', 0):+.2f}%  "
                f"roll={ts.get('roll_yield_pct', 0):+.2f}%  "
                f"R²={ts.get('curve_r_squared', 0):.3f}  "
                f"stress={ts.get('stress_flag', False)}"
            )
        except Exception as _ts_exc:
            log.warning(f"Term structure failed (non-fatal): {_ts_exc}")

    # ── Macro Nowcasting Composite ────────────────────────────────────────────
    # Fuses 8 macro components (real yields, copper-gold, COT, DCC stress,
    # geo, vol regime, ETF flows, sentiment) into a single bull/bear score.
    if not dry_run:
        try:
            from scripts.macro_nowcast import run_macro_nowcast
            mn = run_macro_nowcast()
            top = mn.get("top_drivers", [{}])[0] if mn.get("top_drivers") else {}
            log.info(
                f"Macro nowcast: regime={mn.get('regime')}  "
                f"score={mn.get('composite_score', 0):+.3f}  "
                f"top_driver={top.get('name', 'n/a')}={top.get('value', 0):+.2f}"
            )
        except Exception as _mn_exc:
            log.warning(f"Macro nowcast failed (non-fatal): {_mn_exc}")

    # ── ETF Flow Tracker ──────────────────────────────────────────────────────
    # Signed-dollar-volume proxy across GLD/IAU/GLDM/BAR + SLV/SIVR.
    if not dry_run:
        try:
            from scripts.etf_flow_tracker import run_etf_flow_tracker
            ef = run_etf_flow_tracker()
            log.info(
                f"ETF flows: headline={ef.get('headline')}  "
                f"gold 7d=${ef.get('gold_bucket', {}).get('flow_7d_usd', 0)/1e9:+.2f}B  "
                f"silver 7d=${ef.get('silver_bucket', {}).get('flow_7d_usd', 0)/1e9:+.2f}B  "
                f"divergent={ef.get('divergent', False)}"
            )
        except Exception as _ef_exc:
            log.warning(f"ETF flow tracker failed (non-fatal): {_ef_exc}")

    # ── Geopolitical Event Detector ───────────────────────────────────────────
    # Spike + regime classification on Perplexity pplx_geo_risk history.
    if not dry_run:
        try:
            from scripts.geopolitical_detector import run_geopolitical_detector
            ge = run_geopolitical_detector()
            log.info(
                f"Geopolitical: regime={ge.get('regime')}  "
                f"score={ge.get('current_score', 0):.2f}  "
                f"priority={ge.get('priority')}  "
                f"event={ge.get('event_flag', False)}"
            )
        except Exception as _ge_exc:
            log.warning(f"Geopolitical detector failed (non-fatal): {_ge_exc}")

    # ── Central Bank Speech Analyzer ──────────────────────────────────────────
    # Pulls live Perplexity macro scores (fed/geo/phys/macro) and persists a
    # historical CSV for trend / regime-shift analytics.
    if not dry_run:
        try:
            from scripts.cb_speech_analyzer import run_cb_speech
            cb = run_cb_speech(record=True)
            log.info(
                f"CB speech: fed_regime={cb.get('fed_regime')}  "
                f"pplx_fed={cb.get('fed_latest', 0):+.2f}  "
                f"geo_risk={cb.get('series', {}).get('pplx_geo_risk', {}).get('latest', 0):.2f}  "
                f"shift={cb.get('regime_shift_detected', False)}"
            )
        except Exception as _cb_exc:
            log.warning(f"CB speech analyzer failed (non-fatal): {_cb_exc}")

    # ── News Sentiment NLP ────────────────────────────────────────────────────
    # Aggregates oracle_history.csv into per-ticker sentiment regimes.
    if not dry_run:
        try:
            from scripts.news_sentiment import run_news_sentiment
            ns = run_news_sentiment()
            agg = ns.get("aggregate", {})
            log.info(
                f"News sentiment: consensus={agg.get('consensus_regime', 'n/a')}  "
                f"avg={agg.get('avg_sentiment', 0):.3f}  "
                f"dispersion={agg.get('dispersion', 0):.3f}  "
                f"divergent={agg.get('divergent', False)}"
            )
        except Exception as _ns_exc:
            log.warning(f"News sentiment failed (non-fatal): {_ns_exc}")

    # ── Strategy Capacity Analyzer ────────────────────────────────────────────
    # AUM caps where impact + physical premium erode the alpha by 25/50/100%.
    if not dry_run:
        try:
            from scripts.capacity_analyzer import run_capacity_analyzer
            cap = run_capacity_analyzer(ticker=ticker)
            log.info(
                f"Capacity: α={cap.get('expected_alpha_pct', 0):+.2f}%  "
                f"physical_cap_25%=${cap.get('thresholds_physical', {}).get('decay_25pct', {}).get('aum_cap_usd', 0):,.0f}  "
                f"paper_cap_25%=${cap.get('thresholds_paper', {}).get('decay_25pct', {}).get('aum_cap_usd', 0):,.0f}"
            )
        except Exception as _cap_exc:
            log.warning(f"Capacity analyzer failed (non-fatal): {_cap_exc}")

    # ── Dynamic Stop-Loss Optimizer ───────────────────────────────────────────
    # Backtests 4 stop methods (ATR 2.0×, ATR 2.5×, chandelier, %-based);
    # vol regime decides which is recommended for current entries.
    if not dry_run:
        try:
            from scripts.stop_loss_optimizer import run_stop_loss_optimizer
            sl = run_stop_loss_optimizer(ticker=ticker)
            log.info(
                f"Stop-loss: method={sl.get('final_recommendation')}  "
                f"price=${sl.get('final_stop_price', 0):,.2f}  "
                f"dist={sl.get('final_stop_distance_pct', 0):.2f}%  "
                f"regime={sl.get('vol_regime')}"
            )
        except Exception as _sl_exc:
            log.warning(f"Stop-loss optimizer failed (non-fatal): {_sl_exc}")

    # ── Adverse Selection Detector ────────────────────────────────────────────
    # Per-hour-of-day markout proxy from hourly bars; flags worst/best
    # sessions for liquidity-demanding execution.
    if not dry_run:
        try:
            from scripts.adverse_selection import run_adverse_selection
            asv = run_adverse_selection(ticker=ticker)
            sess = asv.get("session_summary", {})
            worst_sess = max(sess.items(), key=lambda kv: kv[1].get("avg_adverse_score", 0)) if sess else (None, {})
            log.info(
                f"Adverse selection: worst={worst_sess[0]} "
                f"(score {worst_sess[1].get('avg_adverse_score', 0):.2f})  "
                f"top-risk hour UTC={asv.get('worst_hours', [{}])[0].get('hour_utc') if asv.get('worst_hours') else 'n/a'}"
            )
        except Exception as _as_exc:
            log.warning(f"Adverse selection failed (non-fatal): {_as_exc}")

    # ── Smart Order Router ────────────────────────────────────────────────────
    # Picks TWAP / VWAP / POV / SINGLE based on participation rate and urgency,
    # generates slice schedule and per-algo expected cost.
    if not dry_run:
        try:
            from scripts.smart_order_router import run_smart_order_router
            sor = run_smart_order_router(ticker=ticker, notional=25_000.0, urgency="medium")
            rc = sor.get("recommended_cost", {})
            log.info(
                f"SOR: algo={sor.get('recommended_algo')}  "
                f"horizon={sor.get('horizon_minutes')}m  "
                f"slices={sor.get('n_slices')}  "
                f"cost={rc.get('total_oneway_bps', 0):.1f}bp  "
                f"partic={sor.get('participation_60min', 0):.2f}%"
            )
        except Exception as _sor_exc:
            log.warning(f"Smart order router failed (non-fatal): {_sor_exc}")

    # ── Bayesian Model Averaging ──────────────────────────────────────────────
    # Posterior weights on alpha sources from rolling directional log-Bayes
    # factors vs a 50/50 null; mathematically coherent ensemble.
    if not dry_run:
        try:
            from scripts.bayesian_model_averaging import run_bma
            b = run_bma(ticker=ticker)
            bt = b.get("backtest", {}).get("bma", {})
            log.info(
                f"BMA: top={b.get('top_source')}  "
                f"Sharpe={bt.get('sharpe', 0):+.3f}  "
                f"vol={bt.get('ann_vol_pct', 0):.2f}%  "
                f"DD={bt.get('max_drawdown_pct', 0):.1f}%"
            )
        except Exception as _bma_exc:
            log.warning(f"BMA failed (non-fatal): {_bma_exc}")

    # ── Macro Regime Classifier ───────────────────────────────────────────────
    # 4-quadrant growth × inflation classifier; Goldilocks / Reflation /
    # Stagflation / Deflation each maps to specific asset tilts.
    if not dry_run:
        try:
            from scripts.macro_regime import run_macro_regime
            mr = run_macro_regime()
            log.info(
                f"Macro regime: {mr.get('quadrant')} "
                f"(conf {mr.get('confidence', 0):.1%})  "
                f"G={mr.get('growth_score', 0):+.2f}  "
                f"I={mr.get('inflation_score', 0):+.2f}"
            )
        except Exception as _mr_exc:
            log.warning(f"Macro regime failed (non-fatal): {_mr_exc}")

    # ── Structural Break Detector ─────────────────────────────────────────────
    # CUSUM + binary-segmentation mean breaks + variance regime breaks on gold.
    if not dry_run:
        try:
            from scripts.structural_breaks import run_structural_breaks
            sb = run_structural_breaks(ticker=ticker)
            s = sb.get("summary", {})
            log.info(
                f"Structural breaks: CUSUM={'BREAK' if s.get('cusum_break') else 'stable'}  "
                f"mean={s.get('n_mean_breaks', 0)}  var={s.get('n_variance_breaks', 0)}  "
                f"last={s.get('most_recent_break', 'none')} "
                f"({s.get('days_since_last_break', 0)}d ago)"
            )
        except Exception as _sb_exc:
            log.warning(f"Structural breaks failed (non-fatal): {_sb_exc}")

    # ── DCC-GARCH Dynamic Conditional Correlations ────────────────────────────
    # Time-varying correlation matrix for the multi-asset universe; flags
    # pairs that have moved >2σ from their long-run correlation (regime stress).
    if not dry_run:
        try:
            from scripts.dcc_garch import run_dcc_garch
            dcc = run_dcc_garch()
            log.info(
                f"DCC-GARCH: a={dcc.get('dcc_params', {}).get('a', 0):.4f} "
                f"b={dcc.get('dcc_params', {}).get('b', 0):.4f}  "
                f"avg corr now={dcc.get('avg_pairwise_corr_now', 0):+.3f} "
                f"vs LR={dcc.get('avg_pairwise_corr_long_run', 0):+.3f}  "
                f"stressed={dcc.get('n_stressed', 0)} pairs"
            )
        except Exception as _dcc_exc:
            log.warning(f"DCC-GARCH failed (non-fatal): {_dcc_exc}")

    # ── Vol Targeting & Risk Budgeting ────────────────────────────────────────
    # Maps current realised vol to a target via leverage; allocates risk budget
    # across alpha sources by IR (edge-tilt) and inverse-vol (equal-risk).
    if not dry_run:
        try:
            from scripts.vol_target_budget import run_vol_target
            vt = run_vol_target()
            log.info(
                f"Vol target: current={vt.get('current_vol_pct', 0):.2f}%  "
                f"target={vt.get('target_vol_pct', 12):.1f}%  "
                f"lev={vt.get('leverage_capped', 1):.2f}×  "
                f"action={vt.get('guidance', {}).get('leverage_action', 'MAINTAIN')}"
            )
        except Exception as _vt_exc:
            log.warning(f"Vol target failed (non-fatal): {_vt_exc}")

    # ── Mean-CVaR LP Optimizer ────────────────────────────────────────────────
    # Rockafellar-Uryasev tail-aware allocator; minimises CVaR_α with optional
    # mean-return target. Pure scipy.linprog on the daily return scenarios.
    if not dry_run:
        try:
            from scripts.mean_cvar_optimizer import run_mean_cvar
            mc = run_mean_cvar()
            mn = mc.get("metrics", {}).get("min_cvar", {})
            mn2 = mc.get("metrics", {}).get("mean_cvar", {})
            log.info(
                f"Mean-CVaR: min_cvar Sharpe={mn.get('sharpe', 0):+.3f} vol={mn.get('ann_vol_pct', 0):.2f}%  "
                f"mean_cvar Sharpe={mn2.get('sharpe', 0):+.3f} vol={mn2.get('ann_vol_pct', 0):.2f}% "
                f"CVaR={mn2.get('cvar_pct', 0):.2f}%/d"
            )
        except Exception as _mc_exc:
            log.warning(f"Mean-CVaR optimizer failed (non-fatal): {_mc_exc}")

    # ── Black-Litterman Bayesian Portfolio ───────────────────────────────────
    # Combines CAPM equilibrium prior with macro / LSTM / pair / momentum views;
    # produces posterior μ and optimal weights.
    if not dry_run:
        try:
            from scripts.black_litterman import run_black_litterman
            bl = run_black_litterman()
            m = bl.get("metrics", {}).get("black_litterman", {})
            log.info(
                f"Black-Litterman: views={bl.get('n_views', 0)}  "
                f"Sharpe={m.get('sharpe', 0):+.3f}  "
                f"vol={m.get('ann_vol_pct', 0):.2f}%  "
                f"DD={m.get('max_drawdown_pct', 0):.1f}%"
            )
        except Exception as _bl_exc:
            log.warning(f"Black-Litterman failed (non-fatal): {_bl_exc}")

    # ── HRP Allocator ─────────────────────────────────────────────────────────
    # López de Prado Hierarchical Risk Parity on multi-asset universe;
    # compares HRP vs equal-weight vs inverse-vol on Sharpe / DD / div ratio.
    if not dry_run:
        try:
            from scripts.hrp_allocator import run_hrp
            hrp = run_hrp()
            m = hrp.get("metrics", {})
            log.info(
                f"HRP: Sharpe={m.get('hrp', {}).get('sharpe', 0):+.3f}  "
                f"vol={m.get('hrp', {}).get('ann_vol_pct', 0):.2f}%  "
                f"DD={m.get('hrp', {}).get('max_drawdown_pct', 0):.1f}%  "
                f"div={m.get('hrp', {}).get('diversification_ratio', 0):.2f}x  "
                f"vs EqW Sharpe={m.get('equal_weight', {}).get('sharpe', 0):+.3f}"
            )
        except Exception as _hrp_exc:
            log.warning(f"HRP allocator failed (non-fatal): {_hrp_exc}")

    # ── Signal Decay Half-Life Analyzer ───────────────────────────────────────
    # Per-signal IC across horizons + exponential decay fit + alpha-decay flag.
    if not dry_run:
        try:
            from scripts.signal_decay import run_signal_decay
            sd = run_signal_decay(ticker=ticker)
            ranked = sd.get("ranked_by_ic", [])
            top = ranked[0] if ranked else None
            top_data = sd.get("signals", {}).get(top, {}) if top else {}
            n_dec = len(sd.get("decaying_signals", []))
            n_str = len(sd.get("strengthening_signals", []))
            log.info(
                f"Signal decay: top={top}  "
                f"IC={top_data.get('best_horizon_ic', 0):+.4f}  "
                f"½-life={top_data.get('half_life_days', 0):.1f}d  "
                f"rebal={top_data.get('rebalance_days', 0)}d  "
                f"decay flags={n_dec}↓ {n_str}↑"
            )
        except Exception as _sd_exc:
            log.warning(f"Signal decay failed (non-fatal): {_sd_exc}")

    # ── Volatility Surface Monitor ────────────────────────────────────────────
    # Realized vol term structure across 5d / 10d / 21d / 63d / 252d horizons,
    # vol-of-vol, regime classification, curve shape, and Kelly/stop guidance.
    if not dry_run:
        try:
            from scripts.vol_surface import run_vol_surface
            vs = run_vol_surface(ticker=ticker)
            log.info(
                f"Vol surface: regime={vs.get('vol_regime')} phase={vs.get('phase')} "
                f"curve={vs.get('curve_shape')} kelly_mult={vs.get('actions', {}).get('kelly_fraction_multiplier', 1):.2f}× "
                f"stop={vs.get('actions', {}).get('stop_atr_multiplier', 2):.2f}×ATR"
            )
        except Exception as _vs_exc:
            log.warning(f"Vol surface failed (non-fatal): {_vs_exc}")

    # ── Alpha Attribution Engine ──────────────────────────────────────────────
    # Decomposes return potential into 5 independent alpha sources (LSTM
    # momentum, macro overlay, regime filter, technical, mean-reversion);
    # reports per-source Sharpe / IR and Choueifaty-Coignard diversification.
    if not dry_run:
        try:
            from scripts.alpha_attribution import run_alpha_attribution
            aa = run_alpha_attribution(ticker=ticker)
            ranked = aa.get("ranked_by_sharpe", [])
            top = ranked[0] if ranked else None
            top_full = aa.get("full_history", {}).get(top, {}) if top else {}
            top_ir = aa.get("information_ratios", {}).get(top, {}) if top else {}
            es = aa.get("combined", {}).get("equal_weight_summary", {})
            div = aa.get("combined", {}).get("diversification_ratio", 0)
            log.info(
                f"Alpha attribution: top={top}  "
                f"Sharpe={top_full.get('sharpe', 0):+.3f}  "
                f"IR={top_ir.get('information_ratio', 0):+.3f}  "
                f"blend Sharpe={es.get('sharpe', 0):+.3f}  "
                f"div={div:.2f}x"
            )
        except Exception as _aa_exc:
            log.warning(f"Alpha attribution failed (non-fatal): {_aa_exc}")

    # ── Transaction Cost Analysis (TCA) ──────────────────────────────────────
    # Almgren-Chriss + UAE physical premium; portfolio-level across metals + halal universe.
    if not dry_run:
        try:
            from scripts.transaction_cost_model import run_portfolio_tca
            shadow = _read_shadow_portfolio()
            pf_val = float(shadow.get("portfolio_value", STARTING_CAPITAL))
            tca = run_portfolio_tca(top_n=10, portfolio_value=pf_val, metals_physical=True)
            agg = tca.get("aggregate", {})
            log.info(f"TCA: avg one-way {agg.get('avg_oneway_cost_bps', 0):.1f}bp  "
                     f"range {agg.get('min_oneway_cost_bps', 0):.1f}-{agg.get('max_oneway_cost_bps', 0):.1f}bp  "
                     f"({agg.get('vol_regime', 'unknown')})")
        except Exception as _tca_exc:
            log.warning(f"TCA engine failed (non-fatal): {_tca_exc}")

    # ── Executive Briefing (Chief of Staff) ───────────────────────────────────
    if not dry_run:
        try:
            from scripts.executive_briefer import run_briefer
            run_briefer()
        except Exception as _eb_exc:
            log.warning(f"Executive briefer failed (non-fatal): {_eb_exc}")

    # ── Position Reconciliation (Phase XII Stage 64) ─────────────────────────
    if not dry_run:
        try:
            from scripts.position_reconciler import run_reconciler
            pr = run_reconciler(use_ibkr=False)  # shadow-only by default
            log.info(
                f"Position reconciler: status={pr.get('status')}  "
                f"broker={pr.get('diff', {}).get('n_broker', 0)}  "
                f"shadow={pr.get('diff', {}).get('n_shadow', 0)}  "
                f"drift={pr.get('diff', {}).get('n_drift_total', 0)}"
            )
        except Exception as _pr_exc:
            log.warning(f"Position reconciler failed (non-fatal): {_pr_exc}")

    # ── Slippage Tracker (Phase XII Stage 65) ────────────────────────────────
    if not dry_run:
        try:
            from scripts.slippage_tracker import run_slippage
            slp = run_slippage()
            log.info(
                f"Slippage: paired={slp.get('n_paired', 0)}/{slp.get('n_audit_orders', 0)}  "
                f"match_rate={slp.get('fill_match_rate_pct', 0):.1f}%  "
                f"avg={slp.get('all_time', {}).get('avg_slippage_bps', 'n/a')}"
            )
        except Exception as _slp_exc:
            log.warning(f"Slippage tracker failed (non-fatal): {_slp_exc}")

    # ── System Health (Phase XII Stage 62) — last, after everything refreshes ─
    if not dry_run:
        try:
            from scripts.system_health import run_system_health
            sh = run_system_health()
            log.info(
                f"System health: {sh.get('overall_status')}  "
                f"flags={sh.get('n_flags_total', 0)}  "
                f"critical={sh.get('by_severity', {}).get('CRITICAL', 0)}  "
                f"high={sh.get('by_severity', {}).get('HIGH', 0)}"
            )
        except Exception as _sh_exc:
            log.warning(f"System health failed (non-fatal): {_sh_exc}")

    # ── Operator Runbook (Phase XII Stage 66) ────────────────────────────────
    if not dry_run:
        try:
            from scripts.operator_runbook import run_operator_runbook
            rb = run_operator_runbook()
            log.info(
                f"Runbook: {rb.get('n_checklist_ok')}/{rb.get('n_checklist')} ✅  "
                f"flags={len(rb.get('risk_flags', []))}  "
                f"→ {rb.get('markdown_path')}"
            )
        except Exception as _rb_exc:
            log.warning(f"Operator runbook failed (non-fatal): {_rb_exc}")

    # ── Phase XVIII: Crisis Detector (Stage 80) ──────────────────────────────
    # MUST run before the selector so the live selector can read the
    # crisis_tier from data/crisis_detector.json.
    if not dry_run:
        try:
            from scripts.crisis_detector import classify_from_engines
            cr = classify_from_engines()
            log.info(
                f"Crisis detector: tier={cr.get('tier')}  "
                f"score={cr.get('score')}  "
                f"price={cr.get('price_score')}  bump={cr.get('engine_bump')}"
            )
        except Exception as _cr_exc:
            log.warning(f"Crisis detector failed (non-fatal): {_cr_exc}")

    # ── Phase XIV: Alpha Stacker (Stage 72) ──────────────────────────────────
    # Meta-engine that fuses every signal-producing engine's JSON output into
    # a single conviction-weighted decision.
    if not dry_run:
        try:
            from scripts.alpha_stacker import run_alpha_stacker
            asr = run_alpha_stacker()
            d = asr.get("decision", {}) or {}
            log.info(
                f"Alpha Stacker: {d.get('direction')}  "
                f"conviction={d.get('conviction_score', 0):+.4f} [{d.get('conviction_tier')}]  "
                f"size={d.get('recommended_size_pct', 0):.1f}%  "
                f"signals={asr.get('stack', {}).get('n_signals', 0)}  "
                f"flags={asr.get('n_risk_flags', 0)}"
            )
        except Exception as _as_exc:
            log.warning(f"Alpha Stacker failed (non-fatal): {_as_exc}")

    # ── Phase XIV: Strategy Selector (Stage 73) ──────────────────────────────
    if not dry_run:
        try:
            from scripts.strategy_selector import run_strategy_selector
            ssr = run_strategy_selector()
            log.info(
                f"Strategy Selector: {ssr.get('strategy')}  "
                f"final size={ssr.get('final_size_pct', 0):.2f}%  "
                f"regime={ssr.get('regime_context', {}).get('hmm_state')}/"
                f"{ssr.get('regime_context', {}).get('vol_regime')}"
            )
        except Exception as _ss_exc:
            log.warning(f"Strategy Selector failed (non-fatal): {_ss_exc}")

    # ── Phase XIV: Performance Targeter (Stage 74) ───────────────────────────
    if not dry_run:
        try:
            from scripts.performance_targeter import run_performance_targeter
            pt = run_performance_targeter()
            log.info(
                f"Targeter: MTD {pt.get('progress', {}).get('actual_progress_pct', 0):+.2f}%  "
                f"vs expected {pt.get('progress', {}).get('expected_progress_pct', 0):+.2f}%  "
                f"({pt.get('progress', {}).get('track_status')})  "
                f"risk×{pt.get('risk_multiplier', {}).get('final', 1.0):.2f}"
            )
        except Exception as _pt_exc:
            log.warning(f"Performance Targeter failed (non-fatal): {_pt_exc}")

    # ── Phase XIV: Multi-Strategy Paper Trader (Stage 75) ────────────────────
    # MUST run AFTER alpha_stacker + strategy_selector + performance_targeter
    # because it consumes their outputs.
    if not dry_run:
        try:
            from scripts.multi_strategy_trader import run_multi_strategy_trader
            mst = run_multi_strategy_trader()
            _hs = mst.get("hedge_state") or {}
            _hedge_desc = (
                f"{_hs.get('instrument')}@{_hs.get('allocation_pct', 0):.0f}%"
                + (f"({_hs.get('sub_tag')})" if _hs.get("sub_tag") else "")
                if _hs.get("instrument") else "off"
            )
            log.info(
                f"Multi-Strategy Trader: strategy={mst.get('strategy')}  "
                f"equity=${mst.get('book_equity_usd', 0):,.0f}  "
                f"lifetime={mst.get('lifetime_pl_pct', 0):+.2f}%  "
                f"open={mst.get('n_open', 0)}  "
                f"new={mst.get('n_new_trades', 0)}  exits={mst.get('n_exits_this_run', 0)}  "
                f"hedge={_hedge_desc}"
            )
        except Exception as _mst_exc:
            log.warning(f"Multi-Strategy Trader failed (non-fatal): {_mst_exc}")

    # ── Phase XXIII: Conviction Weights Optimizer (Stage 84) ─────────────────
    # Re-weights the five conviction components from historical IC + Sharpe.
    # The strategy backtester reads data/conviction_weights.json at runtime, so
    # this must run BEFORE the backtester. Refresh weekly.
    if not dry_run:
        try:
            from scripts.conviction_weights_optimizer import run as run_cwo
            cwo_path = ROOT / "data" / "conviction_weights.json"
            need_run = True
            if cwo_path.exists():
                age_h = (
                    datetime.now(timezone.utc).timestamp() - cwo_path.stat().st_mtime
                ) / 3600
                need_run = age_h > 168  # 7 days
            if need_run:
                cwo = run_cwo()
                top = max(
                    (cwo.get("weights") or {}).items(),
                    key=lambda kv: kv[1], default=("—", 0),
                )
                log.info(
                    f"Conviction weights: top_component={top[0]} ({top[1]:.3f})  "
                    f"components={len(cwo.get('weights', {}))}"
                )
            else:
                log.info("Conviction weights cache fresh (< 7d) — skipped")
        except Exception as _cwo_exc:
            log.warning(f"Conviction weights optimizer failed (non-fatal): {_cwo_exc}")

    # ── Phase XV: Strategy Backtester (Stage 76) ─────────────────────────────
    # Historical replay of the Phase XIV stack vs ~2 years of gold prices.
    # Refresh weekly: if last run was within 6 days, skip.
    if not dry_run:
        try:
            from scripts.strategy_backtester import run_backtest
            bt_path = ROOT / "data" / "strategy_backtest.json"
            need_run = True
            if bt_path.exists():
                age_h = (
                    datetime.now(timezone.utc).timestamp() - bt_path.stat().st_mtime
                ) / 3600
                need_run = age_h > 144  # 6 days
            if need_run:
                bt = run_backtest()
                log.info(
                    f"Backtest: {bt.get('achievability_verdict')}  "
                    f"cum={bt['performance']['cum_return_pct']:+.2f}%  "
                    f"Sharpe={bt['performance']['sharpe']}  "
                    f"DD={bt['performance']['max_drawdown_pct']:+.2f}%  "
                    f"months@target {bt['monthly']['n_at_or_above_target']}/{bt['monthly']['n_months']}"
                )
            else:
                log.info("Backtest cache fresh (< 6d) — skipped")
        except Exception as _bt_exc:
            log.warning(f"Backtester failed (non-fatal): {_bt_exc}")

    # ── Phase XXVI: ML Conviction PoC (Stage 1) ──────────────────────────────
    # Tests whether LightGBM out-of-sample IC ≥ 0.05 AND Sharpe ≥ rule+0.15.
    # Expensive (downloads 25y of data on first run; cached thereafter). Refresh
    # weekly. The follow-on walk-forward (Stage c) runs immediately after if the
    # predictions CSV exists.
    if not dry_run:
        try:
            from scripts.ml_conviction_poc import run as run_ml_poc
            poc_path = ROOT / "data" / "ml_conviction_poc.json"
            pred_path = ROOT / "data" / "ml_conviction_predictions.csv"
            need_run = True
            if poc_path.exists():
                age_h = (
                    datetime.now(timezone.utc).timestamp() - poc_path.stat().st_mtime
                ) / 3600
                need_run = age_h > 168  # 7 days
            if need_run:
                poc = run_ml_poc()
                cp = poc.get("comparison", {})
                log.info(
                    f"ML Conviction PoC: gate={'PASS' if cp.get('gate_passed') else 'FAIL'}  "
                    f"OOS_IC={cp.get('oos_ic_ml')}  "
                    f"Sharpe_ML={cp.get('sharpe_ml')}  "
                    f"Sharpe_rule={cp.get('sharpe_rule')}"
                )
            else:
                log.info("ML Conviction PoC cache fresh (< 7d) — skipped")

            # Walk-forward runs immediately after if predictions file is present
            if pred_path.exists():
                from scripts.ml_walk_forward import run as run_ml_wf
                wf_path = ROOT / "data" / "ml_walk_forward.json"
                need_wf = need_run or not wf_path.exists()
                if not need_wf and wf_path.exists():
                    age_wf = (
                        datetime.now(timezone.utc).timestamp() - wf_path.stat().st_mtime
                    ) / 3600
                    need_wf = age_wf > 168
                if need_wf:
                    wf = run_ml_wf()
                    log.info(
                        f"ML Walk-Forward: gate={'PASS' if wf.get('gate_passed') else 'FAIL'}  "
                        f"verdict={wf.get('verdict')}  "
                        f"median_ann={wf.get('ml', {}).get('median_ann_pct')}%  "
                        f"avg_sharpe={wf.get('ml', {}).get('avg_sharpe')}"
                    )
                else:
                    log.info("ML Walk-Forward cache fresh (< 7d) — skipped")
        except Exception as _ml_exc:
            log.warning(f"ML Conviction pipeline failed (non-fatal): {_ml_exc}")

    # ── Phase XV: Multi-Asset Backtester (Stage 77) ──────────────────────────
    # Halal-equity book + metals book + combined; vs SPY + 60/40 benchmarks.
    # Refresh weekly.
    if not dry_run:
        try:
            from scripts.multi_asset_backtester import run_multi_asset_backtest
            ma_path = ROOT / "data" / "multi_asset_backtest.json"
            need_run = True
            if ma_path.exists():
                age_h = (
                    datetime.now(timezone.utc).timestamp() - ma_path.stat().st_mtime
                ) / 3600
                need_run = age_h > 144
            if need_run:
                ma = run_multi_asset_backtest()
                c = ma["books"]["combined"]
                spy = ma["benchmarks"]["spy_buyhold"]
                log.info(
                    f"Multi-asset backtest: {ma.get('verdict')}  "
                    f"combined ann={c.get('annualised_pct')}% Sharpe={c.get('sharpe')} "
                    f"DD={c.get('max_drawdown_pct')}%  vs SPY {spy.get('annualised_pct')}% "
                    f"({ma['delta_vs_spy']['annualised_pp']:+.2f}pp)"
                )
            else:
                log.info("Multi-asset backtest cache fresh (< 6d) — skipped")
        except Exception as _ma_exc:
            log.warning(f"Multi-asset backtester failed (non-fatal): {_ma_exc}")

    # ── Phase XX: Multi-Asset Stress Backtester ──────────────────────────────
    # Applies 8 historical crisis windows to the combined metals + halal-equity
    # book. Refresh monthly — pulls long history (cached). This produces the
    # input that treasury_overlay_stress_eval needs for the Phase XXV-F gate.
    if not dry_run:
        try:
            from scripts.multi_asset_stress_backtester import run as run_mast
            mast_path = ROOT / "data" / "multi_asset_stress_backtest.json"
            need_run = True
            if mast_path.exists():
                age_h = (
                    datetime.now(timezone.utc).timestamp() - mast_path.stat().st_mtime
                ) / 3600
                need_run = age_h > 720  # 30 days
            if need_run:
                mast = run_mast()
                log.info(
                    f"Multi-asset stress backtest: verdict={mast.get('verdict')}  "
                    f"windows={mast.get('n_windows')}  note={str(mast.get('note',''))[:80]}"
                )
            else:
                log.info("Multi-asset stress backtest cache fresh (< 30d) — skipped")

            # Treasury overlay stress eval: always runs after the stress backtest
            # refreshes, or when its own output is stale. Feeds the Phase XXV-F gate.
            if mast_path.exists():
                from scripts.treasury_overlay_stress_eval import run as run_tos
                tos_path = ROOT / "data" / "treasury_overlay_stress_eval.json"
                need_tos = need_run or not tos_path.exists()
                if not need_tos and tos_path.exists():
                    age_tos = (
                        datetime.now(timezone.utc).timestamp() - tos_path.stat().st_mtime
                    ) / 3600
                    need_tos = age_tos > 720
                if need_tos:
                    tos = run_tos()
                    log.info(
                        f"Treasury overlay stress eval (Phase XXV-b): "
                        f"verdict={tos.get('verdict')}  "
                        f"delta_sharpe={tos.get('delta_avg_sharpe'):+.3f}  "
                        f"rescued={tos.get('n_rescued')}  regressed={tos.get('n_regressed')}"
                    )
                else:
                    log.info("Treasury overlay stress eval cache fresh (< 30d) — skipped")
        except Exception as _mast_exc:
            log.warning(f"Multi-asset stress / treasury overlay eval failed (non-fatal): {_mast_exc}")

    # ── Phase XVII: Stress Backtester (Stage 79) ─────────────────────────────
    # Multi-regime stress test across 8 historical crisis windows.
    # Refresh monthly — it pulls 25y of history (cached) but the slicing
    # logic only matters when we change the rule cascade.
    if not dry_run:
        try:
            from scripts.stress_backtester import run_stress_test
            st_path = ROOT / "data" / "stress_backtest.json"
            need_run = True
            if st_path.exists():
                age_h = (
                    datetime.now(timezone.utc).timestamp() - st_path.stat().st_mtime
                ) / 3600
                need_run = age_h > 720  # 30 days
            if need_run:
                st = run_stress_test()
                a = st["aggregate"]
                log.info(
                    f"Stress test: {a.get('verdict')}  "
                    f"avg-Sharpe={a.get('avg_sharpe')}  "
                    f"worst-DD={a.get('worst_max_dd')}%  "
                    f"windows={st.get('n_valid')}/{st.get('n_windows')}  "
                    f"best={a.get('best_window')}  worst={a.get('worst_window')}"
                )
            else:
                log.info("Stress test cache fresh (< 30d) — skipped")
        except Exception as _st_exc:
            log.warning(f"Stress backtester failed (non-fatal): {_st_exc}")

    # ── Telegram heartbeat ────────────────────────────────────────────────────
    try:
        from scripts.telegram_notifier import send_heartbeat
        send_heartbeat(state)
        log.info("Telegram heartbeat sent.")
    except Exception as _tg_exc:
        log.warning(f"Telegram heartbeat failed (non-fatal): {_tg_exc}")


if __name__ == "__main__":
    main()
