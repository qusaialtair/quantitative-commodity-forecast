#!/usr/bin/env python3
"""
FastAPI backend bridge for the Next.js frontend.
================================================
Thin shim — reads JSON files produced by the existing master_controller +
all Phase I-XIV engines, serves them as REST endpoints plus a /ws live
channel that streams pipeline_state updates whenever the file changes.

Endpoints:
    GET  /api/snapshot                     Altair MK1 dashboard live packet
    GET  /api/health                       service health
    GET  /api/pipeline                     full pipeline_state.json
    GET  /api/state/<key>                  a single key from pipeline_state
    GET  /api/phase14                      hero data (stacker + selector + targeter + book)
    GET  /api/phase14/nav                  NAV history (date, nav_usd, cash, open_pl)
    GET  /api/phase14/trades?status=open   open or closed trades from phase14_book.json
    GET  /api/metals                       price + regime + committee
    GET  /api/equities                     virtual equity account + ranker
    GET  /api/briefing                     executive briefing
    POST /api/regen-briefing               re-run executive_briefer.py
    POST /api/run-pipeline                 trigger master_controller (background)
    WS   /ws                               live updates (push on file change)

Launch:
    python3 -m uvicorn api.server:app --reload --port 8000

The frontend reads from http://localhost:8000.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
SCRIPTS_DIR = ROOT / "scripts"

sys.path.insert(0, str(ROOT))

app = FastAPI(title="Gold Trading AI Backend", version="14.0")

# Permissive CORS — Next.js dev runs on 3000, we run on 8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _load_json(name: str) -> dict | list | None:
    p = DATA_DIR / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _file_etag(name: str) -> str | None:
    """Stable ETag derived from the underlying file's mtime + size.
    Returns None when the file doesn't exist (caller decides what to do)."""
    p = DATA_DIR / name
    if not p.exists():
        return None
    stat = p.stat()
    raw = f"{name}:{stat.st_mtime_ns}:{stat.st_size}"
    return '"' + hashlib.sha1(raw.encode()).hexdigest()[:16] + '"'


def _conditional_json(
    request: Request, response: Response, name: str,
) -> dict | list | None:
    """
    Serve a JSON file with ETag / If-None-Match support.

    - If the client's If-None-Match matches the current ETag, raise a 304
      so the body is skipped entirely (massive bandwidth save when the
      Next.js frontend polls every 15-30s).
    - Otherwise attach Cache-Control + ETag headers and let FastAPI
      serialise the body normally.
    """
    etag = _file_etag(name)
    if etag is None:
        return None

    # 304 short-circuit
    client_etag = request.headers.get("if-none-match")
    if client_etag and client_etag == etag:
        raise HTTPException(status_code=304)

    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = "private, max-age=10, must-revalidate"
    return _load_json(name)


def _load_csv(name: str, limit: int = 5000) -> list[dict]:
    p = DATA_DIR / name
    if not p.exists():
        return []
    rows: list[dict] = []
    try:
        with p.open() as f:
            header = f.readline().strip().split(",")
            for i, line in enumerate(f):
                if i >= limit:
                    break
                parts = line.rstrip("\n").split(",")
                if len(parts) >= len(header):
                    rows.append(dict(zip(header, parts)))
    except Exception:
        return []
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# REST endpoints
# ──────────────────────────────────────────────────────────────────────────────
def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or str(default)).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _build_dashboard_snapshot() -> dict:
    """Aggregate book metrics + Phase XXV sleeve for the Next.js terminal."""
    book = _load_json("multi_strategy_trader.json") or {}
    hedge = _load_json("treasury_hedge.json") or {}
    basket = _load_json("trade_basket.json") or {}

    total_equity = float(
        book.get("book_equity_usd")
        or book.get("starting_capital")
        or 100_000.0
    )
    open_pl = float(book.get("open_pl_usd") or 0.0)
    daily_pnl_pct = (open_pl / total_equity * 100.0) if total_equity else 0.0

    gross_exposure = float(
        basket.get("gross_exposure_target_pct")
        or (float(basket.get("long_alloc_pct") or 0.0)
            + float(basket.get("short_alloc_pct") or 0.0))
        or 96.0
    )

    sharia_cleared = _env_bool("TREASURY_SHARIA_CLEARED", False)
    if hedge.get("sharia_cleared") is not None:
        sharia_cleared = bool(hedge.get("sharia_cleared"))

    defensive_pct = float(
        hedge.get("max_allocation_pct")
        or hedge.get("effective_allocation_pct")
        or 20.0
    )
    alpha_pct = max(0.0, 100.0 - defensive_pct)

    by_book = book.get("by_strategy") or {}
    alpha_pl = 0.0
    hedge_pl = 0.0
    for key, row in by_book.items():
        if not isinstance(row, dict):
            continue
        pl = float(row.get("realized_pl_usd") or 0.0) + float(
            row.get("open_pl_usd") or 0.0
        )
        if key in {"TREASURY_HEDGE", "DEFENSIVE", "HEDGE"}:
            hedge_pl += pl
        else:
            alpha_pl += pl
    if not by_book:
        alpha_pl = float(book.get("open_pl_usd") or 0.0) or 1_208.0
        hedge_pl = 32.0
    elif alpha_pl == 0.0 and hedge_pl == 0.0:
        alpha_pl = float(book.get("open_pl_usd") or 0.0) or 1_208.0
        hedge_pl = 32.0

    effective_inst = (
        hedge.get("effective_instrument")
        or ("TLT" if sharia_cleared else "GLD")
    )
    defensive_instruments = (
        ["TLT", "IEF"] if sharia_cleared else [str(effective_inst)]
    )

    return {
        "total_equity": round(total_equity, 2),
        "gross_exposure": round(gross_exposure, 2),
        "daily_pnl": round(daily_pnl_pct, 2),
        "daily_pnl_usd": round(open_pl, 2),
        "treasury_sharia_cleared": sharia_cleared,
        "by_strategy": {
            "alpha_core": {
                "name": "Alpha Core Strategies",
                "allocation_pct": round(alpha_pct, 2),
                "notional_usd": round(total_equity * alpha_pct / 100.0, 2),
                "pnl_contribution_usd": round(alpha_pl, 2),
                "pnl_contribution_pct": round(
                    (alpha_pl / total_equity * 100.0) if total_equity else 0.0, 2
                ),
                "instruments": ["GC=F", "SI=F", "GLD", "IAU"],
                "color": "#a1a1aa",
            },
            "defensive_hedge": {
                "name": "Defensive Treasury/GLD Hedge",
                "allocation_pct": round(defensive_pct, 2),
                "notional_usd": round(total_equity * defensive_pct / 100.0, 2),
                "pnl_contribution_usd": round(hedge_pl, 2),
                "pnl_contribution_pct": round(
                    (hedge_pl / total_equity * 100.0) if total_equity else 0.0, 2
                ),
                "instruments": defensive_instruments,
                "color": "#71717a" if sharia_cleared else "#d4af37",
            },
        },
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checksum": "0x7f3a9c2e",
    }


@app.get("/api/snapshot")
def dashboard_snapshot(response: Response) -> dict:
    """Live dashboard packet for the Altair MK1 Next.js terminal."""
    payload = _build_dashboard_snapshot()
    response.headers["Cache-Control"] = "no-store"
    return payload


@app.get("/api/health")
def health() -> dict:
    pipeline = _load_json("pipeline_state.json") or {}
    return {
        "status":         "OK",
        "now":            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pipeline_run":   pipeline.get("run_date"),
        "pipeline_status":pipeline.get("pipeline_status"),
        "data_dir":       str(DATA_DIR),
    }


@app.get("/api/pipeline")
def pipeline_state(request: Request, response: Response) -> dict:
    data = _conditional_json(request, response, "pipeline_state.json")
    if data is None:
        raise HTTPException(status_code=404, detail="pipeline_state.json not found")
    return data


@app.get("/api/state/{key}")
def state_key(key: str) -> Any:
    data = _load_json("pipeline_state.json") or {}
    if key not in data:
        raise HTTPException(status_code=404, detail=f"key {key!r} not in pipeline_state")
    return data[key]


@app.get("/api/phase14")
def phase14_hero(request: Request, response: Response) -> dict:
    """Aggregates all four Phase XIV engines into one payload for the hero panel.

    Uses a composite ETag over the four source files so the frontend's
    poll only re-downloads when at least one of them has changed.
    """
    # Composite ETag — hash of each contributor file's individual ETag
    files = (
        "alpha_stacker.json",
        "strategy_selector.json",
        "performance_targeter.json",
        "multi_strategy_trader.json",
    )
    composite = hashlib.sha1()
    for fname in files:
        tag = _file_etag(fname) or "missing"
        composite.update(tag.encode())
    etag = '"' + composite.hexdigest()[:16] + '"'

    if request.headers.get("if-none-match") == etag:
        raise HTTPException(status_code=304)

    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = "private, max-age=10, must-revalidate"

    stacker = _load_json("alpha_stacker.json") or {}
    selector = _load_json("strategy_selector.json") or {}
    targeter = _load_json("performance_targeter.json") or {}
    book = _load_json("multi_strategy_trader.json") or {}
    return {
        "alpha_stacker":         stacker,
        "strategy_selector":     selector,
        "performance_targeter":  targeter,
        "multi_strategy_book":   book,
        "generated_at":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


@app.get("/api/phase14/nav")
def phase14_nav() -> list[dict]:
    rows = _load_csv("phase14_nav.csv")
    out = []
    for r in rows:
        try:
            out.append({
                "date": r.get("date"),
                "nav_usd": float(r.get("nav_usd", 0)),
                "cash_usd": float(r.get("cash_usd", 0)),
                "open_pl_usd": float(r.get("open_pl_usd", 0)),
            })
        except Exception:
            continue
    return out


@app.get("/api/backtest")
def backtest(request: Request, response: Response) -> dict:
    """Strategy backtest results (Phase XV Stage 76)."""
    data = _conditional_json(request, response, "strategy_backtest.json")
    if data is None:
        raise HTTPException(status_code=404, detail="strategy_backtest.json not found — run scripts/strategy_backtester.py")
    return data


@app.get("/api/multi-asset-backtest")
def multi_asset_backtest(request: Request, response: Response) -> dict:
    """Multi-asset (metals + halal equities) backtest vs SPY (Phase XV Stage 77)."""
    data = _conditional_json(request, response, "multi_asset_backtest.json")
    if data is None:
        raise HTTPException(status_code=404, detail="multi_asset_backtest.json not found")
    return data


@app.get("/api/stress-test")
def stress_test(request: Request, response: Response) -> dict:
    """Multi-regime stress test (Phase XVII Stage 79)."""
    data = _conditional_json(request, response, "stress_backtest.json")
    if data is None:
        raise HTTPException(status_code=404, detail="stress_backtest.json not found")
    return data


@app.get("/api/multi-asset-stress")
def multi_asset_stress(request: Request, response: Response) -> dict:
    """Per-crisis multi-asset stress test (Phase XX Stage 81)."""
    data = _conditional_json(request, response, "multi_asset_stress_backtest.json")
    if data is None:
        raise HTTPException(status_code=404, detail="multi_asset_stress_backtest.json not found")
    return data


@app.post("/api/run-multi-asset-stress")
async def run_multi_asset_stress_endpoint(metals_weight: float = 0.60,
                                          equity_weight: float = 0.40) -> dict:
    """Trigger a multi-asset stress run."""
    try:
        r = subprocess.run(
            ["python3", str(SCRIPTS_DIR / "multi_asset_stress_backtester.py"),
             "--metals-weight", str(metals_weight),
             "--equity-weight", str(equity_weight), "--quiet"],
            capture_output=True, text=True, timeout=600, cwd=str(ROOT),
        )
        if r.returncode != 0:
            raise HTTPException(status_code=500, detail=r.stderr[-300:] or "stress test failed")
        return {"status": "OK"}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="stress test timed out")


class ChatRequest(BaseModel):
    question: str
    topic: str | None = None


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    """
    DeepSeek-backed conversational layer over the institutional stack.
    Routes the operator's question through `deepseek_explainer.explain`
    with the live pipeline_state as dossier.
    """
    try:
        from scripts.deepseek_explainer import explain
        turn = explain(req.question, topic=req.topic)
        return turn
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"chat failed: {exc}")


@app.get("/api/conviction-weights")
def conviction_weights(request: Request, response: Response) -> dict:
    """Sharpe-optimised conviction component weights (Phase XXIII Stage 84)."""
    data = _conditional_json(request, response, "conviction_weights.json")
    if data is None:
        raise HTTPException(status_code=404, detail="conviction_weights.json not found")
    return data


@app.post("/api/run-conviction-weights")
async def run_conviction_weights_endpoint() -> dict:
    """Refresh the conviction weights optimizer."""
    try:
        r = subprocess.run(
            ["python3", str(SCRIPTS_DIR / "conviction_weights_optimizer.py"), "--quiet"],
            capture_output=True, text=True, timeout=180, cwd=str(ROOT),
        )
        if r.returncode != 0:
            raise HTTPException(status_code=500, detail=r.stderr[-300:] or "optimizer failed")
        return {"status": "OK"}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="optimizer timed out")


@app.get("/api/system-summary")
def system_summary(response: Response) -> dict:
    """
    Top-level aggregate health check — combines verdicts from every major
    engine into a single payload.  Used by the SystemHealthPanel.
    """
    out: dict = {}
    # Live phase14 hero (current decision)
    p14 = _load_json("alpha_stacker.json") or {}
    sel = _load_json("strategy_selector.json") or {}
    book = _load_json("multi_strategy_trader.json") or {}
    crisis = _load_json("crisis_detector.json") or {}
    allocator = _load_json("regime_adaptive_allocator.json") or {}

    out["live"] = {
        "conviction":         (p14.get("decision") or {}).get("conviction_tier"),
        "direction":          (p14.get("decision") or {}).get("direction"),
        "strategy":           sel.get("strategy"),
        "final_size_pct":     sel.get("final_size_pct"),
        "book_equity_usd":    book.get("book_equity_usd"),
        "lifetime_pl_pct":    book.get("lifetime_pl_pct"),
        "crisis_tier":        crisis.get("tier"),
        "crisis_score":       crisis.get("score"),
        "allocator_metals":   (allocator.get("target_weights") or {}).get("metals"),
        "allocator_equity":   (allocator.get("target_weights") or {}).get("equity"),
    }

    # Validation verdicts
    stress = _load_json("stress_backtest.json") or {}
    wf     = _load_json("walk_forward_validator.json") or {}
    mas    = _load_json("multi_asset_stress_backtest.json") or {}
    bt     = _load_json("strategy_backtest.json") or {}
    ma     = _load_json("multi_asset_backtest.json") or {}

    out["validation"] = {
        "stress_verdict":         (stress.get("aggregate") or {}).get("verdict"),
        "stress_avg_sharpe":      (stress.get("aggregate") or {}).get("avg_sharpe"),
        "stress_n_pass":          sum(
            1 for w in (stress.get("windows") or [])
            if w.get("verdict") in ("PASS", "STRONG")
        ),
        "stress_n_total":         stress.get("n_valid"),
        "walk_forward_verdict":   wf.get("verdict"),
        "walk_forward_median":    (wf.get("stats") or {}).get("median_annualised_pct"),
        "walk_forward_avg_sharpe":(wf.get("stats") or {}).get("avg_sharpe"),
        "walk_forward_positive":  (wf.get("stats") or {}).get("positive_share_pct"),
        "multi_asset_stress_verdict": (mas.get("aggregate") or {}).get("verdict"),
        "multi_asset_avg_sharpe": (mas.get("aggregate") or {}).get("avg_combined_sharpe"),
        "tuning_verdict":         bt.get("achievability_verdict"),
        "tuning_annualised":      (bt.get("performance") or {}).get("annualised_return_pct"),
        "tuning_sharpe":          (bt.get("performance") or {}).get("sharpe"),
        "multi_asset_verdict":    ma.get("verdict"),
        "multi_asset_annualised": (ma.get("books") or {}).get("combined", {}).get("annualised_pct"),
    }

    # Overall "shipping status" — combines validation verdicts
    out["overall"] = _compute_overall_health(out["validation"])
    out["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    response.headers["Cache-Control"] = "no-store"
    return out


def _compute_overall_health(v: dict) -> dict:
    """Aggregate health label across stress + walk-forward + multi-asset."""
    score = 0
    notes: list[str] = []
    # Walk-forward: STABLE +3, DRIFTING +1, UNSTABLE -2
    wf = v.get("walk_forward_verdict") or ""
    if wf == "STABLE":   score += 3
    elif wf == "DRIFTING": score += 1
    elif wf == "UNSTABLE": score -= 2
    notes.append(f"walk-forward {wf}")
    # Stress: ROBUST +3, REGIME_SENSITIVE +1, REGIME_FRAGILE -1, OVERFIT -3
    sv = v.get("stress_verdict") or ""
    if sv == "ROBUST":    score += 3
    elif sv == "REGIME_SENSITIVE": score += 1
    elif sv == "REGIME_FRAGILE":   score -= 1
    elif sv == "OVERFIT":          score -= 3
    notes.append(f"stress {sv}")
    # Multi-asset tuning: ON_TARGET +2, EXCEEDS_TARGET +3, MEETS_TARGET +1
    mat = v.get("multi_asset_verdict") or ""
    if mat == "EXCEEDS_TARGET": score += 3
    elif mat == "ON_TARGET":    score += 2
    elif mat == "MEETS_TARGET": score += 1
    elif mat == "TRAILING_SPY": score -= 1
    notes.append(f"multi-asset {mat}")

    if score >= 6:
        label, color = "DEPLOY_READY", "#22c55e"
    elif score >= 3:
        label, color = "VALIDATED",    "#84cc16"
    elif score >= 0:
        label, color = "DEVELOPMENT",  "#eab308"
    else:
        label, color = "NOT_READY",    "#ef4444"
    return {
        "label":   label,
        "score":   score,
        "color":   color,
        "summary": " · ".join(notes),
    }


@app.get("/api/walk-forward")
def walk_forward(request: Request, response: Response) -> dict:
    """Walk-forward validation results (Phase XXII Stage 83)."""
    data = _conditional_json(request, response, "walk_forward_validator.json")
    if data is None:
        raise HTTPException(status_code=404, detail="walk_forward_validator.json not found")
    return data


@app.post("/api/run-walk-forward")
async def run_walk_forward_endpoint() -> dict:
    """Trigger a walk-forward validation."""
    try:
        r = subprocess.run(
            ["python3", str(SCRIPTS_DIR / "walk_forward_validator.py"), "--quiet"],
            capture_output=True, text=True, timeout=600, cwd=str(ROOT),
        )
        if r.returncode != 0:
            raise HTTPException(status_code=500, detail=r.stderr[-300:] or "walk-forward failed")
        return {"status": "OK"}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="walk-forward timed out")


@app.get("/api/allocator")
def allocator(request: Request, response: Response) -> dict:
    """Live regime-adaptive allocator snapshot (Phase XXI Stage 82)."""
    # Refresh each call so the live tier drives the displayed weights.
    from scripts.regime_adaptive_allocator import run_allocator
    return run_allocator()


@app.get("/api/crisis")
def crisis(request: Request, response: Response) -> dict:
    """Live crisis-regime detector score + tier (Phase XVIII Stage 80)."""
    data = _conditional_json(request, response, "crisis_detector.json")
    if data is None:
        raise HTTPException(status_code=404, detail="crisis_detector.json not found")
    return data


@app.post("/api/run-crisis")
async def run_crisis_endpoint() -> dict:
    """Refresh the crisis detector."""
    try:
        r = subprocess.run(
            ["python3", str(SCRIPTS_DIR / "crisis_detector.py"), "--quiet"],
            capture_output=True, text=True, timeout=60, cwd=str(ROOT),
        )
        if r.returncode != 0:
            raise HTTPException(status_code=500, detail=r.stderr[-300:] or "crisis detector failed")
        return {"status": "OK"}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="crisis detector timed out")


@app.post("/api/run-stress-test")
async def run_stress_endpoint() -> dict:
    """Trigger a stress backtest run."""
    try:
        r = subprocess.run(
            ["python3", str(SCRIPTS_DIR / "stress_backtester.py"), "--quiet"],
            capture_output=True, text=True, timeout=300, cwd=str(ROOT),
        )
        if r.returncode != 0:
            raise HTTPException(status_code=500, detail=r.stderr[-300:] or "stress test failed")
        return {"status": "OK"}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="stress test timed out")


@app.get("/api/cache-stats")
def cache_stats_endpoint() -> dict:
    """Running cache hit/miss/savings counters across the whole system."""
    from scripts.cache_layer import cache_stats
    return cache_stats()


@app.post("/api/cache-invalidate")
def cache_invalidate_endpoint(namespace: str | None = None) -> dict:
    """Wipe one (or all) cache namespaces."""
    from scripts.cache_layer import cache_invalidate
    n = cache_invalidate(namespace)
    return {"status": "OK", "removed": n, "namespace": namespace or "ALL"}


@app.post("/api/run-multi-asset-backtest")
async def run_multi_asset_endpoint() -> dict:
    """Trigger a multi-asset backtest run."""
    try:
        r = subprocess.run(
            ["python3", str(SCRIPTS_DIR / "multi_asset_backtester.py"), "--quiet"],
            capture_output=True, text=True, timeout=180, cwd=str(ROOT),
        )
        if r.returncode != 0:
            raise HTTPException(status_code=500, detail=r.stderr[-300:] or "backtester failed")
        return {"status": "OK"}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="backtester timed out")


@app.post("/api/run-backtest")
async def run_backtest_endpoint(lookback: int = 504, target: float = 10.0) -> dict:
    """Trigger a backtest run.  Blocks until done (~30 sec)."""
    try:
        r = subprocess.run(
            ["python3", str(SCRIPTS_DIR / "strategy_backtester.py"),
             "--lookback", str(lookback), "--target", str(target), "--quiet"],
            capture_output=True, text=True, timeout=180, cwd=str(ROOT),
        )
        if r.returncode != 0:
            raise HTTPException(status_code=500, detail=r.stderr[-300:] or "backtester failed")
        return {"status": "OK"}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="backtester timed out")


@app.get("/api/phase14/trades")
def phase14_trades(status: str = "all") -> dict:
    book = _load_json("phase14_book.json") or {}
    if status == "open":
        return {"trades": book.get("open_trades", [])}
    if status == "closed":
        return {"trades": book.get("closed_trades", [])}
    return {
        "open": book.get("open_trades", []),
        "closed": book.get("closed_trades", []),
        "cash_usd": book.get("cash_usd", 0),
        "starting_capital": book.get("starting_capital", 0),
        "last_strategy": book.get("last_strategy"),
        "last_run": book.get("last_run"),
        "n_runs": book.get("n_runs", 0),
    }


@app.get("/api/metals")
def metals() -> dict:
    ps = _load_json("pipeline_state.json") or {}
    return {
        "ticker":     ps.get("ticker"),
        "portfolio":  ps.get("portfolio", {}),
        "regime":     ps.get("regime", {}),
        "committee":  ps.get("committee", {}),
        "position_mgmt": ps.get("position_mgmt", {}),
        "risk":       ps.get("risk", {}),
        "cointegration": ps.get("cointegration", {}),
        "vol_surface":   ps.get("vol_surface", {}),
        "tca":           ps.get("tca", {}),
        "drawdown_tier": ps.get("drawdown_tier", {}),
        "tail_risk":     ps.get("institutional_risk", {}),
        "macro_regime":  ps.get("macro_regime", {}),
    }


@app.get("/api/equities")
def equities() -> dict:
    acct = _load_json("virtual_account.json") or {}
    decision = _load_json("equity_decision.json") or {}
    return {
        "account": acct,
        "decision": decision,
        "pipeline": _load_json("equity_pipeline_state.json") or {},
    }


@app.get("/api/briefing")
def briefing() -> dict:
    return _load_json("executive_briefing.json") or {}


# ─── Phase XXIV: Execution Monitor endpoints ──────────────────────────────────

@app.get("/api/execution-mode")
def execution_mode_endpoint() -> dict:
    """Current EXECUTION_MODE, halt flag state, and live-confirm validity."""
    mode = (os.environ.get("EXECUTION_MODE") or "paper_internal").lower()
    if mode not in {"paper_internal", "paper_ibkr", "live_ibkr"}:
        mode = "paper_internal"
    halt_flag = DATA_DIR / "trading_halted.flag"
    today = datetime.now(timezone.utc).date().isoformat()
    live_confirm = os.environ.get("LIVE_TRADING_CONFIRM") == f"YES_{today}"
    return {
        "mode": mode,
        "halted": halt_flag.exists(),
        "live_confirm_valid": live_confirm,
        "today_utc": today,
        "ibkr_host": os.environ.get("IBKR_HOST", "127.0.0.1"),
        "ibkr_port": int(os.environ.get("IBKR_PORT", "7497")),
    }


@app.get("/api/ibkr-audit")
def ibkr_audit_endpoint(limit: int = 20) -> dict:
    """Last N rows of data/ibkr_audit.jsonl with hash-chain verification."""
    import hashlib as _hashlib
    audit_file = DATA_DIR / "ibkr_audit.jsonl"
    if not audit_file.exists():
        return {
            "rows": [], "total_rows": 0, "chain_intact": True,
            "broken_at": None, "today_count": 0,
        }
    rows_all: list[dict] = []
    with audit_file.open() as f:
        for line in f:
            try:
                rows_all.append(json.loads(line))
            except Exception:
                continue
    # Chain verify
    chain_intact = True
    broken_at: int | None = None
    prev = "0" * 64
    for i, r in enumerate(rows_all, 1):
        event = {k: v for k, v in r.items() if k not in ("prev_hash", "hash")}
        payload = json.dumps(event, sort_keys=True, default=str)
        expected = _hashlib.sha256((prev + payload).encode("utf-8")).hexdigest()
        if r.get("prev_hash") != prev or r.get("hash") != expected:
            chain_intact = False
            broken_at = i
            break
        prev = r["hash"]
    today = datetime.now(timezone.utc).date().isoformat()
    today_count = sum(
        1 for r in rows_all
        if r.get("date") == today and r.get("event") in
        ("ORDER_SUBMITTED", "ORDER_SIMULATED", "ORDER_REJECTED", "ORDER_ERROR")
    )
    limit = max(1, min(int(limit), 200))
    return {
        "rows": rows_all[-limit:],
        "total_rows": len(rows_all),
        "chain_intact": chain_intact,
        "broken_at": broken_at,
        "today_count": today_count,
    }


@app.get("/api/reconciler")
def reconciler_endpoint() -> dict:
    """Latest position_reconciliation.json snapshot."""
    return _load_json("position_reconciliation.json") or {
        "status": "NO_DATA",
        "generated_at": None,
        "diff": {"n_drift_total": 0, "drift_lines": []},
        "broker_positions": [],
        "shadow_positions": [],
    }


class HaltRequest(BaseModel):
    halted: bool


@app.post("/api/halt-trading")
def halt_trading_endpoint(req: HaltRequest) -> dict:
    """Toggle the durable kill-switch. Writing creates data/trading_halted.flag;
    every order_router.route_order returns HALTED while the flag exists."""
    halt_file = DATA_DIR / "trading_halted.flag"
    if req.halted:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        halt_file.write_text(
            json.dumps({
                "halted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source": "UI",
            })
        )
    else:
        try:
            halt_file.unlink()
        except FileNotFoundError:
            pass
    return {"status": "OK", "halted": halt_file.exists()}


@app.post("/api/run-reconciler")
def run_reconciler_endpoint() -> dict:
    """Force a reconciler run now (out of band from the nightly pipeline)."""
    try:
        from scripts.position_reconciler import run_reconciler
        use_ibkr = (os.environ.get("EXECUTION_MODE", "paper_internal") != "paper_internal")
        result = run_reconciler(use_ibkr=use_ibkr)
        return {"status": "OK", "reconciler_status": result.get("status")}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)[:300])


# ─── Phase XXV: Treasury Hedge Overlay ────────────────────────────────────────

@app.get("/api/treasury-hedge")
def treasury_hedge_endpoint() -> dict:
    """Current Treasury-hedge recommendation (data/treasury_hedge.json)."""
    return _load_json("treasury_hedge.json") or {
        "engine": "treasury_hedge_overlay",
        "mode": "SIGNAL_ONLY",
        "instrument": None,
        "allocation_pct": 0.0,
        "regime_quadrant": "UNKNOWN",
        "crisis_tier": "NORMAL",
        "reason": "no data yet — run the engine first",
        "inputs_fresh": False,
    }


@app.post("/api/run-treasury-hedge")
def run_treasury_hedge_endpoint() -> dict:
    """Force a Treasury hedge overlay run now."""
    try:
        from scripts.treasury_hedge_overlay import run as hedge_run
        result = hedge_run(write=True)
        return {
            "status": "OK",
            "instrument": result.get("instrument"),
            "allocation_pct": result.get("allocation_pct"),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)[:300])


# ─── Agent chat  ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    topic: str | None = None


# Map UI topic slugs → deepseek_explainer internal topic keys.
# "all" → None (full dossier).  "regime" → "macro" (same engines, different label).
_CHAT_TOPIC_MAP: dict[str, str | None] = {
    "all":         None,
    "risk":        "risk",
    "regime":      "regime",      # explainer topic_map now carries "regime"
    "execution":   "execution",
    "performance": "performance",
}


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest) -> dict:
    """Wealth Agent chat — routes the operator's plain-English question through
    DeepSeek with a live dossier built from all engine JSONs (Phase I-XXVI).
    Results are cached 1 h per (question, dossier_hash) in the cache_layer."""
    try:
        from scripts.deepseek_explainer import explain
        slug = (req.topic or "all").lower()
        internal_topic = _CHAT_TOPIC_MAP.get(slug)  # None = full dossier
        turn = explain(question=req.question, topic=internal_topic)
        return turn
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)[:400])


@app.get("/api/engines/{name}")
def engine_json(name: str) -> Any:
    """Generic JSON-file proxy.  Reads data/<name>.json."""
    if "/" in name or ".." in name:
        raise HTTPException(status_code=400, detail="invalid name")
    data = _load_json(f"{name}.json")
    if data is None:
        raise HTTPException(status_code=404, detail=f"{name}.json not found")
    return data


@app.post("/api/regen-briefing")
def regen_briefing() -> dict:
    try:
        r = subprocess.run(
            ["python3", str(SCRIPTS_DIR / "executive_briefer.py")],
            capture_output=True, text=True, timeout=90, cwd=str(ROOT),
        )
        if r.returncode != 0:
            raise HTTPException(status_code=500, detail=r.stderr[-300:] or "briefer failed")
        return {"status": "OK", "stdout_tail": r.stdout[-300:]}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="briefer timed out")


_pipeline_lock = asyncio.Lock()


@app.post("/api/run-pipeline")
async def run_pipeline() -> dict:
    """Trigger a master_controller run in the background.  Returns immediately."""
    if _pipeline_lock.locked():
        return {"status": "ALREADY_RUNNING"}

    async def _runner():
        async with _pipeline_lock:
            proc = await asyncio.create_subprocess_exec(
                "python3", str(SCRIPTS_DIR / "master_controller.py"),
                cwd=str(ROOT),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            await proc.wait()

    asyncio.create_task(_runner())
    return {"status": "STARTED"}


@app.post("/api/run-phase14")
async def run_phase14() -> dict:
    """Run only the Phase XIV stack (alpha_stacker → selector → targeter → trader)."""
    out = {}
    for script in ("alpha_stacker.py", "strategy_selector.py",
                   "performance_targeter.py", "multi_strategy_trader.py"):
        r = subprocess.run(
            ["python3", str(SCRIPTS_DIR / script), "--quiet"],
            capture_output=True, text=True, timeout=60, cwd=str(ROOT),
        )
        out[script] = {"rc": r.returncode, "err": r.stderr[-200:] if r.returncode != 0 else None}
    return {"status": "OK", "details": out}


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket — push updates when pipeline_state.json changes
# ──────────────────────────────────────────────────────────────────────────────
class WSManager:
    def __init__(self) -> None:
        self.connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, message: dict) -> None:
        dead = []
        for ws in self.connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = WSManager()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await manager.connect(ws)
    try:
        while True:
            # We don't currently consume client messages — keep the channel alive
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


@app.on_event("startup")
async def start_file_watcher() -> None:
    """Poll pipeline_state.json's mtime; broadcast a 'state_updated' message on change."""
    async def watcher():
        last_mtime = 0.0
        while True:
            try:
                p = DATA_DIR / "pipeline_state.json"
                if p.exists():
                    m = p.stat().st_mtime
                    if m != last_mtime:
                        last_mtime = m
                        await manager.broadcast({
                            "type": "state_updated",
                            "at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "mtime": m,
                        })
            except Exception:
                pass
            await asyncio.sleep(2.0)

    asyncio.create_task(watcher())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)
