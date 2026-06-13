#!/usr/bin/env python3
"""
Data Quality Monitor  (Phase XIII Stage 69)
=============================================
Verifies the live feeds the platform depends on. The system_health monitor
(Stage 62) checks engine output freshness; this engine drills down to the
RAW data layer.

Checks:
  1. yfinance liveness        fetch GC=F + SPY + DXY closes; non-zero, non-NaN
  2. alt_data freshness       last row of data/alt_data.csv < 72h old
  3. yf gap detection         any day-over-day gap > 10% in the last 21 days
  4. Perplexity cache         macro_narrative.json fresh, valid JSON, score in [0,1]
  5. NaN floor in shadow book gold_oz, cash_usd, portfolio_value all finite
  6. Audit log writeability   can we append to ibkr_audit.jsonl?

Outputs a per-check pass/fail + severity bucket.

Output: data/data_quality.json
"""
from __future__ import annotations

import argparse
import json
import sys
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import yfinance as yf
except ImportError:
    yf = None

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "data_quality.json"
ALT_CSV = DATA_DIR / "alt_data.csv"
PPLX_NARRATIVE = DATA_DIR / "perplexity_cache" / "gold" / "macro_narrative.json"
SHADOW_DB = DATA_DIR / "shadow_book.db"
AUDIT_LOG = DATA_DIR / "ibkr_audit.jsonl"

LINE_W = 62
SEP = "━" * LINE_W


def _check_yf_liveness() -> dict:
    if yf is None:
        return {"check": "yf_liveness", "ok": False, "severity": "CRITICAL",
                "detail": "yfinance not installed"}
    try:
        results = {}
        for t in ("GC=F", "SPY", "DX-Y.NYB"):
            raw = yf.download(t, period="5d", interval="1d",
                              progress=False, auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            close = raw["Close"].dropna()
            if len(close) == 0:
                return {"check": "yf_liveness", "ok": False, "severity": "CRITICAL",
                        "detail": f"{t} returned empty close series"}
            v = float(close.iloc[-1])
            if not np.isfinite(v) or v <= 0:
                return {"check": "yf_liveness", "ok": False, "severity": "CRITICAL",
                        "detail": f"{t} last close {v}"}
            results[t] = round(v, 2)
        return {"check": "yf_liveness", "ok": True, "severity": "OK",
                "detail": f"prices: {results}"}
    except Exception as exc:
        return {"check": "yf_liveness", "ok": False, "severity": "CRITICAL",
                "detail": f"yfinance error: {exc}"[:120]}


def _check_alt_data() -> dict:
    if not ALT_CSV.exists():
        return {"check": "alt_data", "ok": False, "severity": "MEDIUM",
                "detail": "alt_data.csv absent"}
    try:
        df = pd.read_csv(ALT_CSV, index_col=0, parse_dates=True)
        if len(df) == 0:
            return {"check": "alt_data", "ok": False, "severity": "MEDIUM",
                    "detail": "alt_data.csv empty"}
        last_ts = df.index.max()
        age_h = (datetime.now() - last_ts.to_pydatetime()).total_seconds() / 3600.0
        if age_h > 72:
            return {"check": "alt_data", "ok": False, "severity": "MEDIUM",
                    "detail": f"last row {age_h:.1f}h old (> 72h)"}
        return {"check": "alt_data", "ok": True, "severity": "OK",
                "detail": f"last row {age_h:.1f}h ago"}
    except Exception as exc:
        return {"check": "alt_data", "ok": False, "severity": "MEDIUM",
                "detail": f"parse error: {exc}"[:120]}


def _check_yf_gaps() -> dict:
    if yf is None:
        return {"check": "yf_gaps", "ok": False, "severity": "MEDIUM",
                "detail": "yfinance not installed"}
    try:
        raw = yf.download("GC=F", period="2mo", interval="1d",
                          progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        close = raw["Close"].dropna().tail(22)
        if len(close) < 5:
            return {"check": "yf_gaps", "ok": False, "severity": "MEDIUM",
                    "detail": "insufficient history"}
        returns = close.pct_change().abs()
        max_gap = float(returns.max())
        if max_gap > 0.10:
            gap_date = returns.idxmax()
            return {"check": "yf_gaps", "ok": False, "severity": "MEDIUM",
                    "detail": f"GC=F {max_gap:.2%} gap on {gap_date.date()}"}
        return {"check": "yf_gaps", "ok": True, "severity": "OK",
                "detail": f"max 21d gap {max_gap:.2%}"}
    except Exception as exc:
        return {"check": "yf_gaps", "ok": False, "severity": "MEDIUM",
                "detail": f"yf error: {exc}"[:120]}


def _check_perplexity_cache() -> dict:
    if not PPLX_NARRATIVE.exists():
        return {"check": "perplexity_cache", "ok": False, "severity": "MEDIUM",
                "detail": "macro_narrative.json absent"}
    try:
        data = json.loads(PPLX_NARRATIVE.read_text())
        score = data.get("score")
        ts = data.get("_ts")
        if score is None or not 0 <= float(score) <= 1:
            return {"check": "perplexity_cache", "ok": False, "severity": "MEDIUM",
                    "detail": f"score out of range: {score}"}
        if ts:
            age_h = (datetime.now().timestamp() - float(ts)) / 3600.0
            if age_h > 48:
                return {"check": "perplexity_cache", "ok": False, "severity": "MEDIUM",
                        "detail": f"cache {age_h:.1f}h old"}
        return {"check": "perplexity_cache", "ok": True, "severity": "OK",
                "detail": f"score {score:.2f}"}
    except Exception as exc:
        return {"check": "perplexity_cache", "ok": False, "severity": "MEDIUM",
                "detail": f"parse error: {exc}"[:120]}


def _check_shadow_book() -> dict:
    if not SHADOW_DB.exists():
        return {"check": "shadow_book", "ok": False, "severity": "HIGH",
                "detail": "shadow_book.db absent"}
    try:
        conn = sqlite3.connect(SHADOW_DB)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM portfolio_state WHERE id = 1"
        ).fetchone()
        conn.close()
        if not row:
            return {"check": "shadow_book", "ok": False, "severity": "HIGH",
                    "detail": "no portfolio_state row"}
        for key in ("gold_oz", "cash_usd", "portfolio_value"):
            v = row[key] if key in row.keys() else None
            if v is None or not np.isfinite(float(v)):
                return {"check": "shadow_book", "ok": False, "severity": "HIGH",
                        "detail": f"{key} is {v} (non-finite)"}
        return {"check": "shadow_book", "ok": True, "severity": "OK",
                "detail": f"oz={row['gold_oz']}, cash=${row['cash_usd']}"}
    except Exception as exc:
        return {"check": "shadow_book", "ok": False, "severity": "HIGH",
                "detail": f"db error: {exc}"[:120]}


def _check_audit_writable() -> dict:
    try:
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_LOG, "a") as f:
            f.write("")  # no-op append
        return {"check": "audit_writable", "ok": True, "severity": "OK",
                "detail": "append OK"}
    except Exception as exc:
        return {"check": "audit_writable", "ok": False, "severity": "CRITICAL",
                "detail": f"cannot append: {exc}"[:120]}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_data_quality() -> dict:
    checks = [
        _check_yf_liveness(),
        _check_alt_data(),
        _check_yf_gaps(),
        _check_perplexity_cache(),
        _check_shadow_book(),
        _check_audit_writable(),
    ]

    failures = [c for c in checks if not c["ok"]]
    by_severity = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "OK": 0}
    for c in checks:
        by_severity[c["severity"]] += 1

    if by_severity["CRITICAL"] > 0:
        overall = "CRITICAL"
    elif by_severity["HIGH"] > 0:
        overall = "DEGRADED"
    elif by_severity["MEDIUM"] > 0:
        overall = "WARN"
    else:
        overall = "OK"

    result = {
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overall_status": overall,
        "n_checks":       len(checks),
        "n_failures":     len(failures),
        "by_severity":    by_severity,
        "checks":         checks,
        "failures":       failures,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2, default=str))
    _print_report(result)
    return result


def _print_report(r: dict) -> None:
    color = {"OK": "\033[32;1m", "WARN": "\033[33m", "DEGRADED": "\033[31m", "CRITICAL": "\033[31;1m"}.get(r["overall_status"], "")
    print(f"\n{SEP}\n  DATA QUALITY MONITOR\n{SEP}")
    print(f"  Overall:     {color}{r['overall_status']}\033[0m")
    print(f"  Checks:      {r['n_checks']}  failures: {r['n_failures']}")
    print()
    print(f"  PER-CHECK")
    print(f"  {'─' * 56}")
    for c in r["checks"]:
        sym = "✅" if c["ok"] else "❌"
        sev_color = {"OK": "\033[32m", "MEDIUM": "\033[33m", "HIGH": "\033[31m", "CRITICAL": "\033[31;1m"}.get(c["severity"], "")
        print(f"  {sym} {c['check']:<18s}  {sev_color}{c['severity']:>8s}\033[0m  {c['detail'][:36]}")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Quality Monitor")
    args = parser.parse_args()
    run_data_quality()
