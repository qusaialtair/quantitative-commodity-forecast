#!/usr/bin/env python3
"""
agents/equity_swarm/macro_agent.py
==================================
Macro Agent — measures systemic risk and outputs a regime label that
caps the book's gross exposure.

Inputs (no API keys required):
  • ^VIX           — equity-vol index (yfinance)
  • DX-Y.NYB       — US Dollar Index (yfinance, fallback to UUP)
  • ^TNX / ^FVX    — 10y & 2y Treasury constant-maturity (yfinance)
  • ICE BofA US HY OAS — BAMLH0A0HYM2 (FRED CSV, no key needed)

Rule-based regime classifier (deterministic — LLM is not the right tool
for thresholding quantitative signals).
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger("MACRO")

FRED_HY_OAS_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2"


@dataclass
class MacroState:
    vix_spot: float
    vix_20d_change_pct: float
    dxy_spot: float
    dxy_20d_change_pct: float
    ust10y: float
    ust2y: float
    term_spread_10_2: float
    hy_oas_spot: float
    hy_oas_20d_change_bp: float
    regime: str              # RISK_ON | NEUTRAL | RISK_OFF
    target_gross_exposure: float
    notes: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _last_pair(tkr: str, lookback_days: int = 40) -> tuple[float | None, float | None]:
    """Return (spot, value_20_trading_days_ago) for a yfinance ticker."""
    try:
        raw = yf.download(tkr, period=f"{lookback_days}d", interval="1d",
                          auto_adjust=True, progress=False)["Close"].dropna()
        # yfinance may return a DataFrame (when ticker input is a list-like) or a Series.
        s = raw[tkr] if isinstance(raw, pd.DataFrame) else raw
        s = pd.Series(s).dropna()
        if s.empty:
            return None, None
        spot = float(s.iloc[-1])
        prev = float(s.iloc[-21]) if len(s) > 21 else float(s.iloc[0])
        return spot, prev
    except Exception as exc:
        logger.warning("yfinance %s failed: %s", tkr, exc)
        return None, None


def _fred_hy_oas() -> tuple[float | None, float | None]:
    """HY OAS today + 20 business days ago, in percent."""
    try:
        r = requests.get(FRED_HY_OAS_URL, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ["date", "oas"]
        df["oas"] = pd.to_numeric(df["oas"], errors="coerce")
        df = df.dropna().tail(30)
        if df.empty:
            return None, None
        return float(df["oas"].iloc[-1]), float(df["oas"].iloc[-21] if len(df) > 21 else df["oas"].iloc[0])
    except Exception as exc:
        logger.warning("FRED HY OAS fetch failed: %s", exc)
        return None, None


def _classify(vix: float, dxy_chg: float, hy_chg_bp: float,
              term_spread: float) -> tuple[str, float, list[str]]:
    notes: list[str] = []
    score = 0.0

    if vix < 15:   score += 1.5; notes.append(f"VIX {vix:.1f} <15 (calm)")
    elif vix < 20: score += 0.5; notes.append(f"VIX {vix:.1f} (normal)")
    elif vix < 28: score -= 0.5; notes.append(f"VIX {vix:.1f} elevated")
    else:          score -= 1.5; notes.append(f"VIX {vix:.1f} stressed")

    if hy_chg_bp <= -10:  score += 0.5; notes.append(f"HY-OAS tightening {hy_chg_bp:+.0f}bp")
    elif hy_chg_bp >= 40: score -= 1.0; notes.append(f"HY-OAS widening {hy_chg_bp:+.0f}bp (credit stress)")

    if term_spread > 0.25:    score += 0.5; notes.append(f"10-2y spread {term_spread:+.2f} (no inversion)")
    elif term_spread < -0.25: score -= 0.5; notes.append(f"10-2y spread {term_spread:+.2f} (inverted)")

    if dxy_chg > 3:    score -= 0.3; notes.append(f"DXY +{dxy_chg:.1f}% (USD strong, risk headwind)")
    elif dxy_chg < -3: score += 0.3; notes.append(f"DXY {dxy_chg:+.1f}% (USD weak, risk tailwind)")

    if score >= 1.0:
        return "RISK_ON",  1.00, notes
    elif score <= -1.0:
        return "RISK_OFF", 0.30, notes
    else:
        return "NEUTRAL",  0.65, notes


def read_macro() -> MacroState:
    logger.info("Fetching macro pack…")
    vix_spot, vix_prev = _last_pair("^VIX")
    dxy_spot, dxy_prev = _last_pair("DX-Y.NYB")
    if dxy_spot is None:  # fallback: UUP ETF, ~1:1 proxy for direction
        dxy_spot, dxy_prev = _last_pair("UUP")
    tnx,  _ = _last_pair("^TNX")   # 10y × 10
    fvx,  _ = _last_pair("^IRX")   # 13w (short end stand-in for 2y if needed)
    two_y, _ = _last_pair("^FVX")  # 5y — closest yfinance to 2y; fallback
    hy_spot, hy_prev = _fred_hy_oas()

    ust10y = (tnx or 0.0) / 10.0 if tnx else 0.0
    # ^FVX is 5y divided by 10; we use it as 2y stand-in; if missing fall back to ^IRX
    ust2y  = ((two_y or fvx or 0.0) / 10.0) if (two_y or fvx) else 0.0
    vix_chg = ((vix_spot / vix_prev) - 1) * 100 if (vix_spot and vix_prev) else 0.0
    dxy_chg = ((dxy_spot / dxy_prev) - 1) * 100 if (dxy_spot and dxy_prev) else 0.0
    hy_chg_bp = (hy_spot - hy_prev) * 100 if (hy_spot is not None and hy_prev is not None) else 0.0
    term = ust10y - ust2y

    regime, gross, notes = _classify(vix_spot or 18.0, dxy_chg, hy_chg_bp, term)

    state = MacroState(
        vix_spot=round(vix_spot or 0, 2),
        vix_20d_change_pct=round(vix_chg, 2),
        dxy_spot=round(dxy_spot or 0, 2),
        dxy_20d_change_pct=round(dxy_chg, 2),
        ust10y=round(ust10y, 3),
        ust2y=round(ust2y, 3),
        term_spread_10_2=round(term, 3),
        hy_oas_spot=round(hy_spot or 0, 2),
        hy_oas_20d_change_bp=round(hy_chg_bp, 1),
        regime=regime,
        target_gross_exposure=gross,
        notes=notes,
    )
    logger.info("Macro regime=%s  gross-cap=%.2f  VIX=%.1f  HY-OAS=%.2f",
                state.regime, state.target_gross_exposure,
                state.vix_spot, state.hy_oas_spot)
    return state


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(read_macro().as_dict(), indent=2))
