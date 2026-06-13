#!/usr/bin/env python3
"""
Position Manager
=================
ATR-based position management with trailing stops, scale-in/out logic,
and optimal entry timing signals.

Usage:
    python3 scripts/position_manager.py --ticker GC=F
    python3 scripts/position_manager.py --audit
"""

from __future__ import annotations
import argparse, json, sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = DATA_DIR / "position_management.json"

ATR_PERIOD = 14
INITIAL_STOP_MULT = 2.5
TRAILING_STOP_MULT = 2.0
CATASTROPHIC_STOP_MULT = 3.5

TRANCHE_1_PCT = 0.40
TRANCHE_2_PCT = 0.30
TRANCHE_3_PCT = 0.30
TRANCHE_2_ATR_OFFSET = 0.5
TRANCHE_3_ATR_OFFSET = 1.0
BREAKOUT_ATR_OFFSET = 0.5

PYRAMID_MIN_ATR_PROFIT = 1.0
PYRAMID_MAX_LEVELS = 3
PYRAMID_SIZE_RATIO = 0.50

LINE_W = 62
SEP_HEAVY = "\u2501" * LINE_W
SEP_LIGHT = "\u2500" * 36


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StopLevels:
    direction: str
    entry_price: float
    initial_stop: float
    trailing_stop: float
    catastrophic_stop: float
    initial_pct: float
    trailing_pct: float
    catastrophic_pct: float


@dataclass
class TrancheOrder:
    label: str
    amount_usd: float
    target_price: float
    order_type: str
    atr_offset: str


@dataclass
class ScaleSchedule:
    total_usd: float
    tranches: list[TrancheOrder]


@dataclass
class EntrySignal:
    score: int
    reasoning: str
    factors: dict


@dataclass
class PositionAudit:
    status: str
    recommendation: str
    entry_price: float
    current_price: float
    pnl_pct: float
    risk_reward: float
    holding_days: int
    stop_hit: Optional[str]
    reasoning: str


@dataclass
class PyramidDecision:
    should_add: bool
    pyramid_level: int
    recommended_size_pct: float
    reasoning: str


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    return atr


# ---------------------------------------------------------------------------
# Dynamic stops
# ---------------------------------------------------------------------------

def compute_stops(
    entry_price: float,
    atr: float,
    direction: str = "long",
    highest_close: Optional[float] = None,
) -> StopLevels:
    sign = 1.0 if direction == "long" else -1.0
    initial = entry_price - sign * INITIAL_STOP_MULT * atr
    catastrophic = entry_price - sign * CATASTROPHIC_STOP_MULT * atr

    ref = highest_close if highest_close and direction == "long" else entry_price
    if direction == "short" and highest_close:
        ref = highest_close
    trailing = ref - sign * TRAILING_STOP_MULT * atr

    if direction == "long":
        trailing = max(trailing, initial)
    else:
        trailing = min(trailing, initial)

    def pct(stop: float) -> float:
        return round((stop / entry_price - 1.0) * 100, 2)

    return StopLevels(
        direction=direction,
        entry_price=round(entry_price, 2),
        initial_stop=round(initial, 2),
        trailing_stop=round(trailing, 2),
        catastrophic_stop=round(catastrophic, 2),
        initial_pct=pct(initial),
        trailing_pct=pct(trailing),
        catastrophic_pct=pct(catastrophic),
    )


# ---------------------------------------------------------------------------
# Scale-in schedule
# ---------------------------------------------------------------------------

def compute_scale_schedule(
    total_deploy_usd: float,
    atr: float,
    current_price: float,
) -> ScaleSchedule:
    t1_usd = round(total_deploy_usd * TRANCHE_1_PCT, 2)
    t2_usd = round(total_deploy_usd * TRANCHE_2_PCT, 2)
    t3_usd = round(total_deploy_usd * TRANCHE_3_PCT, 2)

    t1_price = round(current_price, 2)
    t2_price = round(current_price - TRANCHE_2_ATR_OFFSET * atr, 2)
    t3_price = round(current_price - TRANCHE_3_ATR_OFFSET * atr, 2)

    tranches = [
        TrancheOrder("Tranche 1", t1_usd, t1_price, "market", "NOW"),
        TrancheOrder("Tranche 2", t2_usd, t2_price, "limit", f"-{TRANCHE_2_ATR_OFFSET} ATR"),
        TrancheOrder("Tranche 3", t3_usd, t3_price, "limit", f"-{TRANCHE_3_ATR_OFFSET} ATR"),
    ]
    return ScaleSchedule(total_usd=total_deploy_usd, tranches=tranches)


# ---------------------------------------------------------------------------
# Entry timing
# ---------------------------------------------------------------------------

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta.clip(upper=0.0))
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _compute_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].replace(0, np.nan)
    cum_tp_vol = (typical * vol).rolling(window).sum()
    cum_vol = vol.rolling(window).sum()
    return cum_tp_vol / cum_vol


def _compute_support(df: pd.DataFrame, lookback: int = 20) -> float:
    return float(df["Low"].iloc[-lookback:].min())


def optimal_entry_signal(df: pd.DataFrame, atr: float) -> EntrySignal:
    close = df["Close"]
    current = float(close.iloc[-1])
    factors: dict = {}

    vwap = _compute_vwap(df, window=20)
    vwap_now = float(vwap.iloc[-1]) if not np.isnan(vwap.iloc[-1]) else current
    vwap_score = np.clip((vwap_now - current) / atr * 25, -25, 25)
    factors["price_vs_vwap"] = round(float(vwap_score), 1)

    rsi = _compute_rsi(close, 14)
    rsi_now = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0
    if rsi_now < 30:
        rsi_score = 25.0
    elif rsi_now < 45:
        rsi_score = 15.0
    elif rsi_now < 55:
        rsi_score = 5.0
    elif rsi_now < 70:
        rsi_score = -5.0
    else:
        rsi_score = -15.0
    factors["rsi"] = round(rsi_now, 1)
    factors["rsi_score"] = round(float(rsi_score), 1)

    recent_vol = float(close.pct_change().iloc[-5:].std())
    hist_vol = float(close.pct_change().iloc[-60:].std()) if len(close) >= 60 else recent_vol
    vol_ratio = recent_vol / hist_vol if hist_vol > 0 else 1.0
    vol_score = np.clip((1.0 - vol_ratio) * 25, -15, 25)
    factors["vol_ratio"] = round(vol_ratio, 3)
    factors["vol_score"] = round(float(vol_score), 1)

    support = _compute_support(df, lookback=20)
    dist_to_support = (current - support) / atr if atr > 0 else 5.0
    support_score = np.clip((2.0 - dist_to_support) * 12.5, -15, 25)
    factors["support"] = round(support, 2)
    factors["dist_to_support_atr"] = round(dist_to_support, 2)
    factors["support_score"] = round(float(support_score), 1)

    raw = 50 + vwap_score + rsi_score + vol_score + support_score
    score = int(np.clip(raw, 0, 100))

    parts = []
    if vwap_score > 5:
        parts.append(f"Price below 20d VWAP")
    elif vwap_score < -5:
        parts.append(f"Price extended above VWAP")
    if rsi_now < 45:
        parts.append(f"RSI pulled back to {rsi_now:.0f}")
    elif rsi_now > 65:
        parts.append(f"RSI elevated at {rsi_now:.0f}")
    if vol_ratio < 0.8:
        parts.append("vol contracting")
    elif vol_ratio > 1.3:
        parts.append("vol expanding (caution)")
    if dist_to_support < 1.5:
        parts.append("near support")
    reasoning = " -- ".join(parts) if parts else "Neutral conditions"
    if score >= 65:
        reasoning += " -- favorable entry conditions"
    elif score <= 35:
        reasoning += " -- unfavorable, consider waiting"

    return EntrySignal(score=score, reasoning=reasoning, factors=factors)


# ---------------------------------------------------------------------------
# Position audit
# ---------------------------------------------------------------------------

def audit_position(
    entry_price: float,
    current_price: float,
    atr: float,
    holding_days: int,
    direction: str = "long",
    highest_close: Optional[float] = None,
) -> PositionAudit:
    stops = compute_stops(entry_price, atr, direction, highest_close)
    sign = 1.0 if direction == "long" else -1.0
    pnl_pct = sign * (current_price / entry_price - 1.0) * 100

    stop_hit = None
    if direction == "long":
        if current_price <= stops.catastrophic_stop:
            stop_hit = "CATASTROPHIC"
        elif current_price <= stops.trailing_stop:
            stop_hit = "TRAILING"
        elif current_price <= stops.initial_stop:
            stop_hit = "INITIAL"
    else:
        if current_price >= stops.catastrophic_stop:
            stop_hit = "CATASTROPHIC"
        elif current_price >= stops.trailing_stop:
            stop_hit = "TRAILING"
        elif current_price >= stops.initial_stop:
            stop_hit = "INITIAL"

    risk = abs(current_price - stops.trailing_stop)
    reward_target = entry_price + sign * CATASTROPHIC_STOP_MULT * atr
    reward = abs(reward_target - current_price)
    rr = round(reward / risk, 2) if risk > 0 else 0.0

    if stop_hit:
        recommendation = "EXIT"
        reasoning = f"{stop_hit} stop breached at ${stops.trailing_stop if stop_hit == 'TRAILING' else stops.initial_stop if stop_hit == 'INITIAL' else stops.catastrophic_stop}"
    elif pnl_pct > 3.0 and holding_days > 5:
        recommendation = "TIGHTEN_STOP"
        reasoning = f"Profitable ({pnl_pct:+.1f}%) over {holding_days}d -- tighten trailing stop"
    elif pnl_pct > 5.0:
        recommendation = "TAKE_PARTIAL"
        reasoning = f"Strong gain ({pnl_pct:+.1f}%) -- consider booking 25-50% profit"
    elif pnl_pct < -2.0 and rr < 1.0:
        recommendation = "EXIT"
        reasoning = f"Underwater ({pnl_pct:+.1f}%) with poor R:R ({rr}) -- cut"
    else:
        recommendation = "HOLD"
        reasoning = f"Position within normal range ({pnl_pct:+.1f}%), R:R {rr}"

    return PositionAudit(
        status="OPEN",
        recommendation=recommendation,
        entry_price=round(entry_price, 2),
        current_price=round(current_price, 2),
        pnl_pct=round(pnl_pct, 2),
        risk_reward=rr,
        holding_days=holding_days,
        stop_hit=stop_hit,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Pyramiding
# ---------------------------------------------------------------------------

def should_pyramid(
    current_position_pnl_pct: float,
    atr: float,
    trend_strength: float,
    entry_price: float = 0.0,
    current_level: int = 0,
) -> PyramidDecision:
    atr_pct = (atr / entry_price * 100) if entry_price > 0 else 2.0
    profit_in_atrs = current_position_pnl_pct / atr_pct if atr_pct > 0 else 0.0

    if current_level >= PYRAMID_MAX_LEVELS:
        return PyramidDecision(
            should_add=False,
            pyramid_level=current_level,
            recommended_size_pct=0.0,
            reasoning=f"Max pyramid levels ({PYRAMID_MAX_LEVELS}) reached",
        )

    if profit_in_atrs < PYRAMID_MIN_ATR_PROFIT:
        return PyramidDecision(
            should_add=False,
            pyramid_level=current_level,
            recommended_size_pct=0.0,
            reasoning=f"Position only {profit_in_atrs:.1f} ATR in profit (need >{PYRAMID_MIN_ATR_PROFIT})",
        )

    if trend_strength < 0.5:
        return PyramidDecision(
            should_add=False,
            pyramid_level=current_level,
            recommended_size_pct=0.0,
            reasoning=f"Trend strength {trend_strength:.2f} too weak (need >0.50)",
        )

    size_pct = PYRAMID_SIZE_RATIO * 100 * (0.75 ** current_level)
    next_level = current_level + 1

    return PyramidDecision(
        should_add=True,
        pyramid_level=next_level,
        recommended_size_pct=round(size_pct, 1),
        reasoning=(
            f"Profit {profit_in_atrs:.1f} ATR, trend {trend_strength:.2f} "
            f"-- add pyramid L{next_level} at {size_pct:.0f}% of original"
        ),
    )


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _fetch_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is required: pip install yfinance")
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for {ticker}")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    return data


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _fmt_price(p: float) -> str:
    return f"${p:,.2f}"


def _print_report(
    ticker: str,
    current_price: float,
    atr_val: float,
    stops: StopLevels,
    schedule: ScaleSchedule,
    entry_sig: EntrySignal,
) -> None:
    print(f"\n{SEP_HEAVY}")
    print(f"  POSITION MANAGER -- {ticker}")
    print(SEP_HEAVY)
    print()
    print(f"  ATR({ATR_PERIOD}):          {_fmt_price(atr_val)}")
    print(f"  Current Price:    {_fmt_price(current_price)}")
    print()
    print(f"  STOP LEVELS ({stops.direction} position)")
    print(f"  {SEP_LIGHT}")
    print(f"  Initial stop:     {_fmt_price(stops.initial_stop)}  ({stops.initial_pct:+.2f}%)")
    print(f"  Trailing stop:    {_fmt_price(stops.trailing_stop)}  ({stops.trailing_pct:+.2f}%)")
    print(f"  Catastrophic:     {_fmt_price(stops.catastrophic_stop)}  ({stops.catastrophic_pct:+.2f}%)")
    print()
    print(f"  SCALE-IN SCHEDULE ({_fmt_price(schedule.total_usd)} to deploy)")
    print(f"  {SEP_LIGHT}")
    for t in schedule.tranches:
        print(f"  {t.label}:  {_fmt_price(t.amount_usd)}  @ {_fmt_price(t.target_price)}  ({t.order_type}, {t.atr_offset})")
    print()
    print(f"  ENTRY QUALITY:    {entry_sig.score}/100")
    print(f"  Reasoning: {entry_sig.reasoning}")
    print(SEP_HEAVY)
    print()


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------

def _build_output(
    ticker: str,
    current_price: float,
    atr_val: float,
    stops: StopLevels,
    schedule: ScaleSchedule,
    entry_sig: EntrySignal,
    audit: Optional[PositionAudit] = None,
    pyramid: Optional[PyramidDecision] = None,
) -> dict:
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "current_price": round(current_price, 2),
        "atr": round(atr_val, 2),
        "atr_period": ATR_PERIOD,
        "stops": asdict(stops),
        "scale_schedule": asdict(schedule),
        "entry_signal": asdict(entry_sig),
    }
    if audit:
        out["audit"] = asdict(audit)
    if pyramid:
        out["pyramid"] = asdict(pyramid)
    return out


def _save_json(payload: dict) -> None:
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, default=str))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_position_manager(
    ticker: str = "GC=F",
    deploy_usd: float = 25000.0,
    direction: str = "long",
    entry_price: Optional[float] = None,
    holding_days: int = 0,
) -> dict:
    df = _fetch_data(ticker)
    atr_series = compute_atr(df, ATR_PERIOD)
    atr_val = float(atr_series.iloc[-1])
    current_price = float(df["Close"].iloc[-1])

    if entry_price is None:
        entry_price = current_price

    highest_close = float(df["Close"].iloc[-20:].max()) if len(df) >= 20 else current_price

    stops = compute_stops(entry_price, atr_val, direction, highest_close)
    schedule = compute_scale_schedule(deploy_usd, atr_val, current_price)
    entry_sig = optimal_entry_signal(df, atr_val)

    audit = None
    if holding_days > 0:
        audit = audit_position(entry_price, current_price, atr_val, holding_days, direction, highest_close)

    atr_pct = (atr_val / entry_price * 100) if entry_price > 0 else 2.0
    pnl_pct = (current_price / entry_price - 1.0) * 100 if direction == "long" else (entry_price / current_price - 1.0) * 100
    pyramid = should_pyramid(pnl_pct, atr_val, 0.6, entry_price, 0)

    _print_report(ticker, current_price, atr_val, stops, schedule, entry_sig)

    if audit:
        print(f"  POSITION AUDIT")
        print(f"  {SEP_LIGHT}")
        print(f"  Recommendation:   {audit.recommendation}")
        print(f"  P&L:              {audit.pnl_pct:+.2f}%")
        print(f"  Risk/Reward:      {audit.risk_reward}")
        print(f"  {audit.reasoning}")
        print()

    if pyramid.should_add:
        print(f"  PYRAMID SIGNAL")
        print(f"  {SEP_LIGHT}")
        print(f"  Level:            {pyramid.pyramid_level}")
        print(f"  Size:             {pyramid.recommended_size_pct}% of original")
        print(f"  {pyramid.reasoning}")
        print()

    payload = _build_output(ticker, current_price, atr_val, stops, schedule, entry_sig, audit, pyramid)
    _save_json(payload)
    print(f"  Saved: {OUTPUT_JSON}")
    print()
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ATR-based position manager")
    parser.add_argument("--ticker", default="GC=F", help="Yahoo Finance ticker")
    parser.add_argument("--deploy", type=float, default=25000.0, help="Total USD to deploy")
    parser.add_argument("--direction", choices=["long", "short"], default="long")
    parser.add_argument("--entry", type=float, default=None, help="Entry price (defaults to current)")
    parser.add_argument("--days", type=int, default=0, help="Holding days (triggers audit)")
    parser.add_argument("--audit", action="store_true", help="Run audit with default 5-day hold")
    args = parser.parse_args()

    days = args.days
    if args.audit and days == 0:
        days = 5

    try:
        run_position_manager(
            ticker=args.ticker,
            deploy_usd=args.deploy,
            direction=args.direction,
            entry_price=args.entry,
            holding_days=days,
        )
    except Exception as exc:
        print(f"\n  [ERROR] {exc}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
