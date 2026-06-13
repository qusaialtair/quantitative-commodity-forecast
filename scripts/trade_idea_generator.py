#!/usr/bin/env python3
"""
Trade Idea Generator  (Phase XI Stage 57)
==========================================
Synthesises the full 43-engine state into a single daily trade card. This
is what shows up at the top of the operator's home menu every morning.

Decision logic (vote + scale):

  1. SIDE        Vote-aggregate the directional engines:
                   BMA top signal × hit-rate
                   IC/IR top with deployable flag
                   Macro nowcast composite
                   MTF confluence
                   Macro regime quadrant
                 → BUY / SELL / HOLD with conviction score in [-1, +1]

  2. TICKER      Pick the asset:
                   physical metals (gold default) when BUY/SELL conviction high
                   top halal equity when macro regime supports equities
                   cash/HOLD otherwise

  3. SIZE        Stack of caps applied in order:
                   Kelly fraction × vol_surface multiplier
                   × drawdown_tier multiplier
                   × vol_target leverage
                   × MRM champion confidence
                   × min(physical_capacity_cap, paper_capacity_cap)

  4. PRICES      entry = current spot
                 stop  = stop_loss_optimizer output
                 target = conformal_intervals 95% upper bound

  5. REASONING   Bullet list citing each contributing engine.

Output: data/trade_idea.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "trade_idea.json"

# Default portfolio assumptions
DEFAULT_PORTFOLIO_USD = 100_000.0
DEFAULT_HORIZON_DAYS = 21

LINE_W = 62
SEP = "━" * LINE_W


def _load(name: str) -> dict:
    p = DATA_DIR / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Directional voting
# ---------------------------------------------------------------------------
def _side_vote() -> tuple[str, float, list]:
    """
    Aggregate directional signals into a single side with conviction in [-1, +1].
    Positive = BUY, negative = SELL, near zero = HOLD.
    """
    votes = []
    reasoning = []

    # BMA top signal
    bma = _load("bma_weights.json")
    if bma:
        top = bma.get("top_source")
        if top:
            per = next(
                (s for s in bma.get("per_source", []) if s.get("source") == top),
                {},
            )
            hit = float(per.get("hit_rate") or 0.5)
            # Sign of vote depends on whether the signal is long-biased
            # We approximate: hit > 0.5 → bullish lean
            v = (hit - 0.5) * 2  # [-1, +1]
            votes.append(v)
            reasoning.append(
                f"BMA top={top}, hit_rate={hit:.3f} → vote {v:+.2f}"
            )

    # IC/IR top
    ii = _load("ic_ir_tracker.json")
    if ii:
        ranked = ii.get("ranked_by_ir", [])
        if ranked:
            top = ranked[0]
            per = ii.get("per_signal", {}).get(top, {})
            ic = float(per.get("ic_63d") or 0)
            v = max(min(ic * 5, 1.0), -1.0)
            votes.append(v)
            reasoning.append(
                f"IC/IR top={top}, IC(63d)={ic:+.4f} → vote {v:+.2f}"
            )

    # Alpha attribution top
    aa = _load("alpha_attribution.json")
    if aa:
        ranked = aa.get("ranked_by_sharpe", [])
        if ranked:
            top = ranked[0]
            per = aa.get("full_history", {}).get(top, {})
            sharpe = float(per.get("sharpe") or 0)
            v = max(min(sharpe * 0.6, 1.0), -1.0)
            votes.append(v)
            reasoning.append(
                f"Alpha top={top}, Sharpe={sharpe:+.3f} → vote {v:+.2f}"
            )

    # Macro nowcast
    nc = _load("macro_nowcast.json")
    if nc:
        score = float(nc.get("composite_score") or 0)
        v = max(min(score / 1.5, 1.0), -1.0)
        votes.append(v)
        reasoning.append(
            f"Macro nowcast={nc.get('regime', '?')}, score={score:+.3f} → vote {v:+.2f}"
        )

    # MTF confluence
    mtf = _load("mtf_confluence.json")
    if mtf:
        score = float(mtf.get("confluence", {}).get("score") or 0)
        v = max(min(score / 100, 1.0), -1.0)
        votes.append(v)
        reasoning.append(
            f"MTF confluence={mtf.get('confluence', {}).get('level', '?')}, "
            f"score={int(score):+d}/100 → vote {v:+.2f}"
        )

    # Macro regime gold tilt (from asset_tilts)
    mr = _load("macro_regime.json")
    if mr:
        tilt = float(mr.get("asset_tilts", {}).get("GC=F", 0))
        v = max(min(tilt / 2.0, 1.0), -1.0)
        votes.append(v)
        reasoning.append(
            f"Macro quadrant={mr.get('quadrant', '?')}, gold tilt={int(tilt):+d} → vote {v:+.2f}"
        )

    # Aggregate
    if not votes:
        return "HOLD", 0.0, ["No directional signals available"]

    avg = sum(votes) / len(votes)
    if avg > 0.20:
        side = "BUY"
    elif avg < -0.20:
        side = "SELL"
    else:
        side = "HOLD"
    return side, float(avg), reasoning


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------
def _size_position(
    conviction: float, portfolio_usd: float,
) -> tuple[float, float, list]:
    """
    Stack caps to get final position % of portfolio.
    Returns (size_pct, size_usd, reasoning_lines).
    """
    reasoning = []
    base_pct = abs(conviction) * 25.0  # max 25% at full conviction
    reasoning.append(f"Base from conviction |{conviction:+.2f}|: {base_pct:.1f}%")

    # Kelly fraction multiplier from kelly_sizing
    kelly = _load("kelly_sizing.json")
    if kelly:
        kelly_pct = float(kelly.get("sizing", {}).get("final_position_pct") or 0)
        # Use kelly_pct as a ceiling
        old = base_pct
        base_pct = min(base_pct, kelly_pct)
        if base_pct < old:
            reasoning.append(f"Kelly cap: {kelly_pct:.1f}% (was {old:.1f}%)")

    # Vol surface Kelly mult
    vs = _load("vol_surface.json")
    if vs:
        km = float(vs.get("actions", {}).get("kelly_fraction_multiplier") or 1.0)
        base_pct *= km
        reasoning.append(f"Vol surface Kelly mult × {km:.2f}: {base_pct:.2f}%")

    # Drawdown tier multiplier
    dd = _load("drawdown_controller.json")
    if dd:
        dm = float(dd.get("sizing_multiplier") or 1.0)
        base_pct *= dm
        reasoning.append(
            f"Drawdown tier {dd.get('tier_name', '?')} mult × {dm:.2f}: {base_pct:.2f}%"
        )

    # Vol-target leverage
    vt = _load("vol_target_budget.json")
    if vt:
        lev = float(vt.get("leverage_capped") or 1.0)
        base_pct *= lev
        reasoning.append(
            f"Vol-target leverage × {lev:.2f} "
            f"({vt.get('guidance', {}).get('leverage_action', '?')}): {base_pct:.2f}%"
        )

    # Capacity check (physical)
    cap = _load("capacity_analyzer.json")
    if cap:
        cap_usd = cap.get("thresholds_physical", {}).get("decay_25pct", {}).get("aum_cap_usd")
        if cap_usd and cap_usd > 0:
            max_pct = cap_usd / portfolio_usd * 100
            if base_pct > max_pct:
                old = base_pct
                base_pct = max_pct
                reasoning.append(
                    f"Capacity cap (physical, 25% α decay): {cap_usd:,.0f} = {max_pct:.2f}% "
                    f"(was {old:.2f}%)"
                )

    # Hard floor/ceiling
    base_pct = max(0.0, min(40.0, base_pct))
    size_usd = portfolio_usd * base_pct / 100.0
    return round(base_pct, 3), round(size_usd, 2), reasoning


# ---------------------------------------------------------------------------
# Ticker selection
# ---------------------------------------------------------------------------
def _select_ticker(side: str, conviction: float) -> str:
    """For now, default to gold for metal trades; halal-equity for risk-on."""
    if side == "HOLD":
        return "CASH"
    # Macro regime tilt → pick metals vs equities
    mr = _load("macro_regime.json")
    quadrant = mr.get("quadrant", "NEUTRAL")
    if quadrant in ("STAGFLATION", "DEFLATION"):
        return "GC=F"
    if quadrant == "GOLDILOCKS":
        return "SPY"  # tilt to equity
    if quadrant == "REFLATION":
        return "GC=F"  # both ok; gold is closer to our core
    return "GC=F"


# ---------------------------------------------------------------------------
# Entry, stop, target
# ---------------------------------------------------------------------------
def _entry_stop_target(ticker: str) -> dict:
    out = {"entry": None, "stop": None, "target": None, "horizon_days": DEFAULT_HORIZON_DAYS}
    if ticker == "CASH":
        return out

    # Stop-loss engine ran on GC=F by default; the data is authoritative for gold.
    sl = _load("stop_loss_optimizer.json")
    if ticker == "GC=F" and sl:
        out["entry"] = sl.get("current_price")
        out["stop"] = sl.get("final_stop_price")
        out["stop_method"] = sl.get("final_recommendation")
        out["stop_distance_pct"] = sl.get("final_stop_distance_pct")
    else:
        # Live-fetch entry from yfinance and derive a 2.5×ATR stop placeholder
        try:
            import yfinance as yf
            raw = yf.download(ticker, period="60d", interval="1d",
                              progress=False, auto_adjust=True)
            if isinstance(raw.columns, type(raw.columns)) and hasattr(raw.columns, "droplevel"):
                try:
                    raw.columns = raw.columns.droplevel(1)
                except Exception:
                    pass
            close = raw["Close"].dropna()
            high = raw["High"].dropna()
            low = raw["Low"].dropna()
            entry = float(close.iloc[-1])
            # Simple 14-day ATR
            tr = (high - low).rolling(14).mean()
            atr_val = float(tr.dropna().iloc[-1]) if len(tr.dropna()) else entry * 0.02
            out["entry"] = round(entry, 2)
            out["stop"] = round(entry - 2.5 * atr_val, 2)
            out["stop_method"] = "atr_2_5_live"
            out["stop_distance_pct"] = round(2.5 * atr_val / entry * 100, 3)
        except Exception:
            pass

    # Target = conformal upper 95% bound applied to whatever entry we just set
    co = _load("conformal_intervals.json")
    if co and out["entry"]:
        upper_pct = co.get("live_intervals", {}).get("alpha_05", {}).get("upper_pct")
        if upper_pct is not None:
            out["target"] = round(float(out["entry"]) * (1 + upper_pct / 100.0), 2)
            out["target_horizon_days"] = co.get("horizon", 5)

    return out


# ---------------------------------------------------------------------------
# Risk flags
# ---------------------------------------------------------------------------
def _risk_flags() -> list:
    flags = []
    dd = _load("drawdown_controller.json")
    if dd and dd.get("tier_name") in ("DEFENSIVE", "CRITICAL", "EMERGENCY"):
        flags.append(f"Drawdown tier {dd.get('tier_name')}")
    vs = _load("vol_surface.json")
    if vs and vs.get("vol_regime") == "EXTREME":
        flags.append("Vol regime EXTREME")
    sb = _load("structural_breaks.json")
    if sb and sb.get("summary", {}).get("cusum_break"):
        flags.append("CUSUM structural break")
    dcc = _load("dcc_garch.json")
    if dcc and (dcc.get("n_stressed") or 0) > 0:
        flags.append(f"DCC stress: {dcc.get('n_stressed')} pairs")
    ge = _load("geopolitical_events.json")
    if ge and ge.get("priority") == "HIGH":
        flags.append(f"Geopolitical HIGH ({ge.get('regime')})")
    ts = _load("term_structure.json")
    if ts and ts.get("stress_flag"):
        flags.append(f"Term structure stress ({ts.get('curve_shape')})")
    return flags


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_trade_idea(portfolio_usd: float = DEFAULT_PORTFOLIO_USD) -> dict:
    # Override portfolio from shadow book if available
    ps = _load("pipeline_state.json")
    pf = ps.get("portfolio", {})
    pv = float(pf.get("portfolio_value") or portfolio_usd)

    side, conviction, side_reasoning = _side_vote()
    size_pct, size_usd, size_reasoning = _size_position(conviction, pv)
    ticker = _select_ticker(side, conviction)
    prices = _entry_stop_target(ticker)
    flags = _risk_flags()

    if abs(conviction) > 0.5:
        conviction_label = "HIGH"
    elif abs(conviction) > 0.2:
        conviction_label = "MEDIUM"
    else:
        conviction_label = "LOW"

    # If conviction LOW or side HOLD, force size to 0
    if side == "HOLD":
        size_pct = 0
        size_usd = 0

    trade_card = {
        "side":              side,
        "ticker":            ticker,
        "size_pct":          size_pct,
        "size_usd":          size_usd,
        "conviction":        conviction_label,
        "conviction_score":  round(conviction, 4),
        "entry_price":       prices["entry"],
        "stop_price":        prices.get("stop"),
        "stop_method":       prices.get("stop_method"),
        "stop_distance_pct": prices.get("stop_distance_pct"),
        "target_price":      prices.get("target"),
        "target_horizon_days":prices.get("target_horizon_days", DEFAULT_HORIZON_DAYS),
        "horizon_days":      DEFAULT_HORIZON_DAYS,
    }

    # MRM champion stamp
    mrm = _load("mrm_champion.json")
    if mrm:
        trade_card["champion_signal"] = mrm.get("current_champion")
        trade_card["mrm_decision"] = mrm.get("decision")

    result = {
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_date":       ps.get("run_date"),
        "portfolio_usd":  pv,
        "trade_card":     trade_card,
        "reasoning": {
            "side_vote":     side_reasoning,
            "size_stack":    size_reasoning,
        },
        "risk_flags":     flags,
        "n_risk_flags":   len(flags),
        "ibkr_ready":     ticker != "CASH" and side in ("BUY", "SELL"),
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2, default=str))
    _print_report(result)
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_report(r: dict) -> None:
    tc = r["trade_card"]
    side_color = {
        "BUY":  "\033[32;1m",
        "SELL": "\033[31;1m",
        "HOLD": "\033[36m",
    }.get(tc["side"], "\033[0m")
    conv_color = {
        "HIGH":   "\033[32m",
        "MEDIUM": "\033[33m",
        "LOW":    "\033[31m",
    }.get(tc["conviction"], "\033[0m")

    print(f"\n{SEP}\n  TRADE IDEA — {tc['ticker']}\n{SEP}")
    print(f"  Side:           {side_color}{tc['side']}\033[0m")
    print(f"  Conviction:     {conv_color}{tc['conviction']}\033[0m  "
          f"(score {tc['conviction_score']:+.3f})")
    print(f"  Size:           {tc['size_pct']:.2f}% of portfolio  "
          f"(${tc['size_usd']:,.2f})")
    if tc.get("entry_price"):
        print(f"  Entry:          ${tc['entry_price']:,.2f}")
    if tc.get("stop_price"):
        print(f"  Stop:           ${tc['stop_price']:,.2f}  "
              f"({tc['stop_distance_pct']:.2f}% below, method {tc.get('stop_method')})")
    if tc.get("target_price"):
        print(f"  Target:         ${tc['target_price']:,.2f}  "
              f"({tc['target_horizon_days']}d horizon)")
    print(f"  Champion signal:{tc.get('champion_signal', 'n/a')}")
    print(f"  IBKR-ready:     {r['ibkr_ready']}")
    print()

    print(f"  SIDE VOTE")
    print(f"  {'─' * 56}")
    for line in r["reasoning"]["side_vote"]:
        print(f"    • {line}")
    print()

    print(f"  SIZE STACK")
    print(f"  {'─' * 56}")
    for line in r["reasoning"]["size_stack"]:
        print(f"    • {line}")
    print()

    if r["risk_flags"]:
        print(f"  RISK FLAGS ({len(r['risk_flags'])})")
        for f in r["risk_flags"]:
            print(f"    ⚠ {f}")
    else:
        print(f"  No active risk flags")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trade Idea Generator")
    parser.add_argument("--portfolio", type=float, default=DEFAULT_PORTFOLIO_USD)
    args = parser.parse_args()
    run_trade_idea(portfolio_usd=args.portfolio)
