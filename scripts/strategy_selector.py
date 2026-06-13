#!/usr/bin/env python3
"""
Strategy Selector  (Phase XIV Stage 73)
========================================
Picks the right strategy class for the current regime instead of mechanically
HOLDing when the HMM says VOLATILE.  Five strategy classes:

    TREND          long/short directional via Alpha Stacker conviction
    MEAN_REVERSION fade extended moves (Bollinger, RSI, OU drift)
    PAIRS          cointegration-driven spread trades
    VOL_SHORT      sell premium when realised << implied (carry harvesting)
    TAIL_HEDGE     protective puts / inverse exposure
    CASH           full defence

Selection logic (deterministic — every rule is auditable):

    1.  Pull regime from current_regime.json (HMM state + probabilities).
    2.  Pull vol_surface.json   for vol regime and term-structure shape.
    3.  Pull macro_regime.json  for quadrant.
    4.  Pull alpha_stacker.json for conviction & tier.
    5.  Pull cointegration_engine.json + pairs_trader.json for actionable
        spread signals.
    6.  Run the rule cascade:

         a)  drawdown_controller tier ∈ {CRITICAL, SEVERE} → CASH
         b)  data_quality FAIL or econ blackout → CASH
         c)  geopolitical regime = EXTREME → TAIL_HEDGE
         d)  cointegration: ≥1 z>2 actionable signal → PAIRS (if conviction MEDIUM+)
         e)  vol_regime ∈ {EXTREME, ELEVATED}  and  HMM=VOLATILE
                 - conviction VERY_HIGH → TREND (we trust the stacker)
                 - else MEAN_REVERSION (fade the move)
         f)  vol_regime = LOW and term structure normal contango
                 → VOL_SHORT (carry harvesting, premium decay)
         g)  conviction tier in {HIGH, VERY_HIGH} → TREND
         h)  conviction tier MEDIUM and HMM = BULLISH/BEARISH → TREND
         i)  default → MEAN_REVERSION at LOW size

    7.  Compute final_size_pct = stacker_size × regime_size_multiplier ×
        drawdown_size_multiplier.

Output: data/strategy_selector.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.crisis_detector import apply_guard as _crisis_apply_guard  # noqa: E402

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "strategy_selector.json"


STRATEGY_DESCRIPTIONS = {
    "TREND":           "Ride the conviction.  Long or short the metal/equity in direction of Alpha Stacker.",
    "MEAN_REVERSION":  "Fade extended moves.  Buy oversold dips, sell overbought rips around the regime mean.",
    "PAIRS":           "Trade the cointegrated spread, not the direction.  Z-score driven entries.",
    "VOL_SHORT":       "Sell options premium when realised vol is well below implied — collect time decay.",
    "TAIL_HEDGE":      "Protective puts / inverse exposure.  Pay carry to insure against shock.",
    "CASH":            "Stand aside.  Risk controls override edge.",
}


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


def _rv_from_vol_surface(vs: dict, horizon: str, default: float = 0.0) -> float:
    """Read realised vol for a horizon ('5d', '21d', ...) from vol_surface.json.

    Supports BOTH schemas that have shipped:
      new:  term_structure.rv_21d            (current vol_surface.py writer)
      old:  term_structure.realised_vol.21d_pct
    The old read path silently returned the default for months — which fed
    rv_realised=0 into the IV-RV gap and biased the cascade toward VOL_SHORT
    in EXTREME vol. Schema-robust reads fix that class of failure.
    """
    ts = vs.get("term_structure") or {}
    v = _safe(ts.get(f"rv_{horizon}"), 0.0)
    if v > 0:
        return v
    legacy = ts.get("realised_vol") or {}
    v = _safe(legacy.get(f"{horizon}_pct"), 0.0)
    return v if v > 0 else default


def _regime_size_multiplier(
    vol_regime: str, hmm_state: str, tier: str, vol_spike_ratio: float = 1.0,
) -> float:
    """Regime scaling.  Tuned 2026-05-17: lean in harder when conviction is
    HIGH or VERY_HIGH (we 'earn' the right to take regime risk).

    Phase XXVII: a fresh volatility ignition (EWMA-fast vol running hot vs
    the 21d baseline) proportionally de-risks BEFORE the slow vol-regime
    label catches up — this is what was missing in the 2026-06 gold break.
    """
    vol = (vol_regime or "").upper()
    hmm = (hmm_state or "").upper()
    base = 1.0
    if vol == "EXTREME":
        base *= 0.40
    elif vol == "ELEVATED":
        base *= 0.60
    elif vol == "LOW":
        base *= 1.15
    if hmm == "VOLATILE":
        base *= 0.70
    elif hmm == "BEARISH":
        base *= 0.90
    # Conviction-aware tilt
    if tier == "VERY_HIGH":
        base *= 1.40
    elif tier == "HIGH":
        base *= 1.20
    elif tier == "LOW":
        base *= 0.75
    # Vol-ignition damping: ratio 1.25 → no cut; 2.0+ → halve.
    if vol_spike_ratio > 1.25:
        base *= max(0.50, 1.25 / vol_spike_ratio)
    return max(0.10, min(1.60, base))


def _drawdown_size_multiplier() -> tuple[float, str]:
    dd = _load("drawdown_controller.json")
    mult = _safe(dd.get("sizing_multiplier"), 1.0)
    tier = (dd.get("tier_name") or dd.get("tier") or "NORMAL").upper()
    return float(min(max(mult, 0.0), 1.0)), tier


def _select_strategy(
    *,
    conviction: float,
    tier: str,
    vol_regime: str,
    hmm_state: str,
    macro_quadrant: str,
    actionable_pairs: list,
    geo_regime: str,
    dd_tier: str,
    data_quality_fail: bool,
    econ_blackout: bool,
    tail_premium_pct: float,
    term_curve: str,
    rv_realised: float,
    iv_implied: float,
    extension_z: float = 0.0,
    macro_tilt: float = 0.0,
    direction: str = "HOLD",
    vol_spike_ratio: float = 1.0,
    rsi: float | None = None,
    macd_hist: float | None = None,
) -> tuple[str, list[str]]:
    """Returns (strategy_name, [reasoning_bullets]).

    Rule cascade — re-tuned 2026-05-17 based on Phase XV backtest evidence
    that MEAN_REVERSION (used 71% of days) lost -2.49% while TREND
    (used 8.4%) gained +9.82%.  New defaults bias toward TREND and
    VOL_SHORT; MEAN_REVERSION now requires a real extension.

    Phase XXVII (2026-06): fast momentum/volatility filter.
      - Falling-knife veto: never TREND-long into bearish fast momentum
        (RSI<40, MACD histogram negative) while volatility is igniting.
      - Vol-ignition gate: VOL_SHORT is forbidden when EWMA-fast vol runs
        ≥1.35× the 21d baseline — selling vol into a spike is how the June
        gold break was missed.
      - Momentum-confirmed TREND: MEDIUM conviction may trade WITH decisive
        fast momentum even before the slow HMM label flips.
    """
    bullets = []
    vol = (vol_regime or "").upper()
    hmm = (hmm_state or "").upper()
    dirn = (direction or "HOLD").upper()
    vol_igniting = vol_spike_ratio >= 1.35
    momentum_bearish = (
        rsi is not None and macd_hist is not None and rsi < 40.0 and macd_hist < 0.0
    )
    momentum_bullish = (
        rsi is not None and macd_hist is not None and rsi > 60.0 and macd_hist > 0.0
    )

    # ── Hard kill-switches (unchanged) ─────────────────────────────────────
    if dd_tier in ("CRITICAL", "SEVERE"):
        bullets.append(f"Drawdown tier {dd_tier} → defensive CASH")
        return "CASH", bullets
    if data_quality_fail:
        bullets.append("Data quality FAIL → cannot trust signals → CASH")
        return "CASH", bullets
    if econ_blackout:
        bullets.append("Economic-calendar blackout window → CASH")
        return "CASH", bullets

    # ── Tail-hedge overrides ───────────────────────────────────────────────
    if (geo_regime or "").upper() == "EXTREME":
        bullets.append("Geopolitical regime EXTREME → TAIL_HEDGE")
        return "TAIL_HEDGE", bullets
    if tail_premium_pct > 200:
        bullets.append(f"Fat-tail premium {tail_premium_pct:.0f}% vs Gaussian → TAIL_HEDGE")
        return "TAIL_HEDGE", bullets

    # ── Stat-arb (only when there's a real spread signal) ──────────────────
    if actionable_pairs:
        strong = [p for p in actionable_pairs if abs(_safe(p.get("z_score"))) >= 2.0]
        if strong and tier in ("MEDIUM", "HIGH", "VERY_HIGH"):
            bullets.append(
                f"Cointegration: {len(strong)} pair(s) |z|≥2.0 and tier {tier} → PAIRS"
            )
            return "PAIRS", bullets

    # ── Falling-knife veto (Phase XXVII) ────────────────────────────────────
    # A long signal into decisively bearish fast momentum while vol ignites is
    # the exact pattern that preceded the June 2026 drawdown. Stand aside and
    # let the knife hit the floor — VERY_HIGH conviction is the only exception
    # (it has earned regime risk), and it still gets size-damped by the
    # vol-ignition multiplier.
    if (
        dirn == "BUY"
        and momentum_bearish
        and vol_igniting
        and tier != "VERY_HIGH"
    ):
        bullets.append(
            f"Falling-knife veto: BUY signal vs RSI {rsi:.0f}, MACD hist "
            f"{macd_hist:+.2f}%, fast-vol {vol_spike_ratio:.2f}× baseline → CASH"
        )
        return "CASH", bullets

    # ── TREND has priority whenever conviction or macro is aligned ─────────
    # Backtest showed TREND is the highest-edge strategy.  Lower the bar.
    if tier in ("HIGH", "VERY_HIGH"):
        bullets.append(f"Conviction tier {tier} → TREND")
        return "TREND", bullets

    if tier == "MEDIUM" and hmm in ("BULLISH", "BEARISH"):
        bullets.append(f"Conviction MEDIUM aligned with HMM {hmm} → TREND")
        return "TREND", bullets

    # NEW: also take TREND when macro_quadrant strongly favours an asset tilt
    if tier == "MEDIUM" and abs(macro_tilt) >= 1.0:
        bullets.append(
            f"Conviction MEDIUM + macro {macro_quadrant} tilt {macro_tilt:+.1f} → TREND"
        )
        return "TREND", bullets

    # ── Momentum-confirmed TREND (Phase XXVII) ──────────────────────────────
    # The HMM needs days of evidence to flip state; fast momentum does not.
    # MEDIUM conviction trading WITH a decisive RSI/MACD reading may ride the
    # move before the slow regime label catches up — this is the "catch the
    # drop" path for fresh breakdowns (SELL + bearish momentum).
    if tier == "MEDIUM" and (
        (dirn == "SELL" and momentum_bearish)
        or (dirn == "BUY" and momentum_bullish and not vol_igniting)
    ):
        bullets.append(
            f"Conviction MEDIUM + fast momentum confirms {dirn} "
            f"(RSI {rsi:.0f}, MACD hist {macd_hist:+.2f}%) → TREND"
        )
        return "TREND", bullets

    # ── MEAN_REVERSION only with a TRULY extreme extension ─────────────────
    # Backtest (2024-2026 gold trend) showed even |z|>=1.5 fades lost money
    # because the trend persisted. Raise the bar to |z|>=2.0 and require
    # vol regime not be ELEVATED/EXTREME (mean-rev is broken in those).
    if (
        abs(extension_z) >= 2.0
        and tier != "VERY_LOW"
        and vol in ("LOW", "NORMAL")
    ):
        bullets.append(
            f"Price |z|={abs(extension_z):.2f}σ stretched + vol {vol}, tier {tier} → MEAN_REVERSION"
        )
        return "MEAN_REVERSION", bullets

    # ── VOL_SHORT in low-vol or range-bound conditions ─────────────────────
    # Phase XXVII vol-ignition gate: NEVER sell volatility while EWMA-fast
    # vol is running ≥1.35× the 21d baseline. The IV-RV gap looks juicy in
    # precisely those moments (RV hasn't caught up yet) — it's a trap.
    iv_rv_gap = (iv_implied or 0.0) - (rv_realised or 0.0)
    if vol == "LOW" and not vol_igniting:
        bullets.append(f"Vol LOW (RV21={rv_realised:.1f}%) → VOL_SHORT (collect premium)")
        return "VOL_SHORT", bullets
    if vol == "NORMAL" and iv_rv_gap > 2.0 and not vol_igniting:
        bullets.append(f"Vol NORMAL, IV-RV gap {iv_rv_gap:+.1f}% → VOL_SHORT")
        return "VOL_SHORT", bullets
    if vol_igniting and vol in ("LOW", "NORMAL"):
        bullets.append(
            f"Fast-vol ignition {vol_spike_ratio:.2f}× baseline — VOL_SHORT "
            f"suppressed → CASH until vol stabilises"
        )
        return "CASH", bullets

    # ── VOLATILE + EXTREME/ELEVATED  ────────────────────────────────────────
    # No extension, no conviction → don't fade indiscriminately.  Step aside.
    if vol in ("EXTREME", "ELEVATED") and hmm == "VOLATILE":
        if tier in ("LOW", "VERY_LOW"):
            bullets.append(
                f"Vol {vol} + HMM {hmm} with weak conviction and no extension → CASH"
            )
            return "CASH", bullets
        if vol_igniting:
            bullets.append(
                f"Vol {vol} + HMM {hmm}, fast-vol {vol_spike_ratio:.2f}× igniting "
                f"— premium selling unsafe → CASH"
            )
            return "CASH", bullets
        # MEDIUM conviction in volatile regime — prefer VOL_SHORT (short premium)
        bullets.append(
            f"Vol {vol} + HMM {hmm}, conviction {tier}, no extension → VOL_SHORT"
        )
        return "VOL_SHORT", bullets

    # ── Hard fallback for VERY_LOW ─────────────────────────────────────────
    if tier == "VERY_LOW":
        bullets.append("Conviction VERY_LOW with no override → CASH")
        return "CASH", bullets

    # ── Default: VOL_SHORT (was MEAN_REVERSION) ────────────────────────────
    if vol_igniting:
        bullets.append(
            f"Default fallback blocked by fast-vol ignition "
            f"{vol_spike_ratio:.2f}× → CASH"
        )
        return "CASH", bullets
    bullets.append(
        f"Default fallback for tier {tier}, vol {vol}, HMM {hmm} → VOL_SHORT"
    )
    return "VOL_SHORT", bullets


# ──────────────────────────────────────────────────────────────────────────────
# Public entrypoint
# ──────────────────────────────────────────────────────────────────────────────
def run_strategy_selector() -> dict:
    """Select strategy class from regime inputs and write ``data/strategy_selector.json``."""
    stacker = _load("alpha_stacker.json")
    decision = stacker.get("decision", {}) or {}
    conviction = _safe(decision.get("conviction_score"))
    tier = (decision.get("conviction_tier") or "VERY_LOW").upper()
    direction = (decision.get("direction") or "HOLD").upper()
    stacker_size_pct = _safe(decision.get("recommended_size_pct"))

    regime_root = _load("current_regime.json")
    # current_regime.json is nested by ticker; default to GC=F
    if "GC=F" in regime_root:
        regime = regime_root["GC=F"]
    else:
        regime = regime_root
    hmm_state = (
        regime.get("state_label")
        or regime.get("state")
        or regime.get("hmm_state")
        or "UNKNOWN"
    ).upper()

    vs = _load("vol_surface.json")
    vol_regime = (vs.get("vol_regime") or "NORMAL").upper()
    actions = vs.get("actions") or {}
    rv_kelly_mult = _safe(actions.get("kelly_fraction_multiplier"), 1.0)

    mr = _load("macro_regime.json")
    macro_q = (mr.get("quadrant") or "UNKNOWN").upper()

    ce = _load("cointegration_engine.json")
    actionable_pairs = ce.get("actionable_signals", []) or []

    ge = _load("geopolitical_detector.json")
    geo_regime = (ge.get("regime") or "NORMAL").upper()

    dq = _load("data_quality.json")
    data_quality_fail = (dq.get("overall_status") or "").upper() == "FAIL"

    eg = _load("economic_calendar.json")
    econ_blackout = bool(eg.get("blocked_today", False))

    tr = _load("tail_risk_engine.json")
    tail_premium_pct = _safe(
        (tr.get("tail_risk") or {}).get("tail_fatness_premium_pct"), 0.0
    )

    ts = _load("term_structure.json")
    term_curve = (ts.get("curve_shape") or "").upper()

    op = _load("options_pricer.json")
    iv_implied = _safe(op.get("sigma")) * 100  # decimal → %

    # Realised vol — schema-robust read (Phase XXVII fix: the old
    # realised_vol.21d_pct path no longer exists and returned 0.0).
    rv_21d = _rv_from_vol_surface(vs, "21d", default=0.0)
    rv_5d = _rv_from_vol_surface(vs, "5d", default=rv_21d)

    # ── Fast momentum / vol-ignition dials (Phase XXVII) ────────────────
    # Primary source: crisis_detector fast_metrics (EWMA-based, price-only).
    # Fallback: rv_5d / rv_21d from vol_surface when the detector hasn't
    # run yet. Missing momentum data disables the veto (defensive default).
    crisis_fm = _load("crisis_detector.json").get("fast_metrics") or {}
    vol_spike_ratio = _safe(crisis_fm.get("vol_spike_ratio"), 0.0)
    if vol_spike_ratio <= 0:
        vol_spike_ratio = (rv_5d / rv_21d) if (rv_5d > 0 and rv_21d > 0) else 1.0
    rsi_fast = crisis_fm.get("rsi_14")
    rsi_fast = float(rsi_fast) if rsi_fast is not None else None
    macd_fast = crisis_fm.get("macd_hist_pct")
    macd_fast = float(macd_fast) if macd_fast is not None else None

    dd_mult, dd_tier = _drawdown_size_multiplier()

    # ── New inputs (Phase XV tuning) ────────────────────────────────────
    # Extension z-score: how many sigmas is current price from SMA20?
    # Pull from pipeline_state's position_mgmt if available, else compute live.
    extension_z = 0.0
    try:
        pm = (_load("pipeline_state.json").get("position_mgmt") or {})
        cur_p = _safe(pm.get("current_price"))
        atr = _safe(pm.get("atr"))
        # ATR ≈ std-equivalent over the bar count.  As a proxy for σ vs SMA,
        # use 14d ATR * sqrt(14) as a vol estimate; not perfect but order-of-magnitude
        if cur_p > 0 and atr > 0:
            # Compute SMA20 + std20 via yfinance for accuracy
            try:
                import yfinance as yf
                hist = yf.Ticker("GC=F").history(period="60d", interval="1d", auto_adjust=True)
                if hist is not None and not hist.empty and len(hist) >= 20:
                    closes = hist["Close"].tail(20)
                    sma20 = float(closes.mean())
                    std20 = float(closes.std())
                    if std20 > 1e-6:
                        extension_z = (cur_p - sma20) / std20
            except Exception:
                pass
    except Exception:
        pass

    # Macro tilt: pull gold-tilt from macro_regime
    macro_tilt = _safe((mr.get("asset_tilts") or {}).get("gold"))

    strategy, bullets = _select_strategy(
        conviction=conviction,
        tier=tier,
        vol_regime=vol_regime,
        hmm_state=hmm_state,
        macro_quadrant=macro_q,
        actionable_pairs=actionable_pairs,
        geo_regime=geo_regime,
        dd_tier=dd_tier,
        data_quality_fail=data_quality_fail,
        econ_blackout=econ_blackout,
        tail_premium_pct=tail_premium_pct,
        term_curve=term_curve,
        rv_realised=rv_21d,
        iv_implied=iv_implied,
        extension_z=extension_z,
        macro_tilt=macro_tilt,
        direction=direction,
        vol_spike_ratio=vol_spike_ratio,
        rsi=rsi_fast,
        macd_hist=macd_fast,
    )

    regime_mult = _regime_size_multiplier(
        vol_regime, hmm_state, tier, vol_spike_ratio=vol_spike_ratio,
    )

    # Non-directional strategies get a minimum size floor (they don't need
    # alpha-stacker conviction — they harvest from regime structure)
    if strategy in ("MEAN_REVERSION", "VOL_SHORT", "PAIRS"):
        base_size = max(stacker_size_pct, 25.0)  # floor at 25%
    elif strategy == "TAIL_HEDGE":
        base_size = max(stacker_size_pct, 15.0)
    else:
        base_size = stacker_size_pct

    final_size_pct = base_size * regime_mult * dd_mult * rv_kelly_mult
    # Final clamp
    final_size_pct = max(0.0, min(100.0, final_size_pct))

    # ── Crisis guard (Phase XVIII) ─────────────────────────────────────────
    # Override the entire decision when the price-driven crisis detector
    # is in STRESS or CRISIS tier.  This is the fix for the Phase XVII
    # stress-test failures (2008 GFC, 2015 China rout, 2022 inflation rout).
    crisis = _load("crisis_detector.json")
    crisis_tier  = (crisis.get("tier") or "NORMAL").upper()
    crisis_score = _safe(crisis.get("score"), 0.0)
    crisis_reason = None
    if crisis_tier != "NORMAL":
        new_strategy, new_size, reason = _crisis_apply_guard(
            strategy, final_size_pct, crisis_tier,
        )
        if reason:
            crisis_reason = reason
            bullets.append(f"Crisis guard: {reason}")
        strategy = new_strategy
        final_size_pct = new_size

    out = {
        "schema_version": "1.0",
        "engine":         "strategy_selector",
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy":       strategy,
        "strategy_description": STRATEGY_DESCRIPTIONS.get(strategy, ""),
        "direction":      direction,
        "final_size_pct": round(final_size_pct, 2),
        "size_stack": {
            "stacker_recommended_pct": round(stacker_size_pct, 2),
            "regime_multiplier":       round(regime_mult, 3),
            "drawdown_multiplier":     round(dd_mult, 3),
            "vol_kelly_multiplier":    round(rv_kelly_mult, 3),
            "final_size_pct":          round(final_size_pct, 2),
        },
        "regime_context": {
            "hmm_state":    hmm_state,
            "vol_regime":   vol_regime,
            "macro_quadrant": macro_q,
            "term_curve":   term_curve,
            "geo_regime":   geo_regime,
            "drawdown_tier": dd_tier,
            "data_quality_fail": data_quality_fail,
            "econ_blackout": econ_blackout,
            "tail_premium_pct": round(tail_premium_pct, 1),
            "rv_realised_pct": round(rv_21d, 2),
            "rv_fast_pct":   round(rv_5d, 2),
            "iv_implied_pct": round(iv_implied, 2),
            "iv_rv_gap_pct": round(iv_implied - rv_21d, 2),
            "extension_z":   round(extension_z, 3),
            "macro_tilt":    round(macro_tilt, 2),
            "vol_spike_ratio": round(vol_spike_ratio, 3),
            "rsi_14":        rsi_fast,
            "macd_hist_pct": macd_fast,
            "crisis_tier":   crisis_tier,
            "crisis_score":  round(crisis_score, 4),
            "crisis_guard_applied": crisis_reason,
        },
        "alpha_stacker": {
            "conviction_score":     round(conviction, 4),
            "conviction_tier":      tier,
            "direction":            direction,
            "stacker_size_pct":     round(stacker_size_pct, 2),
            "n_signals":            (stacker.get("stack") or {}).get("n_signals", 0),
            "n_risk_flags":         stacker.get("n_risk_flags", 0),
            "risk_flags":           stacker.get("risk_flags", []),
        },
        "reasoning":      bullets,
        "actionable_pairs_count": len(actionable_pairs),
        "strong_pairs_count":     len(
            [p for p in actionable_pairs if abs(_safe(p.get("z_score"))) >= 2.0]
        ),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(out, indent=2))
    return out


def main() -> int:
    """CLI entrypoint for the strategy selector."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    out = run_strategy_selector()
    if args.quiet:
        return 0
    print("=" * 64)
    print(f"STRATEGY SELECTOR  ({out['generated_at']})")
    print("=" * 64)
    print(f"  Strategy        : {out['strategy']}")
    print(f"  Description     : {out['strategy_description']}")
    print(f"  Direction       : {out['direction']}")
    print(f"  Final size      : {out['final_size_pct']:.2f}% of risk budget")
    print()
    print("Size stack:")
    for k, v in out["size_stack"].items():
        print(f"  {k:<28s} = {v}")
    print()
    print("Regime context:")
    for k, v in out["regime_context"].items():
        print(f"  {k:<22s} : {v}")
    print()
    print("Reasoning:")
    for b in out["reasoning"]:
        print(f"  • {b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
