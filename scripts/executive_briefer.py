#!/usr/bin/env python3
"""
scripts/executive_briefer.py
==============================
Chief of Staff briefing module.

Reads both portfolio states (equities + metals), feeds raw JSON to DeepSeek
acting as the Chief of Staff, and saves a concise 3-paragraph executive
summary to data/executive_briefing.json.

Called at the end of the master_controller pipeline, or standalone:
    python3 scripts/executive_briefer.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BRIEFING_FILE    = ROOT / "data" / "executive_briefing.json"
VIRTUAL_ACCT     = ROOT / "data" / "virtual_account.json"
PIPELINE_STATE   = ROOT / "data" / "pipeline_state.json"
EQUITY_DECISION  = ROOT / "data" / "equity_decision.json"
CURRENT_REGIME   = ROOT / "data" / "current_regime.json"
ORACLE_HISTORY   = ROOT / "data" / "oracle_history.csv"

# ── DeepSeek client ───────────────────────────────────────────────────────────
_DS_KEY = os.getenv("DEEPSEEK_API_KEY", "")

_SYSTEM_PROMPT = """\
You are the Chief of Staff for a private fund manager overseeing a two-book \
operation: a physical gold/metals mandate and a Sharia-compliant equity portfolio.

Write a concise 3-paragraph executive briefing from the raw fund data provided.

Paragraph 1 — Portfolio Status: Total AUM across both books, allocation breakdown \
by asset class, and the key open P&L metrics.

Paragraph 2 — Market Regime: Summarise the HMM regime for metals, the macro oracle \
sentiment score, the equity market regime (RISK_ON / RISK_OFF), and what the \
combined picture means for the fund.

Paragraph 3 — Strategic Assessment: State the current CIO action on each book, any \
active vetoes or overrides, and the single most important risk or opportunity \
the principal should be aware of today.

Paragraph 4 — Model Intelligence: Summarise the LSTM model health (verdict, fine-tune \
trend), the proving ground ensemble consensus (do the tri-horizon models agree?), and \
any cross-asset anomalies (structural breaks in gold correlations, unusual gold beta). \
Flag anything that requires attention.

Paragraph 5 — Treasury Hedge Sleeve (Phase XXV): Report the current hedge sleeve \
status — mode (SIGNAL_ONLY vs LOCAL_ACTIVE), the effective instrument held \
(TLT/IEF if Sharia is cleared, or GLD fallback tagged sharia_fallback_gld if not), \
notional allocation as a percentage of book equity, the macro regime and crisis tier \
driving the recommendation, and the realized P&L contribution or drag from the \
TREASURY_HEDGE strategy tag versus the core alpha strategies. If the sleeve is off, \
note the regime context. If Sharia is not cleared and the GLD fallback is active, \
flag it explicitly so the principal knows sovereign debt is not being held.

Style: Direct, factual, no bullet points, no hedging. Written like a seasoned chief \
of staff briefing a time-constrained principal. Maximum 400 words total."""


def _load_snapshot() -> dict:
    """Collect all fund state into a single dict for the LLM prompt."""
    snap: dict = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    # ── Equities book ─────────────────────────────────────────────────────────
    if VIRTUAL_ACCT.exists():
        acct = json.loads(VIRTUAL_ACCT.read_text())
        cash = float(acct.get("cash_balance", 0))
        positions = acct.get("positions", {})
        snap["equities"] = {
            "cash_buffer_usd": round(cash, 2),
            "positions": {
                sym: {
                    "qty":      round(float(p["qty"]), 4),
                    "avg_cost": round(float(p["avg_cost"]), 2),
                }
                for sym, p in positions.items()
                if float(p.get("qty", 0)) > 0
            },
            "last_updated": acct.get("last_updated", ""),
        }

    # ── Equity CIO decision ───────────────────────────────────────────────────
    if EQUITY_DECISION.exists():
        eq = json.loads(EQUITY_DECISION.read_text())
        snap["equity_decision"] = {
            "market_regime":   eq.get("market_regime"),
            "spsk_override":   eq.get("spsk_override"),
            "decision_date":   eq.get("decision_date"),
            "selections": [
                {
                    "ticker":    s["ticker"],
                    "weight":    s["weight"],
                    "sector":    s.get("sector", ""),
                    "vam_score": s.get("vam_score", 0),
                }
                for s in eq.get("selections", [])
            ],
        }

    # ── Metals book ───────────────────────────────────────────────────────────
    if PIPELINE_STATE.exists():
        ps = json.loads(PIPELINE_STATE.read_text())
        pf = ps.get("portfolio", {})
        rg = ps.get("regime", {})
        cm = ps.get("committee", {})
        snap["metals"] = {
            "portfolio": {
                "state":           pf.get("state", "FIAT"),
                "gold_oz":         round(float(pf.get("gold_oz", 0)), 4),
                "cash_usd":        round(float(pf.get("cash_usd", 0)), 2),
                "portfolio_value": round(float(pf.get("portfolio_value", 0)), 2),
                "unrealised_pnl":  round(float(pf.get("unrealised_pnl", 0)), 2),
            },
            "regime": {
                "hmm_state":    rg.get("hmm_state", "UNKNOWN"),
                "veto_active":  rg.get("hmm_veto_active", False),
                "p_bullish":    round(float(rg.get("p_bullish", 0)), 3),
                "p_volatile":   round(float(rg.get("p_volatile", rg.get("p_ranging", 0))), 3),
                "p_bearish":    round(float(rg.get("p_bearish", 0)), 3),
            },
            "committee": {
                "action":            cm.get("action_taken", "HOLD_METAL"),
                "oracle_score":      cm.get("oracle_score"),
                "quant_conviction":  cm.get("quant_conviction"),
                "macro_conviction":  cm.get("macro_conviction"),
                "veto_active":       cm.get("veto_active", False),
            },
        }

    # ── Latest oracle score ───────────────────────────────────────────────────
    try:
        import pandas as pd
        df = pd.read_csv(ORACLE_HISTORY, parse_dates=["date"])
        gc = df[df["ticker"] == "GC=F"].sort_values("date")
        si = df[df["ticker"] == "SI=F"].sort_values("date")
        snap["oracle_latest"] = {
            "gold_score":   float(gc["score"].iloc[-1]) if not gc.empty else None,
            "silver_score": float(si["score"].iloc[-1]) if not si.empty else None,
        }
    except Exception:
        pass

    # ── LSTM model diagnostics ───────────────────────────────────────────
    _lstm_meta = ROOT / "data" / "gold_lstm_meta.json"
    _lstm_audit = ROOT / "data" / "lstm_audit.json"
    try:
        if _lstm_meta.exists():
            meta = json.loads(_lstm_meta.read_text())
            snap["lstm_diagnostics"] = {
                "trained_at":          meta.get("trained_at", ""),
                "best_val_loss":       meta.get("best_val_loss"),
                "last_fine_tune":      meta.get("last_fine_tuned", ""),
                "last_fine_tune_loss": meta.get("last_fine_tune_loss"),
                "n_features":          meta.get("n_features"),
            }
        if _lstm_audit.exists():
            audit = json.loads(_lstm_audit.read_text())
            snap.setdefault("lstm_diagnostics", {})["verdict"] = audit.get("verdict", "UNKNOWN")
    except Exception:
        pass

    # ── Proving ground ensemble ──────────────────────────────────────────
    _pg_path = ROOT / "data" / "proving_ground_predictions.json"
    try:
        if _pg_path.exists():
            pg = json.loads(_pg_path.read_text())
            snap["proving_ground"] = {
                k: {"pred_pct": v.get("pred_pct"), "horizon": v.get("horizon")}
                for k, v in pg.items() if k.startswith("GCF")
            }
    except Exception:
        pass

    # ── Correlation monitor ──────────────────────────────────────────────
    _corr_path = ROOT / "data" / "correlation_report.json"
    try:
        if _corr_path.exists():
            corr = json.loads(_corr_path.read_text())
            snap["cross_asset"] = {
                "gold_dxy_corr":     corr.get("correlations_21d", {}).get("DX-Y.NYB"),
                "gold_silver_ratio": corr.get("gold_silver_ratio"),
                "gold_beta_spx":     corr.get("gold_beta_spx"),
                "regime_signal":     corr.get("regime_signal"),
            }
    except Exception:
        pass

    # ── Monte Carlo simulation ──────────────────────────────────────────
    _mc_path = ROOT / "data" / "monte_carlo_simulation.json"
    try:
        if _mc_path.exists():
            mc = json.loads(_mc_path.read_text())
            snap["monte_carlo"] = {
                "horizon_days":    mc.get("horizon_days"),
                "prob_positive":   mc.get("probabilities", {}).get("positive_return"),
                "var_95_pct":      mc.get("risk", {}).get("var_95_pct"),
                "cvar_95_pct":     mc.get("risk", {}).get("cvar_95_pct"),
                "mean_return_pct": mc.get("terminal", {}).get("mean_return_pct"),
            }
    except Exception:
        pass

    # ── Kelly sizing ────────────────────────────────────────────────────
    _kelly_path = ROOT / "data" / "kelly_sizing.json"
    try:
        if _kelly_path.exists():
            ks = json.loads(_kelly_path.read_text())
            snap["kelly_sizing"] = {
                "edge":             ks.get("kelly", {}).get("edge"),
                "should_trade":     ks.get("kelly", {}).get("should_trade"),
                "final_pct":        ks.get("sizing", {}).get("final_position_pct"),
                "deploy_usd":       ks.get("sizing", {}).get("deploy_usd"),
            }
    except Exception:
        pass

    # ── Multi-timeframe confluence ──────────────────────────────────────
    _mtf_path = ROOT / "data" / "mtf_confluence.json"
    try:
        if _mtf_path.exists():
            mtf = json.loads(_mtf_path.read_text())
            snap["mtf_confluence"] = {
                "level": mtf.get("confluence", {}).get("level"),
                "score": mtf.get("confluence", {}).get("score"),
                "bullish_tfs": mtf.get("confluence", {}).get("bullish_tfs"),
                "bearish_tfs": mtf.get("confluence", {}).get("bearish_tfs"),
            }
    except Exception:
        pass

    # ── Treasury hedge sleeve (Phase XXV) ──────────────────────────────
    _mst_path = ROOT / "data" / "multi_strategy_trader.json"
    _th_path  = ROOT / "data" / "treasury_hedge.json"
    try:
        mst_data = json.loads(_mst_path.read_text()) if _mst_path.exists() else {}
        th_data  = json.loads(_th_path.read_text())  if _th_path.exists()  else {}
        hs = mst_data.get("hedge_state") or {}
        by_s = mst_data.get("by_strategy") or {}
        hedge_realized = (by_s.get("TREASURY_HEDGE") or {}).get("realized_pl_usd")
        alpha_realized = sum(
            (v.get("realized_pl_usd") or 0.0)
            for k, v in by_s.items() if k != "TREASURY_HEDGE"
        ) if by_s else None
        snap["treasury_hedge_sleeve"] = {
            "mode":               hs.get("mode") or th_data.get("mode"),
            "instrument":         hs.get("instrument"),
            "sub_tag":            hs.get("sub_tag"),
            "gate_action":        hs.get("gate_action") or th_data.get("gate_action"),
            "notional_usd":       mst_data.get("hedge_notional_usd"),
            "fraction_pct":       round((mst_data.get("hedge_fraction") or 0.0) * 100, 1),
            "regime_quadrant":    th_data.get("regime_quadrant"),
            "crisis_tier":        th_data.get("crisis_tier"),
            "rec_instrument":     th_data.get("instrument"),
            "rec_alloc_pct":      th_data.get("allocation_pct"),
            "sharia_cleared":     th_data.get("sharia_cleared"),
            "hedge_realized_pl":  hedge_realized,
            "alpha_realized_pl":  alpha_realized,
            "by_strategy_summary": {
                k: {
                    "realized_pl_usd":   v.get("realized_pl_usd"),
                    "open_notional_usd": v.get("open_notional_usd"),
                    "n_open":            v.get("n_open"),
                    "n_closed":          v.get("n_closed"),
                }
                for k, v in by_s.items()
            },
        }
    except Exception:
        pass

    return snap


def run_briefer(dry_run: bool = False) -> dict:
    """
    Generate an executive briefing and save to data/executive_briefing.json.
    Returns the full result dict.
    """
    if not _DS_KEY:
        logger.error("DEEPSEEK_API_KEY not configured — briefer disabled.")
        return {"error": "DEEPSEEK_API_KEY not set", "briefing": None}

    from openai import OpenAI
    client = OpenAI(api_key=_DS_KEY, base_url="https://api.deepseek.com")

    snap = _load_snapshot()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    user_msg = (
        f"Date: {today}\n\n"
        f"RAW FUND DATA (JSON):\n{json.dumps(snap, indent=2)}\n\n"
        "Write the executive briefing now."
    )

    logger.info("Calling DeepSeek Chief of Staff...")
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=500,
            temperature=0.3,
        )
        briefing_text = resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("DeepSeek call failed: %s", exc)
        return {"error": str(exc), "briefing": None}

    result = {
        "generated_at":   snap["generated_at"],
        "run_date":        today,
        "briefing":        briefing_text,
        "data_snapshot":   snap,
    }

    if not dry_run:
        BRIEFING_FILE.parent.mkdir(parents=True, exist_ok=True)
        BRIEFING_FILE.write_text(json.dumps(result, indent=2))
        logger.info("Briefing saved → %s", BRIEFING_FILE)

    # Always print to stdout so master_controller log captures it
    print("\n" + "=" * 60)
    print("  EXECUTIVE BRIEFING —", today)
    print("=" * 60)
    print(briefing_text)
    print("=" * 60 + "\n")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Executive Briefer — Chief of Staff module")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print briefing without saving to disk")
    args = parser.parse_args()
    result = run_briefer(dry_run=args.dry_run)
    if result.get("error"):
        sys.exit(1)
