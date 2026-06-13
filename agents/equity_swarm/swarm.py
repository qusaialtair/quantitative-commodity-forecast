#!/usr/bin/env python3
"""
agents/equity_swarm/swarm.py
============================
Stage-3 orchestrator — runs the multi-agent LLM swarm end-to-end.

Pipeline:
  1. Load Stage-2 ranker output          (data/equity_ranker/ranking_latest.parquet)
  2. Take top-N candidates (default 50)
  3. In parallel:
       · MacroAgent    (one call — market-wide)
       · SentimentAgent (Perplexity Sonar Pro per ticker, 6h cached)
       · InsiderAgent   (SEC EDGAR Form 4, per ticker)
  4. Hand everything to the CIO synthesiser (Gemini 2.5 Pro + fractional Kelly)
  5. Write:
       · data/equity_decision.json       (overwrites the V1 format)
       · data/equity_swarm/swarm_{date}.json  (immutable archive)

CLI
---
  python3 -m agents.equity_swarm.swarm                  # default: top 50
  python3 -m agents.equity_swarm.swarm --top 20         # dev / small run
  python3 -m agents.equity_swarm.swarm --no-sentiment   # skip Perplexity calls
  python3 -m agents.equity_swarm.swarm --no-insider     # skip SEC scraping
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from agents.equity_swarm.cio import synthesize  # noqa: E402
from agents.equity_swarm.insider_agent import fetch_insiders  # noqa: E402
from agents.equity_swarm.macro_agent import read_macro  # noqa: E402
from agents.equity_swarm.sentiment_agent import fetch_sentiment  # noqa: E402
from models.macro_regime import load_macro_state  # noqa: E402  — Phase 5 Crash Shield

logger = logging.getLogger("SWARM")

DATA_DIR       = ROOT / "data"
RANKING_LATEST = DATA_DIR / "equity_ranker" / "ranking_latest.parquet"
ARCHIVE_DIR    = DATA_DIR / "equity_swarm"
DECISION_FILE  = DATA_DIR / "equity_decision.json"
LOG_DIR        = DATA_DIR / "logs"
for _d in (ARCHIVE_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _load_top(path: Path, top_n: int) -> pd.DataFrame:
    if not path.exists():
        logger.error("Ranking not found: %s  (run Stage 2 first)", path)
        sys.exit(2)
    df = pd.read_parquet(path)
    df = df.head(top_n).copy()
    # Index is already `ticker` from Stage 1; confirm for safety.
    if df.index.name != "ticker":
        df = df.set_index("ticker")
    logger.info("Loaded top-%d from %s", len(df), path.name)
    return df


def _atomic_write_json(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.rename(path)


def run(top_n: int, skip_sentiment: bool, skip_insider: bool) -> dict:
    wall = time.time()
    top = _load_top(RANKING_LATEST, top_n)
    tickers = list(top.index)

    sent_input = [
        {"ticker": t,
         "name":   top.at[t, "name"]   if "name"   in top.columns else t,
         "sector": top.at[t, "sector"] if "sector" in top.columns else "Unknown"}
        for t in tickers
    ]

    # ── Run three agents in parallel ─────────────────────────────────────────
    logger.info("━" * 60)
    logger.info("Running agent swarm for %d tickers…", len(tickers))
    logger.info("━" * 60)

    with ThreadPoolExecutor(max_workers=3) as pool:
        f_macro  = pool.submit(read_macro)
        f_sent   = (pool.submit(fetch_sentiment, sent_input)
                    if not skip_sentiment else None)
        f_ins    = (pool.submit(fetch_insiders, tickers)
                    if not skip_insider else None)

        macro   = f_macro.result().as_dict()
        sent    = f_sent.result() if f_sent else {}
        insider = f_ins.result()  if f_ins  else {}

    # ── Phase 5 Crash Shield: HMM macro state overlay ────────────────────────
    # The HMM (models/macro_regime.py) is the QUANTITATIVE regime classifier.
    # It ALWAYS outranks the rule-based macro agent on gross-exposure sizing:
    # defensive wins, so gross_cap = min(rule_gross, hmm_gross).
    macro_hmm = load_macro_state()
    if macro_hmm:
        rule_gross = float(macro.get("target_gross_exposure", 0.65))
        hmm_gross  = float(macro_hmm.get("target_gross_exposure", 0.65))
        macro["hmm_regime"]               = macro_hmm.get("regime")
        macro["hmm_state_probs"]          = macro_hmm.get("state_probs", {})
        macro["hmm_target_gross_exposure"] = hmm_gross
        macro["hmm_features"]             = macro_hmm.get("features", {})
        # Hard gate — take the MORE defensive of the two.
        blended = min(rule_gross, hmm_gross)
        if blended < rule_gross:
            logger.warning("HMM overrides rule-based macro: gross %.2f → %.2f (%s)",
                           rule_gross, blended, macro_hmm.get("regime"))
        macro["target_gross_exposure"] = blended
        # If the HMM flags RISK_OFF, regime label surfaces that even if the
        # rule-based agent is neutral — the swarm prompt reads this field.
        if macro_hmm.get("regime") == "RISK_OFF":
            macro["regime"] = "RISK_OFF"
    else:
        logger.warning("macro_state.json not found — HMM crash shield inactive. "
                       "Run `python3 -m models.macro_regime --infer` first.")

    # ── CIO synthesis (Gemini + Kelly) ───────────────────────────────────────
    logger.info("━" * 60)
    logger.info("CIO synthesising portfolio…  regime=%s  gross_cap=%.2f",
                macro.get("regime"), macro.get("target_gross_exposure", 0.65))
    logger.info("━" * 60)
    decision = synthesize(top, macro=macro, insider=insider,
                          sentiment=sent, top_n=top_n)
    decision["elapsed_s"] = round(time.time() - wall, 1)

    # ── Persist ──────────────────────────────────────────────────────────────
    stamp = date.today().isoformat()
    archive = ARCHIVE_DIR / f"swarm_{stamp}.json"
    _atomic_write_json(decision, archive)
    _atomic_write_json(decision, DECISION_FILE)

    logger.info("Decision → %s", DECISION_FILE)
    logger.info("Archive  → %s", archive)
    return decision


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-Agent LLM Swarm  (Stage 3)")
    ap.add_argument("--top",           type=int, default=50)
    ap.add_argument("--no-sentiment",  action="store_true",
                    help="skip Perplexity sentiment scan")
    ap.add_argument("--no-insider",    action="store_true",
                    help="skip SEC Form-4 scan")
    args = ap.parse_args()

    log_file = LOG_DIR / f"equity_swarm_{date.today().isoformat()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s UTC [%(name)-7s] %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )

    decision = run(args.top, args.no_sentiment, args.no_insider)

    # Compact summary print
    print()
    print(f"Regime:        {decision['macro_regime']}   "
          f"gross-cap {decision['macro_gross_cap']:.2f}")
    print(f"Equity gross:  {decision['equity_gross']*100:5.1f}%")
    print(f"Cash / sukuk:  {decision['cash_sukuk']*100:5.1f}%")
    print(f"Positions:     {decision['n_positions']}")
    if decision["positions"]:
        print("\nTop 10 positions:")
        for p in decision["positions"][:10]:
            print(f"  {p['ticker']:6s}  w={p['weight']*100:4.1f}%  "
                  f"conv={p['conviction']:+.2f}  "
                  f"ai={p['ai_score']:.2f}  "
                  f"sent={p['sent_signal']:<4s}  "
                  f"{p['name'][:30]}")


if __name__ == "__main__":
    main()
