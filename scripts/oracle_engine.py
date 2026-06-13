#!/usr/bin/env python3
"""
PHASE 6 — The Oracle Logger
=============================
Queries Perplexity Sonar Pro for current macro sentiment on Gold and Silver.
Returns a strict 0.0 (bearish) → 1.0 (bullish) score and appends it to
data/oracle_history.csv — the Continuous Learning Data Lake.

Usage:
    python3 scripts/oracle_engine.py               # score both metals
    python3 scripts/oracle_engine.py --ticker GC=F # single metal

The CSV grows one row per run.  After 6+ months of data, feed it into the
trainer via the ### PHASE 9: 5-FEATURE INTEGRATION ### block in metals_trainer.py
"""

import os, sys, json, argparse, csv
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from scripts.cache_layer import cached  # noqa: E402

PERPLEXITY_KEY   = os.getenv("PERPLEXITY_API_KEY", "")
ORACLE_CSV       = ROOT / "data" / "oracle_history.csv"
ORACLE_CSV.parent.mkdir(parents=True, exist_ok=True)

METAL_NAMES = {"GC=F": "gold", "SI=F": "silver"}

SYSTEM_PROMPT = """You are a senior macro economist.
Analyze the CURRENT global macroeconomic environment for the specified commodity.
Consider: Fed policy, USD strength, inflation, geopolitical risk, ETF flows, and China demand.
Respond with ONLY a JSON object in this exact format, no extra text:
{"score": <float between 0.0 and 1.0>, "reasoning": "<one sentence>"}
Where 0.0 = extremely bearish, 0.5 = neutral, 1.0 = extremely bullish."""


# 4-hour TTL: macro sentiment updates daily; intraday volatility doesn't move
# the score meaningfully.  Saves ~5 Perplexity calls per day per metal.
@cached(namespace="perplexity", ttl_seconds=4 * 3600)
def query_oracle(metal_name: str) -> dict:
    """Call Perplexity and parse the strict JSON score."""
    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": (
                 f"What is your current macro sentiment score for {metal_name}? "
                 f"Return only the JSON object."
             )},
        ],
        "temperature": 0.1,
        "max_tokens":  120,
    }
    resp = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers={"Authorization": f"Bearer {PERPLEXITY_KEY}",
                 "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()

    # Strip any markdown fences
    raw = raw.strip("`").strip()
    if raw.startswith("json"):
        raw = raw[4:].strip()

    data = json.loads(raw)
    score = float(data["score"])
    if not (0.0 <= score <= 1.0):
        raise ValueError(f"Score {score} out of [0,1] range")
    return {"score": round(score, 4), "reasoning": data.get("reasoning", "")}


def append_to_csv(ticker: str, score: float, reasoning: str):
    """Append one row to oracle_history.csv, creating header if new."""
    today     = date.today().isoformat()
    file_new  = not ORACLE_CSV.exists()
    with open(ORACLE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if file_new:
            writer.writerow(["date", "ticker", "score", "reasoning"])
        writer.writerow([today, ticker, score, reasoning])


def run_oracle(tickers: list[str]) -> dict:
    """Score each ticker, log to CSV, return scores dict."""
    scores = {}
    for ticker in tickers:
        name = METAL_NAMES.get(ticker, ticker)
        print(f"  🔮  Querying Oracle for {ticker} ({name}) …", end="", flush=True)
        try:
            result = query_oracle(name)
            score  = result["score"]
            reasoning = result["reasoning"]
            append_to_csv(ticker, score, reasoning)
            scores[ticker] = score
            level = "🟢 Bullish" if score >= 0.6 else ("🔴 Bearish" if score <= 0.4 else "🟡 Neutral")
            print(f"  score={score:.2f}  {level}")
            print(f"      {reasoning}")
        except Exception as exc:
            print(f"  ❌  {exc}")
            scores[ticker] = None
    return scores


def get_latest_scores(tickers: list[str] | None = None) -> dict:
    """
    Read the most recent score for each ticker from oracle_history.csv.
    Returns {ticker: score or None}.
    Used by the Streamlit UI to load live oracle scores without re-querying.
    """
    tickers = tickers or list(METAL_NAMES.keys())
    scores  = {t: None for t in tickers}
    if not ORACLE_CSV.exists():
        return scores
    try:
        import pandas as pd
        df = pd.read_csv(ORACLE_CSV, parse_dates=["date"])
        for ticker in tickers:
            sub = df[df["ticker"] == ticker].sort_values("date")
            if not sub.empty:
                scores[ticker] = float(sub.iloc[-1]["score"])
    except Exception:
        pass
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=None,
                        help="GC=F or SI=F (default: both)")
    args = parser.parse_args()

    tickers = [args.ticker] if args.ticker else list(METAL_NAMES.keys())

    print("\n" + "━" * 60)
    print("  WORLDWIDE AI METALS — Oracle Engine")
    print(f"  Scoring: {tickers}")
    print("━" * 60)

    scores = run_oracle(tickers)

    print("\n" + "━" * 60)
    print(f"  ✅  Oracle log updated → {ORACLE_CSV}")
    print("━" * 60 + "\n")
