#!/usr/bin/env python3
"""
PHASE 7 — The Daily Oracle Scout
===================================
Queries Perplexity Sonar Pro for structured daily intelligence on Gold and Silver.
Produces a 0.0–1.0 sentiment score + 3 key bullets and appends to
data/raw_memory_log.json — the raw feed for the Weekly Archivist.

Usage:
    python3 scripts/oracle_scout.py               # scout both metals
    python3 scripts/oracle_scout.py --ticker GC=F # single metal
"""

import os, sys, json, argparse
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

PERPLEXITY_KEY  = os.getenv("PERPLEXITY_API_KEY", "")
RAW_MEMORY_LOG  = ROOT / "data" / "raw_memory_log.json"
RAW_MEMORY_LOG.parent.mkdir(parents=True, exist_ok=True)

METAL_NAMES = {"GC=F": "gold", "SI=F": "silver"}

SCOUT_SYSTEM = """You are a senior macro intelligence analyst.
Analyze the CURRENT global environment for the specified commodity.
Consider: Fed policy, USD strength, inflation, geopolitical risk, ETF flows, China demand, technical levels.
Respond with ONLY a JSON object in this exact format, no extra text:
{
  "score": <float 0.0-1.0>,
  "bullets": ["<key catalyst 1>", "<key catalyst 2>", "<key catalyst 3>"],
  "summary": "<one sentence macro thesis>"
}
Where 0.0 = extremely bearish, 0.5 = neutral, 1.0 = extremely bullish."""


def query_scout(metal_name: str) -> dict:
    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": SCOUT_SYSTEM},
            {"role": "user",
             "content": (
                 f"Provide current macro intelligence for {metal_name}. "
                 f"Return only the JSON object."
             )},
        ],
        "temperature": 0.1,
        "max_tokens":  300,
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
    raw = raw.strip("`").strip()
    if raw.startswith("json"):
        raw = raw[4:].strip()

    data    = json.loads(raw)
    score   = float(data["score"])
    if not (0.0 <= score <= 1.0):
        raise ValueError(f"Score {score} out of [0,1] range")

    return {
        "score":   round(score, 4),
        "bullets": data.get("bullets", []),
        "summary": data.get("summary", ""),
    }


def _load_log() -> list:
    if not RAW_MEMORY_LOG.exists():
        return []
    with open(RAW_MEMORY_LOG) as f:
        return json.load(f)


def _save_log(entries: list) -> None:
    with open(RAW_MEMORY_LOG, "w") as f:
        json.dump(entries, f, indent=2)


def run_scout(tickers: list) -> dict:
    today   = date.today().isoformat()
    entries = _load_log()
    scores  = {}

    for ticker in tickers:
        name = METAL_NAMES.get(ticker, ticker)
        print(f"  🔭  Scouting {ticker} ({name}) …", end="", flush=True)
        try:
            result = query_scout(name)
            entry  = {
                "date":    today,
                "ticker":  ticker,
                "score":   result["score"],
                "bullets": result["bullets"],
                "summary": result["summary"],
            }
            entries.append(entry)
            scores[ticker] = result["score"]
            level = "🟢 Bullish" if result["score"] >= 0.6 else (
                    "🔴 Bearish" if result["score"] <= 0.4 else "🟡 Neutral")
            print(f"  score={result['score']:.2f}  {level}")
            print(f"      {result['summary']}")
            for b in result["bullets"]:
                print(f"      • {b}")
        except Exception as exc:
            print(f"  ❌  {exc}")
            scores[ticker] = None

    _save_log(entries)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=None, help="GC=F or SI=F (default: both)")
    args   = parser.parse_args()

    tickers = [args.ticker] if args.ticker else list(METAL_NAMES.keys())

    print("\n" + "━" * 60)
    print("  WORLDWIDE AI METALS — Daily Oracle Scout")
    print(f"  Scouting: {tickers}")
    print("━" * 60)

    scores = run_scout(tickers)

    print("\n" + "━" * 60)
    print(f"  ✅  Raw memory log updated → {RAW_MEMORY_LOG}")
    print("━" * 60 + "\n")
