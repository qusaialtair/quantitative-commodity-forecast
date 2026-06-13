#!/usr/bin/env python3
"""
PHASE 7 — The Weekly Archivist
================================
Reads data/raw_memory_log.json (daily oracle scout entries),
uses DeepSeek Reasoner (deepseek-reasoner) to compress and synthesize them
into a weighted master_context.json — the persistent memory bank.

Run weekly (or manually):
    python3 scripts/memory_consolidator.py

Output: data/master_context.json
Schema:
    {
      "last_consolidated": "2026-03-28",
      "entries": [
        {
          "date":        "2026-03-28",
          "ticker":      "GC=F",
          "summary":     "<2-sentence synthesis>",
          "score":       0.72,
          "relevance":   0.9,
          "key_themes":  ["Fed pivot", "safe haven demand"]
        }, ...
      ],
      "macro_thesis": "<overall paragraph synthesis>"
    }
"""

import os, sys, json, argparse
from datetime import date
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

ROOT         = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

RAW_LOG_PATH  = ROOT / "data" / "raw_memory_log.json"
MASTER_PATH   = ROOT / "data" / "master_context.json"

DS_KEY = os.getenv("DEEPSEEK_API_KEY", "")
client = OpenAI(api_key=DS_KEY, base_url="https://api.deepseek.com") if DS_KEY else None

MAX_RAW_ENTRIES = 90  # feed last 90 daily entries into the archivist


ARCHIVIST_SYSTEM = """You are the Chief Memory Archivist for a precious metals AI trading system.
Your role: compress, synthesize, and weight raw daily intelligence logs into a structured memory bank.

You will receive a JSON array of raw daily oracle entries. For each unique ticker, produce a compressed
entry array where each entry synthesizes related daily signals into a coherent intelligence fragment.

Respond with ONLY a JSON object in this exact format:
{
  "entries": [
    {
      "date":        "<most recent date covered>",
      "ticker":      "<GC=F or SI=F>",
      "summary":     "<2-sentence synthesis of key macro drivers>",
      "score":       <float 0.0-1.0 weighted average>,
      "relevance":   <float 0.0-1.0 recency/importance weight>,
      "key_themes":  ["<theme1>", "<theme2>", "<theme3>"]
    }
  ],
  "macro_thesis": "<one paragraph overall synthesis of the macro environment for precious metals>"
}

Rules:
- Produce at most 20 total entries (top by recency and significance)
- relevance = 1.0 for entries from last 7 days, decay by 0.1 per week older
- Preserve specific data points: prices, policy decisions, geopolitical events
- Do NOT hallucinate — only synthesize what is in the raw data"""


def _load_raw() -> list:
    if not RAW_LOG_PATH.exists():
        return []
    with open(RAW_LOG_PATH) as f:
        return json.load(f)


def _save_master(data: dict) -> None:
    with open(MASTER_PATH, "w") as f:
        json.dump(data, f, indent=2)


def consolidate(dry_run: bool = False) -> dict:
    raw_entries = _load_raw()
    if not raw_entries:
        print("  ⚠️  No raw memory entries found. Run oracle_scout.py first.")
        return {}

    # Take last MAX_RAW_ENTRIES
    feed = raw_entries[-MAX_RAW_ENTRIES:]
    print(f"  📚  Feeding {len(feed)} raw entries to DeepSeek Reasoner…")

    if dry_run:
        print("  🏃  Dry run — skipping API call.")
        return {}

    if client is None:
        print("  ❌  DEEPSEEK_API_KEY not set.")
        return {}

    prompt = (
        f"Here are {len(feed)} raw daily oracle intelligence entries for precious metals. "
        f"Consolidate them into the structured memory bank format:\n\n"
        f"{json.dumps(feed, indent=2)}"
    )

    resp = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": ARCHIVIST_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
        max_tokens=4096,
    )

    raw_out = resp.choices[0].message.content.strip()
    raw_out = raw_out.strip("`").strip()
    if raw_out.startswith("json"):
        raw_out = raw_out[4:].strip()

    consolidated = json.loads(raw_out)
    master = {
        "last_consolidated": date.today().isoformat(),
        "raw_entries_processed": len(feed),
        **consolidated,
    }

    _save_master(master)
    print(f"  ✅  master_context.json updated → {len(master.get('entries', []))} entries")
    print(f"  📝  Macro thesis: {master.get('macro_thesis', '')[:120]}…")
    return master


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse raw log but don't call DeepSeek")
    args = parser.parse_args()

    print("\n" + "━" * 60)
    print("  WORLDWIDE AI METALS — Weekly Memory Archivist")
    print("━" * 60)

    result = consolidate(dry_run=args.dry_run)

    print("━" * 60 + "\n")
