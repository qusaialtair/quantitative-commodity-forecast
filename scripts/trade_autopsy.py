#!/usr/bin/env python3
"""
PHASE 7 — The Reflexion Engine: Sunday Autopsy
================================================
Reads data/decision_log.json for decisions made 5–7 days ago.
Fetches the current live price via yfinance.
Sends an autopsy prompt to DeepSeek Reasoner.
Saves the resulting operational rule to data/lessons_learned.json.

Run weekly (e.g. every Sunday):
    python3 scripts/trade_autopsy.py

Or force-run on all decisions regardless of age:
    python3 scripts/trade_autopsy.py --force
"""

import os, sys, json, argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import yfinance as yf
from openai import OpenAI
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

DS_KEY  = os.getenv("DEEPSEEK_API_KEY", "")
client  = OpenAI(api_key=DS_KEY, base_url="https://api.deepseek.com") if DS_KEY else None

DECISION_LOG  = ROOT / "data" / "decision_log.json"
LESSONS_FILE  = ROOT / "data" / "lessons_learned.json"
AUTOPSY_LOG   = ROOT / "data" / "autopsy_log.json"

AUTOPSY_WINDOW_MIN = 5   # days old
AUTOPSY_WINDOW_MAX = 7   # days old

AUTOPSY_SYSTEM = """You are an AI performing a rigorous Trade Autopsy for a precious metals trading system.
Your job: evaluate whether a past CIO decision was correct given what the price did,
then produce exactly ONE strict operational rule to reinforce success or penalize failure.

Rules for your output:
- Be brutally honest — no softening language
- If the advice was correct: start with "REINFORCE:"
- If the advice was wrong: start with "AVOID:"
- The rule must be actionable and specific (cite price levels, conditions, or signals)
- Output ONLY a JSON object, no extra text:
  {"rule": "<your one-sentence operational rule>", "verdict": "correct|incorrect|neutral"}"""


def _load_decisions() -> list:
    if not DECISION_LOG.exists():
        return []
    with open(DECISION_LOG) as f:
        return json.load(f)


def _load_lessons() -> dict:
    if not LESSONS_FILE.exists():
        return {"lessons": []}
    with open(LESSONS_FILE) as f:
        return json.load(f)


def _save_lessons(data: dict) -> None:
    with open(LESSONS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _load_autopsy_log() -> list:
    if not AUTOPSY_LOG.exists():
        return []
    with open(AUTOPSY_LOG) as f:
        return json.load(f)


def _save_autopsy_log(log: list) -> None:
    with open(AUTOPSY_LOG, "w") as f:
        json.dump(log, f, indent=2)


def _get_live_price(ticker: str) -> float | None:
    try:
        df = yf.download(ticker, period="2d", interval="1d",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, __import__("pandas").MultiIndex):
            df.columns = df.columns.get_level_values(0)
        cl = df["Close"].dropna()
        return float(cl.iloc[-1]) if len(cl) > 0 else None
    except Exception:
        return None


def _days_old(date_str: str) -> int:
    try:
        d = datetime.fromisoformat(date_str.replace("Z", "")).date()
        return (date.today() - d).days
    except Exception:
        return -1


def run_autopsy(force: bool = False) -> int:
    """
    Scan decision_log for decisions 5–7 days old (or all if force=True).
    For each, fetch live price, call DeepSeek Reasoner, save lesson.
    Returns count of autopsies performed.
    """
    if client is None:
        print("  ❌  DEEPSEEK_API_KEY not set — cannot run autopsy.")
        return 0

    decisions    = _load_decisions()
    lessons_data = _load_lessons()
    autopsy_log  = _load_autopsy_log()
    autopsied_ids = {a["decision_id"] for a in autopsy_log}

    if not decisions:
        print("  ℹ️  No decisions in decision_log.json yet.")
        return 0

    count = 0
    for i, dec in enumerate(decisions):
        # Build a stable ID from date + ticker + action
        dec_id = f"{dec.get('timestamp', dec.get('date',''))}_{dec.get('ticker','')}_{i}"

        if dec_id in autopsied_ids:
            continue  # already autopsied

        age = _days_old(dec.get("timestamp", dec.get("date", "")))

        if not force and not (AUTOPSY_WINDOW_MIN <= age <= AUTOPSY_WINDOW_MAX):
            continue

        ticker      = dec.get("ticker", "GC=F")
        old_price   = dec.get("price_at_time", 0.0)
        old_action  = dec.get("action_taken", "HOLD")
        old_reason  = dec.get("original_reasoning", "")
        old_date    = dec.get("date", "")

        print(f"  🔬  Autopsy: {ticker} · {old_action} @ ${old_price:,.2f} on {old_date} "
              f"({age}d ago) …", end="", flush=True)

        live_price = _get_live_price(ticker)
        if live_price is None:
            print("  ⚠️  Could not fetch live price — skipping.")
            continue

        price_chg = ((live_price - old_price) / old_price * 100) if old_price > 0 else 0.0
        name      = "Gold" if ticker == "GC=F" else "Silver"

        user_msg = (
            f"You are performing a Trade Autopsy for {name} ({ticker}).\n\n"
            f"{age} days ago, you told the user to [{old_action}] at ${old_price:,.2f} because:\n"
            f"\"{old_reason}\"\n\n"
            f"The price is now ${live_price:,.2f} ({price_chg:+.2f}% change).\n\n"
            f"Did your advice make the user a profit, save them from a crash, or cause them to lose money?\n"
            f"Write a strict, one-sentence operational rule to reinforce this success or penalize this failure.\n"
            f"Do not repeat mistakes. Return only the JSON object."
        )

        try:
            resp = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": AUTOPSY_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=4000,
            )

            msg = resp.choices[0].message
            raw = (msg.content or "").strip()

            if not raw:
                import re as _re
                rc = getattr(msg, "reasoning_content", "") or ""
                m  = _re.search(r'\{[^{}]*"rule"[^{}]*\}', rc, _re.DOTALL)
                if m:
                    raw = m.group(0)

            if not raw:
                print("  ⚠️  Empty response from DeepSeek — skipping.")
                continue

            raw = raw.strip("`").strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

            result  = json.loads(raw)
            rule    = result.get("rule", "")
            verdict = result.get("verdict", "neutral")

            # Append lesson
            lesson = {
                "date":          date.today().isoformat(),
                "ticker":        ticker,
                "decision_date": old_date,
                "old_action":    old_action,
                "old_price":     old_price,
                "new_price":     live_price,
                "price_change":  round(price_chg, 4),
                "verdict":       verdict,
                "rule":          rule,
            }
            lessons_data["lessons"].append(lesson)
            _save_lessons(lessons_data)

            # Record in autopsy log so we don't re-run this decision
            autopsy_log.append({"decision_id": dec_id, "autopsied_on": date.today().isoformat()})
            _save_autopsy_log(autopsy_log)

            verdict_icon = "✅" if verdict == "correct" else ("❌" if verdict == "incorrect" else "🟡")
            print(f"  {verdict_icon} {verdict.upper()}")
            print(f"      Rule: {rule}")
            count += 1

        except Exception as exc:
            print(f"  ❌  DeepSeek error: {exc}")

    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Autopsy all un-autopsied decisions regardless of age")
    args = parser.parse_args()

    print("\n" + "━" * 60)
    print("  WORLDWIDE AI METALS — Reflexion Engine: Trade Autopsy")
    if args.force:
        print("  Mode: FORCE (all unprocessed decisions)")
    else:
        print(f"  Mode: standard ({AUTOPSY_WINDOW_MIN}–{AUTOPSY_WINDOW_MAX} day window)")
    print("━" * 60)

    n = run_autopsy(force=args.force)

    print("\n" + "━" * 60)
    if n == 0:
        print("  ℹ️  No decisions in the autopsy window. Run again in 5–7 days.")
        print("       Use --force to process all decisions regardless of age.")
    else:
        print(f"  ✅  {n} autopsy/autopsies completed → {LESSONS_FILE}")
    print("━" * 60 + "\n")
