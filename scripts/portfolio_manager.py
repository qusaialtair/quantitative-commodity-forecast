#!/usr/bin/env python3
"""
Portfolio Manager — Tool Function + DeepSeek Direct Advisor
=============================================================
Two responsibilities:
  1. Tool function  — update_portfolio_json() for DeepSeek tool-calling
  2. Direct advisor — ask_portfolio_advisor() queries deepseek-chat with full
     portfolio context to give a direct, specific answer to any question

Schema per ticker:
    {
        "shares":           float,
        "avg_cost":         float,
        "acquisition_date": str (ISO 8601),
        "strategy_mandate": str,
        "status":           str
    }
"""

from __future__ import annotations
import os, json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
from pathlib import Path
from datetime import date, datetime, timezone

from openai import OpenAI
from dotenv import load_dotenv

ROOT           = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

PORTFOLIO_PATH    = ROOT / "data" / "portfolio.json"
VIRTUAL_ACCT_PATH = ROOT / "data" / "virtual_account.json"
PORTFOLIO_PATH.parent.mkdir(parents=True, exist_ok=True)

_AED_TO_USD = 1.0 / 3.6725  # conversion factor

DS_KEY  = os.getenv("DEEPSEEK_API_KEY", "")
_client = OpenAI(api_key=DS_KEY, base_url="https://api.deepseek.com") if DS_KEY else None

VALID_TICKERS = {"GC=F", "SI=F"}


# ── File I/O ───────────────────────────────────────────────────────────────────

def _load() -> dict:
    if not PORTFOLIO_PATH.exists():
        return {t: {"shares": 0, "avg_cost": 0.0,
                    "acquisition_date": "", "strategy_mandate": "active_swing",
                    "status": "active_swing"}
                for t in VALID_TICKERS}
    with open(PORTFOLIO_PATH) as f:
        return json.load(f)


def _save(data: dict) -> None:
    with open(PORTFOLIO_PATH, "w") as f:
        json.dump(data, f, indent=2)


# ── Virtual account I/O ────────────────────────────────────────────────────────

def _load_virtual() -> dict:
    if not VIRTUAL_ACCT_PATH.exists():
        return {"cash_balance": 0.0, "positions": {}, "last_updated": ""}
    with open(VIRTUAL_ACCT_PATH) as f:
        return json.load(f)


def _save_virtual(data: dict) -> None:
    data["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with open(VIRTUAL_ACCT_PATH, "w") as f:
        json.dump(data, f, indent=2)


# ── Tool function (DeepSeek tool-calling schema) ───────────────────────────────

def update_portfolio_json(
    ticker:            str,
    shares:            float,
    avg_cost:          float,
    acquisition_date:  str  = "",
    strategy_mandate:  str  = "active_swing",
) -> dict:
    """
    Update the portfolio entry for a given ticker.

    Parameters
    ----------
    ticker            : "GC=F" (Gold) or "SI=F" (Silver)
    shares            : units held after this update (0 = flat/exited)
    avg_cost          : average cost basis per unit in USD
    acquisition_date  : ISO 8601 date of entry. Defaults to today if shares > 0.
    strategy_mandate  : free-text mandate, e.g. "active_swing", "long_hold"

    Returns
    -------
    {"success": bool, "ticker": str, "portfolio": dict}
    """
    if ticker not in VALID_TICKERS:
        return {"success": False,
                "error": f"Unknown ticker '{ticker}'. Valid: {VALID_TICKERS}"}

    portfolio = _load()

    if not acquisition_date and shares > 0:
        acquisition_date = date.today().isoformat()

    portfolio[ticker] = {
        "shares":           round(float(shares), 6),
        "avg_cost":         round(float(avg_cost), 4),
        "acquisition_date": acquisition_date,
        "strategy_mandate": strategy_mandate,
        "status":           "active_swing",
    }
    _save(portfolio)
    return {"success": True, "ticker": ticker, "portfolio": portfolio[ticker]}


def get_portfolio_summary() -> dict:
    """Return the full portfolio dict — used by the Wealth Agent on startup."""
    return _load()


def get_virtual_cash() -> float:
    """Return the current cash_balance from virtual_account.json."""
    return float(_load_virtual().get("cash_balance", 0.0))


# ── Live price tool ────────────────────────────────────────────────────────────

def get_live_price(ticker: str) -> float:
    """
    Fetch the latest closing price for a ticker via yfinance (8-second timeout).

    Parameters
    ----------
    ticker : yfinance symbol, e.g. "GC=F" (Gold) or "SI=F" (Silver)

    Returns
    -------
    float — price in USD per native unit (troy oz for precious metals)

    Raises
    ------
    RuntimeError if the fetch fails or times out.
    """
    def _fetch() -> float:
        import yfinance as yf
        h = yf.Ticker(ticker).history(period="2d")
        if h.empty:
            raise ValueError(f"No data returned for {ticker!r}")
        return float(h["Close"].dropna().iloc[-1])

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(_fetch).result(timeout=8)
    except _FuturesTimeout:
        raise RuntimeError(f"Price fetch timed out for {ticker!r} (>8s)")
    except Exception as exc:
        raise RuntimeError(f"Could not fetch price for {ticker!r}: {exc}")


# ── FIAT balance tool ──────────────────────────────────────────────────────────

def update_fiat_balance(action: str, amount: float, currency: str = "USD") -> dict:
    """
    Deposit or withdraw cash in the virtual fund account.

    Parameters
    ----------
    action   : "deposit" or "withdraw"
    amount   : amount in the given currency
    currency : "USD" (default) or "AED" (auto-converted at 1 USD = 3.6725 AED)

    Returns
    -------
    dict with success flag, USD amount applied, and new cash_balance.
    """
    if action not in ("deposit", "withdraw"):
        return {"success": False, "error": "action must be 'deposit' or 'withdraw'"}
    if amount <= 0:
        return {"success": False, "error": "amount must be positive"}
    if currency not in ("USD", "AED"):
        return {"success": False, "error": f"currency must be 'USD' or 'AED', got '{currency}'"}

    usd_amount = round(amount * _AED_TO_USD, 2) if currency == "AED" else round(amount, 2)

    acct = _load_virtual()
    cash = float(acct.get("cash_balance", 0.0))

    if action == "deposit":
        new_cash = cash + usd_amount
    else:
        if cash < usd_amount - 0.001:
            return {"success": False,
                    "error": f"Insufficient cash: ${cash:,.2f} available, ${usd_amount:,.2f} needed"}
        new_cash = max(0.0, cash - usd_amount)

    acct["cash_balance"] = round(new_cash, 2)
    _save_virtual(acct)

    return {
        "success":              True,
        "action":               action,
        "amount_original":      amount,
        "currency":             currency,
        "amount_usd":           usd_amount,
        "new_cash_balance_usd": round(new_cash, 2),
    }


# ── Trade execution tool ───────────────────────────────────────────────────────

def execute_trade(action: str, ticker: str, fiat_amount: float) -> dict:
    """
    Execute a virtual metals trade against the fund's cash balance.

    Fetches the live price internally, computes quantity, updates positions
    and cash_balance in virtual_account.json.

    Parameters
    ----------
    action      : "buy" or "sell"
    ticker      : "GC=F" (Gold) or "SI=F" (Silver)
    fiat_amount : USD amount to spend (buy) or target proceeds (sell)

    Returns
    -------
    dict with success flag, executed price, quantity, and updated balances.
    """
    if action not in ("buy", "sell"):
        return {"success": False, "error": "action must be 'buy' or 'sell'"}
    if ticker not in VALID_TICKERS:
        return {"success": False, "error": f"ticker must be one of {VALID_TICKERS}"}
    if fiat_amount <= 0:
        return {"success": False, "error": "fiat_amount must be positive"}

    # Always fetch live price — never use a cached or remembered value
    try:
        price = get_live_price(ticker)
    except RuntimeError as exc:
        return {"success": False, "error": f"Price fetch failed: {exc}"}

    qty = fiat_amount / price  # troy oz

    acct      = _load_virtual()
    cash      = float(acct.get("cash_balance", 0.0))
    positions = acct.get("positions", {})
    existing  = positions.get(ticker, {})
    ex_qty    = float(existing.get("qty",      0.0))
    ex_cost   = float(existing.get("avg_cost", 0.0))

    if action == "buy":
        if cash < fiat_amount - 0.01:
            return {"success": False,
                    "error": f"Insufficient cash: ${cash:,.2f} available, ${fiat_amount:,.2f} needed"}
        new_qty  = ex_qty + qty
        new_cost = ((ex_qty * ex_cost) + fiat_amount) / new_qty if new_qty > 0 else price
        positions[ticker] = {"qty": round(new_qty, 6), "avg_cost": round(new_cost, 4)}
        acct["cash_balance"] = round(cash - fiat_amount, 2)

    else:  # sell
        if ex_qty < qty - 0.0001:
            return {"success": False,
                    "error": f"Insufficient position: {ex_qty:.4f} oz held, {qty:.4f} oz needed"}
        new_qty = ex_qty - qty
        if new_qty < 0.0001:
            positions.pop(ticker, None)   # fully flat
            new_qty = 0.0
        else:
            positions[ticker] = {"qty": round(new_qty, 6), "avg_cost": round(ex_cost, 4)}
        acct["cash_balance"] = round(cash + fiat_amount, 2)

    acct["positions"] = positions
    _save_virtual(acct)

    return {
        "success":              True,
        "action":               action,
        "ticker":               ticker,
        "qty_oz":               round(qty, 6),
        "price_usd_per_oz":     round(price, 4),
        "fiat_usd":             round(fiat_amount, 2),
        "new_cash_balance_usd": acct["cash_balance"],
        "position_after":       positions.get(ticker, {"qty": 0, "avg_cost": 0}),
        "note":                 "1 troy oz = 31.1034768 g",
    }


# ── DeepSeek direct advisor ───────────────────────────────────────────────────

def _build_portfolio_context() -> str:
    """Assemble full context block for the advisor: portfolio + oracle + lessons."""
    portfolio = _load()

    # Oracle scores
    oracle_txt = ""
    oracle_path = ROOT / "data" / "oracle_history.csv"
    if oracle_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(oracle_path, parse_dates=["date"])
            lines = []
            for ticker in ["GC=F", "SI=F"]:
                sub = df[df["ticker"] == ticker].sort_values("date")
                if not sub.empty:
                    row = sub.iloc[-1]
                    lines.append(
                        f"  {ticker}: score={row['score']:.2f} "
                        f"({row['date'].date()}) — {row['reasoning']}"
                    )
            oracle_txt = "\n".join(lines)
        except Exception:
            oracle_txt = "Oracle data unavailable."

    # Lessons learned
    lessons_txt = ""
    lessons_path = ROOT / "data" / "lessons_learned.json"
    if lessons_path.exists():
        try:
            with open(lessons_path) as f:
                ld = json.load(f)
            rules = [
                f"  {i+1}. [{l.get('date','')} {l.get('ticker','')}] {l.get('rule','')}"
                for i, l in enumerate(ld.get("lessons", [])[-10:])
            ]
            lessons_txt = "\n".join(rules) if rules else "No lessons yet."
        except Exception:
            lessons_txt = "Could not load lessons."

    # Master context
    master_txt = ""
    mc_path = ROOT / "data" / "master_context.json"
    if mc_path.exists():
        try:
            with open(mc_path) as f:
                mc = json.load(f)
            master_txt = mc.get("macro_thesis", "")
        except Exception:
            master_txt = ""

    return f"""## Current Portfolio (data/portfolio.json)
{json.dumps(portfolio, indent=2)}

## Latest Oracle Macro Scores
{oracle_txt or "No oracle data yet."}

## Lessons Learned (Reflexion Engine)
{lessons_txt}

## Macro Memory Thesis
{master_txt or "No consolidated memory yet."}"""


ADVISOR_SYSTEM = """You are the Portfolio Advisor — a direct, hyper-specific AI wealth manager
for a high-net-worth principal in the UAE with Sharia-compliant investment principles.
You have full visibility into the user's metals portfolio, oracle macro scores, and your own
lessons learned from past trade autopsies.

Rules:
- Be direct and specific — cite exact prices, dates, percentages
- No disclaimers, no hedging, no filler
- Respect Sharia principles: physical metals, no leverage, no derivatives
- If you recommend an action, be explicit about the entry/exit price and timing
- Always reference the user's actual position data in your answer"""


def ask_portfolio_advisor(question: str, extra_context: str = "") -> str:
    """
    Ask a direct question to the DeepSeek portfolio advisor.
    Automatically loads full portfolio context.

    Parameters
    ----------
    question      : The user's question
    extra_context : Optional extra data to inject (e.g. live LSTM predictions)

    Returns
    -------
    str — DeepSeek's direct answer
    """
    if _client is None:
        return "⚠️ DeepSeek API key not configured. Set DEEPSEEK_API_KEY in .env"

    ctx = _build_portfolio_context()
    if extra_context:
        ctx += f"\n\n## Additional Context\n{extra_context}"

    user_msg = f"{ctx}\n\n## Question\n{question}"

    try:
        resp = _client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": ADVISOR_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.4,
            max_tokens=1024,
        )
        return resp.choices[0].message.content or "(no response)"
    except Exception as exc:
        return f"Advisor error: {exc}"


# ── DeepSeek tool schema ───────────────────────────────────────────────────────

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "update_portfolio_json",
        "description": (
            "Update or record a metals portfolio position in the local portfolio.json file. "
            "Call this whenever the user confirms a trade, changes their cost basis, "
            "or adjusts their position in Gold (GC=F) or Silver (SI=F)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "enum": ["GC=F", "SI=F"],
                    "description": "The metal ticker: GC=F for Gold, SI=F for Silver.",
                },
                "shares": {
                    "type": "number",
                    "description": "Units held after this update. Use 0 to mark as flat/exited.",
                },
                "avg_cost": {
                    "type": "number",
                    "description": "Average cost per unit in USD.",
                },
                "acquisition_date": {
                    "type": "string",
                    "description": "ISO 8601 date of entry, e.g. '2026-03-28'. Defaults to today.",
                },
                "strategy_mandate": {
                    "type": "string",
                    "description": (
                        "Trading mandate for this position, e.g. 'active_swing', "
                        "'long_hold', 'Sharia_compliant_physical'."
                    ),
                },
            },
            "required": ["ticker", "shares", "avg_cost"],
        },
    },
}


UPDATE_FIAT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "update_fiat_balance",
        "description": (
            "Deposit or withdraw cash in the virtual fund account (virtual_account.json). "
            "Supports USD and AED — AED is auto-converted at 1 USD = 3.6725 AED. "
            "Call this when the user wants to add or remove capital from the fund."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["deposit", "withdraw"],
                    "description": "'deposit' to add funds, 'withdraw' to remove funds.",
                },
                "amount": {
                    "type": "number",
                    "description": "Amount to deposit or withdraw in the stated currency.",
                },
                "currency": {
                    "type": "string",
                    "enum": ["USD", "AED"],
                    "description": "Currency of amount. AED is converted to USD automatically.",
                },
            },
            "required": ["action", "amount"],
        },
    },
}

EXECUTE_TRADE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "execute_trade",
        "description": (
            "Execute a virtual metals trade (buy or sell) using the fund's cash balance. "
            "Fetches the live price automatically. "
            "Deducts from (buy) or credits to (sell) cash_balance in virtual_account.json. "
            "Call this when the user wants to trade Gold or Silver with a specific dollar amount."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["buy", "sell"],
                    "description": "'buy' to purchase, 'sell' to liquidate.",
                },
                "ticker": {
                    "type": "string",
                    "enum": ["GC=F", "SI=F"],
                    "description": "Asset ticker: 'GC=F' for Gold, 'SI=F' for Silver.",
                },
                "fiat_amount": {
                    "type": "number",
                    "description": "USD amount to spend (buy) or receive (sell). Quantity is calculated from live price.",
                },
            },
            "required": ["action", "ticker", "fiat_amount"],
        },
    },
}

GET_LIVE_PRICE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_live_price",
        "description": (
            "Fetch the current live market price for a metals ticker via yfinance. "
            "Returns USD per troy ounce for precious metals (GC=F, SI=F). "
            "You MUST call this before calculating any trade cost, total value, or "
            "portfolio update — never use a cached or remembered price for live maths."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": (
                        "yfinance ticker symbol. Use 'GC=F' for Gold, 'SI=F' for Silver."
                    ),
                },
            },
            "required": ["ticker"],
        },
    },
}

# ── CLI test ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if "--ask" in sys.argv:
        idx = sys.argv.index("--ask")
        q   = " ".join(sys.argv[idx+1:]) if idx + 1 < len(sys.argv) else "What is my current P&L?"
        print("\n💬 Asking Portfolio Advisor:", q)
        print("─" * 60)
        print(ask_portfolio_advisor(q))
        print("─" * 60)
    else:
        print("Portfolio:", json.dumps(_load(), indent=2))
        print("\nTool schema:", TOOL_SCHEMA["function"]["name"])
        print("\nRun with --ask 'your question' to test the advisor.")
