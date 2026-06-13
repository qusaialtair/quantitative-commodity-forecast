"""
PHASE 7 — The Wealth Agent
============================
Agentic chat interface powered by DeepSeek (deepseek-chat) with tool calling.
Loads full portfolio context on startup and can execute portfolio updates live.

Tools available to the agent:
  • update_portfolio_json  — write position changes to portfolio.json
"""

import os, sys, json
from pathlib import Path

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

load_dotenv(ROOT / ".env")

from portfolio_manager import (
    update_portfolio_json,
    get_live_price,
    update_fiat_balance,
    execute_trade,
    get_portfolio_summary,
    get_virtual_cash,
    ask_portfolio_advisor,
    TOOL_SCHEMA,
    GET_LIVE_PRICE_SCHEMA,
    UPDATE_FIAT_SCHEMA,
    EXECUTE_TRADE_SCHEMA,
)

_ALL_TOOLS = [TOOL_SCHEMA, GET_LIVE_PRICE_SCHEMA, UPDATE_FIAT_SCHEMA, EXECUTE_TRADE_SCHEMA]

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fund Manager — Wealth Agent",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── DeepSeek client ────────────────────────────────────────────────────────────
_DS_KEY = os.getenv("DEEPSEEK_API_KEY", "")
_client = OpenAI(api_key=_DS_KEY, base_url="https://api.deepseek.com") if _DS_KEY else None

# ── Load context files ─────────────────────────────────────────────────────────
def _load_master_context() -> str:
    p = ROOT / "data" / "master_context.json"
    if not p.exists():
        return "No consolidated memory available yet."
    try:
        with open(p) as f:
            mc = json.load(f)
        snippets = []
        for entry in mc.get("entries", [])[:10]:
            snippets.append(f"[{entry.get('date','')}] {entry.get('summary','')}")
        return "\n".join(snippets) if snippets else "Memory bank is empty."
    except Exception:
        return "Could not read master context."


def _load_oracle_scores() -> str:
    p = ROOT / "data" / "oracle_history.csv"
    if not p.exists():
        return "No oracle scores logged yet."
    try:
        import pandas as pd
        df = pd.read_csv(p, parse_dates=["date"])
        lines = []
        for ticker in ["GC=F", "SI=F"]:
            sub = df[df["ticker"] == ticker].sort_values("date")
            if not sub.empty:
                row = sub.iloc[-1]
                lines.append(
                    f"{ticker}: score={row['score']:.2f} "
                    f"({row['date'].date()}) — {row['reasoning']}"
                )
        return "\n".join(lines) if lines else "No scores yet."
    except Exception:
        return "Could not read oracle scores."


def _load_lessons() -> str:
    p = ROOT / "data" / "lessons_learned.json"
    if not p.exists():
        return "No lessons learned yet — Reflexion Engine pending first autopsy."
    try:
        with open(p) as f:
            data = json.load(f)
        rules = [
            f"  {i+1}. [{l.get('date','')} {l.get('ticker','')}] {l.get('rule','')}"
            for i, l in enumerate(data.get("lessons", [])[-10:])
        ]
        return "\n".join(rules) if rules else "Lessons file empty."
    except Exception:
        return "Could not load lessons."


def _build_system_prompt() -> str:
    portfolio = get_portfolio_summary()
    master    = _load_master_context()
    oracle    = _load_oracle_scores()
    lessons   = _load_lessons()

    portfolio_str = json.dumps(portfolio, indent=2)

    return f"""You are the Wealth Agent — a world-class AI wealth manager and metals trading advisor.
You serve a high-net-worth principal in the UAE who follows Sharia-compliant investment principles.
You have full authority to advise on Gold (GC=F) and Silver (SI=F) positions.

## Current Portfolio
```json
{portfolio_str}
```

## Latest Oracle Macro Scores (Perplexity Sonar Pro)
{oracle}

## Consolidated Market Memory (most recent entries)
{master}

## Lessons Learned (Reflexion Engine — your past trade autopsies)
{lessons}

## Your Tools
1. `get_live_price(ticker)` — Fetches the real-time market price in USD per troy oz.
2. `execute_trade(action, ticker, fiat_amount)` — Buys or sells Gold/Silver from the fund cash balance. Fetches live price internally. Updates virtual_account.json.
3. `update_fiat_balance(action, amount, currency)` — Deposits or withdraws cash (USD or AED) in the fund account.
4. `update_portfolio_json(ticker, shares, avg_cost, ...)` — Records physical holdings in portfolio.json (use only for recording real physical purchases already made).

## Execution Rules (CRITICAL)
- You are a Portfolio Execution Agent. You CAN and SHOULD transact directly.
- ALWAYS confirm the trade details (ticker, quantity, cost, cash remaining) with the user BEFORE calling execute_trade or update_fiat_balance, UNLESS the user explicitly says "do it", "confirm", or "execute".
- If the user asks to deposit AED, convert to USD (1 USD = 3.6725 AED) via update_fiat_balance with currency="AED".
- If the user asks to buy/sell an asset: fetch price → calculate → confirm → execute.
- Never quote a price from memory. Always call get_live_price first.

## Live Price Rule
Before ANY calculation involving a price: call get_live_price. No exceptions.

## Unit Conversions
- Gold and Silver are priced in USD per troy oz. 1 troy oz = 31.1034768 g.
- "1 kg of silver" = 1000 / 31.1034768 = 32.15 troy oz.
- Convert grams/kg to oz before recording: oz = grams / 31.1034768.

## Fund Structure
- virtual_account.json: cash_balance (available FIAT) + virtual metals positions (execute_trade).
- portfolio.json: physical real-world metals holdings (update_portfolio_json).

## Rules
- Be concise but substantive — no filler
- Respect Sharia principles: no leveraged speculation, physical-backed where possible
- CRITICAL: Adhere to your Lessons Learned — they are hard constraints from your real performance
"""


# ── Tool dispatcher ────────────────────────────────────────────────────────────
def _dispatch_tool(tool_name: str, tool_args: dict) -> str:
    if tool_name == "get_live_price":
        ticker = tool_args.get("ticker", "")
        try:
            price = get_live_price(ticker)
            return json.dumps({
                "ticker": ticker,
                "price_usd_per_oz": round(price, 4),
                "note": "Price in USD per troy oz. 1 troy oz = 31.1034768 g.",
            })
        except RuntimeError as exc:
            return json.dumps({"error": str(exc)})

    if tool_name == "update_fiat_balance":
        result = update_fiat_balance(**tool_args)
        return json.dumps(result)

    if tool_name == "execute_trade":
        result = execute_trade(**tool_args)
        return json.dumps(result)

    if tool_name == "update_portfolio_json":
        result = update_portfolio_json(**tool_args)
        return json.dumps(result)

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


# ── Agentic loop ───────────────────────────────────────────────────────────────
def _run_agent(messages: list) -> str:
    """Run DeepSeek with tool calling until final text response."""
    if _client is None:
        return "⚠️ DeepSeek API key not configured. Set DEEPSEEK_API_KEY in .env"

    loop_messages = list(messages)

    for _ in range(8):  # max 8 tool-call rounds (price fetch + portfolio write = min 2)
        resp = _client.chat.completions.create(
            model="deepseek-chat",
            messages=loop_messages,
            tools=_ALL_TOOLS,
            tool_choice="auto",
            temperature=0.5,
            max_tokens=1024,
        )

        choice = resp.choices[0]

        if choice.finish_reason == "tool_calls":
            # Append assistant message with tool_calls
            loop_messages.append(choice.message.model_dump())

            # Execute each tool call
            for tc in choice.message.tool_calls:
                args   = json.loads(tc.function.arguments)
                result = _dispatch_tool(tc.function.name, args)
                loop_messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result,
                })
        else:
            # Final text response
            return choice.message.content or "(no response)"

    return "Agent loop exhausted — please rephrase your request."


# ── Session state ──────────────────────────────────────────────────────────────
if "wa_messages" not in st.session_state:
    st.session_state.wa_messages = []

if "wa_system" not in st.session_state:
    st.session_state.wa_system = _build_system_prompt()

# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
:root { --gold: #C9A84C; --mono: 'Space Mono', monospace; }
#MainMenu, footer, header, .stDeployButton { visibility: hidden !important; }
section[data-testid="stSidebar"],
[data-testid="stSidebarNav"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stExpandSidebarButton"],
[data-testid="collapsedControl"] { display: none !important; }
.block-container { padding: 1rem 2rem 4rem !important; max-width: 1580px !important; }
/* Bloomberg nav */
.fn-topnav {
    background: #050505; border-bottom: 1px solid #181818;
    display: flex; align-items: center;
    padding: 0 24px; height: 44px;
    margin: -1rem -2rem 0 -2rem;
}
.fn-brand {
    font-family: var(--mono); font-size: 11px; font-weight: 700;
    letter-spacing: .18em; color: var(--gold); text-transform: uppercase; flex-shrink: 0;
}
.fn-brand-sep { width:1px; height:16px; background:#252525; margin:0 16px; flex-shrink:0; }
.fn-page-id   { font-family:var(--mono); font-size:10px; letter-spacing:.14em; color:#2a2a2a; text-transform:uppercase; }
.fn-ts        { font-family:var(--mono); font-size:9px; color:#222; margin-left:auto; }
.fn-nav-row {
    background: #080808; border-bottom: 1px solid #1a1a1a;
    margin: 0 -2rem 1.6rem -2rem; padding: 0 .5rem;
}
.fn-nav-row [data-testid="stHorizontalBlock"] { background:#080808 !important; gap:0 !important; }
.fn-nav-row [data-testid="column"]            { background:#080808 !important; padding:0 !important; }
.fn-nav-row .stButton > button {
    background: transparent !important; border: none !important;
    border-bottom: 2px solid transparent !important; border-radius: 0 !important;
    color: #3e3e3e !important; font-family: var(--mono) !important;
    font-size: 10px !important; font-weight: 700 !important;
    letter-spacing: .12em !important; text-transform: uppercase !important;
    padding: 11px 22px 9px !important; box-shadow: none !important;
    white-space: nowrap !important; width: 100% !important;
    transition: color .1s, border-color .1s, background .1s !important;
}
.fn-nav-row .stButton > button:hover {
    background: #0f0f0f !important; color: #a0a0a0 !important;
    border-bottom-color: rgba(201,168,76,.35) !important; box-shadow: none !important;
}
.fn-nav-row .stButton > button:disabled {
    background: #0c0c0c !important; color: var(--gold) !important;
    border-bottom-color: var(--gold) !important;
    opacity: 1 !important; cursor: default !important; box-shadow: none !important;
}
.wa-header { font-size: 1.3rem; font-weight: 700; margin-bottom: 0.25rem; }
.wa-sub    { font-size: 0.85rem; color: #444; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Top nav ────────────────────────────────────────────────────────────────────
from datetime import datetime as _dt, timezone as _tz
st.markdown(
    f'<div class="fn-topnav">'
    f'<span class="fn-brand">Fund Manager</span>'
    f'<span class="fn-brand-sep"></span>'
    f'<span class="fn-page-id">Wealth Agent</span>'
    f'<span class="fn-ts">{_dt.now(_tz.utc).strftime("%Y-%m-%d  %H:%M UTC")}</span>'
    f'</div>',
    unsafe_allow_html=True,
)
def _safe_switch(candidates: list[str]) -> None:
    last_err = None
    for p in candidates:
        try:
            st.switch_page(p); return
        except Exception as e:
            last_err = e
    st.error("Navigation failed — relaunch with `streamlit run ui/app.py`.")
    if last_err: st.caption(f"detail: {last_err}")

st.markdown('<div class="fn-nav-row">', unsafe_allow_html=True)
_n0, _n1, _n2, _n3, _nsp = st.columns([1, 1, 1, 1.5, 7])
with _n0:
    if st.button("Home", key="topnav_Home"):
        _safe_switch(["pages/0_Home.py", "0_Home.py", "ui/pages/0_Home.py"])
with _n1:
    if st.button("Metals", key="topnav_Metals"):
        _safe_switch(["app.py", "../app.py", "ui/app.py"])
with _n2:
    if st.button("Equities", key="topnav_Equities"):
        _safe_switch(["pages/2_Equity.py", "2_Equity.py", "ui/pages/2_Equity.py"])
with _n3:
    st.button("Wealth Agent", key="topnav_WealthAgent", disabled=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="wa-header">Wealth Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="wa-sub">DeepSeek AI · Portfolio-aware · Tool-enabled</div>', unsafe_allow_html=True)

col_chat, col_ctx = st.columns([3, 1])

with col_ctx:
    with st.expander("Portfolio Context", expanded=True):
        _cash = get_virtual_cash()
        st.markdown(
            f'<div style="background:#0d1a0d;border:1px solid #22c55e44;border-radius:6px;'
            f'padding:8px 12px;margin-bottom:10px;font-family:\'Space Mono\',monospace;">'
            f'<div style="font-size:9px;letter-spacing:.14em;color:#22c55e;margin-bottom:4px">AVAILABLE FIAT</div>'
            f'<div style="font-size:16px;font-weight:700;color:#d8d8d8">${_cash:,.2f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        portfolio = get_portfolio_summary()
        for ticker, pos in portfolio.items():
            name  = "Gold" if ticker == "GC=F" else "Silver"
            shr   = pos.get("shares", 0)
            cost  = pos.get("avg_cost", 0.0)
            if shr > 0:
                st.markdown(f"**{name}** `{ticker}`  \n{shr} units @ ${cost:,.2f}")
            else:
                st.markdown(f"**{name}** `{ticker}`  \nFlat — no position")

    with st.expander("Oracle Scores"):
        st.markdown(_load_oracle_scores())

    if st.button("Refresh Context", key="wa_refresh_ctx"):
        st.session_state.wa_system = _build_system_prompt()
        st.toast("Context refreshed")

    if st.button("Clear Chat", key="wa_clear_chat"):
        st.session_state.wa_messages = []
        st.rerun()

with col_chat:
    # Render chat history
    for msg in st.session_state.wa_messages:
        role = msg["role"]
        if role in ("user", "assistant"):
            with st.chat_message(role):
                st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Ask about your metals portfolio…"):
        # Append user message
        st.session_state.wa_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build full message list with system prompt
        full_messages = [
            {"role": "system", "content": st.session_state.wa_system},
            *st.session_state.wa_messages,
        ]

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = _run_agent(full_messages)
            st.markdown(reply)

        st.session_state.wa_messages.append({"role": "assistant", "content": reply})
