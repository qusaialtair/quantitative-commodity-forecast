#!/usr/bin/env python3
"""
scripts/telegram_notifier.py
=============================
Lightweight synchronous Telegram alerting module.
Uses requests.post() to the Bot API — no async libraries, no heavy dependencies.

Message types:
  send_urgent(api_name, error, context)   — URGENT alert for API failures
  send_heartbeat(pipeline_state)          — daily summary after pipeline completes
  send_alert(text, level)                 — low-level primitive for any custom message
  notify_execution_summary(trades, ...)   — PM-authorize execution broadcast
  notify_compliance_shift(prev, new, ...) — treasury Sharia gate transition
  notify_system_halted(source)            — operator emergency halt

All functions are fire-and-forget: they NEVER raise. A Telegram failure
cannot cascade into a pipeline failure.

Deduplication: the same API alert will not be sent more than once every
ALERT_COOLDOWN_S seconds (prevents alert storms from parallel agent calls).

Setup:
  1. Create a bot via Telegram's @BotFather → get TELEGRAM_BOT_TOKEN
  2. Start a chat with your bot, then visit:
       https://api.telegram.org/bot<TOKEN>/getUpdates
     to find your TELEGRAM_CHAT_ID
  3. Add both to .env:
       TELEGRAM_BOT_TOKEN=123456789:ABCdef...
       TELEGRAM_CHAT_ID=987654321       (your personal chat ID, negative for groups)

Test:
  python3 scripts/telegram_notifier.py --test
  python3 scripts/telegram_notifier.py --test-urgent DeepSeek
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import json

import requests

# ── Portfolio file paths (for heartbeat — reads real holdings, not shadow book) ─
_PORTFOLIO_FILE    = ROOT / "data" / "portfolio.json"
_VIRTUAL_ACCT_FILE = ROOT / "data" / "virtual_account.json"


def _read_physical_holdings() -> dict:
    """Return physical metals holdings from portfolio.json (source of truth)."""
    if _PORTFOLIO_FILE.exists():
        try:
            return json.loads(_PORTFOLIO_FILE.read_text())
        except Exception:
            pass
    return {}


def _read_virtual_cash() -> float:
    """Return available cash balance from virtual_account.json."""
    if _VIRTUAL_ACCT_FILE.exists():
        try:
            return float(json.loads(_VIRTUAL_ACCT_FILE.read_text()).get("cash_balance", 0.0))
        except Exception:
            pass
    return 0.0


# ── Configuration ─────────────────────────────────────────────────────────────

_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID",   "")
_API_URL    = f"https://api.telegram.org/bot{_BOT_TOKEN}/sendMessage"
_TIMEOUT_S  = 8            # Telegram API call timeout
_MAX_MSG    = 4000         # Telegram limit is 4096; leave headroom
_ERROR_TRUNC = 350         # max chars for raw exception message in alerts

# Deduplication: don't repeat the same API alert within this window
ALERT_COOLDOWN_S = 120
_last_alert: dict[str, float] = {}   # api_name → last-sent epoch

_ENABLED = bool(_BOT_TOKEN and _CHAT_ID)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _truncate(text: str, n: int) -> str:
    text = str(text).strip()
    return text if len(text) <= n else text[:n] + "…"


def _html_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


_OVERRIDE_FILE = ROOT / "data" / "operator_override.json"


def is_execution_cleared_today() -> bool:
    """True when operator_override.json has AUTHORIZE for today's UTC date."""
    today = datetime.now(timezone.utc).date().isoformat()
    if not _OVERRIDE_FILE.exists():
        return False
    try:
        override = json.loads(_OVERRIDE_FILE.read_text())
    except Exception:
        return False
    return (
        str(override.get("action", "")).upper() == "AUTHORIZE"
        and override.get("cleared_for_date") == today
    )


def _is_debounced(key: str) -> bool:
    """Return True (and do NOT update timer) if this key is still in cooldown."""
    now  = time.time()
    last = _last_alert.get(key, 0.0)
    if now - last < ALERT_COOLDOWN_S:
        return True
    _last_alert[key] = now
    return False


def _post(text: str) -> bool:
    """
    POST a message to Telegram. Returns True on success.
    Never raises — all errors are caught and printed to stderr.
    """
    if not _ENABLED:
        print(
            f"[telegram_notifier] Disabled (TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID "
            f"not set in .env). Message would have been:\n{text[:200]}",
            file=sys.stderr,
        )
        return False

    payload = {
        "chat_id":    _CHAT_ID,
        "text":       text[:_MAX_MSG],
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(_API_URL, json=payload, timeout=_TIMEOUT_S)
        if not resp.ok:
            print(
                f"[telegram_notifier] Telegram API returned {resp.status_code}: "
                f"{resp.text[:200]}",
                file=sys.stderr,
            )
            return False
        return True
    except Exception as exc:
        print(f"[telegram_notifier] Failed to send message: {exc}", file=sys.stderr)
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def send_alert(text: str, level: str = "INFO") -> bool:
    """
    Send any arbitrary message. level is used only for the emoji prefix.
    Low-level primitive — use send_urgent() or send_heartbeat() for standard alerts.
    """
    icons = {"URGENT": "🚨", "WARNING": "⚠️", "INFO": "ℹ️", "OK": "✅"}
    icon  = icons.get(level.upper(), "ℹ️")
    return _post(f"{icon} {text}")


def send_urgent(
    api_name: str,
    error:    Exception | str,
    context:  str = "",
) -> bool:
    """
    Send an URGENT alert for an API failure (429, 500, billing cap, etc.).
    Deduplicated per api_name — at most once per ALERT_COOLDOWN_S seconds.

    Parameters
    ----------
    api_name : human-readable API label, e.g. "DeepSeek", "Perplexity"
    error    : the exception or error string
    context  : extra context, e.g. "model=deepseek-reasoner" or "gold/fed_sentiment"
    """
    if _is_debounced(api_name):
        return False   # silent — already alerted recently for this API

    err_str = _truncate(repr(error) if isinstance(error, Exception) else str(error),
                        _ERROR_TRUNC)

    # Classify the failure type for a clearer subject line
    err_lower = err_str.lower()
    if "429" in err_str or "rate" in err_lower or "quota" in err_lower:
        subject = "RATE LIMIT / QUOTA EXCEEDED"
    elif "billing" in err_lower or "balance" in err_lower or "payment" in err_lower:
        subject = "BILLING CAP REACHED"
    elif "500" in err_str or "502" in err_str or "503" in err_str:
        subject = "SERVER ERROR (5xx)"
    else:
        subject = "API FAILURE"

    lines = [
        "🚨 <b>URGENT — API FAILURE</b>",
        "",
        f"<b>API:</b>    <code>{api_name}</code>",
        f"<b>Type:</b>   <code>{subject}</code>",
        f"<b>Error:</b>  <code>{err_str}</code>",
    ]
    if context:
        lines.append(f"<b>Context:</b> <code>{_truncate(context, 200)}</code>")
    lines += [
        f"<b>Time:</b>   <code>{_now_utc()}</code>",
        "",
        "Pipeline has been notified. Check API billing dashboard.",
    ]
    return _post("\n".join(lines))


def send_heartbeat(pipeline_state: dict) -> bool:
    """
    Send a daily summary after master_controller.py finishes.
    Reads directly from the pipeline_state.json structure.
    """
    status = pipeline_state.get("pipeline_status", "UNKNOWN")
    date   = pipeline_state.get("run_date",        "")
    ticker = pipeline_state.get("ticker",           "GC=F")

    pf  = pipeline_state.get("portfolio",  {})
    rg  = pipeline_state.get("regime",     {})
    rk  = pipeline_state.get("risk",       {})
    cm  = pipeline_state.get("committee",  {})
    stg = pipeline_state.get("stages",     {})

    # ── Status icon ───────────────────────────────────────────────────────────
    if status == "SUCCESS":
        icon, subj = "✅", "Pipeline Complete"
    elif status == "PARTIAL":
        icon, subj = "⚠️", "Pipeline PARTIAL"
    elif status == "ABORTED":
        icon, subj = "🚨", "Pipeline ABORTED"
    else:
        icon, subj = "ℹ️", f"Pipeline {status}"

    # ── Portfolio — physical metals + Phase XIV multi-strategy book ───────────
    _phys      = _read_physical_holdings()
    _gold_e    = _phys.get("GC=F", {})
    _silver_e  = _phys.get("SI=F", {})
    gold_oz    = float(_gold_e.get("shares",   0.0))
    gold_cost  = float(_gold_e.get("avg_cost", 0.0))
    silver_oz  = float(_silver_e.get("shares", 0.0))
    virt_cash  = _read_virtual_cash()

    _p14_path = ROOT / "data" / "phase14_book.json"
    _p14: dict = {}
    if _p14_path.exists():
        try:
            _p14 = json.loads(_p14_path.read_text())
        except Exception:
            pass
    _open_trades = _p14.get("open_trades") or []
    _hedge = _p14.get("hedge_state") or {}
    _gld_inst = _hedge.get("instrument") or "GLD"

    unreal    = float(pf.get("unrealised_pnl", _p14.get("open_pl_usd", 0) or 0))
    unreal_p  = float(pf.get("unrealised_pct", 0))
    pf_state  = (
        "ACTIVE" if gold_oz > 0 or silver_oz > 0 or _open_trades else "FIAT"
    )
    pnl_sign  = "+" if unreal >= 0 else ""
    pnl_icon  = "📈" if unreal >= 0 else "📉"

    # ── Regime ────────────────────────────────────────────────────────────────
    hmm_state = str(rg.get("hmm_state",       "UNKNOWN"))
    hmm_veto  = bool(rg.get("hmm_veto_active", False))
    p_bull    = float(rg.get("p_bullish",      0))
    p_vol     = float(rg.get("p_volatile",     rg.get("p_ranging", 0)))
    p_bear    = float(rg.get("p_bearish",      0))

    regime_icons = {"BULLISH": "🟢", "VOLATILE": "🟡", "BEARISH": "🔴"}
    rg_icon = regime_icons.get(hmm_state, "⚪")

    # Show active-state probability with 1 decimal place.
    # :.0% would round 0.9997 → "100%" — masking the true float value.
    if hmm_state == "BULLISH":
        prob_str = f"P(bull)={p_bull:.1%}  P(vol)={p_vol:.1%}  P(bear)={p_bear:.1%}"
    elif hmm_state in ("VOLATILE", "RANGING"):
        prob_str = f"P(vol)={p_vol:.1%}  P(bull)={p_bull:.1%}  P(bear)={p_bear:.1%}"
    elif hmm_state == "BEARISH":
        prob_str = f"P(bear)={p_bear:.1%}  P(vol)={p_vol:.1%}  P(bull)={p_bull:.1%}"
    else:
        prob_str = f"{p_bull:.1%} / {p_vol:.1%} / {p_bear:.1%}"

    # ── Committee ─────────────────────────────────────────────────────────────
    action  = str(cm.get("action_taken",    "HOLD_METAL"))
    q_conv  = cm.get("quant_conviction")
    m_conv  = cm.get("macro_conviction")
    o_scr   = cm.get("oracle_score")
    veto_a  = bool(cm.get("veto_active",   False))

    action_icons = {
        "ACCUMULATE":     "🟢 ACCUMULATE",
        "HOLD_METAL":     "⚪ HOLD_METAL",
        "STRATEGIC_EXIT": "🔴 STRATEGIC_EXIT",
        "RE_ENTER":       "🟢 RE_ENTER",
    }
    action_str = action_icons.get(action, action)

    # ── Risk ──────────────────────────────────────────────────────────────────
    var_95   = rk.get("var_95_daily")
    tw       = rk.get("target_weight")
    override = bool(rk.get("var_override", False))

    var_str = f"{float(var_95):.2%}/day" if var_95 is not None else "N/A"
    tw_str  = f"{float(tw):.1%}"         if tw       is not None else "N/A"

    # ── Stage summary ─────────────────────────────────────────────────────────
    n_ok     = sum(1 for v in stg.values() if v.get("status") in ("OK", "DRY_RUN"))
    n_total  = len(stg)
    failed   = [k for k, v in stg.items() if v.get("status") == "FAILED"]
    total_s  = sum(v.get("duration_s", 0) for v in stg.values())
    abort_r  = pipeline_state.get("abort_reason", "")

    # ── Build message ─────────────────────────────────────────────────────────
    lines = [
        f"{icon} <b>{subj} — {date}</b>",
        f"<b>Ticker:</b> <code>{ticker}</code>",
        "",
        "─── Committee ───────────────────",
        f"<b>Action:</b>  <code>{action_str}</code>",
    ]

    if q_conv is not None and m_conv is not None:
        lines.append(
            f"<b>Conviction:</b> "
            f"<code>Quant {'+' if int(q_conv)>=0 else ''}{q_conv}  |  "
            f"Macro {'+' if int(m_conv)>=0 else ''}{m_conv}</code>"
        )
    if o_scr is not None:
        lines.append(f"<b>Oracle score:</b> <code>{float(o_scr):.2f}</code>")
    if veto_a:
        lines.append("<b>VETO:</b> <code>Active — deploy suppressed</code>")

    lines += [
        "",
        "─── Regime ──────────────────────",
        f"<b>HMM state:</b>  <code>{rg_icon} {hmm_state}"
        f"{'  (veto)' if hmm_veto else ''}</code>",
        f"<b>P(active state):</b> <code>{prob_str}</code>",
        "",
        "─── Risk ────────────────────────",
        f"<b>VaR 95%:</b>     <code>{var_str}</code>",
        f"<b>Target weight:</b> <code>{tw_str}</code>"
        + ("<code>  [VaR override]</code>" if override else ""),
        "",
        "─── Portfolio ───────────────────",
        f"<b>State:</b>      <code>{pf_state}</code>",
        f"<b>Gold:</b>       <code>{gold_oz:.4f} oz"
        + (f" @ ${gold_cost:,.0f} avg" if gold_oz > 0 and gold_cost > 0 else "") + "</code>",
    ]
    if silver_oz > 0:
        lines.append(f"<b>Silver:</b>     <code>{silver_oz:.4f} oz</code>")
    if _open_trades:
        tickers = ", ".join(
            f"{t.get('ticker', '?')}" for t in _open_trades[:5]
        )
        lines.append(f"<b>Phase XIV:</b>  <code>{_html_escape(tickers)}</code>")
    if _hedge.get("instrument"):
        lines.append(
            f"<b>Hedge sleeve:</b> <code>{_html_escape(str(_gld_inst))} "
            f"{float(_hedge.get('allocation_pct') or 0):.0f}%</code>"
        )
    lines += [
        f"<b>FIAT Cash:</b>  <code>${virt_cash:,.2f}</code>",
        f"<b>Open P&amp;L:</b>  <code>{pnl_icon} {pnl_sign}${unreal:,.0f} ({pnl_sign}{unreal_p:.2f}%)</code>",
        "",
        "─── Pipeline ────────────────────",
        f"<b>Stages:</b> <code>{n_ok}/{n_total} OK  ({total_s:.0f}s)</code>",
    ]

    if failed:
        lines.append(f"<b>Failed:</b> <code>{', '.join(failed)}</code>")
    if abort_r:
        lines.append(f"<b>Abort:</b>  <code>{_truncate(abort_r, 200)}</code>")

    lines.append(f"\n<i>{_now_utc()}</i>")
    return _post("\n".join(lines))


def notify_execution_summary(
    trades: list[dict],
    *,
    source: str = "order_router",
    execution_mode: str | None = None,
    book_equity_usd: float | None = None,
) -> bool:
    """
    Broadcast a formatted summary of executed trades (Event A).
    ``trades`` items may include ticker, side, qty, notional_usd, price,
    strategy, status, sub_tag. Never raises.
    """
    if not trades:
        return False

    mode = execution_mode or os.getenv("EXECUTION_MODE", "paper_internal")
    n = len(trades)

    lines = [
        "✅ <b>EXECUTION CONFIRMED</b>",
        "",
        f"<b>Source:</b>  <code>{_html_escape(source)}</code>",
        f"<b>Mode:</b>    <code>{_html_escape(mode)}</code>",
        f"<b>Trades:</b>  <code>{n}</code>",
    ]
    if book_equity_usd is not None:
        lines.append(f"<b>Book NAV:</b> <code>${book_equity_usd:,.2f}</code>")
    lines.append("")

    for i, t in enumerate(trades[:25], 1):
        side = _html_escape(t.get("side", "—"))
        ticker = _html_escape(t.get("ticker", "—"))
        qty = t.get("qty")
        qty_s = f"{float(qty):.4f}" if qty is not None else "—"
        notional = t.get("notional_usd")
        notional_s = f"${float(notional):,.0f}" if notional is not None else ""
        status = _html_escape(t.get("status", t.get("ibkr_status", "EXECUTED")))
        strategy = _html_escape(t.get("strategy", ""))
        sub = t.get("sub_tag")
        row = (
            f"{i}. <b>{side}</b> <code>{ticker}</code>  "
            f"<code>{qty_s}</code>"
        )
        if notional_s:
            row += f"  <code>{notional_s}</code>"
        row += f"  <i>{status}</i>"
        if strategy:
            row += f"  [{strategy}]"
        if sub:
            row += f"  <code>{_html_escape(sub)}</code>"
        lines.append(row)

    if n > 25:
        lines.append(f"<i>…and {n - 25} more</i>")

    lines.append(f"\n<i>{_now_utc()}</i>")
    return _post("\n".join(lines))


def notify_compliance_shift(
    previous_gate: str | None,
    new_gate: str,
    *,
    allocation_pct: float = 0.0,
    effective_instrument: str | None = None,
    regime_quadrant: str | None = None,
    crisis_tier: str | None = None,
) -> bool:
    """
    Alert on treasury hedge gate_action transition (Event B). Never raises.
    """
    prev = (previous_gate or "").upper() or "UNKNOWN"
    new = (new_gate or "").upper()
    if not new or prev == new:
        return False

    pct = allocation_pct if allocation_pct > 0 else 20.0

    if prev == "CLEARED_SOVEREIGN" and new == "SHARIA_FALLBACK_GLD":
        headline = (
            "⚠️ COMPLIANCE SHIFT: Sovereign Debt Gate Locked. "
            f"{pct:.0f}% Defensive Allocation rerouted to Physical Gold."
        )
    elif new == "SHARIA_FALLBACK_GLD":
        headline = (
            "⚠️ COMPLIANCE SHIFT: Sovereign Debt Gate Locked. "
            f"{pct:.0f}% Defensive Allocation rerouted to Physical Gold."
        )
    elif new == "CLEARED_SOVEREIGN":
        headline = (
            "✅ COMPLIANCE SHIFT: Sovereign debt gate cleared. "
            "Defensive sleeve may route to TLT/IEF."
        )
    else:
        headline = f"⚠️ COMPLIANCE SHIFT: <code>{_html_escape(prev)}</code> → <code>{_html_escape(new)}</code>"

    lines = [
        f"<b>{headline}</b>",
        "",
        f"<b>Previous gate:</b> <code>{_html_escape(prev)}</code>",
        f"<b>New gate:</b>      <code>{_html_escape(new)}</code>",
    ]
    if effective_instrument:
        lines.append(
            f"<b>Effective:</b>     <code>{_html_escape(effective_instrument)}</code>  "
            f"<code>{pct:.1f}%</code>"
        )
    if regime_quadrant:
        lines.append(f"<b>Regime:</b>        <code>{_html_escape(regime_quadrant)}</code>")
    if crisis_tier:
        lines.append(f"<b>Crisis tier:</b>   <code>{_html_escape(crisis_tier)}</code>")
    lines.append(f"\n<i>{_now_utc()}</i>")
    return _post("\n".join(lines))


def notify_system_halted(
    source: str = "EXECUTIVE_OVERRIDE",
    *,
    briefing: dict | None = None,
) -> bool:
    """High-priority operator halt alert (Event C). Never raises."""
    lines = [
        "🛑 <b>SYSTEM HALTED BY OPERATOR</b>",
        "All routing suspended — liquidate-to-cash posture armed.",
        "",
        f"<b>Source:</b> <code>{_html_escape(source)}</code>",
        f"<b>Time:</b>   <code>{_now_utc()}</code>",
        "",
        "order_router returns HALTED until AUTHORIZE clears the flag.",
    ]
    if briefing:
        lines += ["", "─── Market Snapshot ─────────────"]
        for key, label in (
            ("market", "Market"),
            ("holdings", "Holdings"),
            ("watchlist", "Watchlist"),
        ):
            val = briefing.get(key) or briefing.get("summary", "")
            if val:
                lines.append(f"<b>{label}:</b> {_html_escape(_truncate(str(val), 280))}")
    return _post("\n".join(lines))


def notify_operator_authorize(
    pipeline_status: dict,
    briefing: dict | None = None,
) -> bool:
    """Telegram after AUTHORIZE + master_controller run completes. Never raises."""
    exit_code = pipeline_status.get("exit_code")
    success = pipeline_status.get("success", exit_code == 0)
    icon = "✅" if success else "⚠️"

    lines = [
        f"{icon} <b>PIPELINE AUTHORIZED — master_controller finished</b>",
        "",
        f"<b>Exit code:</b> <code>{exit_code}</code>",
        f"<b>Source:</b>    <code>{_html_escape(str(pipeline_status.get('source', 'AUTHORIZE')))}</code>",
        f"<b>Started:</b>   <code>{_html_escape(str(pipeline_status.get('started_at', '—')))}</code>",
        f"<b>Finished:</b>  <code>{_html_escape(str(pipeline_status.get('finished_at', '—')))}</code>",
    ]

    if briefing:
        lines += ["", "─── DeepSeek Executive Summary ──"]
        for key, label in (
            ("market", "Gold / Silver"),
            ("holdings", "Held Positions"),
            ("watchlist", "Watchlist"),
            ("action", "Operator Action"),
        ):
            val = briefing.get(key)
            if val:
                lines.append(f"<b>{label}:</b> {_html_escape(_truncate(str(val), 320))}")
        if not any(briefing.get(k) for k in ("market", "holdings", "watchlist")):
            summary = briefing.get("summary")
            if summary:
                lines.append(_html_escape(_truncate(str(summary), 1200)))

    err_tail = pipeline_status.get("stderr_tail") or ""
    if not success and err_tail:
        lines += ["", f"<b>stderr:</b> <code>{_html_escape(_truncate(err_tail, 400))}</code>"]

    lines.append(f"\n<i>{_now_utc()}</i>")
    return _post("\n".join(lines))


def notify_post_authorize_execution(summary_path: Path | None = None) -> bool:
    """
    After an AUTHORIZE-triggered pipeline, read multi_strategy_trader.json and
    broadcast any trades from the latest run. Never raises.
    """
    if not is_execution_cleared_today():
        return False

    path = summary_path or (ROOT / "data" / "multi_strategy_trader.json")
    if not path.exists():
        return False
    try:
        summary = json.loads(path.read_text())
    except Exception:
        return False

    n_new = int(summary.get("n_new_trades") or 0)
    n_hedge = int(summary.get("n_hedge_trades") or 0)
    if n_new <= 0 and n_hedge <= 0:
        return False

    try:
        from scripts.order_router import pop_routed_orders
        routed = pop_routed_orders()
    except Exception:
        routed = []

    trades: list[dict] = list(routed)
    if not trades:
        book_path = ROOT / "data" / "phase14_book.json"
        open_by_id: dict[str, dict] = {}
        if book_path.exists():
            try:
                book = json.loads(book_path.read_text())
                for t in book.get("open_trades") or []:
                    tid = t.get("trade_id")
                    if tid:
                        open_by_id[str(tid)] = t
            except Exception:
                pass
        for tid in summary.get("new_trade_ids") or []:
            t = open_by_id.get(str(tid), {})
            trades.append({
                "ticker":       t.get("ticker") or (str(tid).split("-")[-1] if "-" in str(tid) else "—"),
                "side":         t.get("side", "LONG"),
                "qty":          t.get("qty"),
                "notional_usd": t.get("notional_usd"),
                "status":       t.get("ibkr_status", "EXECUTED"),
                "strategy":     t.get("strategy") or summary.get("strategy"),
                "sub_tag":      t.get("sub_tag"),
            })
        if n_hedge > 0:
            hs = summary.get("hedge_state") or {}
            trades.append({
                "ticker":       hs.get("instrument") or "GLD",
                "side":         "LONG",
                "status":       "EXECUTED",
                "strategy":     "TREASURY_HEDGE",
                "sub_tag":      hs.get("sub_tag"),
                "notional_usd": summary.get("hedge_notional_usd"),
            })

    return notify_execution_summary(
        trades,
        source="AUTHORIZE pipeline",
        book_equity_usd=summary.get("book_equity_usd"),
    )


def send_training_ready() -> bool:
    """Send TRAINING SUFFICIENT notification."""
    lines = [
        "✅ <b>TRAINING SUFFICIENT</b>",
        "",
        "You may proceed with real trading integration.",
        "",
        "All models trained and validated:",
        "• GoldLSTM-v1 — val_loss healthy, fine-tune trend improving",
        "• HMM Regime Detector — 3-state (BULLISH/VOLATILE/BEARISH) fitted",
        "• Proving Ground — 6 tri-horizon models (GC=F + SI=F) trained",
        "• Signal Engine — 20+ factor quantitative analysis active",
        "• Risk Manager — VaR + vol-adjusted sizing operational",
        "",
        f"<i>{_now_utc()}</i>",
    ]
    return _post("\n".join(lines))


# ── CLI test tool ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Telegram Notifier — test and diagnostic tool")
    parser.add_argument("--test",        action="store_true",
        help="Send a test heartbeat using current pipeline_state.json")
    parser.add_argument("--test-urgent", metavar="API_NAME", default="",
        help="Send a mock URGENT alert for the named API")
    parser.add_argument("--send",        metavar="MESSAGE", default="",
        help="Send a raw message")
    parser.add_argument("--training-ready", action="store_true",
        help="Send TRAINING SUFFICIENT notification")
    args = parser.parse_args()

    if not args.test and not args.test_urgent and not args.send and not args.training_ready:
        parser.print_help()
        return

    if not _ENABLED:
        print("ERROR: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env")
        sys.exit(1)

    if args.test_urgent:
        class _MockError(Exception):
            pass
        err = _MockError(
            "openai.RateLimitError: You exceeded your current quota. "
            "Please check your plan and billing details."
        )
        ok = send_urgent(args.test_urgent, err, context="model=deepseek-reasoner (mock)")
        print("Urgent alert sent." if ok else "Failed — check stderr.")

    elif args.test:
        ps_path = ROOT / "data" / "pipeline_state.json"
        if not ps_path.exists():
            # Send a minimal synthetic heartbeat for testing
            import json
            ps = {
                "run_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "pipeline_status": "SUCCESS",
                "ticker": "GC=F",
                "stages": {k: {"status": "OK", "duration_s": 10} for k in
                           ["harvester", "regime", "metal_logic", "risk", "shadow"]},
                "portfolio": {
                    "gold_oz": 0.0, "cash_usd": 100000.0,
                    "portfolio_value": 100000.0, "last_spot": 3050.0,
                    "state": "FIAT", "unrealised_pnl": 0.0, "unrealised_pct": 0.0,
                },
                "regime": {
                    "hmm_state": "BEARISH", "hmm_veto_active": True,
                    "p_bullish": 0.0, "p_volatile": 0.0, "p_bearish": 1.0,
                },
                "risk": {
                    "approved_action": "HOLD_METAL", "target_weight": 0.0,
                    "deploy_usd": 0.0, "var_95_daily": -0.028, "var_override": False,
                },
                "committee": {
                    "action_taken": "HOLD_METAL", "quant_conviction": 4,
                    "macro_conviction": 3, "oracle_score": 0.45,
                    "veto_active": True, "decision_date": "2026-03-30",
                },
            }
        else:
            import json
            ps = json.loads(ps_path.read_text())
        ok = send_heartbeat(ps)
        print("Heartbeat sent." if ok else "Failed — check stderr.")

    elif args.training_ready:
        ok = send_training_ready()
        print("Training-ready notification sent." if ok else "Failed — check stderr.")

    elif args.send:
        ok = send_alert(args.send)
        print("Message sent." if ok else "Failed — check stderr.")


if __name__ == "__main__":
    main()
