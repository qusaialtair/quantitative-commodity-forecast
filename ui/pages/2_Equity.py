"""
ui/pages/2_Equity.py
====================
Phase 2 — The Universal Bloomberg Terminal for the Sharia-compliant equity
fund. Three-view dashboard reading strictly from the frozen Phase 1/3
contract:

    data/equity_ranker/ranking_latest.parquet   (Stage 2 ranking)
    data/equity_decision.json                   (Stage 3 CIO decision)
    data/equity_features/latest.parquet         (optional, for display)

Tabs
----
  Master God-View  — AUM, regime gauge, gross/net exposure, sector
                     heatmap, top-10 positions.
  Global Screener  — full ranking grid with Sharia gate, sparklines,
                     progress bars, sector / min-ai_score filters.
  Ticker Terminal  — Plotly candles + radar + CIO thesis; execution
                     routing BLOCKED when sharia_pass is False.

Sharia gate is enforced at the UI boundary: any ticker with sharia_pass
not equal to True is (a) excluded from the default screener universe,
(b) cannot be routed to the execution queue, (c) surfaced in a rejected
tally so the operator sees how many tickers the strict AAOIFI gate is
ejecting.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

RANKING_PATH  = ROOT / "data" / "equity_ranker" / "ranking_latest.parquet"
DECISION_PATH = ROOT / "data" / "equity_decision.json"
FEATURES_PATH = ROOT / "data" / "equity_features" / "latest.parquet"
EXEC_QUEUE    = ROOT / "data" / "execution_queue.json"
VAULT_PATH    = ROOT / "data" / "virtual_account.json"
TRADE_LOG_V2  = ROOT / "data" / "trade_log_v2.jsonl"

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE SHELL
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Equity Terminal — Sharia Fund",
    layout="wide",
    initial_sidebar_state="expanded",
)

SECTOR_COLORS = {
    "Technology":             "#4F8EF7",
    "Healthcare":             "#34D399",
    "Energy":                 "#F59E0B",
    "Industrials":            "#818CF8",
    "Materials":              "#FB923C",
    "Consumer Staples":       "#A78BFA",
    "Consumer Discretionary": "#F472B6",
    "Financials":             "#38BDF8",
    "Utilities":              "#6EE7B7",
    "Real Estate":            "#FCD34D",
    "Communication Services": "#C084FC",
    "Basic Materials":        "#FB923C",
    "Unknown":                "#374151",
}
GOLD      = "#D4AF37"
GOLD_DIM  = "#8A6F1F"
GREEN     = "#22C55E"
RED       = "#EF4444"
AMBER     = "#F59E0B"
BG        = "#0A0A0A"
PANEL     = "#111111"
BORDER    = "#1E1E1E"
TEXT      = "#D8D8D8"
MUTED     = "#5A5A5A"


def _inject_css() -> None:
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {{
    --bg: {BG}; --panel: {PANEL}; --border: {BORDER};
    --text: {TEXT}; --muted: {MUTED};
    --gold: {GOLD}; --green: {GREEN}; --red: {RED}; --amber: {AMBER};
}}
html, body, [class*="css"], .stApp {{
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}}
.block-container {{ padding-top: 1.2rem !important; max-width: 100% !important; }}
h1, h2, h3, h4 {{ color: var(--text) !important; letter-spacing: -0.01em; }}
.eq-header {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 14px 18px; background: linear-gradient(90deg, #111 0%, #0a0a0a 100%);
    border: 1px solid var(--border); border-radius: 6px; margin-bottom: 14px;
}}
.eq-brand {{
    font-family: 'JetBrains Mono', monospace; font-size: 15px; font-weight: 600;
    color: var(--gold); letter-spacing: 0.15em; text-transform: uppercase;
}}
.eq-sub {{ color: var(--muted); font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; }}
.eq-kpi {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
    padding: 14px 16px;
}}
.eq-kpi .lbl {{ color: var(--muted); font-size: 10px; letter-spacing: 0.15em;
    text-transform: uppercase; margin-bottom: 4px; }}
.eq-kpi .val {{ font-family: 'JetBrains Mono', monospace; font-size: 22px;
    font-weight: 600; color: var(--text); }}
.eq-kpi .delta {{ font-family: 'JetBrains Mono', monospace; font-size: 11px; }}
.regime-banner {{
    display: inline-block; padding: 6px 14px; border-radius: 4px;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    font-weight: 600; letter-spacing: 0.2em;
}}
.regime-RISK_ON  {{ background: rgba(34,197,94,0.12);  color: {GREEN}; border: 1px solid {GREEN}; }}
.regime-NEUTRAL  {{ background: rgba(245,158,11,0.12); color: {AMBER}; border: 1px solid {AMBER}; }}
.regime-RISK_OFF {{ background: rgba(239,68,68,0.12);  color: {RED};   border: 1px solid {RED}; }}
.sharia-pass {{
    background: rgba(34,197,94,0.15); color: {GREEN}; border: 1px solid {GREEN};
    padding: 3px 10px; border-radius: 3px; font-size: 10px;
    font-family: 'JetBrains Mono', monospace; font-weight: 600; letter-spacing: 0.1em;
}}
.sharia-fail {{
    background: rgba(239,68,68,0.15); color: {RED}; border: 1px solid {RED};
    padding: 3px 10px; border-radius: 3px; font-size: 10px;
    font-family: 'JetBrains Mono', monospace; font-weight: 600; letter-spacing: 0.1em;
}}
.block-banner {{
    background: rgba(239,68,68,0.08); border: 1px solid {RED};
    color: {RED}; padding: 14px 16px; border-radius: 6px;
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    margin: 10px 0;
}}
.panel {{
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; padding: 16px;
}}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid var(--border); }}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important; color: var(--muted) !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important;
    letter-spacing: 0.15em; text-transform: uppercase; padding: 8px 18px !important;
}}
.stTabs [aria-selected="true"] {{ color: var(--gold) !important;
    border-bottom: 2px solid var(--gold) !important; }}
[data-testid="stSidebar"] {{ background: #0d0d0d !important; border-right: 1px solid var(--border); }}
button[kind="primary"] {{ background: var(--gold) !important; color: #000 !important;
    border: none !important; font-weight: 600 !important; letter-spacing: 0.08em; }}
button[disabled] {{ opacity: 0.35 !important; cursor: not-allowed !important; }}
</style>
""",
        unsafe_allow_html=True,
    )


_inject_css()

# ══════════════════════════════════════════════════════════════════════════════
#  LOADERS (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=600, show_spinner=False)
def load_ranking() -> pd.DataFrame:
    if not RANKING_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(RANKING_PATH)
    if df.index.name != "ticker":
        df = df.set_index("ticker") if "ticker" in df.columns else df
    # Strict gate contract: missing sharia_pass column OR missing value → ejected.
    if "sharia_pass" not in df.columns:
        df["sharia_pass"] = False   # no fundamentals → cannot certify Sharia
    df["sharia_pass"] = df["sharia_pass"].fillna(False).astype(bool)
    return df


@st.cache_data(ttl=600, show_spinner=False)
def load_decision() -> dict:
    if not DECISION_PATH.exists():
        return {}
    return json.loads(DECISION_PATH.read_text())


@st.cache_data(ttl=120, show_spinner=False)
def load_vault() -> dict:
    """Live execution state — the ACTUAL book, not the CIO target.

    Returns {} if the vault has never been bootstrapped. The UI must
    handle this honestly (empty book, not dummy data).
    """
    if not VAULT_PATH.exists():
        return {}
    try:
        return json.loads(VAULT_PATH.read_text())
    except Exception:
        return {}


@st.cache_data(ttl=900, show_spinner=False)
def load_price_history(tickers: tuple[str, ...], period: str = "6mo") -> dict:
    """Thin yfinance cache for candle / sparkline drawing."""
    if not tickers:
        return {}
    try:
        import yfinance as yf
        raw = yf.download(list(tickers), period=period, interval="1d",
                          auto_adjust=True, progress=False, group_by="ticker")
    except Exception:
        return {}

    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    # yfinance returns MultiIndex columns [ticker][field] when group_by='ticker'
    # is set, regardless of single vs multi-ticker input. Flatten defensively.
    has_multi = isinstance(raw.columns, pd.MultiIndex)
    for t in tickers:
        try:
            sub = raw[t] if has_multi else raw
            sub = sub.dropna(how="all")
            # Standardise column names to the set we'll consume downstream.
            needed = {"Open", "High", "Low", "Close"}
            if not needed.issubset(set(sub.columns)):
                continue
            if not sub.empty:
                out[t] = sub
        except Exception:
            continue
    return out


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_live_prices(tickers: tuple[str, ...]) -> dict[str, float]:
    """Batch-fetch latest closing prices for vault tickers. Cached 5 min."""
    if not tickers:
        return {}
    try:
        import yfinance as yf
        raw = yf.download(list(tickers), period="5d", interval="1d",
                          auto_adjust=True, progress=False, group_by="ticker")
    except Exception:
        return {}
    if raw is None or raw.empty:
        return {}
    has_multi = isinstance(raw.columns, pd.MultiIndex)
    out: dict[str, float] = {}
    for t in tickers:
        try:
            sub = raw[t] if has_multi else raw
            cl = sub["Close"].dropna()
            if not cl.empty:
                out[t] = float(cl.iloc[-1])
        except Exception:
            continue
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
ranking = load_ranking()
decision = load_decision()

now = datetime.now(timezone.utc)
regime = (decision.get("macro_regime") or "NEUTRAL").upper()
regime_cap = decision.get("macro_gross_cap", 0.65)

st.markdown(
    f"""
<div class="eq-header">
  <div>
    <div class="eq-brand">SHARIA EQUITY TERMINAL · v2.0</div>
    <div class="eq-sub">{now.strftime('%A · %d %b %Y · %H:%M UTC')}</div>
  </div>
  <div style="text-align: right;">
    <div class="regime-banner regime-{regime}">REGIME · {regime} · CAP {regime_cap:.0%}</div>
    <div class="eq-sub" style="margin-top:6px;">
      DECISION LAST WRITTEN · {decision.get('generated_at', '—')}
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if ranking.empty:
    st.error("No ranking artefact found. Run Stage 2 first: "
             "`python3 -m models.equity_ranker.ranker rank`")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  STALENESS GUARD — the UI never fabricates. If the live decision JSON is
#  out of sync with the live ranking (e.g., decision written before the strict
#  Sharia flip, or before a universe rebuild), we surface a red banner rather
#  than let Master God-View silently contradict the Screener.
# ══════════════════════════════════════════════════════════════════════════════
def _sync_status(ranking_df: pd.DataFrame, decision_obj: dict) -> tuple[str, str] | None:
    """Return (severity, message) if the two artefacts disagree; else None.

    severity ∈ {"error", "warn"}
    """
    if not decision_obj:
        return ("warn",
                "No Stage 3 decision on disk yet. Master God-View will be empty "
                "until `python3 -m agents.equity_swarm.swarm` is run.")

    positions = decision_obj.get("positions") or []
    if not positions:
        return None

    halal_now = set(ranking_df.index[ranking_df["sharia_pass"].astype(bool)])
    pos_tkrs  = {p.get("ticker") for p in positions}
    in_universe = pos_tkrs & set(ranking_df.index)
    halal_hits  = pos_tkrs & halal_now

    # 1. Decision references tickers that no longer exist in the ranking file.
    if not in_universe:
        return ("error",
                f"DECISION OUT OF SYNC · {len(pos_tkrs)} positions reference "
                f"tickers absent from the current ranking file. Re-run the "
                f"swarm against the live ranking: "
                f"`python3 -m agents.equity_swarm.swarm`.")

    # 2. Decision was taken under the old lenient Sharia policy — every sized
    #    ticker would now fail the strict gate.
    if not halal_hits:
        return ("error",
                f"STRICT SHARIA DESYNC · {len(pos_tkrs)} positions in the book "
                f"were sized under the old lenient gate; zero survive the "
                f"current AAOIFI strict screen. Rebuild features with "
                f"fundamentals and re-run the swarm.")

    # 3. Partial overlap — some positions are now non-compliant.
    stale_pct = 1.0 - len(halal_hits) / len(pos_tkrs)
    if stale_pct >= 0.25:
        return ("warn",
                f"DECISION PARTIALLY STALE · "
                f"{len(pos_tkrs) - len(halal_hits)}/{len(pos_tkrs)} positions no "
                f"longer pass the strict Sharia gate. Execution router will "
                f"block them; re-run the swarm to refresh the book.")

    return None


_sync = _sync_status(ranking, decision)
if _sync is not None:
    sev, msg = _sync
    color = RED if sev == "error" else AMBER
    st.markdown(
        f'<div class="block-banner" style="border-color:{color};color:{color};'
        f'background:rgba({239 if sev=="error" else 245},'
        f'{68 if sev=="error" else 158},'
        f'{68 if sev=="error" else 11},0.08);">'
        f'<b>SYSTEM ADVISORY</b><br>{msg}</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — filters (Sharia gate is locked ON by default but visible)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### FILTERS")
    halal_only = st.toggle("Sharia-compliant only (AAOIFI)", value=True,
                           help="Strict gate: NaN / missing fundamentals are ejected.")
    if not halal_only:
        st.warning(
            "Sharia filter disabled — non-compliant tickers visible for audit "
            "only. Execution routing remains blocked for sharia_pass=False.",
            icon=None,
        )

    sector_opts = sorted([s for s in ranking.get("sector", pd.Series([], dtype=str)).dropna().unique()])
    sector_sel = st.multiselect("Sector", options=sector_opts, default=[])
    min_ai = st.slider("Min ai_score", 0.0, 1.0, 0.0, 0.05)
    top_n = st.number_input("Show top-N", min_value=10, max_value=500, value=50, step=10)

    st.markdown("---")
    st.markdown("### UNIVERSE GATE")
    total = len(ranking)
    halal = int(ranking["sharia_pass"].sum())
    ejected = total - halal
    st.markdown(
        f"""
<div style="font-family:'JetBrains Mono',monospace;font-size:11px;line-height:1.8;">
  <div>UNIVERSE&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>{total}</b></div>
  <div style="color:{GREEN};">HALAL PASS&nbsp;&nbsp;<b>{halal}</b></div>
  <div style="color:{RED};">EJECTED&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>{ejected}</b></div>
</div>
""",
        unsafe_allow_html=True,
    )

# Build filtered working set for screener/terminal
universe = ranking.copy()
if halal_only:
    universe = universe[universe["sharia_pass"]]
if sector_sel and "sector" in universe.columns:
    universe = universe[universe["sector"].isin(sector_sel)]
universe = universe[universe.get("ai_score", pd.Series(0, index=universe.index)) >= min_ai]
universe = universe.head(int(top_n))


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_master, tab_screener, tab_terminal = st.tabs(
    ["Master God-View", "Global Screener", "Ticker Terminal"]
)


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 1 — MASTER GOD-VIEW
# ──────────────────────────────────────────────────────────────────────────────
def render_master() -> None:
    # ─── LIVE EXECUTION VAULT — the ACTUAL book. Never fabricate. ──────────
    vault = load_vault()
    macro = decision.get("macro_snapshot") or {}

    if not vault:
        st.markdown(
            '<div class="block-banner"><b>VAULT NOT INITIALIZED</b><br>'
            'No virtual_account.json on disk. Run '
            '<code>python3 -m execution.virtual_trader bootstrap --cash 1000000</code> '
            'to create the live execution book.</div>',
            unsafe_allow_html=True,
        )
        return

    # Book == dict keyed by ticker → position record. Build a positions list
    # compatible with the V1 render path while joining in decision context
    # (name, sector, conviction, sent_signal, thesis) as enrichment only.
    dec_by_tkr = {p["ticker"]: p for p in (decision.get("positions") or [])}

    raw_positions = vault.get("positions", {})
    tickers_needing_price = tuple(
        t for t, p in raw_positions.items()
        if "current_price" not in p and p.get("qty", 0) > 0
    )
    live_prices = _fetch_live_prices(tickers_needing_price) if tickers_needing_price else {}

    vault_positions: list[dict] = []
    total_unrealized_pnl = 0.0

    for t, p in raw_positions.items():
        qty       = p.get("qty", 0)
        if qty <= 0:
            continue
        entry     = p.get("current_price", p.get("avg_entry_price", p.get("avg_cost", 0.0)))
        cur_price = p.get("current_price", live_prices.get(t, entry))
        avg_cost  = p.get("avg_entry_price", p.get("avg_cost", 0.0))
        mv        = qty * cur_price
        cost      = qty * avg_cost
        u_pnl     = mv - cost
        total_unrealized_pnl += u_pnl
        meta = dec_by_tkr.get(t, {})
        vault_positions.append({
            "ticker":            t,
            "name":              meta.get("name", t),
            "sector":            meta.get("sector", "Unknown"),
            "qty":               qty,
            "avg_entry_price":   avg_cost,
            "current_price":     cur_price,
            "market_value":      mv,
            "weight":            None,  # filled once total_equity is known
            "unrealized_pnl":    u_pnl,
            "unrealized_pnl_pct": (u_pnl / cost * 100 if cost else 0.0),
            "trailing_stop":     p.get("trailing_stop_price", 0.0),
            "highest_seen":      p.get("highest_price_since_entry", 0.0),
            "atr_at_entry":      p.get("atr_at_entry", 0.0),
            "conviction":        meta.get("conviction"),
            "sent_signal":       meta.get("sent_signal", "HOLD"),
        })

    total_equity = float(vault.get("total_equity") or 0.0)
    cash_balance = float(vault.get("cash_balance") or 0.0)
    starting     = float(vault.get("starting_cash") or total_equity or 1.0)
    invested     = total_equity - cash_balance
    life_pnl     = total_equity - starting
    life_pnl_pct = (life_pnl / starting) * 100.0 if starting else 0.0
    cash_pct     = (cash_balance / total_equity) if total_equity > 0 else 1.0
    eq_pct       = 1.0 - cash_pct

    # Fill live weight per position now that total_equity is known.
    for row in vault_positions:
        row["weight"] = (row["market_value"] / total_equity) if total_equity else 0.0
    vault_positions.sort(key=lambda r: -r["market_value"])

    # --- KPI row ---------------------------------------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    pnl_color = GREEN if life_pnl >= 0 else RED
    kpi_defs = [
        (c1, "TOTAL EQUITY",  f"${total_equity:,.0f}",  None),
        (c2, "INVESTED",      f"{eq_pct:.1%}",          GREEN if eq_pct > 0 else MUTED),
        (c3, "CASH / SUKUK",  f"{cash_pct:.1%}",        GOLD),
        (c4, "LIFETIME P&L",  f"{life_pnl_pct:+.2f}%",  pnl_color),
        (c5, "POSITIONS",     f"{len(vault_positions)}", None),
    ]
    for col, lbl, val, color in kpi_defs:
        color_css = f"color:{color};" if color else ""
        col.markdown(
            f'<div class="eq-kpi"><div class="lbl">{lbl}</div>'
            f'<div class="val" style="{color_css}">{val}</div></div>',
            unsafe_allow_html=True,
        )

    last_rb = vault.get("last_rebalance") or "never"
    gen_at  = vault.get("generated_at") or "—"
    st.markdown(
        f'<div class="eq-sub" style="margin-top:4px;">'
        f'VAULT MARKED · {gen_at}   &nbsp;|&nbsp;   LAST REBALANCE · {last_rb}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    # --- Row: regime gauge + exposure donut ------------------------------------
    col_gauge, col_donut, col_macro = st.columns([1.1, 1.1, 1.4])

    with col_gauge:
        vix = float(macro.get("vix_spot") or 0)
        g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=vix,
            number={"font": {"family": "JetBrains Mono", "size": 32, "color": TEXT}},
            gauge={
                "axis": {"range": [0, 50], "tickcolor": MUTED, "tickfont": {"color": MUTED, "size": 10}},
                "bar": {"color": GOLD, "thickness": 0.22},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 15],  "color": "rgba(34,197,94,0.18)"},
                    {"range": [15, 20], "color": "rgba(245,158,11,0.14)"},
                    {"range": [20, 28], "color": "rgba(245,158,11,0.25)"},
                    {"range": [28, 50], "color": "rgba(239,68,68,0.25)"},
                ],
                "threshold": {"line": {"color": TEXT, "width": 2},
                              "thickness": 0.8, "value": vix},
            },
            title={"text": "VIX", "font": {"color": MUTED, "size": 12, "family": "JetBrains Mono"}},
        ))
        g.update_layout(height=230, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                        margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(g, width="stretch", config={"displayModeBar": False})

    with col_donut:
        # LIVE exposure from the vault. If fund is 100% cash, donut shows it.
        d = go.Figure(go.Pie(
            values=[eq_pct, cash_pct], labels=["Equity", "Cash / Sukuk"],
            hole=0.70, direction="clockwise", sort=False,
            marker=dict(colors=[GREEN, GOLD], line=dict(color=BG, width=2)),
            textinfo="label+percent",
            textfont=dict(family="JetBrains Mono", color=TEXT, size=11),
        ))
        d.update_layout(
            height=230, paper_bgcolor=PANEL, plot_bgcolor=PANEL, showlegend=False,
            margin=dict(l=10, r=10, t=30, b=10),
            title=dict(text="VAULT · EXPOSURE",
                       font=dict(color=MUTED, size=12, family="JetBrains Mono"),
                       x=0.5, xanchor="center"),
        )
        st.plotly_chart(d, width="stretch", config={"displayModeBar": False})

    with col_macro:
        rows = [
            ("VIX spot",        f"{macro.get('vix_spot', 0):.2f}",     f"{macro.get('vix_20d_change_pct',0):+.1f}%"),
            ("DXY spot",        f"{macro.get('dxy_spot', 0):.2f}",     f"{macro.get('dxy_20d_change_pct',0):+.1f}%"),
            ("UST 10Y",         f"{macro.get('ust10y', 0):.2f}%",      None),
            ("UST 2Y",          f"{macro.get('ust2y', 0):.2f}%",       None),
            ("10Y-2Y spread",   f"{macro.get('term_spread_10_2', 0):+.2f}", None),
            ("HY-OAS",          f"{macro.get('hy_oas_spot', 0):.2f}%", f"{macro.get('hy_oas_20d_change_bp',0):+.0f} bp"),
        ]
        rows_html = "".join(
            f'<tr><td style="color:{MUTED};padding:4px 8px;font-size:11px;">{k}</td>'
            f'<td style="font-family:JetBrains Mono;padding:4px 8px;color:{TEXT};">{v}</td>'
            f'<td style="font-family:JetBrains Mono;padding:4px 8px;color:{MUTED};font-size:11px;">{d or ""}</td></tr>'
            for k, v, d in rows
        )
        st.markdown(
            f'<div class="panel"><div class="eq-sub" style="margin-bottom:6px;">MACRO SNAPSHOT</div>'
            f'<table style="width:100%;border-collapse:collapse;">{rows_html}</table></div>',
            unsafe_allow_html=True,
        )

    # --- Sector heatmap + top-10 ----------------------------------------------
    st.markdown("")
    col_heat, col_top = st.columns([1.3, 1.7])

    with col_heat:
        st.markdown('<div class="eq-sub">SECTOR WEIGHTS · LIVE VAULT</div>',
                    unsafe_allow_html=True)
        if vault_positions:
            pos_df = pd.DataFrame(vault_positions)
            sec = pos_df.groupby("sector")["weight"].sum().sort_values(ascending=True)
            bar = go.Figure(go.Bar(
                x=sec.values, y=sec.index, orientation="h",
                marker=dict(color=[SECTOR_COLORS.get(s, SECTOR_COLORS["Unknown"]) for s in sec.index]),
                text=[f"{v*100:.1f}%" for v in sec.values], textposition="outside",
                textfont=dict(family="JetBrains Mono", color=TEXT, size=10),
            ))
            bar.update_layout(
                height=max(220, 30 * len(sec) + 60),
                paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                margin=dict(l=10, r=40, t=10, b=10),
                xaxis=dict(showgrid=False, tickformat=".0%", color=MUTED),
                yaxis=dict(showgrid=False, color=TEXT),
            )
            st.plotly_chart(bar, width="stretch", config={"displayModeBar": False})
        else:
            st.info("Vault is flat — 100% cash. No sector exposure to display.")

    with col_top:
        st.markdown('<div class="eq-sub">TOP 10 POSITIONS · LIVE P&amp;L</div>',
                    unsafe_allow_html=True)
        if vault_positions:
            top10 = pd.DataFrame(vault_positions[:10]).copy()
            # Distance-to-stop as a % of current price — positive means headroom
            # above the trailing stop, the cushion before a forced exit fires.
            top10["stop_dist_pct"] = np.where(
                top10["current_price"] > 0,
                (top10["current_price"] - top10["trailing_stop"]) / top10["current_price"] * 100.0,
                0.0,
            )
            # Vault stores weight as decimal; ProgressColumn's printf is literal.
            top10["weight"] = top10["weight"].astype(float) * 100.0
            disp = top10[["ticker", "name", "sector", "weight", "current_price",
                          "unrealized_pnl", "unrealized_pnl_pct",
                          "trailing_stop", "stop_dist_pct"]]
            st.dataframe(
                disp,
                hide_index=True,
                width="stretch",
                column_config={
                    "ticker": st.column_config.TextColumn("TKR", width="small"),
                    "name":   st.column_config.TextColumn("Name", width="medium"),
                    "sector": st.column_config.TextColumn("Sector", width="small"),
                    "weight": st.column_config.ProgressColumn(
                        "Weight %", format="%.2f%%",
                        min_value=0.0, max_value=8.0,
                        help="Live vault weight (position cap = 8%)"),
                    "current_price": st.column_config.NumberColumn(
                        "Last", format="$%.2f"),
                    "unrealized_pnl": st.column_config.NumberColumn(
                        "U.P&L $", format="$%+,.0f",
                        help="Unrealized dollar P&L vs average entry"),
                    "unrealized_pnl_pct": st.column_config.NumberColumn(
                        "U.P&L %", format="%+.2f%%"),
                    "trailing_stop": st.column_config.NumberColumn(
                        "ATR Stop", format="$%.2f",
                        help="2×ATR trailing stop — forced exit below this"),
                    "stop_dist_pct": st.column_config.ProgressColumn(
                        "Stop Δ", format="%.1f%%",
                        min_value=0.0, max_value=25.0,
                        help="Headroom above the trailing stop"),
                },
            )
        else:
            st.info("Vault is flat — no open positions.")


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 2 — GLOBAL SCREENER
# ──────────────────────────────────────────────────────────────────────────────
def render_screener() -> None:
    total_u = len(ranking)
    halal_u = int(ranking["sharia_pass"].sum())
    shown = len(universe)

    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val, color in [
        (c1, "UNIVERSE", f"{total_u}", None),
        (c2, "HALAL PASS", f"{halal_u}", GREEN),
        (c3, "EJECTED", f"{total_u - halal_u}", RED),
        (c4, "SHOWN", f"{shown}", GOLD),
    ]:
        cs = f"color:{color};" if color else ""
        col.markdown(
            f'<div class="eq-kpi"><div class="lbl">{lbl}</div>'
            f'<div class="val" style="{cs}">{val}</div></div>',
            unsafe_allow_html=True,
        )

    if universe.empty:
        st.warning("No tickers match current filters.")
        return

    # --- Sparkline source: last 21 close values per ticker --------------------
    tickers = tuple(universe.index.tolist())
    hist = load_price_history(tickers, period="2mo")
    spark = {
        t: (hist[t]["Close"].tail(21).tolist() if t in hist and not hist[t].empty else [])
        for t in tickers
    }

    grid = universe.reset_index().copy()
    grid["Spark21d"]    = grid["ticker"].map(spark)
    grid["Sharia"]      = grid["sharia_pass"].map(lambda b: "PASS" if b else "FAIL")
    # Safe column access for optional cols
    for col in ("name", "sector", "beta_252d", "vol_63d_ann",
                "return_21d", "prob_beats_median", "expected_return_10d",
                "ai_score", "ai_drivers_top3"):
        if col not in grid.columns:
            grid[col] = None

    grid["Drivers"] = grid["ai_drivers_top3"].apply(
        lambda lst: ", ".join(lst[:3]) if isinstance(lst, (list, tuple)) else ""
    )

    # Contract emits returns / vol as decimals. Streamlit's %.Nf%% is literal
    # printf and does NOT multiply by 100 — pre-scale any column we want to
    # show as a percentage, then show with the % suffix.
    for col in ("expected_return_10d", "return_21d", "vol_63d_ann"):
        if col in grid.columns:
            grid[col] = pd.to_numeric(grid[col], errors="coerce") * 100.0

    display = grid[[
        "ticker", "name", "sector", "Sharia", "ai_score",
        "prob_beats_median", "expected_return_10d", "Spark21d",
        "return_21d", "vol_63d_ann", "beta_252d", "Drivers",
    ]]

    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        height=min(720, 45 + 36 * len(display)),
        column_config={
            "ticker":              st.column_config.TextColumn("TKR", width="small"),
            "name":                st.column_config.TextColumn("Name", width="medium"),
            "sector":              st.column_config.TextColumn("Sector", width="small"),
            "Sharia":              st.column_config.TextColumn(
                                      "Sharia", width="small",
                                      help="AAOIFI strict: PASS = debt≤33%, cash≤33%, data present"),
            "ai_score":            st.column_config.ProgressColumn(
                                      "AI", format="%.2f", min_value=0.0, max_value=1.0),
            "prob_beats_median":   st.column_config.NumberColumn("P>median", format="%.2f"),
            "expected_return_10d": st.column_config.NumberColumn("E[r10d] %", format="%+.2f%%"),
            "Spark21d":            st.column_config.LineChartColumn(
                                      "Spark 21d", width="medium",
                                      help="Last 21 trading-day close"),
            "return_21d":          st.column_config.NumberColumn("r21d %", format="%+.2f%%"),
            "vol_63d_ann":         st.column_config.NumberColumn("σ ann %", format="%.1f%%"),
            "beta_252d":           st.column_config.NumberColumn("β", format="%.2f"),
            "Drivers":             st.column_config.TextColumn("Drivers", width="medium"),
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 3 — TICKER TERMINAL
# ──────────────────────────────────────────────────────────────────────────────
def _radar_axes(row: pd.Series, ref: pd.DataFrame) -> tuple[list[str], list[float]]:
    """Build 6-axis radar: higher = better. Z-score vs ref universe, clipped."""
    def zclip(col: str, flip: bool = False) -> float:
        if col not in ref.columns or col not in row or pd.isna(row.get(col)):
            return 0.5
        series = pd.to_numeric(ref[col], errors="coerce").dropna()
        if series.empty or series.std(ddof=0) < 1e-9:
            return 0.5
        z = (row[col] - series.mean()) / series.std(ddof=0)
        if flip:
            z = -z
        # squash to [0, 1] via logistic
        return float(1 / (1 + math.exp(-z)))

    axes = [
        ("AI score",         zclip("ai_score")),
        ("P > median",       zclip("prob_beats_median")),
        ("Momentum 12-1",    zclip("momentum_12_1")),
        ("Sharpe 126d",      zclip("sharpe_126d")),
        ("Low volatility",   zclip("vol_63d_ann", flip=True)),
        ("Low drawdown",     zclip("drawdown_current", flip=True)),
    ]
    return [a[0] for a in axes], [a[1] for a in axes]


def _queue_execution(pos: dict) -> str:
    EXEC_QUEUE.parent.mkdir(parents=True, exist_ok=True)
    queue = []
    if EXEC_QUEUE.exists():
        try:
            queue = json.loads(EXEC_QUEUE.read_text()).get("orders", [])
        except Exception:
            queue = []
    queue.append({**pos, "queued_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    EXEC_QUEUE.write_text(json.dumps({"orders": queue}, indent=2, default=str))
    return str(EXEC_QUEUE.relative_to(ROOT))


def render_terminal() -> None:
    # Universe for the selectbox: honors the Sharia-only filter but we still
    # allow the operator to PICK a failing ticker for audit (execution is
    # blocked downstream regardless).
    selectable = ranking if not halal_only else ranking[ranking["sharia_pass"]]
    if selectable.empty:
        st.warning("No tickers in the selectable universe. Loosen filters.")
        return

    # Preserve selection across reruns
    default_t = st.session_state.get("eq_terminal_ticker") or selectable.index[0]
    if default_t not in selectable.index:
        default_t = selectable.index[0]

    left, right = st.columns([1, 4])
    with left:
        ticker = st.selectbox(
            "Ticker", options=list(selectable.index),
            index=list(selectable.index).index(default_t),
            key="eq_terminal_ticker",
        )
    row = ranking.loc[ticker]
    is_halal = bool(row.get("sharia_pass", False))

    # --- Header card ----------------------------------------------------------
    last_close = row.get("last_close", np.nan)
    ai = row.get("ai_score", np.nan)
    p_up = row.get("prob_beats_median", np.nan)
    er = row.get("expected_return_10d", np.nan)
    with right:
        badge = ('<span class="sharia-pass">SHARIA · PASS</span>'
                 if is_halal else '<span class="sharia-fail">SHARIA · FAIL</span>')
        name = row.get("name", ticker) or ticker
        sector = row.get("sector", "Unknown")
        st.markdown(
            f"""
<div class="panel" style="display:flex;justify-content:space-between;align-items:center;">
  <div>
    <div style="font-family:'JetBrains Mono';font-size:22px;font-weight:600;color:{GOLD};">
       {ticker} · <span style="color:{TEXT};">{name}</span>
    </div>
    <div class="eq-sub" style="margin-top:4px;">{sector}</div>
  </div>
  <div style="text-align:right;">
    <div style="font-family:'JetBrains Mono';font-size:26px;color:{TEXT};">
      ${last_close:.2f}
    </div>
    <div style="margin-top:6px;">{badge}</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    # KPI strip
    k1, k2, k3, k4 = st.columns(4)
    for col, lbl, val in [
        (k1, "AI SCORE",            f"{ai:.2f}" if pd.notna(ai) else "—"),
        (k2, "P > MEDIAN",          f"{p_up:.2f}" if pd.notna(p_up) else "—"),
        (k3, "E[r · 10d]",           f"{er*100:+.2f}%" if pd.notna(er) else "—"),
        (k4, "β · 252d",             f"{row.get('beta_252d', float('nan')):.2f}"
                                     if pd.notna(row.get("beta_252d")) else "—"),
    ]:
        col.markdown(
            f'<div class="eq-kpi"><div class="lbl">{lbl}</div>'
            f'<div class="val">{val}</div></div>',
            unsafe_allow_html=True,
        )

    # --- Candles + radar -------------------------------------------------------
    col_chart, col_radar = st.columns([1.9, 1.1])

    with col_chart:
        hist = load_price_history((ticker,), period="6mo")
        df = hist.get(ticker, pd.DataFrame())
        if df.empty:
            st.info(f"No price history for {ticker}.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"],
                increasing_line_color=GREEN, decreasing_line_color=RED,
                increasing_fillcolor=GREEN, decreasing_fillcolor=RED,
                name=ticker, showlegend=False,
            ))
            fig.update_layout(
                height=420, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                margin=dict(l=10, r=10, t=30, b=10),
                title=dict(text="6M · DAILY", font=dict(color=MUTED, size=11,
                           family="JetBrains Mono"), x=0.01),
                xaxis=dict(rangeslider=dict(visible=False), gridcolor=BORDER,
                           color=MUTED, showspikes=True, spikecolor=GOLD,
                           spikethickness=1, spikedash="dot"),
                yaxis=dict(gridcolor=BORDER, color=MUTED, side="right"),
            )
            st.plotly_chart(fig, width="stretch",
                            config={"displayModeBar": False})

    with col_radar:
        axes, vals = _radar_axes(row, ranking)
        vals_loop = vals + [vals[0]]
        axes_loop = axes + [axes[0]]
        rad = go.Figure(go.Scatterpolar(
            r=vals_loop, theta=axes_loop, fill="toself",
            line=dict(color=GOLD, width=2),
            fillcolor="rgba(212,175,55,0.22)",
            name=ticker,
        ))
        rad.update_layout(
            height=420, paper_bgcolor=PANEL, plot_bgcolor=PANEL, showlegend=False,
            margin=dict(l=40, r=40, t=40, b=20),
            title=dict(text="FACTOR RADAR · vs UNIVERSE",
                       font=dict(color=MUTED, size=11, family="JetBrains Mono"),
                       x=0.5, xanchor="center"),
            polar=dict(
                bgcolor=PANEL,
                radialaxis=dict(range=[0, 1], showticklabels=False,
                                gridcolor=BORDER, linecolor=BORDER),
                angularaxis=dict(color=MUTED, gridcolor=BORDER,
                                 tickfont=dict(family="JetBrains Mono", size=9)),
            ),
        )
        st.plotly_chart(rad, width="stretch",
                        config={"displayModeBar": False})

    # --- CIO thesis (from decision) or on-demand -----------------------------
    st.markdown('<div class="eq-sub" style="margin-top:8px;">CIO THESIS</div>',
                unsafe_allow_html=True)
    book_pos = next((p for p in (decision.get("positions") or [])
                     if p.get("ticker") == ticker), None)
    if book_pos:
        conv = book_pos.get("conviction", 0.0)
        conv_color = GREEN if conv > 0 else (RED if conv < 0 else MUTED)
        st.markdown(
            f"""
<div class="panel">
  <div style="color:{MUTED};font-size:11px;font-family:JetBrains Mono;">
    WEIGHT <span style="color:{GOLD};">{book_pos.get('weight',0)*100:.2f}%</span>
    &nbsp;·&nbsp; CONVICTION
    <span style="color:{conv_color};">{conv:+.2f}</span>
    &nbsp;·&nbsp; SENT {book_pos.get('sent_signal','HOLD')}
  </div>
  <div style="margin-top:10px;color:{TEXT};line-height:1.6;">
    {book_pos.get('thesis','—')}
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="panel" style="color:{MUTED};">'
            f'Ticker is not in the current book. The CIO did not size a position.'
            f'</div>',
            unsafe_allow_html=True,
        )

    # --- Execution router ------------------------------------------------------
    st.markdown('<div class="eq-sub" style="margin-top:14px;">EXECUTION ROUTER</div>',
                unsafe_allow_html=True)
    if not is_halal:
        st.markdown(
            f'<div class="block-banner"><b>BLOCKED · SHARIA GATE</b><br>'
            f'Ticker {ticker} failed the strict AAOIFI screen (debt_to_mktcap, '
            f'cash_to_mktcap, or missing fundamental data). Routing to execution '
            f'is permanently disabled for non-compliant securities.</div>',
            unsafe_allow_html=True,
        )
        st.button("Route to Execution", disabled=True, use_container_width=True)
    elif book_pos is None:
        st.info("Not in current book — CIO did not size. Re-run the swarm "
                "if you want this ticker evaluated.")
        st.button("Route to Execution", disabled=True, use_container_width=True)
    else:
        if st.button("Route to Execution", type="primary", use_container_width=True):
            path = _queue_execution({
                "ticker": ticker,
                "name":   book_pos.get("name", ticker),
                "weight": book_pos.get("weight"),
                "conviction": book_pos.get("conviction"),
                "sharia_pass": True,
                "regime": regime,
            })
            st.success(f"Queued. Appended to {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  RENDER
# ══════════════════════════════════════════════════════════════════════════════
with tab_master:
    render_master()

with tab_screener:
    render_screener()

with tab_terminal:
    render_terminal()
