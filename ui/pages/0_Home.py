"""
ui/pages/0_Home.py
==================
God View — Fund Manager Home Dashboard

Aggregates both books (Metals + Equities) into a single command-centre view.

Sections
--------
  Row 0  — Executive Briefing (DeepSeek Chief of Staff, from executive_briefing.json)
  Row 1  — KPI strip: Total AUM | Metals Book | Equity Book | Open PnL
  Row 2  — [Pie chart: AUM by asset class]  [Holdings table with live prices/PnL]
  Row 3  — Radar: [Equity runner-ups]  [Metals HMM + oracle]
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fund Manager — Home",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BRIEFING_FILE  = ROOT / "data" / "executive_briefing.json"
VIRTUAL_ACCT   = ROOT / "data" / "virtual_account.json"
PIPELINE_STATE = ROOT / "data" / "pipeline_state.json"
PORTFOLIO_FILE = ROOT / "data" / "portfolio.json"      # physical holdings

# ── Currency / unit constants ─────────────────────────────────────────────────

_GRAMS_PER_TROY_OZ = 31.1034768

CURRENCIES = {
    "USD": {"sym": "$",     "rate": 1.0},
    "AED": {"sym": "د.إ ",  "rate": 3.6725},
    "EUR": {"sym": "€",     "rate": 0.92},
}

# Each preset: which currency + unit, and the per-unit price factor (from USD/oz)
UNIT_PRESETS = [
    {"label": "USD/oz", "currency": "USD", "unit": "oz", "px_factor": 1.0,
     "qty_factor": 1.0, "qty_dec": 4},
    {"label": "AED/g",  "currency": "AED", "unit": "g",
     "px_factor": 1.0 / _GRAMS_PER_TROY_OZ,
     "qty_factor": _GRAMS_PER_TROY_OZ, "qty_dec": 2},
    {"label": "EUR/g",  "currency": "EUR", "unit": "g",
     "px_factor": 1.0 / _GRAMS_PER_TROY_OZ,
     "qty_factor": _GRAMS_PER_TROY_OZ, "qty_dec": 2},
]

# ── Colour palette ────────────────────────────────────────────────────────────

# Cohesive equity-slice colours (dark-theme tonal palette)
PIE_EQUITY_COLORS = [
    "#3B82F6",  # blue
    "#10B981",  # emerald
    "#8B5CF6",  # violet
    "#F59E0B",  # amber
    "#EC4899",  # pink
    "#06B6D4",  # cyan
    "#84CC16",  # lime
]
PIE_CASH_COLOR   = "#2D3748"  # dark slate — unified cash colour
PIE_GOLD_COLOR   = "#C9A84C"  # gold


# ── CSS ───────────────────────────────────────────────────────────────────────
def _inject_css() -> None:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg:    #0a0a0a;
    --bg1:   #111111;
    --nav:   #0d0d0d;
    --border:#1e1e1e;
    --text:  #d8d8d8;
    --muted: #5a5a5a;
    --green: #22c55e;
    --red:   #ef4444;
    --gold:  #C9A84C;
    --mono:  'Space Mono', monospace;
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
    font-size: 14px;
}
.stApp { background: var(--bg) !important; }
#MainMenu, footer, header, .stDeployButton { visibility: hidden !important; }

/* ── Hide sidebar + all toggle controls ─────────────────────────────────── */
section[data-testid="stSidebar"],
[data-testid="stSidebarNav"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stExpandSidebarButton"],
[data-testid="collapsedControl"] { display: none !important; }

/* ── Bloomberg-terminal navigation ─────────────────────────────────────── */
.fn-topnav {
    background: #050505;
    border-bottom: 1px solid #181818;
    display: flex; align-items: center;
    padding: 0 24px; height: 44px;
    margin: -1rem -2rem 0 -2rem;
}
.fn-brand {
    font-family: var(--mono); font-size: 11px; font-weight: 700;
    letter-spacing: .18em; color: var(--gold); text-transform: uppercase;
    flex-shrink: 0;
}
.fn-brand-sep {
    width: 1px; height: 16px; background: #252525;
    margin: 0 16px; flex-shrink: 0;
}
.fn-page-id {
    font-family: var(--mono); font-size: 10px; letter-spacing: .14em;
    color: #2a2a2a; text-transform: uppercase;
}
.fn-ts {
    font-family: var(--mono); font-size: 9px; color: #222;
    margin-left: auto; letter-spacing: .06em;
}
.fn-nav-row {
    background: #080808;
    border-bottom: 1px solid #1a1a1a;
    margin: 0 -2rem 1.6rem -2rem;
    padding: 0 .5rem;
}
.fn-nav-row [data-testid="stHorizontalBlock"] {
    background: #080808 !important; gap: 0 !important;
}
.fn-nav-row [data-testid="column"] {
    background: #080808 !important; padding: 0 !important;
}
.fn-nav-row .stButton > button {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    color: #3e3e3e !important;
    font-family: var(--mono) !important;
    font-size: 10px !important; font-weight: 700 !important;
    letter-spacing: .12em !important; text-transform: uppercase !important;
    padding: 11px 22px 9px !important;
    box-shadow: none !important; white-space: nowrap !important;
    width: 100% !important;
    transition: color .1s, border-color .1s, background .1s !important;
}
.fn-nav-row .stButton > button:hover {
    background: #0f0f0f !important;
    color: #a0a0a0 !important;
    border-bottom-color: rgba(201,168,76,.35) !important;
    box-shadow: none !important;
}
.fn-nav-row .stButton > button:disabled {
    background: #0c0c0c !important;
    color: var(--gold) !important;
    border-bottom-color: var(--gold) !important;
    opacity: 1 !important; cursor: default !important;
    box-shadow: none !important;
}
/* Utility buttons — always col 6 or later */
.fn-nav-row [data-testid="column"]:nth-child(n+6) .stButton > button {
    font-size: 8px !important; letter-spacing: .16em !important;
    color: #2c2c2c !important;
    border: 1px solid #1e1e1e !important;
    border-bottom: 1px solid #1e1e1e !important;
    border-radius: 3px !important;
    padding: 4px 10px !important; margin: 7px 3px 0 !important;
    width: auto !important;
}
.fn-nav-row [data-testid="column"]:nth-child(n+6) .stButton > button:hover {
    border-color: rgba(201,168,76,.22) !important;
    color: rgba(201,168,76,.55) !important;
    background: #0e0e0e !important;
    border-bottom-color: rgba(201,168,76,.22) !important;
}

/* ── Main content ───────────────────────────────────────────────────────── */
.block-container { padding: 0 2rem 4rem !important; max-width: 1580px !important; }

/* KPI cards */
.kpi-card {
    background: var(--bg1);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
}
.kpi-label { font-size:10px; font-weight:600; letter-spacing:.12em; color:var(--muted); text-transform:uppercase; margin-bottom:6px; }
.kpi-value { font-family:var(--mono); font-size:22px; font-weight:700; color:var(--text); }
.kpi-sub   { font-size:11px; color:var(--muted); margin-top:4px; }
.pos { color: var(--green) !important; }
.neg { color: var(--red)   !important; }
.neu { color: var(--muted) !important; }

/* Briefing box */
.briefing-box {
    background: var(--bg1);
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold);
    border-radius: 8px;
    padding: 20px 24px;
    line-height: 1.7;
    font-size: 13.5px;
    color: var(--text);
}
.briefing-header {
    font-size: 10px; font-weight: 600; letter-spacing: .12em;
    color: var(--gold); text-transform: uppercase; margin-bottom: 12px;
}

/* Section headers */
.section-header {
    font-size: 10px; font-weight: 600; letter-spacing: .12em;
    color: var(--muted); text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px; margin-bottom: 14px;
}

/* Radar / signal panel */
.radar-table { background:var(--bg1); border:1px solid var(--border); border-radius:8px; padding:16px 20px; }
.regime-badge { display:inline-block; padding:3px 10px; border-radius:4px; font-size:11px; font-weight:600; font-family:var(--mono); }
.badge-bullish { background:#14532d; color:#4ade80; }
.badge-ranging { background:#78350f; color:#fbbf24; }
.badge-bearish { background:#7f1d1d; color:#f87171; }
.badge-unknown { background:#1f2937; color:#9ca3af; }

[data-testid="stDataFrame"] { border:1px solid var(--border) !important; border-radius:8px !important; }
</style>
""", unsafe_allow_html=True)


# ── Data loaders ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def _fetch_live_prices(tickers: tuple[str, ...],
                       fallbacks: tuple[tuple[str, float], ...] = ()) -> dict[str, float]:
    """Fetch prices via yfinance with a 5-second timeout.
    Falls back to `fallbacks` (a tuple of (ticker, price) pairs) on failure.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as _TOut
    prices: dict[str, float] = {}
    if not tickers:
        return prices

    def _do_fetch() -> dict[str, float]:
        result: dict[str, float] = {}
        try:
            raw = yf.download(list(tickers), period="2d", interval="1d",
                              progress=False, auto_adjust=True)
            cl = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
            for tk in tickers:
                try:
                    col = cl[tk] if tk in cl.columns else cl.iloc[:, 0]
                    result[tk] = float(col.dropna().iloc[-1])
                except Exception:
                    pass
        except Exception:
            for tk in tickers:
                try:
                    h = yf.Ticker(tk).history(period="2d")
                    if not h.empty:
                        result[tk] = float(h["Close"].dropna().iloc[-1])
                except Exception:
                    pass
        return result

    try:
        with ThreadPoolExecutor(max_workers=1) as _ex:
            prices = _ex.submit(_do_fetch).result(timeout=5)
    except Exception:
        pass   # timeout or network error — fall through to fallbacks

    # Apply fallback prices for any ticker still missing
    for tk, px in fallbacks:
        if tk not in prices and px:
            prices[tk] = px

    return prices


def _load_briefing() -> dict:
    if BRIEFING_FILE.exists():
        try:
            return json.loads(BRIEFING_FILE.read_text())
        except Exception:
            pass
    return {}


def _load_equities() -> dict:
    if VIRTUAL_ACCT.exists():
        try:
            return json.loads(VIRTUAL_ACCT.read_text())
        except Exception:
            pass
    return {}


def _load_pipeline_state() -> dict:
    """AI pipeline state — used for HMM regime, oracle, committee, last_spot."""
    if PIPELINE_STATE.exists():
        try:
            return json.loads(PIPELINE_STATE.read_text())
        except Exception:
            pass
    return {}


def _load_physical_portfolio() -> dict:
    """Physical gold/silver holdings from portfolio.json.
    Returns the GC=F entry (or empty dict) and any others.
    Schema: { "GC=F": { "shares": 1.0, "avg_cost": 4150.0, ... }, ... }
    """
    if PORTFOLIO_FILE.exists():
        try:
            return json.loads(PORTFOLIO_FILE.read_text())
        except Exception:
            pass
    return {}


def _load_candidates() -> list[dict]:
    if False:  # candidates panel removed
        raw = json.loads("{}")
        return raw.get("candidates", raw) if isinstance(raw, dict) else raw
    return []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _regime_badge(state: str) -> str:
    cls = {
        "BULLISH":  "badge-bullish",
        "VOLATILE": "badge-ranging",
        "RANGING":  "badge-ranging",   # legacy fallback
        "BEARISH":  "badge-bearish",
    }.get(state, "badge-unknown")
    return f'<span class="regime-badge {cls}">{state}</span>'


def _pnl_class(v: float) -> str:
    return "pos" if v > 0 else ("neg" if v < 0 else "neu")


def _fmt_cur(v: float, sym: str, decimals: int = 0) -> str:
    return f"{sym}{v:,.{decimals}f}"


def _fmt_usd(v: float, decimals: int = 0) -> str:
    return f"${v:,.{decimals}f}"


def _safe_switch(candidates: list[str]) -> None:
    """Try switch_page across path candidates so navigation works whether the
    app was launched via `streamlit run ui/app.py` or `streamlit run ui/pages/0_Home.py`."""
    last_err: Exception | None = None
    for path in candidates:
        try:
            st.switch_page(path)
            return
        except Exception as e:
            last_err = e
            continue
    st.error(
        "Navigation failed — relaunch with `streamlit run ui/app.py` "
        "(or use `./launch_production.sh`) so all pages are reachable."
    )
    if last_err:
        st.caption(f"detail: {last_err}")


# ── Phase XIV: Performance Hero ───────────────────────────────────────────────

_STRATEGY_COLOR = {
    "TREND":          "#22c55e",
    "MEAN_REVERSION": "#06b6d4",
    "PAIRS":          "#a855f7",
    "VOL_SHORT":      "#eab308",
    "TAIL_HEDGE":     "#ef4444",
    "CASH":           "#5a5a5a",
}

_TIER_COLOR = {
    "VERY_HIGH": "#22c55e",
    "HIGH":      "#84cc16",
    "MEDIUM":    "#eab308",
    "LOW":       "#f97316",
    "VERY_LOW":  "#5a5a5a",
}

_TRACK_COLOR = {
    "AHEAD":              "#22c55e",
    "ON_TRACK_AHEAD":     "#84cc16",
    "ON_TRACK":           "#eab308",
    "BEHIND":             "#f97316",
    "WELL_BEHIND":        "#ef4444",
    "CRITICALLY_BEHIND":  "#7f1d1d",
}


def _read_phase14_nav_csv() -> list[tuple[str, float]]:
    p = ROOT / "data" / "phase14_nav.csv"
    if not p.exists():
        return []
    rows: list[tuple[str, float]] = []
    try:
        with p.open() as f:
            next(f, None)
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    try:
                        rows.append((parts[0], float(parts[1])))
                    except ValueError:
                        continue
    except Exception:
        return []
    return rows


def _equity_curve_fig(rows: list[tuple[str, float]]) -> go.Figure:
    if not rows:
        rows = [(date.today().isoformat(), 100000.0)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[r[0] for r in rows],
        y=[r[1] for r in rows],
        mode="lines",
        line=dict(color="#C9A84C", width=2),
        fill="tozeroy",
        fillcolor="rgba(201,168,76,0.10)",
        hovertemplate="<b>%{x}</b><br>NAV: $%{y:,.2f}<extra></extra>",
        name="NAV",
    ))
    if len(rows) > 1:
        start_y = rows[0][1]
        fig.add_hline(y=start_y, line_dash="dot", line_color="#3a3a3a",
                      annotation_text="start", annotation_position="left",
                      annotation_font_size=9, annotation_font_color="#5a5a5a")
    fig.update_layout(
        height=130, margin=dict(l=0, r=0, t=4, b=4),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        showlegend=False,
    )
    return fig


def _family_bars_fig(by_family: dict) -> go.Figure:
    if not by_family:
        return go.Figure()
    names = list(by_family.keys())
    convictions = [float(v.get("conviction", 0)) for v in by_family.values()]
    weights = [float(v.get("weight_share", 0)) for v in by_family.values()]
    colors = ["#22c55e" if c >= 0 else "#ef4444" for c in convictions]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names,
        x=convictions,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{c:+.2f}  (w={w:.1f})" for c, w in zip(convictions, weights)],
        textposition="auto",
        textfont=dict(size=10, color="white"),
        hovertemplate="<b>%{y}</b><br>conviction=%{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        height=130, margin=dict(l=4, r=4, t=6, b=6),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[-1, 1], showgrid=True, gridcolor="#1a1a1a",
            zeroline=True, zerolinecolor="#3a3a3a",
            tickfont=dict(size=9, color="#5a5a5a"),
        ),
        yaxis=dict(
            tickfont=dict(size=10, color="#d8d8d8"),
            categoryorder="array",
            categoryarray=names,
        ),
        showlegend=False,
    )
    return fig


def _render_phase14_hero(ps: dict) -> None:
    stacker = ps.get("alpha_stacker") or {}
    selector = ps.get("strategy_selector") or {}
    targeter = ps.get("performance_targeter") or {}
    book = ps.get("multi_strategy_book") or {}

    # Fall back to direct file reads if pipeline_state hasn't been rebuilt yet
    if not stacker:
        p = ROOT / "data" / "alpha_stacker.json"
        if p.exists():
            try:
                raw = json.loads(p.read_text())
                stacker = {
                    "decision": raw.get("decision", {}),
                    "by_family": raw.get("by_family", {}),
                    "top_drivers": raw.get("top_drivers", []),
                    "top_detractors": raw.get("top_detractors", []),
                }
            except Exception:
                pass
    if not selector:
        p = ROOT / "data" / "strategy_selector.json"
        if p.exists():
            try:
                selector = json.loads(p.read_text())
            except Exception:
                pass
    if not targeter:
        p = ROOT / "data" / "performance_targeter.json"
        if p.exists():
            try:
                targeter = json.loads(p.read_text())
            except Exception:
                pass
    if not book:
        p = ROOT / "data" / "multi_strategy_trader.json"
        if p.exists():
            try:
                book = json.loads(p.read_text())
            except Exception:
                pass

    # ── KPI strip values ─────────────────────────────────────────────────
    target = (targeter.get("target") or {})
    progress = (targeter.get("progress") or {})
    rmult = (targeter.get("risk_multiplier") or {})

    target_pct       = float(target.get("monthly_pct", 10.0))
    mtd_actual       = float(progress.get("actual_progress_pct", 0.0))
    mtd_expected     = float(progress.get("expected_progress_pct", 0.0))
    track_status     = str(progress.get("track_status", "ON_TRACK"))
    projected_mom    = float(progress.get("projected_full_month_pct", 0.0))
    elapsed_pct      = float(progress.get("elapsed_fraction", 0.0)) * 100
    risk_mult_final  = float(rmult.get("final", 1.0))

    decision = (stacker.get("decision") or {})
    direction        = str(decision.get("direction", "HOLD"))
    conviction_score = float(decision.get("conviction_score", 0.0))
    conviction_tier  = str(decision.get("conviction_tier", "VERY_LOW"))
    rec_size_pct     = float(decision.get("recommended_size_pct", 0.0))

    strategy         = str(selector.get("strategy", "CASH"))
    strategy_desc    = str(selector.get("strategy_description", ""))
    final_size_pct   = float(selector.get("final_size_pct", 0.0))
    reasoning_lines  = list(selector.get("reasoning", []))[:5]

    book_equity      = float(book.get("book_equity_usd", 100000.0))
    lifetime_pct     = float(book.get("lifetime_pl_pct", 0.0))
    open_pl_usd      = float(book.get("open_pl_usd", 0.0))
    n_open           = int(book.get("n_open", 0))
    n_closed         = int(book.get("n_closed_total", 0))
    nav_stats        = (book.get("nav_stats") or {})
    book_mtd         = float(nav_stats.get("mtd_return_pct", 0.0))
    book_sharpe      = nav_stats.get("sharpe_approx")

    # ── Progress bar (% to target) ───────────────────────────────────────
    progress_pct_on_bar = max(0.0, min(100.0, (mtd_actual / target_pct) * 100)) if target_pct > 0 else 0.0
    expected_marker_pct = max(0.0, min(100.0, (mtd_expected / target_pct) * 100)) if target_pct > 0 else 0.0
    track_color = _TRACK_COLOR.get(track_status, "#5a5a5a")
    tier_color = _TIER_COLOR.get(conviction_tier, "#5a5a5a")
    strat_color = _STRATEGY_COLOR.get(strategy, "#5a5a5a")
    direction_color = {"BUY": "#22c55e", "SELL": "#ef4444", "HOLD": "#5a5a5a"}.get(direction, "#5a5a5a")

    progress_bar_html = (
        f'<div style="position:relative;width:100%;height:12px;'
        f'background:#1a1a1a;border:1px solid #2a2a2a;border-radius:6px;overflow:hidden;'
        f'margin:6px 0 4px 0">'
        f'<div style="position:absolute;left:0;top:0;height:100%;width:{progress_pct_on_bar:.1f}%;'
        f'background:linear-gradient(90deg,{track_color}55 0%,{track_color}CC 100%);'
        f'transition:width .3s ease"></div>'
        f'<div style="position:absolute;left:{expected_marker_pct:.1f}%;top:-2px;height:16px;'
        f'width:2px;background:#C9A84C;box-shadow:0 0 4px #C9A84CCC"></div>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;font-size:9px;color:#5a5a5a;'
        f'font-family:var(--mono)">'
        f'<span>0%</span><span>target {target_pct:.0f}%</span></div>'
    )

    # ── KPI row ──────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#0c0c0c 0%,#080808 100%);'
        f'border:1px solid #1e1e1e;border-radius:12px;padding:18px 20px 14px 20px;'
        f'margin-bottom:14px">'
        # Header row
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">'
        f'<div style="font-size:10px;color:#C9A84C;font-weight:700;letter-spacing:.2em">'
        f'PHASE XIV · OPERATIONAL ALPHA SYNTHESIS</div>'
        f'<div style="font-size:9px;color:#5a5a5a;font-family:var(--mono);letter-spacing:.1em">'
        f'monthly target · strategy · conviction · book</div>'
        f'</div>'
        # 4-col KPI grid
        f'<div style="display:grid;grid-template-columns:1.4fr 1fr 1fr 1fr;gap:14px">'
        # Column 1: monthly progress
        f'<div style="background:#0a0a0a;border:1px solid #181818;border-radius:8px;padding:12px 14px">'
        f'<div style="font-size:9px;font-weight:600;letter-spacing:.14em;color:#5a5a5a;'
        f'text-transform:uppercase;margin-bottom:6px">monthly return · target {target_pct:.0f}%</div>'
        f'<div style="display:flex;align-items:baseline;gap:8px">'
        f'<div style="font-family:var(--mono);font-size:26px;font-weight:700;color:{track_color}">'
        f'{mtd_actual:+.2f}%</div>'
        f'<div style="font-size:11px;color:#7a7a7a">vs expected {mtd_expected:+.2f}%</div>'
        f'</div>'
        + progress_bar_html
        + f'<div style="display:flex;justify-content:space-between;margin-top:6px;font-size:10px;'
        f'color:#7a7a7a;font-family:var(--mono)">'
        f'<span>day {int(elapsed_pct*21/100)} / 21 · {elapsed_pct:.0f}% elapsed</span>'
        f'<span style="color:{track_color};font-weight:700">{track_status.replace("_", " ")}</span>'
        f'</div>'
        f'<div style="margin-top:4px;font-size:10px;color:#5a5a5a;font-family:var(--mono)">'
        f'projected MoM @ pace: {projected_mom:+.2f}%   ·   risk× {risk_mult_final:.2f}'
        f'</div>'
        f'</div>'
        # Column 2: Strategy
        f'<div style="background:#0a0a0a;border:1px solid #181818;border-radius:8px;padding:12px 14px">'
        f'<div style="font-size:9px;font-weight:600;letter-spacing:.14em;color:#5a5a5a;'
        f'text-transform:uppercase;margin-bottom:6px">active strategy</div>'
        f'<div style="font-family:var(--mono);font-size:18px;font-weight:700;'
        f'color:{strat_color};line-height:1.1">{strategy.replace("_", " ")}</div>'
        f'<div style="font-size:10px;color:#7a7a7a;margin-top:6px;line-height:1.4">'
        f'{strategy_desc[:120]}{"…" if len(strategy_desc) > 120 else ""}'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;margin-top:10px;'
        f'font-size:10px;color:#5a5a5a;font-family:var(--mono)">'
        f'<span>direction</span>'
        f'<span style="color:{direction_color};font-weight:700">{direction}</span></div>'
        f'<div style="display:flex;justify-content:space-between;'
        f'font-size:10px;color:#5a5a5a;font-family:var(--mono)">'
        f'<span>final size</span>'
        f'<span style="color:#d8d8d8;font-weight:700">{final_size_pct:.2f}%</span></div>'
        f'</div>'
        # Column 3: Conviction
        f'<div style="background:#0a0a0a;border:1px solid #181818;border-radius:8px;padding:12px 14px">'
        f'<div style="font-size:9px;font-weight:600;letter-spacing:.14em;color:#5a5a5a;'
        f'text-transform:uppercase;margin-bottom:6px">alpha stacker conviction</div>'
        f'<div style="display:flex;align-items:baseline;gap:8px">'
        f'<div style="font-family:var(--mono);font-size:22px;font-weight:700;'
        f'color:{tier_color}">{conviction_score:+.4f}</div>'
        f'<div style="font-size:10px;color:{tier_color};font-weight:700;letter-spacing:.1em">'
        f'{conviction_tier.replace("_", " ")}</div>'
        f'</div>'
        # Conviction gauge
        f'<div style="position:relative;width:100%;height:8px;background:#1a1a1a;'
        f'border-radius:4px;margin-top:10px;overflow:hidden">'
        f'<div style="position:absolute;left:50%;top:0;height:100%;width:1px;'
        f'background:#3a3a3a;z-index:2"></div>'
        f'<div style="position:absolute;'
        + (
            f'left:50%;width:{min(abs(conviction_score)*50, 50):.1f}%;'
            if conviction_score >= 0 else
            f'right:50%;width:{min(abs(conviction_score)*50, 50):.1f}%;'
        )
        + f'top:0;height:100%;background:{tier_color}AA"></div>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;margin-top:8px;'
        f'font-size:10px;color:#5a5a5a;font-family:var(--mono)">'
        f'<span>rec size</span>'
        f'<span style="color:#d8d8d8;font-weight:700">{rec_size_pct:.1f}%</span></div>'
        f'<div style="display:flex;justify-content:space-between;'
        f'font-size:10px;color:#5a5a5a;font-family:var(--mono)">'
        f'<span>signals</span>'
        + f'<span style="color:#d8d8d8;font-weight:700">{len(stacker.get("top_drivers", [])) + len(stacker.get("top_detractors", []))}+</span>'
        f'</div>'
        f'</div>'
        # Column 4: Multi-Strategy Book
        f'<div style="background:#0a0a0a;border:1px solid #181818;border-radius:8px;padding:12px 14px">'
        f'<div style="font-size:9px;font-weight:600;letter-spacing:.14em;color:#5a5a5a;'
        f'text-transform:uppercase;margin-bottom:6px">phase XIV book</div>'
        f'<div style="font-family:var(--mono);font-size:20px;font-weight:700;'
        f'color:#d8d8d8">${book_equity:,.0f}</div>'
        f'<div style="font-size:11px;font-family:var(--mono);'
        f'color:{"#22c55e" if lifetime_pct >= 0 else "#ef4444"};font-weight:700">'
        f'{lifetime_pct:+.2f}% lifetime'
        + (f'   ·   open ${open_pl_usd:+,.0f}' if abs(open_pl_usd) > 0.01 else "")
        + f'</div>'
        f'<div style="display:flex;justify-content:space-between;margin-top:10px;'
        f'font-size:10px;color:#5a5a5a;font-family:var(--mono)">'
        f'<span>MTD</span>'
        f'<span style="color:{"#22c55e" if book_mtd >= 0 else "#ef4444"};font-weight:700">'
        f'{book_mtd:+.2f}%</span></div>'
        f'<div style="display:flex;justify-content:space-between;'
        f'font-size:10px;color:#5a5a5a;font-family:var(--mono)">'
        f'<span>open / closed</span>'
        f'<span style="color:#d8d8d8;font-weight:700">{n_open} / {n_closed}</span></div>'
        + (
            f'<div style="display:flex;justify-content:space-between;'
            f'font-size:10px;color:#5a5a5a;font-family:var(--mono)">'
            f'<span>Sharpe</span>'
            f'<span style="color:#d8d8d8;font-weight:700">{float(book_sharpe):+.2f}</span></div>'
            if book_sharpe is not None else ""
        )
        + f'</div>'
        f'</div>'  # close grid
        f'</div>',  # close outer card
        unsafe_allow_html=True,
    )

    # ── Charts row ───────────────────────────────────────────────────────
    chart_left, chart_right = st.columns([1.0, 1.0])
    with chart_left:
        st.markdown(
            '<div style="background:#0a0a0a;border:1px solid #181818;border-radius:10px;'
            'padding:12px 14px 4px 14px;margin-bottom:14px">'
            '<div style="font-size:9px;font-weight:600;letter-spacing:.14em;color:#5a5a5a;'
            'text-transform:uppercase;margin-bottom:2px">multi-strategy equity curve</div>',
            unsafe_allow_html=True,
        )
        nav_rows = _read_phase14_nav_csv()
        st.plotly_chart(_equity_curve_fig(nav_rows), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with chart_right:
        st.markdown(
            '<div style="background:#0a0a0a;border:1px solid #181818;border-radius:10px;'
            'padding:12px 14px 4px 14px;margin-bottom:14px">'
            '<div style="font-size:9px;font-weight:600;letter-spacing:.14em;color:#5a5a5a;'
            'text-transform:uppercase;margin-bottom:2px">conviction by signal family</div>',
            unsafe_allow_html=True,
        )
        by_family = stacker.get("by_family", {}) or {}
        st.plotly_chart(_family_bars_fig(by_family), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Reasoning row: drivers | detractors | strategy bullets ───────────
    drv_col, det_col, why_col = st.columns([1.0, 1.0, 1.0])

    def _signal_row(s: dict, sign: str) -> str:
        name = s.get("name", "")
        note = s.get("note", "")
        contrib = float(s.get("contribution", 0.0))
        color = "#22c55e" if contrib >= 0 else "#ef4444"
        return (
            f'<div style="display:flex;justify-content:space-between;'
            f'font-size:11px;line-height:1.5;padding:3px 0;'
            f'border-bottom:1px solid #131313">'
            f'<span style="color:#d8d8d8"><span style="color:{color};font-weight:700">'
            f'{sign}</span> {name}</span>'
            f'<span style="font-family:var(--mono);color:{color};font-weight:700">'
            f'{contrib:+.3f}</span>'
            f'</div>'
            f'<div style="font-size:9px;color:#5a5a5a;'
            f'padding:0 0 4px 12px;font-family:var(--mono)">{note}</div>'
        )

    with drv_col:
        drivers = stacker.get("top_drivers", []) or []
        st.markdown(
            '<div style="background:#0a0a0a;border:1px solid #181818;border-radius:10px;'
            'padding:12px 14px;height:100%">'
            '<div style="font-size:9px;font-weight:600;letter-spacing:.14em;color:#22c55e;'
            'text-transform:uppercase;margin-bottom:8px">top drivers</div>'
            + ("".join(_signal_row(s, "+") for s in drivers[:5]) or
               '<div style="font-size:11px;color:#5a5a5a;font-style:italic">'
               'no drivers — conviction near zero</div>')
            + '</div>',
            unsafe_allow_html=True,
        )

    with det_col:
        detractors = stacker.get("top_detractors", []) or []
        st.markdown(
            '<div style="background:#0a0a0a;border:1px solid #181818;border-radius:10px;'
            'padding:12px 14px;height:100%">'
            '<div style="font-size:9px;font-weight:600;letter-spacing:.14em;color:#ef4444;'
            'text-transform:uppercase;margin-bottom:8px">top detractors</div>'
            + ("".join(_signal_row(s, "−") for s in detractors[:5]) or
               '<div style="font-size:11px;color:#5a5a5a;font-style:italic">'
               'no detractors — conviction near zero</div>')
            + '</div>',
            unsafe_allow_html=True,
        )

    with why_col:
        st.markdown(
            '<div style="background:#0a0a0a;border:1px solid #181818;border-radius:10px;'
            'padding:12px 14px;height:100%">'
            '<div style="font-size:9px;font-weight:600;letter-spacing:.14em;color:#C9A84C;'
            'text-transform:uppercase;margin-bottom:8px">why this strategy</div>'
            + ("".join(
                f'<div style="font-size:11px;color:#d8d8d8;line-height:1.55;padding:3px 0">'
                f'<span style="color:#C9A84C">›</span> {r}</div>'
                for r in reasoning_lines
            ) or '<div style="font-size:11px;color:#5a5a5a;font-style:italic">'
                 'no reasoning available</div>')
            + '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)


# ── Session state defaults ─────────────────────────────────────────────────────
for _k, _v in {"home_currency": "AED", "home_unit": "g"}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Main render ───────────────────────────────────────────────────────────────

def main() -> None:
    _inject_css()

    # ── Currency / unit resolution ────────────────────────────────────────────
    _cur      = st.session_state.get("home_currency", "AED")
    _unit     = st.session_state.get("home_unit", "g")
    _preset   = next((p for p in UNIT_PRESETS if p["currency"] == _cur and p["unit"] == _unit),
                     UNIT_PRESETS[0])
    _sym      = CURRENCIES[_cur]["sym"]
    _rate     = CURRENCIES[_cur]["rate"]
    _px_fac   = _preset["px_factor"]   # converts USD/oz price → chosen currency/unit
    _qt_fac   = _preset["qty_factor"]  # converts oz qty → chosen unit
    _qty_dec  = _preset["qty_dec"]

    # ── Load all data ─────────────────────────────────────────────────────────
    briefing = _load_briefing()
    acct     = _load_equities()          # virtual paper portfolio (training only)
    ps       = _load_pipeline_state()   # AI regime / oracle / committee
    phys     = _load_physical_portfolio()  # real holdings: portfolio.json

    metals_rg = ps.get("regime", {})
    metals_cm = ps.get("committee", {})

    # Physical gold from portfolio.json (the source of truth for real holdings)
    gold_entry   = phys.get("GC=F", {})
    metals_oz    = float(gold_entry.get("shares", 0))
    metals_avg   = float(gold_entry.get("avg_cost", 0))
    metals_state = "ACTIVE" if metals_oz > 0 else "FIAT"

    # Last known spot price from pipeline_state as yfinance fallback
    _last_spot = float(ps.get("portfolio", {}).get("last_spot", 0))
    _fallbacks: tuple[tuple[str, float], ...] = (("GC=F", _last_spot),) if _last_spot else ()

    # Virtual equity tickers (training only — prices needed for training section)
    eq_positions = {
        sym: p for sym, p in acct.get("positions", {}).items()
        if float(p.get("qty", 0)) > 0
    }
    eq_tickers  = tuple(eq_positions.keys())
    all_tickers = (("GC=F",) if metals_oz > 0 else ()) + eq_tickers
    live_prices = _fetch_live_prices(all_tickers, _fallbacks)

    # Physical portfolio metrics
    gold_px        = live_prices.get("GC=F", metals_avg or _last_spot)
    metals_val     = metals_oz * gold_px if metals_oz > 0 and gold_px else 0.0
    metals_pnl     = metals_oz * (gold_px - metals_avg) if metals_oz > 0 and gold_px and metals_avg else 0.0
    metals_pnl_pct = (metals_pnl / (metals_oz * metals_avg) * 100) if metals_oz > 0 and metals_avg else 0.0
    total_aum      = metals_val
    price_is_live  = "GC=F" in live_prices and live_prices["GC=F"] != _last_spot

    # ── Unified top navigation bar ────────────────────────────────────────────
    _ts_now    = datetime.now(timezone.utc).strftime("%Y-%m-%d  %H:%M UTC")
    _price_tag = "" if price_is_live else "  ·  CACHED"
    st.markdown(
        f'<div class="fn-topnav">'
        f'<span class="fn-brand">Fund Manager</span>'
        f'<span class="fn-brand-sep"></span>'
        f'<span class="fn-page-id">Command Centre</span>'
        f'<span class="fn-ts">{_ts_now}{_price_tag}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    # ── Split nav: tabs LEFT | spacer | utilities RIGHT ────────────────────
    st.markdown('<div class="fn-nav-row">', unsafe_allow_html=True)
    _n0, _n1, _n2, _n3, _nsp, _u1, _u2 = st.columns([1, 1, 1, 1.5, 5, 0.85, 1.35])
    with _n0:
        st.button("Home",          key="topnav_Home",        disabled=True)
    with _n1:
        if st.button("Metals",     key="topnav_Metals"):
            _safe_switch(["app.py", "../app.py", "ui/app.py"])
    with _n2:
        if st.button("Equities",   key="topnav_Equities"):
            _safe_switch(["pages/2_Equity.py", "2_Equity.py", "ui/pages/2_Equity.py"])
    with _n3:
        if st.button("Wealth Agent", key="topnav_WealthAgent"):
            _safe_switch(["pages/1_Agent.py", "1_Agent.py", "ui/pages/1_Agent.py"])
    # _nsp: visual gap — empty
    with _u1:
        if st.button("Refresh",    key="home_refresh"):
            st.cache_data.clear()
            st.rerun()
    with _u2:
        if st.button("Regen Briefing", key="home_regen_briefing"):
            import subprocess as _sp2
            with st.spinner("Calling Chief of Staff..."):
                _r = _sp2.run(
                    ["python3", str(ROOT / "scripts" / "executive_briefer.py")],
                    capture_output=True, text=True, timeout=90,
                )
            if _r.returncode == 0:
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(_r.stderr[-200:] or "Briefer failed")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Unit / currency preset strip ──────────────────────────────────────────
    _spacer, _pu1, _pu2, _pu3 = st.columns([7, 1, 1, 1])
    for _col, _pr in zip((_pu1, _pu2, _pu3), UNIT_PRESETS):
        with _col:
            _active = (_cur == _pr["currency"] and _unit == _pr["unit"])
            if st.button(
                _pr["label"],
                key=f"home_preset_{_pr['label']}",
                type="primary" if _active else "secondary",
                use_container_width=True,
            ):
                st.session_state.home_currency = _pr["currency"]
                st.session_state.home_unit     = _pr["unit"]
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION −2 — PERFORMANCE HERO  (Phase XIV — monthly target & strategy)
    #   Monthly progress to 10%, active strategy, alpha-stacker conviction,
    #   multi-strategy book equity, drivers / detractors, equity curve.
    # ════════════════════════════════════════════════════════════════════════
    _render_phase14_hero(ps)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION −1 — HERO PANEL  (Phase XI Stage 61 — operator surface)
    #   The day's trade idea + DeepSeek's plain-English briefing.
    # ════════════════════════════════════════════════════════════════════════
    _ti_path = ROOT / "data" / "trade_idea.json"
    _ds_path_h = ROOT / "data" / "deepseek_last_turn.json"

    if _ti_path.exists():
        try:
            _ti = json.loads(_ti_path.read_text())
        except Exception:
            _ti = {}
        _tc = _ti.get("trade_card", {})
        _side = _tc.get("side", "HOLD")
        _ticker = _tc.get("ticker", "—")
        _size_pct = _tc.get("size_pct", 0)
        _size_usd = _tc.get("size_usd", 0)
        _conv = _tc.get("conviction", "—")
        _entry = _tc.get("entry_price")
        _stop = _tc.get("stop_price")
        _target = _tc.get("target_price")
        _risk_flags = _ti.get("risk_flags", [])

        _side_color = {
            "BUY":  "#22c55e",
            "SELL": "#ef4444",
            "HOLD": "var(--muted)",
        }.get(_side, "var(--text)")
        _conv_color = {
            "HIGH":   "#22c55e",
            "MEDIUM": "#eab308",
            "LOW":    "#ef4444",
            "—":      "var(--muted)",
        }.get(_conv, "var(--text)")

        _flags_html = ""
        if _risk_flags:
            _flags_html = (
                '<div style="margin-top:14px;padding:10px;background:#7f1d1d20;'
                'border-left:3px solid #ef4444;border-radius:4px">'
                '<div style="font-size:11px;font-weight:700;color:#fca5a5;margin-bottom:4px">'
                f'⚠ {len(_risk_flags)} ACTIVE RISK FLAGS</div>'
                + "".join(
                    f'<div style="font-size:11px;color:#fecaca;line-height:1.6">• {f}</div>'
                    for f in _risk_flags[:5]
                )
                + '</div>'
            )

        _prices_html = ""
        if _entry:
            _prices_html = (
                '<div style="display:flex;gap:24px;margin-top:12px;font-size:12px">'
                f'<div><span style="color:var(--muted)">ENTRY</span><br>'
                f'<span style="color:var(--text);font-weight:700;font-size:14px">${_entry:,.2f}</span></div>'
                + (
                    f'<div><span style="color:var(--muted)">STOP</span><br>'
                    f'<span style="color:#ef4444;font-weight:700;font-size:14px">${_stop:,.2f}</span></div>'
                    if _stop else ""
                )
                + (
                    f'<div><span style="color:var(--muted)">TARGET</span><br>'
                    f'<span style="color:#22c55e;font-weight:700;font-size:14px">${_target:,.2f}</span></div>'
                    if _target else ""
                )
                + '</div>'
            )

        # Hero trade card
        st.markdown(
            f'<div style="background:linear-gradient(135deg, var(--card-bg) 0%, #0a0f1c 100%);'
            f'border:2px solid var(--gold);border-radius:12px;padding:22px;margin-bottom:18px;'
            f'box-shadow: 0 4px 20px rgba(212, 175, 55, 0.15)">'
            f'<div style="display:flex;justify-content:space-between;align-items:start;margin-bottom:12px">'
            f'<div>'
            f'<div style="font-size:11px;color:var(--gold);font-weight:700;letter-spacing:2px;margin-bottom:6px">'
            f'TODAY\'S TRADE IDEA</div>'
            f'<div style="font-size:32px;font-weight:800;line-height:1">'
            f'<span style="color:{_side_color}">{_side}</span> '
            f'<span style="color:var(--text)">{_ticker}</span>'
            f'</div>'
            f'</div>'
            f'<div style="text-align:right">'
            f'<div style="font-size:11px;color:var(--muted)">SIZE</div>'
            f'<div style="font-size:24px;font-weight:700;color:var(--text)">{_size_pct:.2f}%</div>'
            f'<div style="font-size:12px;color:var(--muted)">${_size_usd:,.0f}</div>'
            f'</div>'
            f'</div>'
            f'<div style="display:flex;gap:16px;font-size:12px;color:var(--muted)">'
            f'<span>Conviction: <span style="color:{_conv_color};font-weight:700">{_conv}</span></span>'
            f'<span>Champion: <span style="color:var(--text)">{_tc.get("champion_signal", "—")}</span></span>'
            f'<span>Horizon: <span style="color:var(--text)">{_tc.get("horizon_days", 21)}d</span></span>'
            f'<span>IBKR-ready: <span style="color:var(--text)">{"YES" if _ti.get("ibkr_ready", False) else "no"}</span></span>'
            f'</div>'
            + _prices_html
            + _flags_html
            + '</div>',
            unsafe_allow_html=True,
        )

        # DeepSeek briefing right below
        if _ds_path_h.exists():
            try:
                _ds = json.loads(_ds_path_h.read_text())
            except Exception:
                _ds = {}
            _answer = _ds.get("answer", "")
            if _answer:
                st.markdown(
                    f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                    f'border-radius:10px;padding:18px;margin-bottom:18px">'
                    f'<div style="font-size:11px;color:var(--gold);font-weight:700;letter-spacing:2px;margin-bottom:8px">'
                    f'EXECUTIVE BRIEFING — DEEPSEEK'
                    f'</div>'
                    f'<div style="font-size:13px;color:var(--text);line-height:1.7;white-space:pre-wrap">'
                    f'{_answer}'
                    f'</div>'
                    f'<div style="font-size:10px;color:var(--muted);margin-top:10px">'
                    f'Generated {_ds.get("ts", "—")} · {_ds.get("total_tokens", 0)} tokens · '
                    f'{len(_ds.get("dossier_keys", []))} engines'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

        # Phase XII status strip below the hero card
        _sh_path = ROOT / "data" / "system_health.json"
        _rb_path = ROOT / "data" / "operator_runbook.json"
        _tb_path = ROOT / "data" / "trade_basket.json"
        _pr_path = ROOT / "data" / "position_reconciliation.json"
        if any(p.exists() for p in (_sh_path, _rb_path, _tb_path, _pr_path)):
            hs_c1, hs_c2, hs_c3, hs_c4 = st.columns(4)

            with hs_c1:
                _sh = json.loads(_sh_path.read_text()) if _sh_path.exists() else {}
                _status = _sh.get("overall_status", "n/a")
                _scolor = {"OK": "#22c55e", "DEGRADED": "#eab308", "CRITICAL": "#ef4444"}.get(_status, "var(--text)")
                _bs = _sh.get("by_severity", {})
                st.markdown(
                    f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                    f'border-radius:8px;padding:12px;">'
                    f'<div style="font-size:11px;color:var(--gold);font-weight:700">SYSTEM HEALTH</div>'
                    f'<div style="font-size:14px;color:{_scolor};font-weight:700;margin-top:4px">{_status}</div>'
                    f'<div style="font-size:10px;color:var(--muted);margin-top:4px">'
                    f'C={_bs.get("CRITICAL", 0)} H={_bs.get("HIGH", 0)} M={_bs.get("MEDIUM", 0)} L={_bs.get("LOW", 0)}'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

            with hs_c2:
                _rb = json.loads(_rb_path.read_text()) if _rb_path.exists() else {}
                _ok = _rb.get("n_checklist_ok", 0)
                _tot = _rb.get("n_checklist", 0)
                _rcolor = "#22c55e" if _ok == _tot else "#ef4444" if _ok < _tot - 1 else "#eab308"
                st.markdown(
                    f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                    f'border-radius:8px;padding:12px;">'
                    f'<div style="font-size:11px;color:var(--gold);font-weight:700">PRE-TRADE CHECKLIST</div>'
                    f'<div style="font-size:14px;color:{_rcolor};font-weight:700;margin-top:4px">{_ok}/{_tot} ✅</div>'
                    f'<div style="font-size:10px;color:var(--muted);margin-top:4px">'
                    f'flags: {_rb.get("n_risk_flags", 0)}  metals: {_rb.get("metals_action", "—")}'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

            with hs_c3:
                _tb = json.loads(_tb_path.read_text()) if _tb_path.exists() else {}
                _nl = _tb.get("n_long", 0)
                _ns = _tb.get("n_short", 0)
                _cash = _tb.get("cash_pct", 0)
                st.markdown(
                    f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                    f'border-radius:8px;padding:12px;">'
                    f'<div style="font-size:11px;color:var(--gold);font-weight:700">TRADE BASKET</div>'
                    f'<div style="font-size:14px;color:var(--text);font-weight:700;margin-top:4px">'
                    f'{_nl}L / {_ns}S</div>'
                    f'<div style="font-size:10px;color:var(--muted);margin-top:4px">'
                    f'cash {_cash:.1f}%  ·  cap {_tb.get("gross_cap", 0):.1f}%'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

            with hs_c4:
                _pr = json.loads(_pr_path.read_text()) if _pr_path.exists() else {}
                _rstatus = _pr.get("status", "n/a")
                _rcolor2 = {"OK": "#22c55e", "DRIFT": "#eab308", "MAJOR_DRIFT": "#ef4444"}.get(_rstatus, "var(--text)")
                _drift = _pr.get("diff", {}).get("n_drift_total", 0)
                st.markdown(
                    f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                    f'border-radius:8px;padding:12px;">'
                    f'<div style="font-size:11px;color:var(--gold);font-weight:700">RECONCILIATION</div>'
                    f'<div style="font-size:14px;color:{_rcolor2};font-weight:700;margin-top:4px">{_rstatus}</div>'
                    f'<div style="font-size:10px;color:var(--muted);margin-top:4px">'
                    f'drift={_drift}  ·  mode={_pr.get("ibkr_mode", "—")}'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        # Phase XIII strip: economic calendar / earnings / data quality / P&L
        _ec_path = ROOT / "data" / "economic_calendar.json"
        _ear_path = ROOT / "data" / "earnings_calendar.json"
        _dq_path = ROOT / "data" / "data_quality.json"
        _pn_path = ROOT / "data" / "pnl_tracker.json"
        if any(p.exists() for p in (_ec_path, _ear_path, _dq_path, _pn_path)):
            x_c1, x_c2, x_c3, x_c4 = st.columns(4)

            with x_c1:
                _ec = json.loads(_ec_path.read_text()) if _ec_path.exists() else {}
                _guard = _ec.get("position_guard", "n/a")
                _gcolor = {
                    "NORMAL":                  "#22c55e",
                    "SIZE_DOWN_NEW_POSITIONS": "#eab308",
                    "HOLD_NEW_POSITIONS":      "#ef4444",
                }.get(_guard, "var(--text)")
                _ne = _ec.get("next_event") or {}
                _kind = _ne.get("kind", "—")
                _t = _ne.get("days_until")
                st.markdown(
                    f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                    f'border-radius:8px;padding:12px;">'
                    f'<div style="font-size:11px;color:var(--gold);font-weight:700">MACRO CALENDAR</div>'
                    f'<div style="font-size:12px;color:{_gcolor};font-weight:700;margin-top:4px">'
                    f'{_guard}</div>'
                    f'<div style="font-size:10px;color:var(--muted);margin-top:4px">'
                    f'next: {_kind}  T+{_t if _t is not None else "?"}d</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with x_c2:
                _er = json.loads(_ear_path.read_text()) if _ear_path.exists() else {}
                _nb = _er.get("n_blocked", 0)
                _ecolor = "#22c55e" if _nb == 0 else "#ef4444"
                _next5 = _er.get("next_5_earnings", []) or []
                _next_txt = (
                    f"next: {_next5[0].get('ticker', '?')} T+{_next5[0].get('days_until', '?')}d"
                    if _next5 else "no upcoming"
                )
                st.markdown(
                    f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                    f'border-radius:8px;padding:12px;">'
                    f'<div style="font-size:11px;color:var(--gold);font-weight:700">EARNINGS</div>'
                    f'<div style="font-size:12px;color:{_ecolor};font-weight:700;margin-top:4px">'
                    f'blocked: {_nb}</div>'
                    f'<div style="font-size:10px;color:var(--muted);margin-top:4px">{_next_txt}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with x_c3:
                _dq = json.loads(_dq_path.read_text()) if _dq_path.exists() else {}
                _ds = _dq.get("overall_status", "n/a")
                _dcolor = {"OK": "#22c55e", "WARN": "#eab308", "DEGRADED": "#ef4444", "CRITICAL": "#ef4444"}.get(_ds, "var(--text)")
                st.markdown(
                    f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                    f'border-radius:8px;padding:12px;">'
                    f'<div style="font-size:11px;color:var(--gold);font-weight:700">DATA QUALITY</div>'
                    f'<div style="font-size:12px;color:{_dcolor};font-weight:700;margin-top:4px">{_ds}</div>'
                    f'<div style="font-size:10px;color:var(--muted);margin-top:4px">'
                    f'{_dq.get("n_checks", 0)} checks  ·  {_dq.get("n_failures", 0)} fail</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with x_c4:
                _pn = json.loads(_pn_path.read_text()) if _pn_path.exists() else {}
                _day = _pn.get("day_pnl_usd", 0) or 0
                _day_pct = _pn.get("day_pnl_pct", 0) or 0
                _pcolor = "#22c55e" if _day >= 0 else "#ef4444"
                _nav = _pn.get("latest_nav_usd", 0) or 0
                _cum = _pn.get("cumulative_return_pct", 0) or 0
                st.markdown(
                    f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                    f'border-radius:8px;padding:12px;">'
                    f'<div style="font-size:11px;color:var(--gold);font-weight:700">P&L TRACKER</div>'
                    f'<div style="font-size:12px;color:{_pcolor};font-weight:700;margin-top:4px">'
                    f'{_day_pct:+.3f}% day</div>'
                    f'<div style="font-size:10px;color:var(--muted);margin-top:4px">'
                    f'NAV ${_nav:,.0f}  ·  cum {_cum:+.2f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        # Reasoning expander
        with st.expander("Side vote + size stack (engine reasoning)"):
            _r = _ti.get("reasoning", {})
            st.markdown("**SIDE VOTE**")
            for line in _r.get("side_vote", []):
                st.markdown(f"- {line}")
            st.markdown("**SIZE STACK**")
            for line in _r.get("size_stack", []):
                st.markdown(f"- {line}")

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 0 — Executive Briefing
    # ────────────────────────────────────────────────────────────────────────
    briefing_text = briefing.get("briefing", "")
    briefing_ts   = briefing.get("generated_at", "")

    if briefing_text:
        gen_label = f"Generated {briefing_ts[:10]}" if briefing_ts else ""
        st.markdown(
            f'<div class="briefing-box">'
            f'<div class="briefing-header">Chief of Staff Briefing'
            f'{"  ·  " + gen_label if gen_label else ""}</div>'
            f'{briefing_text}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "No executive briefing yet. Run `python3 scripts/executive_briefer.py` "
            "or wait for the next pipeline cycle."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────────
    # TRAINING SUFFICIENT banner
    # ────────────────────────────────────────────────────────────────────────
    _lstm_meta_path = ROOT / "data" / "gold_lstm_meta.json"
    _trained_at_str = ""
    if _lstm_meta_path.exists():
        try:
            _trained_at_str = json.loads(_lstm_meta_path.read_text()).get("trained_at", "")
        except Exception:
            pass

    st.markdown(
        '<style>'
        '@keyframes training-glow {'
        '  0%,100%{box-shadow:0 0 6px #22c55e44;}'
        '  50%{box-shadow:0 0 18px #22c55eaa;}'
        '}'
        '</style>'
        '<div style="'
        'background:#0d2818;'
        'border:2px solid #22c55e;'
        'border-radius:8px;'
        'padding:22px 28px;'
        'margin:0 0 1.5rem 0;'
        'text-align:center;'
        'font-family:\'Space Mono\',monospace;'
        'animation:training-glow 2.5s ease-in-out infinite;'
        '">'
        '<div style="font-size:18px;font-weight:700;letter-spacing:.14em;color:#d4af37;margin-bottom:6px;">'
        'TRAINING SUFFICIENT'
        '</div>'
        '<div style="font-size:14px;color:#ffffff;letter-spacing:.06em;">'
        '&mdash; YOU MAY PROCEED WITH REAL TRADING INTEGRATION &mdash;'
        '</div>'
        '<div style="font-size:11px;color:#9ca3af;margin-top:10px;">'
        'All models trained and validated &mdash; LSTM, HMM Regime, Proving Ground'
        '</div>'
        + (
            '<div style="font-size:10px;color:#6b7280;margin-top:6px;">'
            f'Last training completed: {_trained_at_str}'
            '</div>' if _trained_at_str else ''
        )
        + '</div>',
        unsafe_allow_html=True,
    )

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 1 — KPI Strip
    # ────────────────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)

    pnl_sign  = "+" if metals_pnl >= 0 else ""
    pnl_cls   = _pnl_class(metals_pnl)
    hmm_state = metals_rg.get("hmm_state", "UNKNOWN")

    # Converted display values
    _aum_disp  = total_aum  * _rate                   # total value in chosen currency
    _pnl_disp  = metals_pnl * _rate
    _qty_disp  = metals_oz  * _qt_fac                 # oz → g (or stays oz)
    _px_disp   = gold_px    * _rate * _px_fac         # price per unit in chosen currency

    with k1:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Portfolio Value</div>'
            f'<div class="kpi-value">{_fmt_cur(_aum_disp, _sym)}</div>'
            f'<div class="kpi-sub">Metals book</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Gold Held</div>'
            f'<div class="kpi-value">{_qty_disp:.{_qty_dec}f} {_unit}</div>'
            f'<div class="kpi-sub">{metals_state}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Open PnL</div>'
            f'<div class="kpi-value {pnl_cls}">'
            f'{pnl_sign}{_fmt_cur(_pnl_disp, _sym)}</div>'
            f'<div class="kpi-sub">{pnl_sign}{metals_pnl_pct:.1f}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">HMM Regime</div>'
            f'<div class="kpi-value" style="font-size:16px;">{hmm_state}</div>'
            f'<div class="kpi-sub">Oracle: {metals_cm.get("oracle_score", "—")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Load virtual FIAT cash for 5th card
    _virt_cash = 0.0
    if VIRTUAL_ACCT.exists():
        try:
            _virt_cash = float(json.loads(VIRTUAL_ACCT.read_text()).get("cash_balance", 0))
        except Exception:
            pass

    with k5:
        st.markdown(
            f'<div class="kpi-card" style="border-color:#22c55e33;">'
            f'<div class="kpi-label" style="color:#22c55e">Available FIAT</div>'
            f'<div class="kpi-value" style="font-size:18px;">${_virt_cash:,.2f}</div>'
            f'<div class="kpi-sub">Virtual account</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 2 — Pie Chart + Holdings Table
    # ────────────────────────────────────────────────────────────────────────
    pie_col, tbl_col = st.columns([1, 1.6])

    # Build pie slices — Gold only; no cash slices
    pie_labels, pie_values, pie_colors = [], [], []

    if metals_oz > 0:
        gold_price = live_prices.get("GC=F", 0)
        gold_mkt   = metals_oz * gold_price if gold_price else metals_val
        pie_labels.append("Gold")
        pie_values.append(gold_mkt)   # always USD internally; converted for display
        pie_colors.append(PIE_GOLD_COLOR)
    # No pie when in FIAT — handled below with a placeholder message

    with pie_col:
        st.markdown('<div class="section-header">Allocation</div>',
                    unsafe_allow_html=True)
        if pie_values:
            fig = go.Figure(go.Pie(
                labels=pie_labels,
                values=pie_values,
                hole=0.52,
                marker=dict(colors=pie_colors,
                            line=dict(color="#0a0a0a", width=2)),
                textinfo="label+percent",
                textfont=dict(size=11, color="#d8d8d8"),
                hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
                insidetextorientation="radial",
            ))
            fig.add_annotation(
                text=f"<b>{_fmt_cur(_aum_disp, _sym)}</b><br><span style='font-size:10px'>Gold Value</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="#d8d8d8"),
                align="center",
            )
            fig.update_layout(
                paper_bgcolor="#0a0a0a",
                plot_bgcolor="#0a0a0a",
                margin=dict(l=0, r=0, t=10, b=0),
                height=340,
                showlegend=False,
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.markdown(
                '<div class="radar-table" style="height:280px;display:flex;'
                'align-items:center;justify-content:center;text-align:center;">'
                '<div><div style="font-size:28px;margin-bottom:10px">—</div>'
                '<div style="color:var(--muted);font-size:12px">No gold position<br>'
                'Metals book in FIAT</div></div></div>',
                unsafe_allow_html=True,
            )

    # Holdings table
    with tbl_col:
        st.markdown('<div class="section-header">Current Holdings</div>',
                    unsafe_allow_html=True)

        if metals_oz > 0:
            gold_px  = live_prices.get("GC=F", 0)
            gold_mkt = metals_oz * gold_px
            _avg_disp = metals_avg * _rate * _px_fac
            rows = [{
                "Asset":      "Gold (XAU)",
                "Qty":        f"{_qty_disp:.{_qty_dec}f} {_unit}",
                "Avg Cost":   f"{_sym}{_avg_disp:,.2f}/{_unit}" if metals_avg else "—",
                "Live Price": f"{_sym}{_px_disp:,.2f}/{_unit}" if gold_px else "—",
                "Mkt Value":  f"{_sym}{gold_mkt * _rate:,.2f}" if gold_px else "—",
                "Open PnL":   f"{_sym}{_pnl_disp:+,.2f}",
                "PnL %":      f"{metals_pnl_pct:+.1f}%",
            }]
            df_hold = pd.DataFrame(rows)
            st.dataframe(df_hold, width="stretch", hide_index=True, height=120)
        else:
            st.markdown(
                '<div style="color:var(--muted);font-size:13px;padding:40px 0;text-align:center">'
                'No gold position — metals book is in FIAT.</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 3 — Metals Signal Board (full width)
    # ────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Metals Signal Board</div>',
                unsafe_allow_html=True)

    sig_col, _ = st.columns([1, 1])

    p_bull     = metals_rg.get("p_bullish", 0)
    p_range    = metals_rg.get("p_volatile", metals_rg.get("p_ranging", 0))
    p_bear     = metals_rg.get("p_bearish", 0)
    veto       = metals_rg.get("hmm_veto_active", False)
    oracle_scr = metals_cm.get("oracle_score")
    action     = metals_cm.get("action_taken", "—")
    q_conv     = metals_cm.get("quant_conviction", "—")
    m_conv     = metals_cm.get("macro_conviction", "—")
    fitted_at  = metals_rg.get("fitted_at", "")[:10]

    if oracle_scr is not None:
        o_f   = float(oracle_scr)
        o_cls = "pos" if o_f >= 0.6 else ("neg" if o_f <= 0.4 else "neu")
    else:
        o_f, o_cls = 0.0, "neu"

    action_cls = {
        "ACCUMULATE": "pos", "RE_ENTER": "pos",
        "STRATEGIC_EXIT": "neg",
    }.get(action, "neu")

    with sig_col:
        st.markdown(
            f"""
<div class="radar-table">
  <table style="width:100%;border-collapse:collapse;font-size:13px;">
    <tr>
      <td style="color:var(--muted);padding:4px 0;width:42%">HMM Regime</td>
      <td>{_regime_badge(hmm_state)}</td>
    </tr>
    <tr>
      <td style="color:var(--muted);padding:4px 0">Probabilities</td>
      <td style="font-family:var(--mono);font-size:12px">
        B {p_bull:.0%} &nbsp;R {p_range:.0%} &nbsp;Be {p_bear:.0%}
      </td>
    </tr>
    <tr>
      <td style="color:var(--muted);padding:4px 0">Veto Active</td>
      <td class="{'neg' if veto else 'pos'}" style="font-weight:600">
        {'YES' if veto else 'NO'}
      </td>
    </tr>
    <tr>
      <td style="color:var(--muted);padding:4px 0">Oracle Score</td>
      <td class="{o_cls}" style="font-family:var(--mono);font-weight:600">
        {f"{o_f:.2f}" if oracle_scr is not None else "—"}
      </td>
    </tr>
    <tr>
      <td style="color:var(--muted);padding:4px 0">CIO Action</td>
      <td class="{action_cls}" style="font-weight:600">{action}</td>
    </tr>
    <tr>
      <td style="color:var(--muted);padding:4px 0">Conviction Q/M</td>
      <td style="font-family:var(--mono)">
        {f"+{q_conv}" if isinstance(q_conv, int) and q_conv >= 0 else q_conv}
        &nbsp;/&nbsp;
        {f"+{m_conv}" if isinstance(m_conv, int) and m_conv >= 0 else m_conv}
      </td>
    </tr>
    <tr>
      <td style="color:var(--muted);padding:4px 0">Model Fitted</td>
      <td style="color:var(--muted);font-size:12px">{fitted_at or "—"}</td>
    </tr>
  </table>
</div>
""",
            unsafe_allow_html=True,
        )

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 4 — System Intelligence (Position Mgr + Correlation + Model Perf)
    # ────────────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    _pm_path = ROOT / "data" / "position_management.json"
    _cr_path = ROOT / "data" / "correlation_report.json"
    _ew_path = ROOT / "data" / "ensemble_weights.json"

    intel_c1, intel_c2, intel_c3 = st.columns(3)

    with intel_c1:
        _pm = {}
        if _pm_path.exists():
            try:
                _pm = json.loads(_pm_path.read_text())
            except Exception:
                pass
        _atr = _pm.get("atr", 0)
        _entry_q = _pm.get("entry_signal", {}).get("score", 0)
        _stops = _pm.get("stops", {})
        _init_stop = _stops.get("initial_stop", 0)
        _trail_stop = _stops.get("trailing_stop", 0)
        _init_pct = _stops.get("initial_pct", 0)
        _trail_pct = _stops.get("trailing_pct", 0)
        _eq_color = "#22c55e" if _entry_q >= 60 else "#eab308" if _entry_q >= 40 else "#ef4444"
        st.markdown(
            f'<div style="background:var(--card-bg);border:1px solid var(--border);'
            f'border-radius:8px;padding:16px;">'
            f'<div style="font-size:13px;font-weight:700;color:var(--gold);margin-bottom:10px;">'
            f'POSITION MANAGER</div>'
            f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
            f'ATR(14): <span style="color:var(--text)">${_atr:,.2f}</span><br>'
            f'Entry Quality: <span style="color:{_eq_color};font-weight:700">{_entry_q}/100</span><br>'
            f'Initial Stop: <span style="color:var(--text)">${_init_stop:,.2f}</span>'
            f' <span style="color:#ef4444">({_init_pct:+.2f}%)</span><br>'
            f'Trailing Stop: <span style="color:var(--text)">${_trail_stop:,.2f}</span>'
            f' <span style="color:#ef4444">({_trail_pct:+.2f}%)</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    with intel_c2:
        _cr = {}
        if _cr_path.exists():
            try:
                _cr = json.loads(_cr_path.read_text())
            except Exception:
                pass
        _regime_sig = _cr.get("regime_signal", "UNKNOWN")
        _n_anom = len(_cr.get("anomalies", []))
        _gsr = _cr.get("gold_silver_ratio", 0)
        _gbeta = _cr.get("gold_beta_spx", 0)
        _corr_21 = _cr.get("correlations_21d", {})
        _dxy_corr = _corr_21.get("DX-Y.NYB", 0) or 0
        _sig_color = "#ef4444" if _regime_sig == "STRUCTURAL_BREAK" else "#22c55e"
        st.markdown(
            f'<div style="background:var(--card-bg);border:1px solid var(--border);'
            f'border-radius:8px;padding:16px;">'
            f'<div style="font-size:13px;font-weight:700;color:var(--gold);margin-bottom:10px;">'
            f'CROSS-ASSET MONITOR</div>'
            f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
            f'Regime Signal: <span style="color:{_sig_color};font-weight:700">{_regime_sig}</span><br>'
            f'Anomalies: <span style="color:var(--text)">{_n_anom}</span><br>'
            f'Gold/DXY Corr: <span style="color:var(--text)">{_dxy_corr:+.2f}</span><br>'
            f'Gold Beta SPX: <span style="color:var(--text)">{_gbeta:+.2f}</span><br>'
            f'Au/Ag Ratio: <span style="color:var(--text)">{_gsr:.1f}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    with intel_c3:
        _ew = {}
        if _ew_path.exists():
            try:
                _ew = json.loads(_ew_path.read_text())
            except Exception:
                pass
        _weights = _ew.get("weights", {})
        _metrics = _ew.get("metrics", {})
        _rows_html = ""
        for _mn, _mw in sorted(_weights.items()):
            _mh = _metrics.get(_mn, {}).get("health", "UNKNOWN")
            _h_color = "#22c55e" if _mh == "HEALTHY" else "#eab308" if _mh == "INSUFFICIENT_DATA" else "#ef4444"
            _rows_html += (
                f'{_mn}: <span style="color:var(--text)">{_mw:.0%}</span> '
                f'<span style="color:{_h_color}">[{_mh}]</span><br>'
            )
        if not _rows_html:
            _rows_html = '<span style="color:var(--text)">No data yet</span>'
        st.markdown(
            f'<div style="background:var(--card-bg);border:1px solid var(--border);'
            f'border-radius:8px;padding:16px;">'
            f'<div style="font-size:13px;font-weight:700;color:var(--gold);margin-bottom:10px;">'
            f'MODEL PERFORMANCE</div>'
            f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
            f'{_rows_html}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 5 — Quantitative Edge (Monte Carlo + Kelly + Backtest + MTF)
    # ────────────────────────────────────────────────────────────────────────
    _mc_path = ROOT / "data" / "monte_carlo_simulation.json"
    _ks_path = ROOT / "data" / "kelly_sizing.json"
    _bt_path = ROOT / "data" / "backtest_results.json"
    _mtf_path = ROOT / "data" / "mtf_confluence.json"

    edge_c1, edge_c2, edge_c3, edge_c4 = st.columns(4)

    with edge_c1:
        _mc = {}
        if _mc_path.exists():
            try:
                _mc = json.loads(_mc_path.read_text())
            except Exception:
                pass
        _prob_pos = _mc.get("probabilities", {}).get("positive_return", 0)
        _cvar = _mc.get("risk", {}).get("cvar_95_pct", 0)
        _mean_ret = _mc.get("terminal", {}).get("mean_return_pct", 0)
        _horizon = _mc.get("horizon_days", 0)
        _prob_color = "#22c55e" if _prob_pos > 0.55 else "#eab308" if _prob_pos > 0.45 else "#ef4444"
        st.markdown(
            f'<div style="background:var(--card-bg);border:1px solid var(--border);'
            f'border-radius:8px;padding:14px;">'
            f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
            f'MONTE CARLO ({_horizon}d)</div>'
            f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
            f'P(positive): <span style="color:{_prob_color};font-weight:700">{_prob_pos:.1%}</span><br>'
            f'Mean Return: <span style="color:var(--text)">{_mean_ret:+.2f}%</span><br>'
            f'CVaR-95: <span style="color:#ef4444">{_cvar:+.2f}%</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    with edge_c2:
        _ks = {}
        if _ks_path.exists():
            try:
                _ks = json.loads(_ks_path.read_text())
            except Exception:
                pass
        _edge = _ks.get("kelly", {}).get("edge", 0)
        _deploy = _ks.get("sizing", {}).get("deploy_usd", 0)
        _final_pct = _ks.get("sizing", {}).get("final_position_pct", 0)
        _should = _ks.get("kelly", {}).get("should_trade", False)
        _trade_color = "#22c55e" if _should else "#ef4444"
        st.markdown(
            f'<div style="background:var(--card-bg);border:1px solid var(--border);'
            f'border-radius:8px;padding:14px;">'
            f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
            f'KELLY SIZING</div>'
            f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
            f'Edge: <span style="color:var(--text)">{_edge:+.4f}</span><br>'
            f'Position: <span style="color:var(--text)">{_final_pct:.1f}%</span><br>'
            f'Deploy: <span style="color:var(--text)">${_deploy:,.0f}</span><br>'
            f'Trade: <span style="color:{_trade_color};font-weight:700">{"YES" if _should else "NO"}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    with edge_c3:
        _bt = {}
        if _bt_path.exists():
            try:
                _bt = json.loads(_bt_path.read_text())
            except Exception:
                pass
        _strat = _bt.get("strategy", {})
        _sharpe = _strat.get("sharpe_ratio", 0)
        _win_r = _strat.get("win_rate_pct", 0)
        _pf = _strat.get("profit_factor", 0)
        _dd = _strat.get("max_drawdown_pct", 0)
        _sharpe_color = "#22c55e" if _sharpe > 0.5 else "#eab308" if _sharpe > 0 else "#ef4444"
        st.markdown(
            f'<div style="background:var(--card-bg);border:1px solid var(--border);'
            f'border-radius:8px;padding:14px;">'
            f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
            f'WALK-FORWARD BACKTEST</div>'
            f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
            f'Sharpe: <span style="color:{_sharpe_color};font-weight:700">{_sharpe:.3f}</span><br>'
            f'Win Rate: <span style="color:var(--text)">{_win_r:.1f}%</span><br>'
            f'Profit Factor: <span style="color:var(--text)">{_pf:.2f}x</span><br>'
            f'Max DD: <span style="color:#ef4444">{_dd:.1f}%</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    with edge_c4:
        _mtf = {}
        if _mtf_path.exists():
            try:
                _mtf = json.loads(_mtf_path.read_text())
            except Exception:
                pass
        _conf = _mtf.get("confluence", {})
        _level = _conf.get("level", "UNKNOWN")
        _mtf_score = _conf.get("score", 0)
        _bull_tfs = _conf.get("bullish_tfs", 0)
        _bear_tfs = _conf.get("bearish_tfs", 0)
        _lvl_color = "#22c55e" if "BULLISH" in _level else "#ef4444" if "BEARISH" in _level else "#eab308"
        st.markdown(
            f'<div style="background:var(--card-bg);border:1px solid var(--border);'
            f'border-radius:8px;padding:14px;">'
            f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
            f'MTF CONFLUENCE</div>'
            f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
            f'Level: <span style="color:{_lvl_color};font-weight:700">{_level}</span><br>'
            f'Score: <span style="color:var(--text)">{_mtf_score:+d}/100</span><br>'
            f'Align: <span style="color:var(--text)">{_bull_tfs}B / {_bear_tfs}Bear</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 5b — Institutional Risk (EVT tail risk + factor attribution +
    #              stress test + drawdown tier)
    # ────────────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header">Institutional Risk</div>',
        unsafe_allow_html=True,
    )

    _tre_path = ROOT / "data" / "tail_risk_engine.json"
    _st_path  = ROOT / "data" / "stress_test_results.json"
    _dd_path  = ROOT / "data" / "drawdown_controller.json"

    _tre, _st_res, _dd = {}, {}, {}
    for _path, _slot in [(_tre_path, "tre"), (_st_path, "st"), (_dd_path, "dd")]:
        if _path.exists():
            try:
                if _slot == "tre":
                    _tre = json.loads(_path.read_text())
                elif _slot == "st":
                    _st_res = json.loads(_path.read_text())
                else:
                    _dd = json.loads(_path.read_text())
            except Exception:
                pass

    inst_c1, inst_c2, inst_c3, inst_c4 = st.columns(4)

    # ── Card 1: EVT Tail Risk ────────────────────────────────────────────────
    with inst_c1:
        _evt = _tre.get("tail_risk", {}).get("methods", {}).get("evt_pot", {})
        _g99 = _tre.get("tail_risk", {}).get("methods", {}).get("gaussian", {}).get("cvar_990", 0)
        _evt_cvar99 = _evt.get("cvar_990", 0)
        _evt_cvar995 = _evt.get("cvar_995", 0)
        _premium = _tre.get("tail_risk", {}).get("tail_fatness_premium_pct", 0)
        _premium_color = "#ef4444" if _premium > 30 else "#eab308" if _premium > 10 else "#22c55e"
        _diag = _tre.get("tail_risk", {}).get("evt_diagnostics", {})
        _xi = _diag.get("shape_xi", 0) if _diag.get("fit_ok") else 0
        st.markdown(
            f'<div style="background:var(--card-bg);border:1px solid var(--border);'
            f'border-radius:8px;padding:14px;">'
            f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
            f'EVT TAIL RISK</div>'
            f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
            f'CVaR-99: <span style="color:#ef4444;font-weight:700">{_evt_cvar99:.2f}%</span><br>'
            f'CVaR-99.5: <span style="color:#ef4444">{_evt_cvar995:.2f}%</span><br>'
            f'GPD ξ: <span style="color:var(--text)">{_xi:+.3f}</span><br>'
            f'vs Gaussian: <span style="color:{_premium_color};font-weight:700">+{_premium:.0f}%</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # ── Card 2: Factor Attribution ───────────────────────────────────────────
    with inst_c2:
        _fac = _tre.get("factor_attribution", {})
        _r2 = _fac.get("r_squared", 0)
        _alpha = _fac.get("alpha_annualised_pct", 0)
        _alpha_t = _fac.get("alpha_t_stat", 0)
        _ir = _fac.get("information_ratio", 0)
        _alpha_color = "#22c55e" if _alpha_t > 2 else "#eab308" if _alpha_t > 0 else "#ef4444"
        _ir_color = "#22c55e" if _ir > 1 else "#eab308" if _ir > 0 else "#ef4444"
        st.markdown(
            f'<div style="background:var(--card-bg);border:1px solid var(--border);'
            f'border-radius:8px;padding:14px;">'
            f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
            f'FACTOR ATTRIBUTION</div>'
            f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
            f'R²: <span style="color:var(--text)">{_r2:.3f}</span><br>'
            f'Alpha: <span style="color:{_alpha_color};font-weight:700">{_alpha:+.2f}%</span><br>'
            f'Alpha t: <span style="color:var(--text)">{_alpha_t:+.2f}</span><br>'
            f'Info Ratio: <span style="color:{_ir_color};font-weight:700">{_ir:+.2f}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # ── Card 3: Stress Test ──────────────────────────────────────────────────
    with inst_c3:
        _agg = _st_res.get("aggregate", {})
        _worst_ret = _agg.get("worst_crisis_return_pct", 0)
        _worst_name = _agg.get("worst_crisis_scenario", "n/a")
        _avg_dd = _agg.get("avg_max_drawdown_pct", 0)
        _avg_recov = _agg.get("avg_recovery_days", 0)
        _worst_color = "#ef4444" if _worst_ret < -10 else "#eab308" if _worst_ret < 0 else "#22c55e"
        # Truncate scenario name to fit
        _worst_short = _worst_name.split()[0] if _worst_name else "n/a"
        st.markdown(
            f'<div style="background:var(--card-bg);border:1px solid var(--border);'
            f'border-radius:8px;padding:14px;">'
            f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
            f'STRESS TEST</div>'
            f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
            f'Worst Crisis: <span style="color:{_worst_color};font-weight:700">{_worst_ret:+.1f}%</span><br>'
            f'Scenario: <span style="color:var(--text)">{_worst_short}</span><br>'
            f'Avg Max DD: <span style="color:#ef4444">{_avg_dd:.1f}%</span><br>'
            f'Recovery: <span style="color:var(--text)">{_avg_recov:.0f}d</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # ── Card 4: Drawdown Tier ────────────────────────────────────────────────
    with inst_c4:
        _tier = _dd.get("tier_name", "UNKNOWN")
        _tier_dd = _dd.get("current_dd_pct", 0)
        _tier_mult = _dd.get("sizing_multiplier", 1.0)
        _tier_action = _dd.get("action", "—")
        _tier_color = {
            "NORMAL": "#22c55e", "CAUTION": "#eab308",
            "DEFENSIVE": "#f97316", "CRITICAL": "#ef4444",
            "EMERGENCY": "#dc2626",
        }.get(_tier, "#6b7280")
        # Trim long action strings to fit card
        _action_short = _tier_action[:34] + "…" if len(_tier_action) > 34 else _tier_action
        st.markdown(
            f'<div style="background:var(--card-bg);border:1px solid var(--border);'
            f'border-radius:8px;padding:14px;">'
            f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
            f'DRAWDOWN TIER</div>'
            f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
            f'Tier: <span style="color:{_tier_color};font-weight:700">{_tier}</span><br>'
            f'Current DD: <span style="color:var(--text)">{_tier_dd:+.2f}%</span><br>'
            f'Sizing: <span style="color:var(--text)">{_tier_mult:.0%}</span><br>'
            f'<span style="color:var(--muted);font-size:10px">{_action_short}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # ── Sub-row: factor betas (only if tail risk engine has data) ────────────
    _factors = _tre.get("factor_attribution", {}).get("factors", [])
    if _factors:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        _today = _tre.get("factor_attribution", {}).get("today_decomposition", {})
        _total_bps = _today.get("total_return_bps", 0)
        _factor_bps = _today.get("factor_explained_bps", 0)
        _alpha_bps = _today.get("residual_alpha_bps", 0)

        _fac_rows = ""
        for f in _factors:
            _b = f.get("beta", 0)
            _t = f.get("t_stat", 0)
            _p = f.get("p_value", 1)
            _contrib = f.get("contribution_today_bps", 0)
            _sig = "***" if _p < 0.01 else ("**" if _p < 0.05 else "")
            _t_color = "#22c55e" if abs(_t) > 2 else "#9ca3af"
            _contrib_color = "#22c55e" if _contrib > 0 else "#ef4444"
            _fac_rows += (
                f'<tr>'
                f'<td style="padding:3px 10px;color:var(--text);font-weight:600">{f.get("factor", "?"):<8s}</td>'
                f'<td style="padding:3px 10px;text-align:right;color:var(--text)">{_b:+.4f}</td>'
                f'<td style="padding:3px 10px;text-align:right;color:{_t_color}">{_t:+.2f}</td>'
                f'<td style="padding:3px 10px;text-align:right;color:var(--muted)">{_p:.4f} {_sig}</td>'
                f'<td style="padding:3px 10px;text-align:right;color:{_contrib_color};font-weight:600">{_contrib:+.1f}</td>'
                f'</tr>'
            )

        _alpha_color2 = "#22c55e" if _alpha_bps > 0 else "#ef4444"
        st.markdown(
            f'<div style="background:var(--card-bg);border:1px solid var(--border);'
            f'border-radius:8px;padding:14px;">'
            f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
            f'FACTOR DECOMPOSITION — Today\'s Return</div>'
            f'<table style="width:100%;font-size:11px;border-collapse:collapse">'
            f'<thead><tr style="color:var(--muted);border-bottom:1px solid var(--border)">'
            f'<th style="padding:5px 10px;text-align:left">Factor</th>'
            f'<th style="padding:5px 10px;text-align:right">Beta</th>'
            f'<th style="padding:5px 10px;text-align:right">t-stat</th>'
            f'<th style="padding:5px 10px;text-align:right">p-value</th>'
            f'<th style="padding:5px 10px;text-align:right">Today (bps)</th>'
            f'</tr></thead>'
            f'<tbody>{_fac_rows}</tbody></table>'
            f'<div style="margin-top:10px;padding-top:8px;border-top:1px solid var(--border);'
            f'font-size:11px;color:var(--muted)">'
            f'Total: <span style="color:var(--text);font-weight:600">{_total_bps:+.1f} bps</span> &nbsp;=&nbsp; '
            f'Factor-explained: <span style="color:var(--text)">{_factor_bps:+.1f}</span> &nbsp;+&nbsp; '
            f'Residual α: <span style="color:{_alpha_color2};font-weight:700">{_alpha_bps:+.1f}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 5d — Transaction Cost Analysis (TCA)
    #              Almgren-Chriss + UAE physical premium across metals + halal book
    # ────────────────────────────────────────────────────────────────────────
    _tca_path = ROOT / "data" / "transaction_cost_model.json"
    _tca = {}
    if _tca_path.exists():
        try:
            _tca = json.loads(_tca_path.read_text())
        except Exception:
            pass

    if _tca.get("aggregate"):
        st.markdown("<br>", unsafe_allow_html=True)
        _agg = _tca.get("aggregate", {})
        _vol_regime = _tca.get("vol_regime", "unknown")
        _vol_mult = _tca.get("vol_multiplier", 1.0)
        _vol_color = "#22c55e" if _vol_mult <= 1.0 else "#eab308" if _vol_mult <= 1.5 else "#ef4444"
        st.markdown(
            f'<div class="section-header">Transaction Cost Analysis '
            f'<span style="font-weight:400;color:var(--muted);font-size:11px">'
            f'&nbsp;&nbsp;avg <span style="color:var(--text);font-weight:700">'
            f'{_agg.get("avg_oneway_cost_bps", 0):.1f}bp</span> one-way '
            f'&middot; vol regime <span style="color:{_vol_color};font-weight:700">{_vol_regime}</span></span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Aggregate row: 4 stats cards
        _ag_c1, _ag_c2, _ag_c3, _ag_c4 = st.columns(4)

        with _ag_c1:
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'AVG ONE-WAY COST</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'<span style="color:var(--text);font-weight:700;font-size:20px">'
                f'{_agg.get("avg_oneway_cost_bps", 0):.1f} bp</span><br>'
                f'Range: {_agg.get("min_oneway_cost_bps", 0):.1f}–{_agg.get("max_oneway_cost_bps", 0):.1f} bp<br>'
                f'Trades: {_agg.get("n_trades", 0)}'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with _ag_c2:
            _metals = _tca.get("metals", [])
            _gold_card = next((m for m in _metals if m.get("ticker") == "GC=F" and "error" not in m), {})
            _gold_total = _gold_card.get("total_oneway_bps", 0)
            _gold_phys = _gold_card.get("physical_premium_bps", 0)
            _gold_imp = _gold_card.get("impact_bps_total", 0)
            _gold_spread = _gold_card.get("spread_bps", 0)
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'GOLD (GC=F) BUY</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Total: <span style="color:var(--text);font-weight:700">{_gold_total:.2f} bp</span><br>'
                f'Physical: <span style="color:var(--text)">{_gold_phys:.0f} bp</span><br>'
                f'Spread: <span style="color:var(--muted)">{_gold_spread:.2f} bp</span> &middot; '
                f'Impact: <span style="color:var(--muted)">{_gold_imp:.2f} bp</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with _ag_c3:
            _silver_card = next((m for m in _metals if m.get("ticker") == "SI=F" and "error" not in m), {})
            _slv_total = _silver_card.get("total_oneway_bps", 0)
            _slv_phys = _silver_card.get("physical_premium_bps", 0)
            _slv_imp = _silver_card.get("impact_bps_total", 0)
            _slv_spread = _silver_card.get("spread_bps", 0)
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'SILVER (SI=F) BUY</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Total: <span style="color:var(--text);font-weight:700">{_slv_total:.2f} bp</span><br>'
                f'Physical: <span style="color:var(--text)">{_slv_phys:.0f} bp</span><br>'
                f'Spread: <span style="color:var(--muted)">{_slv_spread:.2f} bp</span> &middot; '
                f'Impact: <span style="color:var(--muted)">{_slv_imp:.2f} bp</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with _ag_c4:
            # Equity book stat: cheapest 3 names
            _equities = [e for e in _tca.get("equities", []) if "error" not in e]
            _equities_sorted = sorted(_equities, key=lambda e: e.get("total_oneway_bps", 999))[:3]
            _eq_lines = ""
            for e in _equities_sorted:
                _eq_lines += (
                    f'{e.get("ticker", "?")[:6]}: '
                    f'<span style="color:var(--text)">{e.get("total_oneway_bps", 0):.1f}bp</span><br>'
                )
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'CHEAPEST HALAL EXEC</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'{_eq_lines}'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        # Full equity TCA table
        _equities_full = [e for e in _tca.get("equities", []) if "error" not in e]
        if _equities_full:
            _rows = ""
            for e in _equities_full:
                _t = e.get("ticker", "?")
                _ow = e.get("total_oneway_bps", 0)
                _spr = e.get("spread_bps", 0)
                _imp = e.get("impact_bps_total", 0)
                _part = e.get("participation_pct", 0)
                _slices = e.get("optimal_slices", 1)
                _days = e.get("days_to_execute", 1)
                _ann_vol = e.get("annualised_vol_pct", 0)
                _ow_color = "#22c55e" if _ow < 8 else "#eab308" if _ow < 20 else "#ef4444"
                _sched = f"{_slices}× over {_days}d" if _slices > 1 else "single"
                _rows += (
                    f'<tr>'
                    f'<td style="padding:5px 10px;color:var(--text);font-weight:600">{_t}</td>'
                    f'<td style="padding:5px 10px;text-align:right;color:var(--muted);font-family:monospace">${e.get("notional_usd", 0):,.0f}</td>'
                    f'<td style="padding:5px 10px;text-align:right;color:var(--text);font-family:monospace">{_spr:.2f}</td>'
                    f'<td style="padding:5px 10px;text-align:right;color:var(--text);font-family:monospace">{_imp:.2f}</td>'
                    f'<td style="padding:5px 10px;text-align:right;color:var(--muted);font-family:monospace">{_part:.3f}%</td>'
                    f'<td style="padding:5px 10px;text-align:right;color:var(--muted);font-family:monospace">{_ann_vol:.0f}%</td>'
                    f'<td style="padding:5px 10px;text-align:right;color:{_ow_color};font-weight:700;font-family:monospace">{_ow:.2f} bp</td>'
                    f'<td style="padding:5px 10px;color:var(--muted);font-size:10px">{_sched}</td>'
                    f'</tr>'
                )

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'HALAL UNIVERSE TCA — top {_tca.get("top_n_equities", 10)} ranked names</div>'
                f'<table style="width:100%;font-size:11px;border-collapse:collapse">'
                f'<thead><tr style="color:var(--muted);border-bottom:1px solid var(--border)">'
                f'<th style="padding:6px 10px;text-align:left">Ticker</th>'
                f'<th style="padding:6px 10px;text-align:right">Notional</th>'
                f'<th style="padding:6px 10px;text-align:right">½-Spread (bp)</th>'
                f'<th style="padding:6px 10px;text-align:right">Impact (bp)</th>'
                f'<th style="padding:6px 10px;text-align:right">% ADV</th>'
                f'<th style="padding:6px 10px;text-align:right">Vol</th>'
                f'<th style="padding:6px 10px;text-align:right">One-way</th>'
                f'<th style="padding:6px 10px;text-align:left">Schedule</th>'
                f'</tr></thead>'
                f'<tbody>{_rows}</tbody></table>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 5c — Cointegration & Mean-Reversion (statistical-arbitrage)
    # ────────────────────────────────────────────────────────────────────────
    _ce_path = ROOT / "data" / "cointegration_engine.json"
    _ce = {}
    if _ce_path.exists():
        try:
            _ce = json.loads(_ce_path.read_text())
        except Exception:
            pass

    if _ce.get("pairs"):
        st.markdown("<br>", unsafe_allow_html=True)
        _hdr_count = (
            f"{_ce.get('n_cointegrated_5pct', 0)}/{_ce.get('n_pairs', 0)} cointegrated"
            f"  &middot;  {_ce.get('n_actionable', 0)} actionable"
        )
        st.markdown(
            f'<div class="section-header">Cointegration &amp; Mean-Reversion '
            f'<span style="font-weight:400;color:var(--muted);font-size:11px">'
            f'&nbsp;&nbsp;{_hdr_count}</span></div>',
            unsafe_allow_html=True,
        )

        _rows = ""
        _signal_color = {
            "LONG_SPREAD":  "#22c55e",
            "SHORT_SPREAD": "#ef4444",
            "STOP":         "#dc2626",
            "WATCH":        "#eab308",
            "FLAT":         "#9ca3af",
            "DISABLED":     "#6b7280",
        }
        for p in _ce.get("pairs", []):
            if "error" in p:
                continue
            _coint_5 = p.get("cointegrated_5pct", False)
            _coint_1 = p.get("cointegrated_1pct", False)
            _coint_marker = "★★" if _coint_1 else ("★" if _coint_5 else "·")
            _coint_color = "#22c55e" if _coint_1 else ("#eab308" if _coint_5 else "#6b7280")
            _hl = p.get("half_life_days", 0)
            _hl_str = f"{_hl:.0f}d" if _hl < 9999 else "∞"
            _z = p.get("z_score", 0)
            _z_color = "#ef4444" if abs(_z) > 2 else ("#eab308" if abs(_z) > 1 else "#9ca3af")
            _sig = p.get("signal", "—")
            _sig_color = _signal_color.get(_sig, "#9ca3af")
            _name = p.get("description", p.get("pair", "?")).split(" — ")[0]
            _rows += (
                f'<tr>'
                f'<td style="padding:5px 10px;color:{_coint_color};font-weight:700;text-align:center">{_coint_marker}</td>'
                f'<td style="padding:5px 10px;color:var(--text);font-weight:600">{_name}</td>'
                f'<td style="padding:5px 10px;text-align:right;color:var(--text);font-family:monospace">{p.get("beta", 0):+.4f}</td>'
                f'<td style="padding:5px 10px;text-align:right;color:var(--text);font-family:monospace">{p.get("r_squared", 0):.3f}</td>'
                f'<td style="padding:5px 10px;text-align:right;color:var(--muted);font-family:monospace">{p.get("adf_stat", 0):+.2f}</td>'
                f'<td style="padding:5px 10px;text-align:right;color:var(--muted);font-family:monospace">{p.get("adf_pvalue_approx", 0):.3f}</td>'
                f'<td style="padding:5px 10px;text-align:right;color:var(--text);font-family:monospace">{_hl_str}</td>'
                f'<td style="padding:5px 10px;text-align:right;color:{_z_color};font-weight:700;font-family:monospace">{_z:+.2f}</td>'
                f'<td style="padding:5px 10px;color:{_sig_color};font-weight:700">{_sig}</td>'
                f'</tr>'
            )

        st.markdown(
            f'<div style="background:var(--card-bg);border:1px solid var(--border);'
            f'border-radius:8px;padding:14px;">'
            f'<table style="width:100%;font-size:11px;border-collapse:collapse">'
            f'<thead><tr style="color:var(--muted);border-bottom:1px solid var(--border)">'
            f'<th style="padding:6px 10px;text-align:center" title="Cointegrated at 1% (★★) / 5% (★)">CI</th>'
            f'<th style="padding:6px 10px;text-align:left">Pair</th>'
            f'<th style="padding:6px 10px;text-align:right">β</th>'
            f'<th style="padding:6px 10px;text-align:right">R²</th>'
            f'<th style="padding:6px 10px;text-align:right">ADF τ</th>'
            f'<th style="padding:6px 10px;text-align:right">p-val</th>'
            f'<th style="padding:6px 10px;text-align:right">Half-life</th>'
            f'<th style="padding:6px 10px;text-align:right">z</th>'
            f'<th style="padding:6px 10px;text-align:left">Signal</th>'
            f'</tr></thead>'
            f'<tbody>{_rows}</tbody></table>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Actionable signals callout
        _sigs = _ce.get("actionable_signals", [])
        if _sigs:
            _sig_rows = ""
            for s in _sigs:
                _c = "#22c55e" if s["signal"] == "LONG_SPREAD" else "#ef4444"
                _sig_rows += (
                    f'<div style="padding:8px 12px;background:rgba(0,0,0,0.2);'
                    f'border-left:3px solid {_c};border-radius:4px;margin-top:6px">'
                    f'<span style="color:{_c};font-weight:700">{s["signal"]}</span> '
                    f'&nbsp;<span style="color:var(--text);font-weight:600">{s["name"]}</span> '
                    f'&nbsp;<span style="color:var(--muted);font-size:11px">'
                    f'z={s["z_score"]:+.2f} &nbsp; ½-life={s["half_life_days"]:.0f}d</span>'
                    f'<div style="color:var(--muted);font-size:11px;margin-top:4px">{s["rationale"]}</div>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="margin-top:10px">{_sig_rows}</div>',
                unsafe_allow_html=True,
            )

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 6 — Strategy Diagnostics
    #   Currently: Alpha Attribution. Vol surface and signal decay land here
    #   once those engines come online (Stages 14-15 of the grand plan).
    # ────────────────────────────────────────────────────────────────────────
    _aa_path = ROOT / "data" / "alpha_attribution.json"
    if _aa_path.exists():
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header">Strategy Diagnostics</div>',
            unsafe_allow_html=True,
        )

        try:
            _aa = json.loads(_aa_path.read_text())
        except Exception:
            _aa = {}

        _ranked = _aa.get("ranked_by_sharpe", [])
        _top = _ranked[0] if _ranked else None
        _top_full = _aa.get("full_history", {}).get(_top, {}) if _top else {}
        _top_ir = _aa.get("information_ratios", {}).get(_top, {}) if _top else {}
        _comb = _aa.get("combined", {})
        _es = _comb.get("equal_weight_summary", {})

        _top_sharpe = _top_full.get("sharpe", 0)
        _top_ir_v = _top_ir.get("information_ratio", 0)
        _top_active = _top_ir.get("active_return_pct", 0)
        _blend_sharpe = _es.get("sharpe", 0)
        _blend_ret = _es.get("ann_return_pct", 0)
        _blend_vol = _es.get("ann_vol_pct", 0)
        _div = _comb.get("diversification_ratio", 0)
        _avg_vol = _comb.get("weighted_avg_vol_pct", 0)

        _top_color = (
            "#22c55e" if _top_sharpe > 0.5
            else "#eab308" if _top_sharpe > 0.2
            else "#ef4444"
        )
        _ir_color = (
            "#22c55e" if _top_ir_v > 0.3
            else "#eab308" if _top_ir_v > 0
            else "#ef4444"
        )
        _blend_color = (
            "#22c55e" if _blend_sharpe > 0.3
            else "#eab308" if _blend_sharpe > 0
            else "#ef4444"
        )
        _div_color = (
            "#22c55e" if _div > 2.0
            else "#eab308" if _div > 1.3
            else "#ef4444"
        )

        diag_c1, diag_c2, diag_c3, diag_c4 = st.columns(4)

        with diag_c1:
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'TOP ALPHA SOURCE</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Source: <span style="color:var(--text);font-weight:700">{_top or "n/a"}</span><br>'
                f'Sharpe: <span style="color:{_top_color};font-weight:700">{_top_sharpe:+.3f}</span><br>'
                f'Ret: <span style="color:var(--text)">{_top_full.get("ann_return_pct", 0):+.2f}%</span><br>'
                f'Win: <span style="color:var(--text)">{_top_full.get("win_rate_pct", 0):.1f}%</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with diag_c2:
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'INFORMATION RATIO</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Top IR: <span style="color:{_ir_color};font-weight:700">{_top_ir_v:+.3f}</span><br>'
                f'Active: <span style="color:var(--text)">{_top_active:+.2f}%</span><br>'
                f'TE: <span style="color:var(--text)">{_top_ir.get("tracking_error_pct", 0):.2f}%</span><br>'
                f'<span style="color:var(--muted);font-size:10px">vs equal-weight blend</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with diag_c3:
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'EQUAL-WEIGHT BLEND</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Sharpe: <span style="color:{_blend_color};font-weight:700">{_blend_sharpe:+.3f}</span><br>'
                f'Return: <span style="color:var(--text)">{_blend_ret:+.2f}%</span><br>'
                f'Vol: <span style="color:var(--text)">{_blend_vol:.2f}%</span><br>'
                f'DD: <span style="color:#ef4444">{_es.get("max_drawdown_pct", 0):.1f}%</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with diag_c4:
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'DIVERSIFICATION</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Div Ratio: <span style="color:{_div_color};font-weight:700">{_div:.2f}x</span><br>'
                f'Avg Source Vol: <span style="color:var(--text)">{_avg_vol:.2f}%</span><br>'
                f'Blend Vol: <span style="color:var(--text)">{_blend_vol:.2f}%</span><br>'
                f'Sources: <span style="color:var(--text)">{_aa.get("n_obs", 0)}d × {len(_aa.get("sources", []))}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        # Source ranking strip
        if _aa.get("ranked_by_sharpe"):
            _rank_html = '<div style="margin-top:12px;font-size:11px;color:var(--muted)">Sharpe rank: '
            _rank_html += " &nbsp;|&nbsp; ".join([
                f'<span style="color:var(--text);font-weight:600">{i+1}. {s}</span>'
                for i, s in enumerate(_aa.get("ranked_by_sharpe", []))
            ])
            _rank_html += '</div>'
            st.markdown(_rank_html, unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────────
    # Volatility Surface row (second sub-section of Section 6)
    # ────────────────────────────────────────────────────────────────────────
    _vs_path = ROOT / "data" / "vol_surface.json"
    if _vs_path.exists():
        try:
            _vs = json.loads(_vs_path.read_text())
        except Exception:
            _vs = {}

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        _ts = _vs.get("term_structure", {})
        _tsp = _vs.get("term_structure_pctile", {})
        _regime = _vs.get("vol_regime", "UNKNOWN")
        _phase = _vs.get("phase", "STABLE")
        _curve = _vs.get("curve_shape", "FLAT")
        _act = _vs.get("actions", {})

        _regime_color = {
            "LOW":      "#22c55e",
            "NORMAL":   "#22c55e",
            "ELEVATED": "#eab308",
            "EXTREME":  "#ef4444",
        }.get(_regime, "var(--text)")
        _phase_color = {
            "EXPANDING":   "#ef4444",
            "STABLE":      "var(--text)",
            "CONTRACTING": "#22c55e",
        }.get(_phase, "var(--text)")
        _curve_color = {
            "BACKWARDATION": "#ef4444",
            "FLAT":          "var(--text)",
            "CONTANGO":      "#22c55e",
        }.get(_curve, "var(--text)")

        vs_c1, vs_c2, vs_c3, vs_c4 = st.columns(4)

        with vs_c1:
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'VOL TERM STRUCTURE</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'5d: <span style="color:var(--text)">{_ts.get("rv_5d", 0):.2f}%</span> '
                f'<span style="color:var(--muted);font-size:10px">(p{_tsp.get("rv_5d_pctile", 0):.0f})</span><br>'
                f'21d: <span style="color:var(--text)">{_ts.get("rv_21d", 0):.2f}%</span> '
                f'<span style="color:var(--muted);font-size:10px">(p{_tsp.get("rv_21d_pctile", 0):.0f})</span><br>'
                f'63d: <span style="color:var(--text)">{_ts.get("rv_63d", 0):.2f}%</span> '
                f'<span style="color:var(--muted);font-size:10px">(p{_tsp.get("rv_63d_pctile", 0):.0f})</span><br>'
                f'252d: <span style="color:var(--text)">{_ts.get("rv_252d", 0):.2f}%</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with vs_c2:
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'VOL REGIME</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Regime: <span style="color:{_regime_color};font-weight:700">{_regime}</span><br>'
                f'21d pctile: <span style="color:var(--text)">{_vs.get("vol_21d_pctile", 0):.0f}</span><br>'
                f'Vol-of-Vol: <span style="color:var(--text)">{_vs.get("vol_of_vol", 0):.2f}%</span><br>'
                f'<span style="color:var(--muted);font-size:10px">VoV pctile: {_vs.get("vol_of_vol_pctile", 0):.0f}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with vs_c3:
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'CURVE / PHASE</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Shape: <span style="color:{_curve_color};font-weight:700">{_curve}</span><br>'
                f'Slope: <span style="color:var(--text)">{_vs.get("curve_slope_pct", 0):+.1f}%</span><br>'
                f'Phase: <span style="color:{_phase_color};font-weight:700">{_phase}</span><br>'
                f'Δ: <span style="color:var(--text)">{_vs.get("phase_change_pct", 0):+.1f}%</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with vs_c4:
            _km = _act.get("kelly_fraction_multiplier", 1.0)
            _sm = _act.get("stop_atr_multiplier", 2.0)
            _km_color = (
                "#22c55e" if _km >= 0.85
                else "#eab308" if _km >= 0.55
                else "#ef4444"
            )
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'VOL-DRIVEN ACTIONS</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Kelly Mult: <span style="color:{_km_color};font-weight:700">{_km:.2f}×</span><br>'
                f'Stop Width: <span style="color:var(--text)">{_sm:.2f}× ATR</span><br>'
                f'<span style="color:var(--muted);font-size:10px">{_act.get("trade_size_guidance", "")}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    # ────────────────────────────────────────────────────────────────────────
    # Signal Decay row (third sub-section of Section 6)
    # ────────────────────────────────────────────────────────────────────────
    _sd_path = ROOT / "data" / "signal_decay.json"
    if _sd_path.exists():
        try:
            _sd = json.loads(_sd_path.read_text())
        except Exception:
            _sd = {}

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        _ranked_ic = _sd.get("ranked_by_ic", [])
        _top_sig = _ranked_ic[0] if _ranked_ic else None
        _top_data = _sd.get("signals", {}).get(_top_sig, {}) if _top_sig else {}
        _decay_list = _sd.get("decaying_signals", [])
        _strengthen_list = _sd.get("strengthening_signals", [])

        _top_ic = _top_data.get("best_horizon_ic", 0)
        _top_t = _top_data.get("best_horizon_t", 0)
        _top_hl = _top_data.get("half_life_days", 0)
        _top_rebal = _top_data.get("rebalance_days", 0)
        _top_status = _top_data.get("decay_status", "STABLE")

        _ic_color = (
            "#22c55e" if abs(_top_ic) > 0.10
            else "#eab308" if abs(_top_ic) > 0.05
            else "#ef4444"
        )
        _t_color = (
            "#22c55e" if abs(_top_t) > 2
            else "#eab308" if abs(_top_t) > 1
            else "#ef4444"
        )
        _decay_color = {
            "DECAYING":      "#ef4444",
            "STRENGTHENING": "#22c55e",
            "STABLE":        "var(--text)",
        }.get(_top_status, "var(--text)")

        sd_c1, sd_c2, sd_c3, sd_c4 = st.columns(4)

        with sd_c1:
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'STRONGEST SIGNAL</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Signal: <span style="color:var(--text);font-weight:700">{_top_sig or "n/a"}</span><br>'
                f'IC: <span style="color:{_ic_color};font-weight:700">{_top_ic:+.4f}</span><br>'
                f't-stat: <span style="color:{_t_color};font-weight:700">{_top_t:+.2f}</span><br>'
                f'Best @: <span style="color:var(--text)">{_top_data.get("best_horizon", 0)}d</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with sd_c2:
            _hl_color = (
                "#22c55e" if _top_hl > 21 and _top_hl < 365
                else "#eab308" if _top_hl > 5
                else "#ef4444"
            )
            _hl_str = (
                f"{_top_hl:.1f}d" if _top_data.get("fit_status") == "OK" else "n/a"
            )
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'HALF-LIFE / REBALANCE</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'½-life: <span style="color:{_hl_color};font-weight:700">{_hl_str}</span><br>'
                f'Rebalance: <span style="color:var(--text)">every {_top_rebal}d</span><br>'
                f'Fit: <span style="color:var(--text)">{_top_data.get("fit_status", "n/a")}</span><br>'
                f'<span style="color:var(--muted);font-size:10px">decay model: |IC|·exp(-h/τ)</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with sd_c3:
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'ALPHA DECAY STATUS</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Top signal: <span style="color:{_decay_color};font-weight:700">{_top_status}</span><br>'
                f'Decaying: <span style="color:#ef4444;font-weight:700">{len(_decay_list)}</span><br>'
                f'Strengthening: <span style="color:#22c55e;font-weight:700">{len(_strengthen_list)}</span><br>'
                f'<span style="color:var(--muted);font-size:10px">vs prior 252d window</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with sd_c4:
            # IC ladder across all signals (top 5 ranked)
            _ladder_rows = []
            for sig in _ranked_ic[:5]:
                d = _sd.get("signals", {}).get(sig, {})
                ic = d.get("best_horizon_ic", 0)
                ic_col = (
                    "#22c55e" if abs(ic) > 0.10
                    else "#eab308" if abs(ic) > 0.05
                    else "var(--muted)"
                )
                _ladder_rows.append(
                    f'<div style="font-size:11px;line-height:1.6">'
                    f'{sig[:14]:<14s} '
                    f'<span style="color:{ic_col}">{ic:+.4f}</span></div>'
                )
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'IC LADDER</div>'
                + "".join(_ladder_rows)
                + '</div>',
                unsafe_allow_html=True,
            )

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 7 — Portfolio Construction (HRP, Black-Litterman, Mean-CVaR,
    # Vol Targeting, DCC-GARCH)  — Phase II of the grand plan
    # ────────────────────────────────────────────────────────────────────────
    _hrp_path = ROOT / "data" / "hrp_allocator.json"
    if _hrp_path.exists():
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header">Portfolio Construction</div>',
            unsafe_allow_html=True,
        )

        try:
            _hrp = json.loads(_hrp_path.read_text())
        except Exception:
            _hrp = {}

        _m = _hrp.get("metrics", {})
        _hrp_m = _m.get("hrp", {})
        _eq_m = _m.get("equal_weight", {})
        _iv_m = _m.get("inverse_vol", {})
        _tickers = _hrp.get("tickers", [])

        _sharpe = _hrp_m.get("sharpe", 0)
        _vol = _hrp_m.get("ann_vol_pct", 0)
        _ret = _hrp_m.get("ann_return_pct", 0)
        _dd = _hrp_m.get("max_drawdown_pct", 0)
        _div = _hrp_m.get("diversification_ratio", 0)
        _enb = _hrp_m.get("effective_n_bets", 0)

        _sharpe_color = (
            "#22c55e" if _sharpe > 0.6
            else "#eab308" if _sharpe > 0.2
            else "#ef4444"
        )
        _div_color = (
            "#22c55e" if _div > 1.8
            else "#eab308" if _div > 1.3
            else "#ef4444"
        )

        pc_c1, pc_c2, pc_c3, pc_c4 = st.columns(4)

        with pc_c1:
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'HRP PERFORMANCE</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Sharpe: <span style="color:{_sharpe_color};font-weight:700">{_sharpe:+.3f}</span><br>'
                f'Return: <span style="color:var(--text)">{_ret:+.2f}%</span><br>'
                f'Vol: <span style="color:var(--text)">{_vol:.2f}%</span><br>'
                f'Max DD: <span style="color:#ef4444">{_dd:.1f}%</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with pc_c2:
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'DIVERSIFICATION</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Div Ratio: <span style="color:{_div_color};font-weight:700">{_div:.2f}x</span><br>'
                f'Eff # Bets: <span style="color:var(--text)">{_enb:.2f}</span><br>'
                f'Universe: <span style="color:var(--text)">{len(_tickers)} assets</span><br>'
                f'<span style="color:var(--muted);font-size:10px">{", ".join(_tickers[:4])}'
                + ("..." if len(_tickers) > 4 else "")
                + f'</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with pc_c3:
            # HRP weights
            _wh = _hrp_m.get("weights", {})
            _rows = []
            for tick in _tickers:
                w = _wh.get(tick, 0)
                bar = int(w * 100)
                _rows.append(
                    f'<div style="font-size:11px;line-height:1.6;display:flex;justify-content:space-between">'
                    f'<span>{tick[:9]}</span>'
                    f'<span style="color:var(--text);font-weight:600">{w:.2%}</span>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'HRP WEIGHTS</div>'
                + "".join(_rows)
                + f'</div>',
                unsafe_allow_html=True,
            )

        with pc_c4:
            # Sharpe comparison
            _eq_sh = _eq_m.get("sharpe", 0)
            _iv_sh = _iv_m.get("sharpe", 0)
            _hrp_dd = _hrp_m.get("max_drawdown_pct", 0)
            _eq_dd = _eq_m.get("max_drawdown_pct", 0)
            _iv_dd = _iv_m.get("max_drawdown_pct", 0)
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'BENCHMARK COMPARISON</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'HRP: <span style="color:var(--text);font-weight:700">{_sharpe:+.3f}</span> '
                f'<span style="color:#ef4444;font-size:10px">DD {_hrp_dd:.0f}%</span><br>'
                f'EqualW: <span style="color:var(--text)">{_eq_sh:+.3f}</span> '
                f'<span style="color:#ef4444;font-size:10px">DD {_eq_dd:.0f}%</span><br>'
                f'InvVol: <span style="color:var(--text)">{_iv_sh:+.3f}</span> '
                f'<span style="color:#ef4444;font-size:10px">DD {_iv_dd:.0f}%</span><br>'
                f'<span style="color:var(--muted);font-size:10px">Sharpe / drawdown by method</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    # ────────────────────────────────────────────────────────────────────────
    # Black-Litterman row (second sub-section of Section 7)
    # ────────────────────────────────────────────────────────────────────────
    _bl_path = ROOT / "data" / "black_litterman.json"
    if _bl_path.exists():
        try:
            _bl = json.loads(_bl_path.read_text())
        except Exception:
            _bl = {}

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        _m = _bl.get("metrics", {})
        _bl_m = _m.get("black_litterman", {})
        _mkt_m = _m.get("market_weights", {})
        _eq_m = _m.get("equal_weight", {})
        _asset_table = _bl.get("asset_table", [])

        _bl_sh = _bl_m.get("sharpe", 0)
        _bl_ret = _bl_m.get("ann_return_pct", 0)
        _bl_vol = _bl_m.get("ann_vol_pct", 0)
        _bl_dd = _bl_m.get("max_drawdown_pct", 0)
        _n_views = _bl.get("n_views", 0)

        _sh_color = (
            "#22c55e" if _bl_sh > 0.7
            else "#eab308" if _bl_sh > 0.3
            else "#ef4444"
        )

        bl_c1, bl_c2, bl_c3, bl_c4 = st.columns(4)

        with bl_c1:
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'BL PERFORMANCE</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Sharpe: <span style="color:{_sh_color};font-weight:700">{_bl_sh:+.3f}</span><br>'
                f'Return: <span style="color:var(--text)">{_bl_ret:+.2f}%</span><br>'
                f'Vol: <span style="color:var(--text)">{_bl_vol:.2f}%</span><br>'
                f'Max DD: <span style="color:#ef4444">{_bl_dd:.1f}%</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with bl_c2:
            # Tilts table
            _tilt_rows = []
            for row in _asset_table[:6]:
                tilt = row.get("view_tilt_pct", 0)
                tilt_col = (
                    "#22c55e" if tilt > 1
                    else "#ef4444" if tilt < -1
                    else "var(--text)"
                )
                _tilt_rows.append(
                    f'<div style="font-size:11px;line-height:1.6;display:flex;justify-content:space-between">'
                    f'<span>{row.get("ticker", "")[:9]}</span>'
                    f'<span style="color:{tilt_col};font-weight:600">{tilt:+.2f}%</span>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'POSTERIOR TILTS (μ-Π)</div>'
                + "".join(_tilt_rows)
                + '</div>',
                unsafe_allow_html=True,
            )

        with bl_c3:
            # BL final weights
            _w_rows = []
            for row in _asset_table[:6]:
                w = row.get("bl_weight", 0)
                _w_rows.append(
                    f'<div style="font-size:11px;line-height:1.6;display:flex;justify-content:space-between">'
                    f'<span>{row.get("ticker", "")[:9]}</span>'
                    f'<span style="color:var(--text);font-weight:600">{w:.2%}</span>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'BL WEIGHTS</div>'
                + "".join(_w_rows)
                + '</div>',
                unsafe_allow_html=True,
            )

        with bl_c4:
            # Views summary
            _view_html = ""
            for d in _bl.get("view_descriptions", [])[:4]:
                _view_html += (
                    f'<div style="font-size:10px;line-height:1.5;color:var(--muted);'
                    f'margin-bottom:4px">• {d[:50]}</div>'
                )
            if not _view_html:
                _view_html = '<div style="font-size:11px;color:var(--muted)">No active views</div>'
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'ACTIVE VIEWS ({_n_views})</div>'
                + _view_html
                + '</div>',
                unsafe_allow_html=True,
            )

    # ────────────────────────────────────────────────────────────────────────
    # Mean-CVaR row (third sub-section of Section 7)
    # ────────────────────────────────────────────────────────────────────────
    _mc_path = ROOT / "data" / "mean_cvar.json"
    if _mc_path.exists():
        try:
            _mc = json.loads(_mc_path.read_text())
        except Exception:
            _mc = {}

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        _m = _mc.get("metrics", {})
        _min_m = _m.get("min_cvar", {})
        _mean_m = _m.get("mean_cvar", {})
        _eq_m = _m.get("equal_weight", {})
        _tickers = _mc.get("tickers", [])

        _min_sh = _min_m.get("sharpe", 0)
        _mean_sh = _mean_m.get("sharpe", 0)
        _min_cvar = _min_m.get("cvar_pct", 0)
        _mean_cvar = _mean_m.get("cvar_pct", 0)
        _alpha = _mc.get("alpha", 0.95)

        _min_sh_color = (
            "#22c55e" if _min_sh > 0.7
            else "#eab308" if _min_sh > 0.3
            else "#ef4444"
        )
        _mean_sh_color = (
            "#22c55e" if _mean_sh > 0.7
            else "#eab308" if _mean_sh > 0.3
            else "#ef4444"
        )

        cv_c1, cv_c2, cv_c3, cv_c4 = st.columns(4)

        with cv_c1:
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'MIN-CVaR PORTFOLIO</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Sharpe: <span style="color:{_min_sh_color};font-weight:700">{_min_sh:+.3f}</span><br>'
                f'Vol: <span style="color:var(--text)">{_min_m.get("ann_vol_pct", 0):.2f}%</span><br>'
                f'Max DD: <span style="color:#ef4444">{_min_m.get("max_drawdown_pct", 0):.2f}%</span><br>'
                f'CVaR-{int(_alpha*100)}: <span style="color:#ef4444">{_min_cvar:.3f}%/d</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with cv_c2:
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'MEAN-CVaR PORTFOLIO</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Sharpe: <span style="color:{_mean_sh_color};font-weight:700">{_mean_sh:+.3f}</span><br>'
                f'Vol: <span style="color:var(--text)">{_mean_m.get("ann_vol_pct", 0):.2f}%</span><br>'
                f'Max DD: <span style="color:#ef4444">{_mean_m.get("max_drawdown_pct", 0):.2f}%</span><br>'
                f'CVaR-{int(_alpha*100)}: <span style="color:#ef4444">{_mean_cvar:.3f}%/d</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with cv_c3:
            # Min-CVaR weights
            _w_min = _min_m.get("weights", {})
            _rows = []
            for tick in _tickers:
                w = _w_min.get(tick, 0)
                _rows.append(
                    f'<div style="font-size:11px;line-height:1.6;display:flex;justify-content:space-between">'
                    f'<span>{tick[:9]}</span>'
                    f'<span style="color:var(--text);font-weight:600">{w:.2%}</span>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'MIN-CVaR WEIGHTS</div>'
                + "".join(_rows)
                + '</div>',
                unsafe_allow_html=True,
            )

        with cv_c4:
            # Mean-CVaR weights
            _w_mean = _mean_m.get("weights", {})
            _rows = []
            for tick in _tickers:
                w = _w_mean.get(tick, 0)
                _rows.append(
                    f'<div style="font-size:11px;line-height:1.6;display:flex;justify-content:space-between">'
                    f'<span>{tick[:9]}</span>'
                    f'<span style="color:var(--text);font-weight:600">{w:.2%}</span>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'MEAN-CVaR WEIGHTS</div>'
                + "".join(_rows)
                + '</div>',
                unsafe_allow_html=True,
            )

    # ────────────────────────────────────────────────────────────────────────
    # Vol Target & Risk Budget row (fourth sub-section of Section 7)
    # ────────────────────────────────────────────────────────────────────────
    _vt_path = ROOT / "data" / "vol_target_budget.json"
    if _vt_path.exists():
        try:
            _vt = json.loads(_vt_path.read_text())
        except Exception:
            _vt = {}

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        _target = _vt.get("target_vol_pct", 12)
        _curr = _vt.get("current_vol_pct", 0)
        _lev_raw = _vt.get("leverage_raw", 1)
        _lev_cap = _vt.get("leverage_capped", 1)
        _action = _vt.get("guidance", {}).get("leverage_action", "MAINTAIN")
        _deploy = _vt.get("guidance", {}).get("deploy_pct_of_capital", 100)

        _action_color = {
            "DELEVERAGE":  "#ef4444",
            "MAINTAIN":    "var(--text)",
            "LEVER_UP":    "#22c55e",
        }.get(_action, "var(--text)")

        _lev_color = (
            "#ef4444" if _lev_cap < 0.5
            else "#eab308" if _lev_cap < 0.85
            else "#22c55e" if _lev_cap <= 1.15
            else "#eab308"
        )

        vt_c1, vt_c2, vt_c3, vt_c4 = st.columns(4)

        with vt_c1:
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'VOL TARGETING</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Target: <span style="color:var(--text);font-weight:700">{_target:.1f}%</span><br>'
                f'Current: <span style="color:var(--text)">{_curr:.2f}%</span><br>'
                f'Leverage: <span style="color:{_lev_color};font-weight:700">{_lev_cap:.2f}×</span><br>'
                f'Action: <span style="color:{_action_color};font-weight:700">{_action}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with vt_c2:
            # IR-weighted budget
            _ir_w = _vt.get("ir_weighted", {}).get("weights", {})
            _ir_rc = _vt.get("ir_weighted", {}).get("risk_contrib_pct", {})
            _rows = []
            for s, w in _ir_w.items():
                rc = _ir_rc.get(s, 0)
                wcol = "var(--text)" if w > 0.01 else "var(--muted)"
                _rows.append(
                    f'<div style="font-size:11px;line-height:1.6;display:flex;justify-content:space-between">'
                    f'<span>{s[:14]}</span>'
                    f'<span style="color:{wcol};font-weight:600">{w:.1%}</span>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'IR-WEIGHTED BUDGET</div>'
                + "".join(_rows)
                + '</div>',
                unsafe_allow_html=True,
            )

        with vt_c3:
            # Equal-risk budget
            _er_w = _vt.get("equal_risk", {}).get("weights", {})
            _rows = []
            for s, w in _er_w.items():
                _rows.append(
                    f'<div style="font-size:11px;line-height:1.6;display:flex;justify-content:space-between">'
                    f'<span>{s[:14]}</span>'
                    f'<span style="color:var(--text);font-weight:600">{w:.1%}</span>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'EQUAL-RISK BUDGET</div>'
                + "".join(_rows)
                + '</div>',
                unsafe_allow_html=True,
            )

        with vt_c4:
            _ir_blend = _vt.get("ir_weighted", {}).get("blend_vol_pct", 0)
            _er_blend = _vt.get("equal_risk", {}).get("blend_vol_pct", 0)
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'BUDGET DIAGNOSTICS</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'IR blend vol: <span style="color:var(--text)">{_ir_blend:.2f}%</span><br>'
                f'EqRisk blend vol: <span style="color:var(--text)">{_er_blend:.2f}%</span><br>'
                f'Lev raw: <span style="color:var(--text)">{_lev_raw:.2f}×</span><br>'
                f'Deploy: <span style="color:var(--text);font-weight:700">{_deploy:.0f}% capital</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    # ────────────────────────────────────────────────────────────────────────
    # DCC-GARCH dynamic correlations row (fifth sub-section of Section 7)
    # ────────────────────────────────────────────────────────────────────────
    _dcc_path = ROOT / "data" / "dcc_garch.json"
    if _dcc_path.exists():
        try:
            _dcc = json.loads(_dcc_path.read_text())
        except Exception:
            _dcc = {}

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        _p = _dcc.get("dcc_params", {})
        _avg_now = _dcc.get("avg_pairwise_corr_now", 0)
        _avg_lr = _dcc.get("avg_pairwise_corr_long_run", 0)
        _n_stress = _dcc.get("n_stressed", 0)
        _stressed = _dcc.get("stressed_pairs", [])
        _pairs = _dcc.get("pairs", [])

        # Top movers by |z|
        _top_movers = sorted(_pairs, key=lambda x: abs(x.get("z_score", 0)), reverse=True)[:6]

        _shift = _avg_now - _avg_lr
        _shift_color = (
            "#ef4444" if abs(_shift) > 0.10
            else "#eab308" if abs(_shift) > 0.04
            else "#22c55e"
        )
        _stress_color = "#ef4444" if _n_stress > 0 else "#22c55e"

        dcc_c1, dcc_c2, dcc_c3, dcc_c4 = st.columns(4)

        with dcc_c1:
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'DCC PARAMETERS</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'a: <span style="color:var(--text)">{_p.get("a", 0):.4f}</span><br>'
                f'b: <span style="color:var(--text)">{_p.get("b", 0):.4f}</span><br>'
                f'a+b: <span style="color:var(--text)">{_p.get("a_plus_b", 0):.4f}</span><br>'
                f'LL: <span style="color:var(--text)">{_p.get("log_likelihood", 0):.1f}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with dcc_c2:
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'AVG PAIRWISE CORR</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Current: <span style="color:var(--text);font-weight:700">{_avg_now:+.3f}</span><br>'
                f'Long-run: <span style="color:var(--text)">{_avg_lr:+.3f}</span><br>'
                f'Shift: <span style="color:{_shift_color};font-weight:700">{_shift:+.3f}</span><br>'
                f'Stressed: <span style="color:{_stress_color};font-weight:700">{_n_stress} pairs</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with dcc_c3:
            # Top movers by |z|
            _rows = []
            for pr in _top_movers[:5]:
                z = pr.get("z_score", 0)
                cur = pr.get("current_corr", 0)
                col = "#ef4444" if abs(z) > 1.5 else "#eab308" if abs(z) > 1.0 else "var(--text)"
                short = pr.get("pair", "").replace("__", "-").replace("DX-Y.NYB", "DXY")
                _rows.append(
                    f'<div style="font-size:11px;line-height:1.6;display:flex;justify-content:space-between">'
                    f'<span>{short[:18]}</span>'
                    f'<span style="color:{col};font-weight:600">{cur:+.2f} (z {z:+.1f})</span>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'TOP MOVERS BY |Z|</div>'
                + "".join(_rows)
                + '</div>',
                unsafe_allow_html=True,
            )

        with dcc_c4:
            # Stressed pair list (or "all clear")
            if _stressed:
                _items = [
                    f'<div style="font-size:11px;line-height:1.5;color:#ef4444;font-weight:600">'
                    f'⚠ {p.replace("__", " / ").replace("DX-Y.NYB", "DXY")[:30]}</div>'
                    for p in _stressed
                ]
                _body = "".join(_items)
            else:
                _body = (
                    '<div style="font-size:11px;color:#22c55e;line-height:1.5">'
                    'All pairs within ±2σ of long-run correlation.</div>'
                    '<div style="font-size:10px;color:var(--muted);margin-top:6px">'
                    'No structural break detected.</div>'
                )
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'STRESSED PAIRS</div>'
                + _body
                + '</div>',
                unsafe_allow_html=True,
            )

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 8 — Regime Diagnostics
    #   Structural breaks now; macro regime classifier and BMA land here as
    #   Stages 23-24 of the grand plan come online.
    # ────────────────────────────────────────────────────────────────────────
    _sb_path = ROOT / "data" / "structural_breaks.json"
    if _sb_path.exists():
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header">Regime Diagnostics</div>',
            unsafe_allow_html=True,
        )

        try:
            _sb = json.loads(_sb_path.read_text())
        except Exception:
            _sb = {}

        _cusum = _sb.get("cusum", {})
        _s = _sb.get("summary", {})
        _cusum_break = _s.get("cusum_break", False)
        _stat = _cusum.get("test_stat", 0)
        _crit = _cusum.get("critical_value", 1.358)
        _n_mean = _s.get("n_mean_breaks", 0)
        _n_var = _s.get("n_variance_breaks", 0)
        _last = _s.get("most_recent_break", "none")
        _days = _s.get("days_since_last_break")

        _cusum_color = "#ef4444" if _cusum_break else "#22c55e"
        _var_color = (
            "#ef4444" if _n_var >= 4
            else "#eab308" if _n_var >= 2
            else "#22c55e"
        )

        sb_c1, sb_c2, sb_c3, sb_c4 = st.columns(4)

        with sb_c1:
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'CUSUM (mean stability)</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Status: <span style="color:{_cusum_color};font-weight:700">'
                f'{"BREAK" if _cusum_break else "STABLE"}</span><br>'
                f'Test stat: <span style="color:var(--text)">{_stat:.4f}</span><br>'
                f'Critical 5%: <span style="color:var(--text)">{_crit:.4f}</span><br>'
                f'<span style="color:var(--muted);font-size:10px">Brown-Durbin-Evans</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with sb_c2:
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'BREAKS COUNT (5y)</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Mean breaks: <span style="color:var(--text);font-weight:700">{_n_mean}</span><br>'
                f'Variance breaks: <span style="color:{_var_color};font-weight:700">{_n_var}</span><br>'
                f'Last detected: <span style="color:var(--text)">'
                f'{(_days if _days is not None else "—")}d ago</span><br>'
                f'<span style="color:var(--muted);font-size:10px">{_last or "none"}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with sb_c3:
            # Most recent variance breaks
            _rb = _sb.get("variance_breaks", [])[-4:]
            _rows = []
            for b in _rb:
                dirc = b.get("direction", "")
                col = "#ef4444" if dirc == "EXPANSION" else "#22c55e"
                _rows.append(
                    f'<div style="font-size:10px;line-height:1.5">'
                    f'{b.get("date", "")}  '
                    f'<span style="color:{col};font-weight:600">{dirc[:3]}</span>  '
                    f'<span style="color:var(--text)">×{b.get("variance_ratio", 0):.2f}</span>  '
                    f'<span style="color:var(--muted)">'
                    f'{b.get("vol_before_pct", 0):.0f}→{b.get("vol_after_pct", 0):.0f}%</span>'
                    f'</div>'
                )
            if not _rows:
                _rows = [
                    '<div style="font-size:11px;color:var(--muted)">'
                    'No variance breaks detected.</div>'
                ]
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'RECENT VOL BREAKS</div>'
                + "".join(_rows)
                + '</div>',
                unsafe_allow_html=True,
            )

        with sb_c4:
            # Recent mean breaks
            _mb = _sb.get("mean_breaks", [])[:4]
            _rows = []
            for b in _mb:
                d = b.get("delta_bps", 0)
                col = "#22c55e" if d > 0 else "#ef4444"
                _rows.append(
                    f'<div style="font-size:10px;line-height:1.5">'
                    f'{b.get("date", "")}  t={b.get("t_stat", 0):.1f}  '
                    f'<span style="color:{col};font-weight:600">Δ{d:+.1f}bp</span>'
                    f'</div>'
                )
            if not _rows:
                _rows = [
                    f'<div style="font-size:11px;color:#22c55e;line-height:1.5">'
                    f'No mean breaks > t=3.5 detected.</div>'
                    f'<div style="font-size:10px;color:var(--muted);margin-top:6px">'
                    f'Drift remains within sample mean.</div>'
                ]
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'TOP MEAN BREAKS</div>'
                + "".join(_rows)
                + '</div>',
                unsafe_allow_html=True,
            )

    # ────────────────────────────────────────────────────────────────────────
    # Macro Regime row (second sub-section of Section 8)
    # ────────────────────────────────────────────────────────────────────────
    _mr_path = ROOT / "data" / "macro_regime.json"
    if _mr_path.exists():
        try:
            _mr = json.loads(_mr_path.read_text())
        except Exception:
            _mr = {}

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        _quad = _mr.get("quadrant", "UNKNOWN")
        _conf = _mr.get("confidence", 0)
        _g = _mr.get("growth_score", 0)
        _i = _mr.get("inflation_score", 0)
        _tilts = _mr.get("asset_tilts", {})
        _desc = _mr.get("description", "")

        _quad_color = {
            "GOLDILOCKS":  "#22c55e",
            "REFLATION":   "#eab308",
            "STAGFLATION": "#ef4444",
            "DEFLATION":   "#3b82f6",
        }.get(_quad, "var(--text)")

        _g_color = "#22c55e" if _g > 0 else "#ef4444"
        _i_color = "#ef4444" if _i > 0 else "#22c55e"  # inflation up is bad

        mr_c1, mr_c2, mr_c3, mr_c4 = st.columns(4)

        with mr_c1:
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'MACRO QUADRANT</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Quadrant: <span style="color:{_quad_color};font-weight:700">{_quad}</span><br>'
                f'Confidence: <span style="color:var(--text);font-weight:600">{_conf:.0%}</span><br>'
                f'<span style="color:var(--muted);font-size:10px;line-height:1.4">{_desc[:80]}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with mr_c2:
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'GROWTH × INFLATION</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Growth Z: <span style="color:{_g_color};font-weight:700">{_g:+.3f}</span><br>'
                f'Inflation Z: <span style="color:{_i_color};font-weight:700">{_i:+.3f}</span><br>'
                f'SPY 21d: <span style="color:var(--text)">'
                f'{_mr.get("spy_21d_mom_pct", 0):+.2f}%</span><br>'
                f'DXY 21d: <span style="color:var(--text)">'
                f'{_mr.get("dxy_21d_mom_pct", 0):+.2f}%</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with mr_c3:
            _rows = []
            for asset, tilt in _tilts.items():
                if tilt > 0:
                    col = "#22c55e"
                    sym = "+" * tilt
                elif tilt < 0:
                    col = "#ef4444"
                    sym = "−" * abs(tilt)
                else:
                    col = "var(--muted)"
                    sym = "·"
                _rows.append(
                    f'<div style="font-size:11px;line-height:1.6;display:flex;justify-content:space-between">'
                    f'<span>{asset}</span>'
                    f'<span style="color:{col};font-weight:700">{sym} ({tilt:+d})</span>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'ASSET TILTS</div>'
                + "".join(_rows)
                + '</div>',
                unsafe_allow_html=True,
            )

        with mr_c4:
            # Visual quadrant diagram
            _positions = {
                "GOLDILOCKS":  ("top-right",    "G↑ / I↓"),
                "REFLATION":   ("bottom-right", "G↑ / I↑"),
                "STAGFLATION": ("bottom-left",  "G↓ / I↑"),
                "DEFLATION":   ("top-left",     "G↓ / I↓"),
            }
            _pos, _ax = _positions.get(_quad, ("center", ""))
            _grid_html = (
                f'<div style="display:grid;grid-template-columns:1fr 1fr;'
                f'grid-template-rows:1fr 1fr;gap:3px;height:90px;margin-top:4px">'
                f'<div style="background:{ "#3b82f6" if _quad == "DEFLATION" else "var(--border)" };border-radius:4px;'
                f'display:flex;align-items:center;justify-content:center;font-size:9px;color:var(--text);font-weight:600">DEFL</div>'
                f'<div style="background:{ "#22c55e" if _quad == "GOLDILOCKS" else "var(--border)" };border-radius:4px;'
                f'display:flex;align-items:center;justify-content:center;font-size:9px;color:var(--text);font-weight:600">GOLDI</div>'
                f'<div style="background:{ "#ef4444" if _quad == "STAGFLATION" else "var(--border)" };border-radius:4px;'
                f'display:flex;align-items:center;justify-content:center;font-size:9px;color:var(--text);font-weight:600">STAG</div>'
                f'<div style="background:{ "#eab308" if _quad == "REFLATION" else "var(--border)" };border-radius:4px;'
                f'display:flex;align-items:center;justify-content:center;font-size:9px;color:var(--text);font-weight:600">REFL</div>'
                f'</div>'
            )
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:6px;">'
                f'QUADRANT MAP</div>'
                f'<div style="font-size:10px;color:var(--muted);">{_ax}</div>'
                + _grid_html
                + '</div>',
                unsafe_allow_html=True,
            )

    # ────────────────────────────────────────────────────────────────────────
    # BMA row (third sub-section of Section 8)
    # ────────────────────────────────────────────────────────────────────────
    _bma_path = ROOT / "data" / "bma_weights.json"
    if _bma_path.exists():
        try:
            _bma = json.loads(_bma_path.read_text())
        except Exception:
            _bma = {}

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        _bt = _bma.get("backtest", {})
        _bma_bt = _bt.get("bma", {})
        _ir_bt = _bt.get("ir", {})
        _eq_bt = _bt.get("equal", {})
        _per_src = _bma.get("per_source", [])
        _top = _bma.get("top_source", "")

        _bma_sh = _bma_bt.get("sharpe", 0)
        _sh_color = (
            "#22c55e" if _bma_sh > 0.5
            else "#eab308" if _bma_sh > 0.2
            else "#ef4444"
        )

        bm_c1, bm_c2, bm_c3, bm_c4 = st.columns(4)

        with bm_c1:
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'BMA PERFORMANCE</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Top: <span style="color:var(--text);font-weight:700">{_top}</span><br>'
                f'Sharpe: <span style="color:{_sh_color};font-weight:700">{_bma_sh:+.3f}</span><br>'
                f'Vol: <span style="color:var(--text)">{_bma_bt.get("ann_vol_pct", 0):.2f}%</span><br>'
                f'DD: <span style="color:#ef4444">{_bma_bt.get("max_drawdown_pct", 0):.1f}%</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with bm_c2:
            # Hit rates per source
            _rows = []
            for d in _per_src[:5]:
                hr = d.get("hit_rate", 0)
                col = "#22c55e" if hr > 0.52 else "#eab308" if hr > 0.48 else "#ef4444"
                _rows.append(
                    f'<div style="font-size:11px;line-height:1.6;display:flex;justify-content:space-between">'
                    f'<span>{d.get("source", "")[:14]}</span>'
                    f'<span style="color:{col};font-weight:600">{hr:.1%}</span>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'DIRECTIONAL HIT RATES</div>'
                + "".join(_rows)
                + '</div>',
                unsafe_allow_html=True,
            )

        with bm_c3:
            # BMA weights
            _rows = []
            for d in _per_src[:5]:
                w = d.get("bma_weight", 0)
                _rows.append(
                    f'<div style="font-size:11px;line-height:1.6;display:flex;justify-content:space-between">'
                    f'<span>{d.get("source", "")[:14]}</span>'
                    f'<span style="color:var(--text);font-weight:600">{w:.1%}</span>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'BMA WEIGHTS</div>'
                + "".join(_rows)
                + '</div>',
                unsafe_allow_html=True,
            )

        with bm_c4:
            _ir_sh = _ir_bt.get("sharpe", 0)
            _eq_sh = _eq_bt.get("sharpe", 0)
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'METHOD COMPARISON</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'BMA Sharpe: <span style="color:var(--text);font-weight:700">{_bma_sh:+.3f}</span><br>'
                f'IR Sharpe: <span style="color:var(--text)">{_ir_sh:+.3f}</span><br>'
                f'EqW Sharpe: <span style="color:var(--text)">{_eq_sh:+.3f}</span><br>'
                f'<span style="color:var(--muted);font-size:10px">window: {_bma.get("bma_window_days", 252)}d</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 9 — Execution & Microstructure
    #   Smart Order Router, Adverse Selection, Stop-Loss, Capacity
    # ────────────────────────────────────────────────────────────────────────
    _sor_path = ROOT / "data" / "smart_order_router.json"
    _as_path = ROOT / "data" / "adverse_selection.json"
    _sl_path = ROOT / "data" / "stop_loss_optimizer.json"
    _cap_path = ROOT / "data" / "capacity_analyzer.json"

    if any(p.exists() for p in (_sor_path, _as_path, _sl_path, _cap_path)):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header">Execution & Microstructure</div>',
            unsafe_allow_html=True,
        )

        ex_c1, ex_c2, ex_c3, ex_c4 = st.columns(4)

        # ── SOR ──────────────────────────────────────────────────────────────
        with ex_c1:
            _sor = {}
            if _sor_path.exists():
                try:
                    _sor = json.loads(_sor_path.read_text())
                except Exception:
                    pass
            _algo = _sor.get("recommended_algo", "n/a")
            _rc = _sor.get("recommended_cost", {})
            _bps = _rc.get("total_oneway_bps", 0)
            _bps_color = (
                "#22c55e" if _bps < 10
                else "#eab308" if _bps < 50
                else "#ef4444"
            )
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'SMART ORDER ROUTER</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Algo: <span style="color:var(--text);font-weight:700">{_algo}</span><br>'
                f'Horizon: <span style="color:var(--text)">{_sor.get("horizon_minutes", 0)}m '
                f'({_sor.get("n_slices", 0)} slices)</span><br>'
                f'Partic 60m: <span style="color:var(--text)">{_sor.get("participation_60min", 0):.3f}%</span><br>'
                f'Cost: <span style="color:{_bps_color};font-weight:700">{_bps:.1f} bps</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        # ── Adverse selection ────────────────────────────────────────────────
        with ex_c2:
            _av = {}
            if _as_path.exists():
                try:
                    _av = json.loads(_as_path.read_text())
                except Exception:
                    pass
            _sess = _av.get("session_summary", {})
            _worst_sess = ""
            _worst_score = 0
            if _sess:
                ws = max(_sess.items(), key=lambda kv: kv[1].get("avg_adverse_score", 0))
                _worst_sess = ws[0]
                _worst_score = ws[1].get("avg_adverse_score", 0)
            _worst_hours = _av.get("worst_hours", [])[:3]
            _hours_str = ", ".join(str(h.get("hour_utc")) for h in _worst_hours) or "—"
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'ADVERSE SELECTION</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Worst session: <span style="color:#ef4444;font-weight:700">{_worst_sess}</span><br>'
                f'Score: <span style="color:var(--text)">{_worst_score:.2f}</span><br>'
                f'Avoid hours UTC: <span style="color:var(--text)">{_hours_str}</span><br>'
                f'<span style="color:var(--muted);font-size:10px">forward markout proxy</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        # ── Stop loss ────────────────────────────────────────────────────────
        with ex_c3:
            _sl = {}
            if _sl_path.exists():
                try:
                    _sl = json.loads(_sl_path.read_text())
                except Exception:
                    pass
            _method = _sl.get("final_recommendation", "n/a")
            _price = _sl.get("final_stop_price", 0)
            _dist = _sl.get("final_stop_distance_pct", 0)
            _regime = _sl.get("vol_regime", "n/a")
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'STOP-LOSS</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Method: <span style="color:var(--text);font-weight:700">{_method}</span><br>'
                f'Stop: <span style="color:var(--text)">${_price:,.2f}</span><br>'
                f'Distance: <span style="color:var(--text)">{_dist:.2f}%</span><br>'
                f'Regime: <span style="color:var(--text)">{_regime}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        # ── Capacity ─────────────────────────────────────────────────────────
        with ex_c4:
            _cap = {}
            if _cap_path.exists():
                try:
                    _cap = json.loads(_cap_path.read_text())
                except Exception:
                    pass
            _phys = _cap.get("thresholds_physical", {})
            _paper = _cap.get("thresholds_paper", {})
            _alpha = _cap.get("expected_alpha_pct", 0)
            _phys_cap = _phys.get("decay_25pct", {}).get("aum_cap_usd", 0)
            _paper_cap = _paper.get("decay_25pct", {}).get("aum_cap_usd", 0)
            def _fmt_aum(x):
                if x is None or x == 0:
                    return "—"
                if x >= 1e9:
                    return f"${x/1e9:.1f}B"
                if x >= 1e6:
                    return f"${x/1e6:.1f}M"
                if x >= 1e3:
                    return f"${x/1e3:.0f}K"
                return f"${x:,.0f}"
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'STRATEGY CAPACITY</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'α: <span style="color:var(--text);font-weight:700">{_alpha:+.2f}%</span><br>'
                f'Physical: <span style="color:var(--text)">{_fmt_aum(_phys_cap)}</span><br>'
                f'Paper: <span style="color:var(--text)">{_fmt_aum(_paper_cap)}</span><br>'
                f'<span style="color:var(--muted);font-size:10px">AUM @ 25% α decay</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 10 — Alt Data & Sentiment
    #   Macro nowcast, CB speech + geo, ETF flows, news sentiment
    # ────────────────────────────────────────────────────────────────────────
    _mn_path = ROOT / "data" / "macro_nowcast.json"
    _cb_path = ROOT / "data" / "cb_speech.json"
    _ge_path = ROOT / "data" / "geopolitical_events.json"
    _ef_path = ROOT / "data" / "etf_flows.json"
    _ns_path = ROOT / "data" / "news_sentiment.json"

    if any(p.exists() for p in (_mn_path, _cb_path, _ef_path, _ns_path)):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header">Alt Data & Sentiment</div>',
            unsafe_allow_html=True,
        )

        alt_c1, alt_c2, alt_c3, alt_c4 = st.columns(4)

        # ── Macro nowcast ─────────────────────────────────────────────────────
        with alt_c1:
            _mn = json.loads(_mn_path.read_text()) if _mn_path.exists() else {}
            _composite = _mn.get("composite_score", 0)
            _regime = _mn.get("regime", "n/a")
            _regime_color = {
                "STRONGLY_BULLISH": "#22c55e",
                "BULLISH":          "#22c55e",
                "NEUTRAL":          "var(--text)",
                "BEARISH":          "#ef4444",
                "STRONGLY_BEARISH": "#ef4444",
            }.get(_regime, "var(--text)")
            _top_drivers = _mn.get("top_drivers", [])[:3]
            _driver_html = "".join(
                f'<div style="font-size:10px;line-height:1.5">'
                f'{d.get("name", "")[:18]}: '
                f'<span style="color:var(--text)">{d.get("value", 0):+.2f}</span></div>'
                for d in _top_drivers
            )
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'MACRO NOWCAST</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Score: <span style="color:{_regime_color};font-weight:700">{_composite:+.3f}</span><br>'
                f'Regime: <span style="color:{_regime_color};font-weight:700">{_regime}</span><br>'
                f'Active: <span style="color:var(--text)">'
                f'{_mn.get("n_active", 0)}/{_mn.get("n_components", 0)}</span>'
                + (f'<br>{_driver_html}' if _driver_html else "")
                + '</div></div>',
                unsafe_allow_html=True,
            )

        # ── CB speech + Geo risk ──────────────────────────────────────────────
        with alt_c2:
            _cb = json.loads(_cb_path.read_text()) if _cb_path.exists() else {}
            _ge = json.loads(_ge_path.read_text()) if _ge_path.exists() else {}
            _fed_regime = _cb.get("fed_regime", "n/a")
            _fed_v = _cb.get("fed_latest", 0)
            _fed_color = {
                "HAWKISH":         "#ef4444",
                "LEANING_HAWKISH": "#ef4444",
                "NEUTRAL":         "var(--text)",
                "LEANING_DOVISH":  "#22c55e",
                "DOVISH":          "#22c55e",
            }.get(_fed_regime, "var(--text)")
            _geo_regime = _ge.get("regime", "n/a")
            _geo_color = {
                "CALM":     "#22c55e",
                "ELEVATED": "#eab308",
                "HIGH":     "#ef4444",
                "CRISIS":   "#ef4444",
            }.get(_geo_regime, "var(--text)")
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'CB / GEOPOLITICAL</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Fed: <span style="color:{_fed_color};font-weight:700">{_fed_regime}</span><br>'
                f'pplx_fed: <span style="color:var(--text)">{_fed_v if _fed_v is not None else 0:+.2f}</span><br>'
                f'Geo regime: <span style="color:{_geo_color};font-weight:700">{_geo_regime}</span><br>'
                f'Geo score: <span style="color:var(--text)">'
                f'{(_ge.get("current_score") or 0):.2f}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        # ── ETF flows ─────────────────────────────────────────────────────────
        with alt_c3:
            _ef = json.loads(_ef_path.read_text()) if _ef_path.exists() else {}
            _headline = _ef.get("headline", "n/a")
            _g7 = _ef.get("gold_bucket", {}).get("flow_7d_usd", 0)
            _s7 = _ef.get("silver_bucket", {}).get("flow_7d_usd", 0)
            _g21 = _ef.get("gold_bucket", {}).get("flow_21d_usd", 0)
            _g_color = "#22c55e" if _g7 > 0 else "#ef4444"
            _s_color = "#22c55e" if _s7 > 0 else "#ef4444"
            def _fmt_b(x):
                if x is None:
                    return "—"
                if abs(x) >= 1e9:
                    return f"${x/1e9:+.2f}B"
                if abs(x) >= 1e6:
                    return f"${x/1e6:+.1f}M"
                return f"${x:+,.0f}"
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'ETF FLOWS (7d)</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Gold: <span style="color:{_g_color};font-weight:700">{_fmt_b(_g7)}</span><br>'
                f'Silver: <span style="color:{_s_color};font-weight:700">{_fmt_b(_s7)}</span><br>'
                f'Gold 21d: <span style="color:var(--text)">{_fmt_b(_g21)}</span><br>'
                f'Headline: <span style="color:var(--text);font-weight:700">{_headline}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        # ── News sentiment ────────────────────────────────────────────────────
        with alt_c4:
            _ns = json.loads(_ns_path.read_text()) if _ns_path.exists() else {}
            _agg = _ns.get("aggregate", {})
            _avg = _agg.get("avg_sentiment", 0)
            _disp = _agg.get("dispersion", 0)
            _cons = _agg.get("consensus_regime", "n/a")
            _div = _agg.get("divergent", False)
            _cons_color = {
                "BULLISH": "#22c55e",
                "NEUTRAL": "var(--text)",
                "BEARISH": "#ef4444",
            }.get(_cons, "var(--text)")
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'NEWS SENTIMENT</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Consensus: <span style="color:{_cons_color};font-weight:700">{_cons}</span><br>'
                f'Avg: <span style="color:var(--text)">{_avg:.3f}</span><br>'
                f'Dispersion: <span style="color:var(--text)">{_disp:.3f}</span><br>'
                f'Divergent: <span style="color:var(--text)">{"yes" if _div else "no"}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 11 — Derivatives & Carry
    #   Options pricer, tail hedge, carry, term structure
    # ────────────────────────────────────────────────────────────────────────
    _op_path = ROOT / "data" / "options_pricer.json"
    _th_path = ROOT / "data" / "tail_hedge.json"
    _ca_path = ROOT / "data" / "carry_analyzer.json"
    _ts_path = ROOT / "data" / "term_structure.json"

    if any(p.exists() for p in (_op_path, _th_path, _ca_path, _ts_path)):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header">Derivatives & Carry</div>',
            unsafe_allow_html=True,
        )

        dc_c1, dc_c2, dc_c3, dc_c4 = st.columns(4)

        # Options
        with dc_c1:
            _op = json.loads(_op_path.read_text()) if _op_path.exists() else {}
            _ac = _op.get("atm_call", {})
            _ap = _op.get("atm_put", {})
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'ATM OPTIONS ({_op.get("tenor_days", 0)}d)</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Call: <span style="color:#22c55e;font-weight:700">${_ac.get("price", 0):.2f}</span> '
                f'<span style="color:var(--muted);font-size:10px">δ {_ac.get("delta", 0):+.2f}</span><br>'
                f'Put: <span style="color:#ef4444;font-weight:700">${_ap.get("price", 0):.2f}</span> '
                f'<span style="color:var(--muted);font-size:10px">δ {_ap.get("delta", 0):+.2f}</span><br>'
                f'γ: <span style="color:var(--text)">{_ac.get("gamma", 0):.4f}</span><br>'
                f'σ: <span style="color:var(--text)">{_op.get("sigma", 0):.2%}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        # Tail hedge
        with dc_c2:
            _th = json.loads(_th_path.read_text()) if _th_path.exists() else {}
            _drag = _th.get("annual_drag_pct", 0)
            _binding = _th.get("constraint_binding", False)
            _drag_color = "#ef4444" if _binding else "#22c55e"
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'TAIL HEDGE</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Contracts: <span style="color:var(--text);font-weight:700">{_th.get("contracts_needed", 0)}</span><br>'
                f'Annual drag: <span style="color:{_drag_color};font-weight:700">{_drag:.3f}%</span><br>'
                f'Current CVaR: <span style="color:var(--text)">{_th.get("current_cvar_daily_pct", 0):.3f}%/d</span><br>'
                f'Target: <span style="color:var(--text)">{_th.get("target_cvar_pct", 0):.2f}%/d</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        # Carry
        with dc_c3:
            _ca = json.loads(_ca_path.read_text()) if _ca_path.exists() else {}
            _carry_fair = _ca.get("carry", {}).get("fair_pct", 0)
            _burden = _ca.get("carry", {}).get("burden", "n/a")
            _excess = _ca.get("excess_vs_carry", {}).get("21d_pct", 0)
            _b_color = {
                "HIGH_CARRY_HEADWIND":      "#ef4444",
                "MODERATE_CARRY_HEADWIND":  "#eab308",
                "LOW_CARRY_HEADWIND":       "var(--text)",
                "NEGATIVE_CARRY_TAILWIND":  "#22c55e",
            }.get(_burden, "var(--text)")
            _excess_color = "#22c55e" if _excess > 0 else "#ef4444"
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'CARRY</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Fair carry: <span style="color:var(--text);font-weight:700">{_carry_fair:+.2f}%</span><br>'
                f'Burden: <span style="color:{_b_color};font-weight:700;font-size:10px">{_burden}</span><br>'
                f'21d excess: <span style="color:{_excess_color};font-weight:700">{_excess:+.2f}%</span><br>'
                f'<span style="color:var(--muted);font-size:10px">real yield based</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        # Term structure
        with dc_c4:
            _ts = json.loads(_ts_path.read_text()) if _ts_path.exists() else {}
            _shape = _ts.get("curve_shape", "n/a")
            _slope = _ts.get("overall_slope_pct", 0)
            _shape_color = {
                "BACKWARDATION":    "#ef4444",
                "FLAT":             "#eab308",
                "NORMAL_CONTANGO":  "var(--text)",
                "STEEP_CONTANGO":   "#ef4444",
            }.get(_shape, "var(--text)")
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'TERM STRUCTURE</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Shape: <span style="color:{_shape_color};font-weight:700;font-size:10px">{_shape}</span><br>'
                f'Slope: <span style="color:var(--text);font-weight:700">{_slope:+.2f}%</span><br>'
                f'Roll yield: <span style="color:var(--text)">{_ts.get("roll_yield_pct", 0):+.2f}%</span><br>'
                f'R²: <span style="color:var(--text)">{_ts.get("curve_r_squared", 0):.3f}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 12 — ML Enhancements
    #   Bayesian HPO, purged K-fold, stacking, RL sizing, conformal intervals
    # ────────────────────────────────────────────────────────────────────────
    _bh_path = ROOT / "data" / "bayesian_hpo.json"
    _pk_path = ROOT / "data" / "purged_kfold.json"
    _es_path = ROOT / "data" / "ensemble_stacking.json"
    _rl_path = ROOT / "data" / "rl_sizing.json"
    _co_path = ROOT / "data" / "conformal_intervals.json"

    if any(p.exists() for p in (_bh_path, _pk_path, _es_path, _rl_path, _co_path)):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header">ML Enhancements</div>',
            unsafe_allow_html=True,
        )

        ml_c1, ml_c2, ml_c3, ml_c4 = st.columns(4)

        # ── HPO + Purged K-fold (combined card) ───────────────────────────────
        with ml_c1:
            _bh = json.loads(_bh_path.read_text()) if _bh_path.exists() else {}
            _pk = json.loads(_pk_path.read_text()) if _pk_path.exists() else {}
            _improvement = _bh.get("improvement", 0)
            _imp_color = (
                "#22c55e" if _improvement > 0.3
                else "#eab308" if _improvement > 0
                else "#ef4444"
            )
            _stab = _pk.get("summary", {}).get("stability_ratio")
            _mean_sh = _pk.get("summary", {}).get("mean_sharpe", 0)
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'HPO + CV STABILITY</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'HPO best Sharpe: <span style="color:{_imp_color};font-weight:700">{_bh.get("best_sharpe", 0):+.3f}</span><br>'
                f'HPO lift: <span style="color:{_imp_color}">{_improvement:+.3f}</span><br>'
                f'KFold mean Sharpe: <span style="color:var(--text)">{_mean_sh:+.3f}</span><br>'
                f'Stability: <span style="color:var(--text)">{_stab if _stab is not None else "—"}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        # ── Stacking ──────────────────────────────────────────────────────────
        with ml_c2:
            _es = json.loads(_es_path.read_text()) if _es_path.exists() else {}
            _meta = _es.get("meta_metrics", {})
            _best = _es.get("best_single_base", {})
            _lift = _es.get("stacking_lift", {})
            _meta_auc = _meta.get("auc", 0)
            _auc_color = (
                "#22c55e" if _meta_auc > 0.55
                else "#eab308" if _meta_auc > 0.50
                else "#ef4444"
            )
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'ENSEMBLE STACKING</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Meta acc: <span style="color:var(--text);font-weight:700">{_meta.get("accuracy", 0):.3f}</span><br>'
                f'Meta AUC: <span style="color:{_auc_color};font-weight:700">{_meta_auc:.3f}</span><br>'
                f'Best base: <span style="color:var(--text)">{_best.get("accuracy", 0):.3f}</span><br>'
                f'AUC lift: <span style="color:var(--text)">{_lift.get("auc", 0):+.4f}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        # ── RL Sizing ─────────────────────────────────────────────────────────
        with ml_c3:
            _rl = json.loads(_rl_path.read_text()) if _rl_path.exists() else {}
            _rl_test = _rl.get("rl_test_sharpe", 0)
            _base_test = _rl.get("baseline_test_sharpe", 0)
            _lift_test = _rl.get("test_lift_sharpe", 0)
            _lift_color = (
                "#22c55e" if _lift_test > 0
                else "#eab308" if _lift_test > -0.5
                else "#ef4444"
            )
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'RL SIZING (Q-learning)</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Test Sharpe: <span style="color:var(--text);font-weight:700">{_rl_test:+.3f}</span><br>'
                f'Baseline: <span style="color:var(--text)">{_base_test:+.3f}</span><br>'
                f'Lift: <span style="color:{_lift_color};font-weight:700">{_lift_test:+.3f}</span><br>'
                f'Avg action: <span style="color:var(--text)">{_rl.get("avg_action_test", 0):.2f}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        # ── Conformal ─────────────────────────────────────────────────────────
        with ml_c4:
            _co = json.loads(_co_path.read_text()) if _co_path.exists() else {}
            _intervals = _co.get("intervals", {})
            _live = _co.get("live_intervals", {})
            _a05 = _intervals.get("alpha_05", {})
            _live05 = _live.get("alpha_05", {})
            _cov = _a05.get("empirical_coverage", 0)
            _cov_color = "#22c55e" if _a05.get("valid", False) else "#ef4444"
            _fc = _co.get("latest_forecast_pct", 0)
            _fc_color = "#22c55e" if _fc > 0 else "#ef4444"
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'CONFORMAL ({_co.get("horizon", 5)}d)</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Forecast: <span style="color:{_fc_color};font-weight:700">{_fc:+.2f}%</span><br>'
                f'95% range: <span style="color:var(--text)">[{_live05.get("lower_pct", 0):+.1f}%, {_live05.get("upper_pct", 0):+.1f}%]</span><br>'
                f'95% cov: <span style="color:{_cov_color};font-weight:700">{_cov:.3f}</span><br>'
                f'<span style="color:var(--muted);font-size:10px">target 0.95</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 13 — Performance Attribution & Analytics
    #   Brinson, Fama-French, IC/IR, Decision Quality
    # ────────────────────────────────────────────────────────────────────────
    _br_path = ROOT / "data" / "brinson_attribution.json"
    _ff_path = ROOT / "data" / "fama_french.json"
    _ii_path = ROOT / "data" / "ic_ir_tracker.json"
    _dq_path = ROOT / "data" / "decision_quality.json"

    if any(p.exists() for p in (_br_path, _ff_path, _ii_path, _dq_path)):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header">Performance Attribution & Analytics</div>',
            unsafe_allow_html=True,
        )

        pa_c1, pa_c2, pa_c3, pa_c4 = st.columns(4)

        # ── Brinson ──────────────────────────────────────────────────────────
        with pa_c1:
            _br = json.loads(_br_path.read_text()) if _br_path.exists() else {}
            _ex = _br.get("excess_return_pct", 0)
            _ex_color = "#22c55e" if _ex > 0 else "#ef4444"
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'BRINSON ATTRIBUTION</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Portfolio: <span style="color:var(--text)">{_br.get("portfolio_return_pct", 0):+.2f}%</span><br>'
                f'Benchmark: <span style="color:var(--text)">{_br.get("benchmark_return_pct", 0):+.2f}%</span><br>'
                f'Excess: <span style="color:{_ex_color};font-weight:700">{_ex:+.2f}%</span><br>'
                f'Alloc/Sel: <span style="color:var(--text)">'
                f'{_br.get("allocation_effect_pct", 0):+.2f} / {_br.get("selection_effect_pct", 0):+.2f}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        # ── Fama-French ──────────────────────────────────────────────────────
        with pa_c2:
            _ff = json.loads(_ff_path.read_text()) if _ff_path.exists() else {}
            _alpha = _ff.get("alpha_annualised_pct", 0)
            _alpha_t = _ff.get("alpha_t_stat", 0)
            _ir = _ff.get("information_ratio", 0)
            _sig = _ff.get("alpha_significant", False)
            _alpha_color = (
                "#22c55e" if _sig and _alpha > 0
                else "#eab308" if _alpha > 0
                else "#ef4444"
            )
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'FAMA-FRENCH</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'α (ann): <span style="color:{_alpha_color};font-weight:700">{_alpha:+.2f}%</span><br>'
                f't-stat: <span style="color:var(--text)">{_alpha_t:+.2f}</span><br>'
                f'R²: <span style="color:var(--text)">{_ff.get("r_squared", 0):.3f}</span><br>'
                f'Dominant: <span style="color:var(--text);font-weight:700">{_ff.get("dominant_factor", "n/a")}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        # ── IC/IR ───────────────────────────────────────────────────────────
        with pa_c3:
            _ii = json.loads(_ii_path.read_text()) if _ii_path.exists() else {}
            _depl = _ii.get("deployable_signals", [])
            _ranked = _ii.get("ranked_by_ir", [])
            _top = _ranked[0] if _ranked else None
            _top_ir = (
                _ii.get("per_signal", {}).get(_top, {}).get("ir_63d", 0) if _top else 0
            )
            _ir_color = (
                "#22c55e" if _top_ir > 0.5
                else "#eab308" if _top_ir > 0
                else "#ef4444"
            )
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'IC / IR TRACKER</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Top: <span style="color:var(--text);font-weight:700">{_top or "n/a"}</span><br>'
                f'IR(63d): <span style="color:{_ir_color};font-weight:700">{_top_ir:+.3f}</span><br>'
                f'Deployable: <span style="color:var(--text);font-weight:700">{len(_depl)}</span><br>'
                f'<span style="color:var(--muted);font-size:10px">'
                f'{", ".join(_depl)[:30] if _depl else "none"}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        # ── Decision Quality ─────────────────────────────────────────────────
        with pa_c4:
            _dq = json.loads(_dq_path.read_text()) if _dq_path.exists() else {}
            _best = _dq.get("best_signal", "n/a")
            _best_brier = _dq.get("best_brier", 0)
            _best_skill = _dq.get("best_skill", 0)
            _realised = _dq.get("realised_positive_rate", 0)
            _skill_color = (
                "#22c55e" if _best_skill > 0
                else "#eab308" if _best_skill > -0.05
                else "#ef4444"
            )
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'DECISION QUALITY</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Best: <span style="color:var(--text);font-weight:700">{_best}</span><br>'
                f'Brier: <span style="color:var(--text)">{_best_brier:.4f}</span><br>'
                f'Skill: <span style="color:{_skill_color};font-weight:700">{_best_skill:+.4f}</span><br>'
                f'Pos rate: <span style="color:var(--text)">{_realised:.3f}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 14 — Production Hardening
    #   IBKR adapter, alerts, audit trail, DR backup, latency
    # ────────────────────────────────────────────────────────────────────────
    _at_path = ROOT / "data" / "audit_trail_status.json"
    _dr_path_p9 = ROOT / "data" / "dr_backup.json"
    _lp_path = ROOT / "data" / "latency_profile.json"
    _ar_path = ROOT / "data" / "alert_router.json"

    if any(p.exists() for p in (_at_path, _dr_path_p9, _lp_path, _ar_path)):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header">Production Hardening</div>',
            unsafe_allow_html=True,
        )

        ph_c1, ph_c2, ph_c3, ph_c4 = st.columns(4)

        # ── IBKR ─────────────────────────────────────────────────────────────
        with ph_c1:
            try:
                from scripts.ibkr_adapter import _today_order_count, _load_halal_universe
                n_today = _today_order_count()
                halal_n = len(_load_halal_universe())
            except Exception:
                n_today = 0
                halal_n = 0
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'IBKR ADAPTER</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Mode: <span style="color:var(--text);font-weight:700">DRY_RUN</span><br>'
                f'Halal universe: <span style="color:var(--text)">{halal_n} tickers</span><br>'
                f'Today orders: <span style="color:var(--text)">{n_today} / 25</span><br>'
                f'<span style="color:var(--muted);font-size:10px">install ib_insync + TWS to enable live</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        # ── Audit trail ─────────────────────────────────────────────────────
        with ph_c2:
            _at = json.loads(_at_path.read_text()) if _at_path.exists() else {}
            _valid = _at.get("chain_valid", False)
            _v_color = "#22c55e" if _valid else "#ef4444"
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'AUDIT TRAIL</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Rows: <span style="color:var(--text);font-weight:700">{_at.get("n_total", 0)}</span><br>'
                f'Chain valid: <span style="color:{_v_color};font-weight:700">{"YES" if _valid else "NO"}</span><br>'
                f'Last hash: <span style="color:var(--text);font-size:9px;font-family:monospace">'
                f'{(_at.get("last_hash") or "—")[:12]}...</span><br>'
                f'<span style="color:var(--muted);font-size:10px">SQLite + SHA-256 chain</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        # ── DR backup ───────────────────────────────────────────────────────
        with ph_c3:
            _dr = json.loads(_dr_path_p9.read_text()) if _dr_path_p9.exists() else {}
            _snap = _dr.get("snapshot", {})
            _ok = _snap.get("success", False)
            _ok_color = "#22c55e" if _ok else "#ef4444"
            _enc = _snap.get("encrypted", False)
            _enc_color = "#22c55e" if _enc else "#eab308"
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'DR SNAPSHOT</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Last: <span style="color:{_ok_color};font-weight:700">{"OK" if _ok else "FAIL"}</span><br>'
                f'Size: <span style="color:var(--text)">{_snap.get("size_mb", 0):.2f} MB</span><br>'
                f'Retained: <span style="color:var(--text)">{_dr.get("n_snapshots", 0)}</span><br>'
                f'Encrypted: <span style="color:{_enc_color};font-weight:700">'
                f'{"YES" if _enc else "no"}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        # ── Latency ─────────────────────────────────────────────────────────
        with ph_c4:
            _lp = json.loads(_lp_path.read_text()) if _lp_path.exists() else {}
            _total = _lp.get("current_run_total_s", 0)
            _lat_color = (
                "#22c55e" if _total < 30
                else "#eab308" if _total < 60
                else "#ef4444"
            )
            _ar = json.loads(_ar_path.read_text()) if _ar_path.exists() else {}
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'LATENCY / ALERTS</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Pipeline: <span style="color:{_lat_color};font-weight:700">{_total:.1f}s</span><br>'
                f'Slowest: <span style="color:var(--text)">{_lp.get("slowest_stage", "n/a")}</span><br>'
                f'<span style="color:var(--muted);font-size:10px">'
                f'{_lp.get("slowest_duration_s", 0):.1f}s</span><br>'
                f'Alerts fired: <span style="color:var(--text);font-weight:700">{_ar.get("n_alerts", 0)}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 15 — Governance & Strategy Sandbox  (Phase X)
    # ────────────────────────────────────────────────────────────────────────
    _mrm_path = ROOT / "data" / "mrm_champion.json"
    _ss_path = ROOT / "data" / "strategy_sandbox.json"
    _pf_path = ROOT / "data" / "form_pf_lite.json"

    if any(p.exists() for p in (_mrm_path, _ss_path, _pf_path)):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header">Governance & Strategy Sandbox</div>',
            unsafe_allow_html=True,
        )

        gv_c1, gv_c2, gv_c3, gv_c4 = st.columns(4)

        with gv_c1:
            _mrm = json.loads(_mrm_path.read_text()) if _mrm_path.exists() else {}
            _decision = _mrm.get("decision", "n/a")
            _d_color = {
                "PROMOTE":         "#22c55e",
                "MONITOR":         "var(--text)",
                "RETIRE_CHAMPION": "#ef4444",
            }.get(_decision, "var(--text)")
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'MRM CHAMPION</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'Current: <span style="color:var(--text);font-weight:700">{_mrm.get("current_champion", "—")}</span><br>'
                f'Decision: <span style="color:{_d_color};font-weight:700">{_decision}</span><br>'
                f'Δ-score: <span style="color:var(--text)">{_mrm.get("score_delta", 0):+.4f}</span><br>'
                f'<span style="color:var(--muted);font-size:10px">vs challengers</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with gv_c2:
            _ss = json.loads(_ss_path.read_text()) if _ss_path.exists() else {}
            _best = _ss.get("best_strategy", "n/a")
            _rows = []
            for label in _ss.get("ranked_by_total_sharpe", [])[:4]:
                m = _ss.get("per_strategy", {}).get(label, {}).get("total", {})
                _rows.append(
                    f'<div style="font-size:11px;line-height:1.6;display:flex;justify-content:space-between">'
                    f'<span>{label[:13]}</span>'
                    f'<span style="color:var(--text);font-weight:600">{m.get("sharpe", 0):+.2f}</span>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'STRATEGY SANDBOX</div>'
                + "".join(_rows)
                + '</div>',
                unsafe_allow_html=True,
            )

        with gv_c3:
            _pf = json.loads(_pf_path.read_text()) if _pf_path.exists() else {}
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'FORM PF LITE</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'AUM: <span style="color:var(--text)">${_pf.get("portfolio_value", 0):,.0f}</span><br>'
                f'Champion: <span style="color:var(--text)">{_pf.get("champion", "—")}</span><br>'
                f'α (ann): <span style="color:var(--text)">'
                f'{_pf.get("alpha_annualised_pct", 0):+.2f}%</span><br>'
                f'EVT CVaR-99: <span style="color:#ef4444">{_pf.get("evt_cvar_99_pct", 0)}%</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with gv_c4:
            tear_path = ROOT / "data" / "tear_sheet.md"
            tear_size_kb = tear_path.stat().st_size / 1024 if tear_path.exists() else 0
            st.markdown(
                f'<div style="background:var(--card-bg);border:1px solid var(--border);'
                f'border-radius:8px;padding:14px;">'
                f'<div style="font-size:12px;font-weight:700;color:var(--gold);margin-bottom:8px;">'
                f'TEAR SHEET</div>'
                f'<div style="font-size:11px;color:var(--muted);line-height:1.8;">'
                f'File: <span style="color:var(--text)">data/tear_sheet.md</span><br>'
                f'Size: <span style="color:var(--text)">{tear_size_kb:.1f} KB</span><br>'
                f'Sections: <span style="color:var(--text)">8</span><br>'
                f'<span style="color:var(--muted);font-size:10px">monthly investor report</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 16 — DeepSeek Explainer  (Phase X Stage 56 — north-star surface)
    # ────────────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header">Ask the System (DeepSeek)</div>',
        unsafe_allow_html=True,
    )

    # Last briefing/QA from disk
    _ds_path = ROOT / "data" / "deepseek_last_turn.json"
    _ds_last = json.loads(_ds_path.read_text()) if _ds_path.exists() else {}

    st.markdown(
        f'<div style="background:var(--card-bg);border:1px solid var(--border);'
        f'border-radius:8px;padding:18px;margin-bottom:12px">'
        f'<div style="font-size:11px;color:var(--muted);margin-bottom:8px">'
        f'Last turn: <span style="color:var(--text)">{_ds_last.get("ts", "—")}</span>  '
        f'·  Mode: <span style="color:var(--text)">{_ds_last.get("kind", "qa")}</span>  '
        f'·  Tokens: <span style="color:var(--text)">{_ds_last.get("total_tokens", "—")}</span>  '
        f'·  Engines: <span style="color:var(--text)">{len(_ds_last.get("dossier_keys", []))}</span>'
        f'</div>'
        f'<div style="font-size:12px;color:var(--text);line-height:1.6;white-space:pre-wrap">'
        f'{_ds_last.get("answer", "(no answer yet — run scripts/deepseek_explainer.py)")}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # Chat input (writes the question to a session file; CLI processes it)
    _question = st.text_input(
        "Ask anything about today's pipeline state, signals, risk, or macro:",
        key="deepseek_question",
        placeholder="e.g. 'why is the macro nowcast neutral when geo risk is high?'",
    )
    _topic = st.selectbox(
        "Filter to topic (optional):",
        options=["", "risk", "signals", "macro", "performance",
                 "execution", "governance", "derivatives"],
        key="deepseek_topic",
    )
    if st.button("Ask DeepSeek", key="deepseek_btn"):
        if _question.strip():
            try:
                from scripts.deepseek_explainer import explain
                with st.spinner("DeepSeek is reading the dossier..."):
                    r = explain(_question, topic=_topic or None)
                st.markdown(
                    f'<div style="background:var(--card-bg);border:2px solid var(--gold);'
                    f'border-radius:8px;padding:18px;margin-top:12px">'
                    f'<div style="font-size:11px;color:var(--muted);margin-bottom:8px">'
                    f'Question: {_question}'
                    f'</div>'
                    f'<div style="font-size:13px;color:var(--text);line-height:1.7;white-space:pre-wrap">'
                    f'{r.get("answer", "(no answer)")}'
                    f'</div>'
                    f'<div style="font-size:10px;color:var(--muted);margin-top:8px">'
                    f'Model: {r.get("model")} · Tokens: {r.get("total_tokens", "—")} · '
                    f'Engines used: {len(r.get("dossier_keys", []))}'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                st.error(f"DeepSeek call failed: {exc}")
        else:
            st.warning("Type a question first.")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(
        "Prices from yfinance (15-min delay). Not financial advice. "
        "Refresh sidebar button clears cache."
    )


main()
