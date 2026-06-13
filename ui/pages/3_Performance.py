"""
Phase XV — Performance & Validation Dashboard
=============================================
Streamlit page surfacing the Phase XV strategy backtest verdict and all
Phase XIV/XXV/XXVI operational metrics that are visible in the Next.js
BacktestPanel — parity for the legacy Streamlit UI.

Sections
--------
1. Phase XV Backtest Verdict   — achievability + Sharpe/DD/return
2. NAV History                 — equity curve from phase14_nav.csv
3. Strategy Attribution        — by-strategy P&L breakdown (incl. TREASURY_HEDGE)
4. Treasury Hedge Sleeve       — Phase XXV status, sharia gate, effective instrument
5. ML Conviction Gate          — Phase XXVI PoC + walk-forward validation
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "data"

st.set_page_config(
    page_title="Performance — Gold Trading AI",
    page_icon="📊",
    layout="wide",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _load(name: str) -> dict:
    p = DATA / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def _age(name: str) -> str:
    p = DATA / name
    if not p.exists():
        return "—"
    dt = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    delta = datetime.now(timezone.utc) - dt
    h = int(delta.total_seconds() / 3600)
    if h < 1:
        return f"{int(delta.total_seconds()/60)}m ago"
    if h < 24:
        return f"{h}h ago"
    return f"{h // 24}d ago"

def _pct(v, sign: bool = True) -> str:
    if v is None:
        return "—"
    try:
        fmt = f"{float(v):+.2f}%" if sign else f"{float(v):.2f}%"
        return fmt
    except Exception:
        return "—"

def _usd(v) -> str:
    if v is None:
        return "—"
    try:
        return f"${float(v):,.2f}"
    except Exception:
        return "—"

# ── Verdict colour maps ───────────────────────────────────────────────────────
VERDICT_COLOUR = {
    "ACHIEVABLE":                  "🟢",
    "STRETCH":                     "🟡",
    "OPTIMISTIC":                  "🟠",
    "ELITE_SYSTEM_TARGET_TOO_HIGH":"🔵",
    "UNREALISTIC":                 "🔴",
}
OVERLAY_COLOUR = {
    "OVERLAY_BENEFICIAL": "🟢",
    "OVERLAY_MIXED":      "🟡",
    "OVERLAY_NEGATIVE":   "🔴",
}

# ══════════════════════════════════════════════════════════════════════════════
# Title
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## Performance & Validation")
st.caption("Phase XIV · XV · XXV · XXVI — parity with Next.js BacktestPanel")

# ══════════════════════════════════════════════════════════════════════════════
# 1. Phase XV Backtest Verdict
# ══════════════════════════════════════════════════════════════════════════════
bt = _load("strategy_backtest.json")
st.divider()
st.markdown("### Phase XV — Strategy Backtest Verdict")

if not bt:
    st.info("strategy_backtest.json not found — run the pipeline or click Run Backtest.")
    if st.button("Run Backtest", key="bt_run"):
        import subprocess, sys
        with st.spinner("Running backtester (~30s) …"):
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "strategy_backtester.py")],
                cwd=str(ROOT), timeout=120,
            )
        st.rerun()
else:
    verdict   = bt.get("achievability_verdict", "—")
    icon      = VERDICT_COLOUR.get(verdict, "⚪")
    note      = bt.get("achievability_note", "")
    perf      = bt.get("performance", {})
    monthly   = bt.get("monthly", {})
    by_strat  = bt.get("by_strategy", [])

    col_v, col_r, col_s, col_d = st.columns(4)
    col_v.metric("Verdict", f"{icon} {verdict}")
    col_r.metric("Cumulative Return", _pct(perf.get("cum_return_pct")))
    col_s.metric("Sharpe", f"{perf.get('sharpe', '—')}")
    col_d.metric("Max Drawdown", _pct(perf.get("max_drawdown_pct")))

    col_a, col_m, col_w = st.columns(3)
    col_a.metric("Ann. Return", _pct(perf.get("annualised_return_pct")))
    col_m.metric(
        "Months at Target",
        f"{monthly.get('n_at_or_above_target', '—')} / {monthly.get('n_months', '—')}",
    )
    col_w.metric("Win Rate", _pct(perf.get("win_days_pct"), sign=False))

    if note:
        st.caption(note)

    # Monthly returns bar chart
    monthly_returns = monthly.get("returns", [])
    if monthly_returns:
        df_m = pd.DataFrame(monthly_returns)
        if "return_pct" in df_m.columns and "month" in df_m.columns:
            df_m["colour"] = df_m["return_pct"].apply(
                lambda x: "green" if x >= 0 else "red"
            )
            with st.expander("Monthly Returns", expanded=True):
                st.bar_chart(df_m.set_index("month")["return_pct"])

    # Strategy breakdown
    if by_strat:
        with st.expander("By-strategy P&L attribution"):
            df_s = pd.DataFrame(by_strat)
            if "total_pl_pct" in df_s.columns:
                df_s["total_pl_pct"] = df_s["total_pl_pct"].round(3)
            st.dataframe(df_s, use_container_width=True, hide_index=True)

    st.caption(f"Backtest data: {_age('strategy_backtest.json')}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. NAV History
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("### Phase XIV — NAV History")

nav_path = DATA / "phase14_nav.csv"
mst = _load("multi_strategy_trader.json")

if nav_path.exists():
    try:
        df_nav = pd.read_csv(nav_path, parse_dates=["date"])
        df_nav = df_nav.sort_values("date").set_index("date")

        col_eq, col_lt, col_sh = st.columns(3)
        col_eq.metric(
            "Book Equity",
            _usd(mst.get("book_equity_usd")),
            delta=_pct(mst.get("lifetime_pl_pct")),
        )
        ns = mst.get("nav_stats") or {}
        col_lt.metric("MTD", _pct(ns.get("mtd_return_pct")))
        col_sh.metric("Sharpe (approx)", f"{ns.get('sharpe_approx', '—')}")

        st.line_chart(df_nav["nav_usd"], use_container_width=True)
        st.caption(
            f"Current strategy: **{mst.get('strategy', '—')}**  ·  "
            f"Open: {mst.get('n_open', 0)}  ·  "
            f"Closed: {mst.get('n_closed_total', 0)}"
        )
    except Exception as e:
        st.warning(f"Could not load NAV history: {e}")
else:
    st.info("phase14_nav.csv not yet generated — run the multi-strategy trader.")

# ══════════════════════════════════════════════════════════════════════════════
# 3. By-Strategy P&L Attribution
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("### Strategy Attribution (live book)")

by_s = mst.get("by_strategy") or {}
if by_s:
    rows = []
    for strat, data in by_s.items():
        rows.append({
            "Strategy":         strat,
            "Realized P&L":     _usd(data.get("realized_pl_usd")),
            "Open Notional":    _usd(data.get("open_notional_usd")),
            "Open Positions":   data.get("n_open", 0),
            "Closed Trades":    data.get("n_closed", 0),
        })
    # Always show TREASURY_HEDGE first
    rows.sort(key=lambda r: (0 if r["Strategy"] == "TREASURY_HEDGE" else 1, r["Strategy"]))

    hedge_realized = (by_s.get("TREASURY_HEDGE") or {}).get("realized_pl_usd", 0.0) or 0.0
    alpha_realized = sum(
        (v.get("realized_pl_usd") or 0.0)
        for k, v in by_s.items() if k != "TREASURY_HEDGE"
    )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Hedge realized P&L",  _usd(hedge_realized))
    c2.metric("Alpha realized P&L",  _usd(alpha_realized))
    c3.metric(
        "Hedge fraction",
        _pct(mst.get("hedge_fraction", 0) * 100, sign=False),
    )
else:
    st.info("No by-strategy data yet. Run the multi-strategy trader first.")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Treasury Hedge Sleeve (Phase XXV)
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("### Phase XXV — Treasury Hedge Sleeve")

from scripts.treasury_hedge_overlay import sanitize_hedge_recommendation

th = sanitize_hedge_recommendation(_load("treasury_hedge.json"))
tes = _load("treasury_overlay_stress_eval.json")
hs = mst.get("hedge_state") or {}

if not th and not hs:
    st.info("treasury_hedge.json not found — run the treasury hedge overlay.")
else:
    mode         = hs.get("mode") or th.get("mode", "SIGNAL_ONLY")
    gate         = hs.get("gate_action") or th.get("gate_action", "—")
    rec_instr    = th.get("instrument") or "—"
    eff_instr    = th.get("effective_instrument") or hs.get("instrument") or "—"
    alloc_pct    = th.get("effective_allocation_pct") or th.get("allocation_pct", 0)
    cleared      = th.get("sharia_cleared", False)
    sub_tag      = hs.get("sub_tag") or th.get("sub_tag")
    hedge_ntl    = mst.get("hedge_notional_usd", 0) or 0
    hedge_frac   = (mst.get("hedge_fraction") or 0) * 100
    quadrant     = th.get("regime_quadrant", "—")
    crisis_tier  = th.get("crisis_tier", "—")
    reason       = th.get("reason", "—")

    sharia_label = "✅ Cleared" if cleared else "❌ Not cleared (GLD fallback active)"
    gate_blocked = hs.get("gate_blocked", False)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mode", mode)
    c2.metric("Rec. instrument", rec_instr)
    c3.metric("Effective instrument", eff_instr)
    c4.metric("Allocation", _pct(alloc_pct, sign=False))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Sharia gate", sharia_label)
    c6.metric("Notional (live)", _usd(hedge_ntl))
    c7.metric("Hedge fraction", _pct(hedge_frac, sign=False))
    c8.metric("Regime", f"{quadrant} / {crisis_tier}")

    if sub_tag:
        st.warning(
            f"**Sharia fallback active** — holding GLD ({sub_tag}) instead of "
            f"coupon-bearing sovereign debt. Set TREASURY_SHARIA_CLEARED=true "
            f"after a fatwa to enable {rec_instr}."
        )
    if gate_blocked:
        st.error(f"Gate blocked: {hs.get('gate_reason', '—')}")

    st.caption(f"Reason: {reason}")

    # Overlay stress eval
    if tes:
        ov = OVERLAY_COLOUR.get(tes.get("verdict", ""), "⚪")
        r, c = tes.get("n_rescued", "—"), tes.get("n_regressed", "—")
        d    = tes.get("delta_avg_sharpe")
        note = tes.get("note", "")
        st.markdown(
            f"**Stress eval (Phase XXV-b):** {ov} `{tes.get('verdict', '—')}`  ·  "
            f"rescued={r}  regressed={c}  "
            f"Δ Sharpe={_pct(d) if d is not None else '—'}  ·  {_age('treasury_overlay_stress_eval.json')}"
        )
        if note:
            st.caption(note)
    else:
        st.caption("Treasury overlay stress eval not yet run (needed for Phase XXV-F gate).")

# ══════════════════════════════════════════════════════════════════════════════
# 5. ML Conviction Gate (Phase XXVI)
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("### Phase XXVI — ML Conviction Gate")

poc = _load("ml_conviction_poc.json")
wf  = _load("ml_walk_forward.json")
cw  = _load("conviction_weights.json")

col_p, col_w = st.columns(2)

with col_p:
    st.markdown("**PoC: ML vs Rule Sharpe**")
    if poc:
        cp = poc.get("comparison", {})
        gate_pass = cp.get("gate_passed", False)
        icon = "✅" if gate_pass else "❌"
        c1, c2, c3 = st.columns(3)
        c1.metric("Gate", f"{icon} {'PASS' if gate_pass else 'FAIL'}")
        c2.metric("OOS IC (ML)", f"{cp.get('oos_ic_ml', '—')}")
        c3.metric("Sharpe ML vs Rule",
                  f"{cp.get('sharpe_ml', '—')} vs {cp.get('sharpe_rule', '—')}")
        st.caption(_age("ml_conviction_poc.json"))
    else:
        st.info("ml_conviction_poc.json not yet generated.")

with col_w:
    st.markdown("**Walk-Forward Validation**")
    if wf:
        gate_wf = wf.get("gate_passed", False)
        icon_wf = "✅" if gate_wf else "❌"
        ml = wf.get("ml", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Gate", f"{icon_wf} {wf.get('verdict', '—')}")
        c2.metric("Median Ann.", _pct(ml.get("median_ann_pct")))
        c3.metric("Avg Sharpe", f"{ml.get('avg_sharpe', '—')}")
        st.caption(_age("ml_walk_forward.json"))
    else:
        st.info("ml_walk_forward.json not yet generated (requires ml_conviction_poc first).")

# Conviction weights table
if cw and cw.get("weights"):
    with st.expander("Conviction weights (Phase XXIII)"):
        w = cw["weights"]
        df_w = pd.DataFrame(
            [{"Component": k, "Weight": round(v, 4), "Share": _pct(v * 100, sign=False)}
             for k, v in sorted(w.items(), key=lambda kv: kv[1], reverse=True)]
        )
        st.dataframe(df_w, use_container_width=True, hide_index=True)
        top = max(w.items(), key=lambda kv: kv[1])
        st.caption(
            f"Top component: **{top[0]}** ({top[1]:.3f})  ·  "
            f"Updated: {_age('conviction_weights.json')}"
        )

# ══════════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "Data refreshes on each pipeline run. "
    "All times UTC. "
    "Phase XV strategy_backtest.json · Phase XXV treasury_hedge.json · "
    "Phase XXVI ml_conviction_poc.json · ml_walk_forward.json"
)
