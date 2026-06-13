# THE GRAND MASTER PLAN
## Gold Trading AI → Tier-1 Hedge Fund

**Mission:** Evolve this codebase into an institutional-grade quantitative trading
platform that BlackRock and Vanguard would beg for. Each phase below adds a
real capability that tier-1 quant shops actually have. We advance one or more
stages per prompt; the auto-continue script (`scripts/auto_continue.py`) keeps
us moving through usage-limit windows with no human in the loop.

> Last updated: 2026-05-12
> Operator: qusai.altair@gmail.com
> Architectural standard: pure NumPy / SciPy / sklearn. No `arch` / `statsmodels`.
> UI standard: dark-gold gradient theme, no emojis (`feedback_ui_style.md`).

---

## NORTH-STAR VISION  (operator brief, 2026-05-12)

This isn't just an analytics platform. The end product is a fully autonomous
**halal trading agent** that:

1. **Trades halal equities through Interactive Brokers (IBKR) on the operator's
   behalf.** Phase IX includes IBKR API integration. The agent generates
   signals, sizes positions through the institutional stack we are building,
   and executes via the IBKR TWS API.

2. **Acquires and stores physical gold and silver (long-term core),** but
   actively rebalances around that core: model identifies opportunities to
   sell into spikes and buy back lower, and vice versa. The 75 bps UAE
   physical premium is already baked into the TCA, capacity, and stop-loss
   engines so every decision factors that real-world cost.

3. **Trains continuously to maximise risk-adjusted profit.** The operator
   funds Perplexity / DeepSeek / IBKR market-data API tokens; the agent
   compounds capital. Phase VII (Bayesian HPO, purged K-fold, RL sizing,
   stacking, conformal intervals) is the active-training backbone. Phase X
   adds champion–challenger model risk management so winning models are
   promoted automatically.

4. **Surfaces everything through a dumbed-down explainer UI.** Home menu is
   the user's interface. Every engine's output is summarised in plain
   English by a **DeepSeek-backed conversational layer**, so the operator
   can ask "what is the macro nowcast telling me?" and get a sourced answer
   that pulls from the actual JSON files we already write. This belongs in
   Phase X, Stage 56 (UI 2.0).

The 56-stage plan below is the *path* to this product. Each engine we build
becomes one of the bricks in the autonomous-trader / explainer-UI stack.

---

## STATE OF THE UNION

### DONE
- Stage 1: Tail Risk + Factor Attribution Engine (`scripts/tail_risk_engine.py`)
- Stage A: Cointegration & Mean-Reversion (`scripts/cointegration_engine.py`)
- Stage B: Realistic Transaction Cost Model (`scripts/transaction_cost_model.py`)
- Monte Carlo: 10K-path GARCH+Student-t simulation (`scripts/monte_carlo.py`)
- Kelly Sizing: Full / Half / Fractional with regime overlay (`scripts/kelly_sizing.py`)
- Multi-Timeframe Confluence (`scripts/mtf_confluence.py`)
- Walk-Forward Backtest with rolling windows (`scripts/walk_forward_backtest.py`)
- Auto-Retrain Triggers on DEGRADING health
- Stress Tester: 5 historical crises + parametric tail (`scripts/stress_tester.py`)
- Drawdown Recovery Controller: 5-tier graduated defense (`scripts/drawdown_controller.py`)

### IN FLIGHT (PHASE I — IMMEDIATE)
- Task 13: Alpha Attribution Engine
- Task 14: Volatility Surface Monitor
- Task 15: Signal Decay Half-Life Analyzer

---

## PHASE I — COMPLETE THE INSTITUTIONAL ENGINE STACK

### Stage 13 — Alpha Attribution Engine
**File:** `scripts/alpha_attribution.py`
Decompose strategy returns into independent alpha sources: LSTM directional
signal, macro overlay (Perplexity oracle), regime filter, technical factors,
proving-ground universe. Compute information ratio per source. Rolling 21d /
63d / 252d windows. Output covariance of source returns to surface
diversification benefit. Wire to dashboard Section 6a.

### Stage 14 — Volatility Surface Monitor
**File:** `scripts/vol_surface.py`
Realized vol term structure (5d / 10d / 21d / 63d / 252d) plus vol-of-vol.
Classify into LOW / NORMAL / ELEVATED / EXTREME regimes. Detect contango /
backwardation in vol curve. Detect vol expansion vs contraction phases.
Output drives Kelly fraction and stop-loss width. Dashboard Section 6b.

### Stage 15 — Signal Decay Half-Life Analyzer
**File:** `scripts/signal_decay.py`
For each model output (LSTM 5d, momentum, mean-reversion, MTF score), measure
predictive power decay across 1d / 3d / 5d / 10d / 21d horizons. Fit
exponential decay to compute half-life. Suggest optimal rebalance frequency.
Identify signals whose half-life has shortened (alpha decay). Dashboard Section 6c.

---

## PHASE II — ADVANCED PORTFOLIO CONSTRUCTION

### Stage 16 — Risk Parity / HRP Allocation
Hierarchical Risk Parity (López de Prado). Cluster-tree weighting on the
metals + halal universe. Compare HRP vs equal-weight vs vol-target weights.

### Stage 17 — Black-Litterman Bayesian Portfolio
Combine equilibrium prior (CAPM-implied returns) with our Perplexity macro
view + LSTM view + mean-reversion view. Confidence-weighted posterior.
Output target weights as a tilt over passive baseline.

### Stage 18 — Mean-CVaR Optimal Construction (Stage E from old plan)
Rockafellar-Uryasev linear program. Replace mean-variance with mean-CVaR.
Tail-optimal allocation that respects the 95th-percentile loss budget.

### Stage 19 — Volatility Targeting & Risk Budgeting
Top-down: target 12% portfolio vol. Allocate vol budget across alpha sources
in proportion to their information ratio. De-leverage when realized vol
exceeds target.

### Stage 20 — DCC-GARCH Dynamic Correlations (Stage D from old plan)
Time-varying correlation matrix between gold, silver, halal equities, USD.
Captures correlation breakdown during crises. Pure-numpy implementation.

---

## PHASE III — REGIME DETECTION & ADAPTATION

### Stage 21 — Hidden Markov Regime Model
3-4 state Markov switching model on macro features (real yields, DXY, vol,
breadth). State posteriors drive regime-conditioned trading rules.

### Stage 22 — Structural Break Detection
Bai-Perron multiple break test + CUSUM on cointegration residuals. Detect
when mean-reversion relationships have broken and stop trading the pair.

### Stage 23 — Macro Regime Classifier
4-quadrant growth/inflation matrix (Bridgewater-style). Classify current
regime from PMI, CPI surprise, real yields, DXY trend. Each quadrant gets
a different gold/silver tilt.

### Stage 24 — Bayesian Model Averaging (Stage C from old plan)
Rolling out-of-sample log-likelihood weights over LSTM, momentum, mean-rev,
HMM-conditioned. Mathematically coherent replacement for ad-hoc ensemble.

---

## PHASE IV — ENHANCED EXECUTION & MICROSTRUCTURE

### Stage 25 — Smart Order Router
TWAP / VWAP / POV slicing. Dynamically choose algo based on urgency and
displayed liquidity. Pre-trade impact estimate from TCA engine.

### Stage 26 — Adverse Selection Detector
Track post-trade markouts at 1m / 5m / 30m. Flag systematic adverse
selection by venue / time-of-day. Drives venue routing.

### Stage 27 — Dynamic Stop-Loss Optimizer
ATR-based, trailing, chandelier exit, and Parabolic SAR variants. Regime-
conditioned: tight stops in vol expansion, wider in mean-reverting regimes.

### Stage 28 — Strategy Capacity Analysis
Estimate the AUM ceiling at which our alpha decays below threshold. Driven
by ADV, our average position size, and impact-cost slippage curve.

---

## PHASE V — ALTERNATIVE DATA & SENTIMENT

### Stage 29 — News Sentiment NLP
Local sentiment model on Bloomberg / Reuters / X feeds. Per-asset sentiment
score with decay. Surprise-vs-baseline z-scores.

### Stage 30 — Central Bank Speech Analyzer
Hawkish-dovish scoring of FOMC / ECB / BoE / PBoC speeches. Real-time
update on day-of via RSS. Drives bond/gold tilts.

### Stage 31 — Geopolitical Event Detector
Auto-classify breaking news into: military, sanctions, energy, trade.
Each category has a calibrated gold-impact prior.

### Stage 32 — ETF Flow Tracker
Daily flow into GLD / SLV / IAU / GLDM. Persistent inflows = retail momentum.
Persistent outflows = bull-market exhaustion.

### Stage 33 — Macro Nowcasting Composite (Stage F from old plan)
Surprise indices, RV vs IV gap, contango. Single composite score driving
the directional bias overlay.

---

## PHASE VI — DERIVATIVES & HEDGING

### Stage 34 — Options Pricing & Greeks
Black-Scholes + binomial tree. Delta, gamma, vega, theta on GLD options.
Foundation for hedge overlays.

### Stage 35 — Tail Risk Hedge Overlay
Rolling 3-month OTM put strategy on GLD. Sized to cap CVaR-95 at fixed level.
Cost-budgeted (max 1.5% drag per year).

### Stage 36 — Carry Trade Analyzer
Gold lease rates, real yields, FX carry. Decompose total return into spot,
carry, and roll components.

### Stage 37 — Term Structure / Gold Futures Curve
GC1 / GC2 / GC3 contango analysis. Flat curve = stress, steep contango =
calm. Drives positioning across spot vs futures.

---

## PHASE VII — MACHINE LEARNING ENHANCEMENTS

### Stage 38 — Bayesian Hyperparameter Optimization
Optuna-style search over LSTM hyperparams. Walk-forward validation as
objective. Resume-able study.

### Stage 39 — Purged K-Fold Cross-Validation
López de Prado purged + embargoed K-fold for time series. Replaces
walk-forward where appropriate. Eliminates label leakage.

### Stage 40 — Ensemble Stacking
LSTM + XGBoost + Transformer (small TCN). Out-of-fold meta-learner.
Each base model has different inductive bias.

### Stage 41 — Reinforcement Learning Sizing Agent (Stage H)
PPO agent that learns position sizing on top of LSTM directional signal.
Reward = realized Sharpe. Replaces / augments Kelly.

### Stage 42 — Conformal Prediction Intervals
Distribution-free prediction intervals around LSTM forecasts. Coverage
guarantee under exchangeability. Drives confidence-weighted sizing.

---

## PHASE VIII — PERFORMANCE ATTRIBUTION & ANALYTICS

### Stage 43 — Brinson Performance Attribution
Allocation effect vs selection effect vs interaction. Attribution against
a benchmark (equal-weight metals, GLD).

### Stage 44 — Fama-French Factor Loading
Regress strategy returns on Mkt-Rf / SMB / HML / Mom / Quality / BAB.
Quantify residual alpha vs factor exposure.

### Stage 45 — Information Coefficient & IR Tracking
Daily IC between LSTM forecast and realized 5d return. Rolling IR.
Decompose strategy IR = breadth × IC.

### Stage 46 — Decision Quality Framework (Stage G from old plan)
Brier score, log loss, calibration plots. Proves the system is calibrated,
not just accurate. Reliability diagrams in dashboard.

---

## PHASE IX — PRODUCTION HARDENING

### Stage 47 — Live Broker Integration
Interactive Brokers (TWS API) or OANDA paper-trading. Order management,
fill reconciliation, position sync. Read-only first, then opt-in live.

### Stage 48 — Real-Time Alerting
Push (already partial), email, SMS via Twilio. Configurable triggers:
regime change, drawdown tier escalation, signal flip.

### Stage 49 — Compliance / Audit Trail (Stage I from old plan)
SQLite append-only schema: every signal, every sizing decision, every
order. Cryptographic chain (hash links). Regulatory-ready.

### Stage 50 — Disaster Recovery / Failover
Automated daily snapshot of `data/` to encrypted off-site backup. Cold-start
playbook tested monthly.

### Stage 51 — Latency Profiling
Pipeline stage timing. Identify the slow stages and optimize. Target:
end-to-end pipeline under 30 seconds.

---

## PHASE XI — OPERATIONALIZATION  (post-56-stage; the daily-use product)

After the 56-stage engine stack was complete, Phase XI built the operator
surface and the live-trading wiring:

### Stage 57 — Trade Idea Generator
**File:** `scripts/trade_idea_generator.py`
Synthesises the full 43-engine state into a single daily trade card: side,
ticker, size, conviction, reasoning. IBKR-ready format. Saved to
`data/trade_idea.json`.

### Stage 58 — Halal Universe Screener
**File:** `scripts/halal_screener.py`
Sector-exclusion + debt-ratio filter over a candidate universe (S&P 100
defaults), produces `data/halal_universe.json` which feeds the IBKR
adapter's pre-trade gate.

### Stage 59 — Physical Metals Rebalancer
**File:** `scripts/metals_rebalancer.py`
Watches the long-term physical gold/silver core for opportunistic
sell-into-spike / buy-back-lower trades. Reads from vol_surface, term
structure, macro_nowcast, cointegration. Tracks open rebalance trades.

### Stage 60 — Continuous Training Orchestrator
**File:** `scripts/continuous_trainer.py`
Background loop: daily Bayesian HPO + RL refit, weekly MRM championship
evaluation, automated promotion when challengers win. All runs logged to
the audit trail.

### Stage 61 — Home UI 2.0
**File:** `ui/pages/0_Home.py` (rewritten)
Polished operator surface. Hero panel: today's trade idea + plain-English
DeepSeek briefing. Below the hero: collapsible engine-evidence sections.
Chat box always visible at top.

---

## PHASE X — GOVERNANCE & RESEARCH

### Stage 52 — Model Risk Management Framework
Champion-challenger tournament. Each model documented with: assumptions,
failure modes, monitoring metrics, retire criteria.

### Stage 53 — Strategy Sandbox
Paper-trading harness for new strategies before they touch live capital.
6-month soak test minimum.

### Stage 54 — Regulatory Reporting (Form PF lite)
Monthly tear sheet: AUM, leverage, top exposures, VaR, concentration.

### Stage 55 — Investor-Facing Tear Sheets
Monthly PDF: returns, attribution, risk metrics, narrative. PyMuPDF
generated. Auto-emailed.

### Stage 56 — UI 2.0 (Stage J from old plan)
Coherent control room. Every engine surfaces its panel. Factor-returns
chart. Live regime indicator. Mobile-responsive layout.

---

## EXECUTION CADENCE

Each prompt advances 1-3 stages. Standard sub-tasks per stage:
1. Build the engine (`scripts/<name>.py`) producing JSON output
2. Wire into `scripts/master_controller.py` as a soft-fail stage
3. Add fields to `_build_pipeline_state` so they appear in `pipeline_state.json`
4. Add UI panel to `ui/pages/0_Home.py` (Section 6+ style)
5. Smoke-test end-to-end (AST parse, JSON valid, snapshot picks up fields)

The auto-continue script handles Claude Code rate-limit windows: when the
5-hour limit hits, it queues a `continue` keystroke at the reset moment and
loops indefinitely until the operator stops it.

## WIN CONDITION

When all 56 stages are checked off, the system has every capability that a
tier-1 multi-strategy quant fund has, scoped to gold / silver / halal-equity:
- Independent alpha sources with attribution
- Tail-aware portfolio construction
- Regime-conditioned execution
- Alternative-data overlay
- Derivatives-hedged tail risk
- Production-grade monitoring and audit
- Governance and investor reporting

At that point we benchmark against AQR / Bridgewater / Two Sigma multi-strat
on returns, Sharpe, max drawdown, and tail-loss measures, and the
"BlackRock would beg for it" claim becomes empirically defensible.
