#!/usr/bin/env python3
"""
RL Position-Sizing Agent  (tabular contextual bandit, CVaR-aware)
=================================================================
A reinforcement-learning sizer on top of the directional signal. State is
discretised into a small grid; action is the long position size in
{0, 0.25, 0.50, 0.75, 1.00}; reward is the next-5d action-weighted return
minus a transaction-cost penalty and a CVaR-95 tail penalty.

Phase VII upgrade (PRIVATE_NOTES §8 item 4): the original agent optimised
in-sample Sharpe only. This version adds a CVaR-95 penalty to the reward so
the agent learns to prefer smaller positions during fat-tail regimes.

Why tabular and not PPO? With ~1,200 daily observations the policy network
would overfit, and the discrete state-action structure is interpretable
("when signal is strong-bullish and vol is high and fat-tail, agent goes 50%").

State features (each bucketed into 3 levels):
  signal      sign of 5d LSTM-momentum proxy    → {-1, 0, +1}
  vol         vol_21d absolute regime            → {LOW, NORMAL, HIGH}
  trend       sign of SMA20-vs-SMA50             → {-1, 0, +1}
  tail_regime vol_21d / vol_63d ratio            → {LOW_TAIL, NORMAL_TAIL, FAT_TAIL}
  Total state count = 3 × 3 × 3 × 3 = 81

Action set: position size in [0, 0.25, 0.50, 0.75, 1.00]  (long-only)

Reward (CVaR-aware):
  reward = a × r_h − tc × |Δa| − λ_cvar × max(0, −a × r_h − |VaR_95(h)|)

  The tail term fires when the action-weighted h-day return falls below the
  5th-percentile loss threshold. λ_cvar controls how aggressively the agent
  avoids the left tail. Default λ_cvar = 2.0 (penalty = 2× the tail excess).

Update:  Q[s, a] += α · (reward − Q[s, a])
Policy:  argmax_a Q[s, a]   (greedy)
Baseline: always-full-position (passive long).

Output: data/rl_sizing.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import yfinance as yf
except ImportError:
    yf = None

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "rl_sizing.json"

DEFAULT_TICKER = "GC=F"
DEFAULT_LOOKBACK = "5y"
DEFAULT_HORIZON = 5
DEFAULT_LR = 0.05
DEFAULT_EPSILON = 0.10
DEFAULT_TC_BPS = 5.0    # bps per unit turnover
DEFAULT_N_EPOCHS = 50
DEFAULT_LAMBDA_CVAR = 2.0  # CVaR-95 penalty weight in reward
ACTION_SET = [0.0, 0.25, 0.50, 0.75, 1.00]

SQ252 = float(np.sqrt(252))
LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Data + state encoding
# ---------------------------------------------------------------------------
def _fetch(ticker: str, lookback: str) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is required")
    raw = yf.download(ticker, period=lookback, interval="1d",
                       progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    return raw[["Close"]].dropna()


def _build_state(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    r5 = close.pct_change(5)
    state = pd.DataFrame(index=df.index)
    state["signal"] = np.sign(r5).fillna(0).astype(int)

    vol21 = close.pct_change().rolling(21).std()
    vol_q33 = vol21.quantile(0.33)
    vol_q66 = vol21.quantile(0.66)
    def _vol_bucket(v):
        if pd.isna(v):
            return 0
        return -1 if v < vol_q33 else (1 if v > vol_q66 else 0)
    state["vol"] = vol21.apply(_vol_bucket).astype(int)

    sma_s = close.rolling(20).mean()
    sma_l = close.rolling(50).mean()
    state["trend"] = (
        ((sma_s > sma_l).astype(int) - (sma_s < sma_l).astype(int))
        .fillna(0).astype(int)
    )

    # 4th feature: tail_regime — vol_21d / vol_63d ratio captures whether
    # short-term realised vol is elevated vs medium-term (fat-tail expansion).
    # FAT_TAIL (+1): ratio > 1.20  → recent vol spike → tail risk elevated
    # LOW_TAIL (-1): ratio < 0.80  → vol compression → tail risk suppressed
    # NORMAL   ( 0): otherwise
    vol63 = close.pct_change().rolling(63).std()
    ratio = (vol21 / vol63.replace(0, np.nan)).fillna(1.0)
    state["tail_regime"] = np.where(ratio > 1.20, 1,
                            np.where(ratio < 0.80, -1, 0)).astype(int)

    return state


def _state_key(row) -> tuple:
    return (int(row["signal"]), int(row["vol"]),
            int(row["trend"]), int(row["tail_regime"]))


# ---------------------------------------------------------------------------
# CVaR helpers
# ---------------------------------------------------------------------------
def _var95_threshold(returns: pd.Series, horizon: int) -> float:
    """5th-percentile of h-day forward returns (negative value = loss threshold).
    Used as the VaR-95 hurdle in the CVaR reward penalty."""
    r = returns.dropna()
    if len(r) < 20:
        return -0.05 * np.sqrt(horizon)   # conservative fallback: -5% × √h
    return float(r.quantile(0.05))


def _cvar95(returns: pd.Series) -> float:
    """Expected Shortfall at 95% confidence (CVaR-95).
    Returns the mean of the worst-5% outcomes, expressed as a positive loss %
    (e.g. 0.032 means the average tail loss is 3.2%)."""
    r = returns.dropna()
    if len(r) < 20:
        return 0.0
    var_05 = float(r.quantile(0.05))
    tail = r[r <= var_05]
    return float(-tail.mean()) if len(tail) > 0 else 0.0


# ---------------------------------------------------------------------------
# Q-learning trainer (contextual bandit — single-step)
# ---------------------------------------------------------------------------
def _train_q(
    states: pd.DataFrame, forward_returns: pd.Series,
    lr: float, epsilon: float, tc_bps: float, n_epochs: int,
    var_threshold: float = -0.05, lambda_cvar: float = DEFAULT_LAMBDA_CVAR,
) -> tuple[dict, dict]:
    Q = {}
    visit = {}
    rng = np.random.default_rng(42)

    n = len(states)
    indices = np.arange(n)
    prev_action_per_state: dict[tuple, float] = {}

    for _ in range(n_epochs):
        rng.shuffle(indices)
        for i in indices:
            if pd.isna(forward_returns.iloc[i]):
                continue
            s = _state_key(states.iloc[i])
            if s not in Q:
                Q[s] = {a: 0.0 for a in ACTION_SET}
                visit[s] = 0
            # Epsilon-greedy
            if rng.random() < epsilon:
                a = float(rng.choice(ACTION_SET))
            else:
                a = max(Q[s].items(), key=lambda kv: kv[1])[0]
            # CVaR-aware reward:
            #   reward = a × r_h − tc − λ × max(0, −a × r_h − |VaR_95|)
            # The CVaR term fires when action-weighted return falls below the
            # 5th-percentile loss threshold. Larger λ_cvar → more tail-averse.
            r_h = float(forward_returns.iloc[i])
            prev_a = prev_action_per_state.get(s, a)
            tc = (tc_bps / 10_000.0) * abs(a - prev_a)
            tail_breach = max(0.0, -(a * r_h) - abs(var_threshold))
            cvar_penalty = lambda_cvar * tail_breach
            reward = a * r_h - tc - cvar_penalty
            # Q update
            Q[s][a] += lr * (reward - Q[s][a])
            visit[s] = visit.get(s, 0) + 1
            prev_action_per_state[s] = a
    return Q, visit


# ---------------------------------------------------------------------------
# Backtest with learned policy
# ---------------------------------------------------------------------------
def _backtest_policy(
    states: pd.DataFrame, forward_returns: pd.Series, Q: dict,
) -> tuple[pd.Series, list]:
    actions = []
    rewards = []
    for i in range(len(states)):
        if pd.isna(forward_returns.iloc[i]):
            actions.append(0.0)
            rewards.append(float("nan"))
            continue
        s = _state_key(states.iloc[i])
        if s not in Q:
            a = 1.0  # default = passive long
        else:
            a = max(Q[s].items(), key=lambda kv: kv[1])[0]
        actions.append(a)
        rewards.append(a * float(forward_returns.iloc[i]))
    return pd.Series(rewards, index=states.index), actions


def _sharpe(returns: pd.Series, horizon: int) -> float:
    # Returns are h-day forward; treat each as one h-day "trade"
    r = returns.dropna()
    if len(r) < 30 or r.std() <= 1e-9:
        return 0.0
    periods_per_year = 252.0 / horizon
    return float(r.mean() * periods_per_year / (r.std() * np.sqrt(periods_per_year)))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_rl_sizing(
    ticker: str = DEFAULT_TICKER,
    lookback: str = DEFAULT_LOOKBACK,
    horizon: int = DEFAULT_HORIZON,
    lr: float = DEFAULT_LR,
    epsilon: float = DEFAULT_EPSILON,
    tc_bps: float = DEFAULT_TC_BPS,
    n_epochs: int = DEFAULT_N_EPOCHS,
    lambda_cvar: float = DEFAULT_LAMBDA_CVAR,
) -> dict:
    df = _fetch(ticker, lookback)
    state = _build_state(df)

    forward_return = df["Close"].pct_change(horizon).shift(-horizon)

    # Train/test split (last 252 rows = test)
    split = max(int(len(state) - 252), int(len(state) * 0.8))
    train_state = state.iloc[:split]
    train_fwd = forward_return.iloc[:split]
    test_state = state.iloc[split:]
    test_fwd = forward_return.iloc[split:]

    # VaR-95 threshold from training distribution (5th percentile of h-day returns)
    var_threshold = _var95_threshold(train_fwd, horizon)

    Q, visit = _train_q(
        train_state, train_fwd, lr, epsilon, tc_bps, n_epochs,
        var_threshold=var_threshold, lambda_cvar=lambda_cvar,
    )

    train_rewards, train_actions = _backtest_policy(train_state, train_fwd, Q)
    test_rewards, test_actions = _backtest_policy(test_state, test_fwd, Q)

    # Baseline: always full position
    baseline_train = train_fwd.dropna()
    baseline_test = test_fwd.dropna()

    rl_train_sharpe = _sharpe(train_rewards, horizon)
    rl_test_sharpe = _sharpe(test_rewards, horizon)
    base_train_sharpe = _sharpe(baseline_train, horizon)
    base_test_sharpe = _sharpe(baseline_test, horizon)

    # CVaR-95 comparison on the test set (the key tail-risk metric)
    rl_test_cvar   = _cvar95(test_rewards)
    base_test_cvar = _cvar95(baseline_test)

    # Distribution of tail_regime on the test set
    if "tail_regime" in test_state.columns:
        tc_counts = test_state["tail_regime"].value_counts().to_dict()
        tail_dist = {
            "fat_tail_pct":  round(100 * tc_counts.get(1,  0) / max(len(test_state), 1), 1),
            "normal_pct":    round(100 * tc_counts.get(0,  0) / max(len(test_state), 1), 1),
            "low_tail_pct":  round(100 * tc_counts.get(-1, 0) / max(len(test_state), 1), 1),
        }
    else:
        tail_dist = {}

    # Average action per tail_regime to show the policy learned to downsize
    fat_tail_actions  = [train_actions[i] for i in range(len(train_state))
                         if train_state["tail_regime"].iloc[i] == 1]
    norm_tail_actions = [train_actions[i] for i in range(len(train_state))
                         if train_state["tail_regime"].iloc[i] == 0]
    low_tail_actions  = [train_actions[i] for i in range(len(train_state))
                         if train_state["tail_regime"].iloc[i] == -1]

    # State table (top 10 by visits); now includes tail_regime
    state_keys = sorted(Q.keys(), key=lambda s: visit.get(s, 0), reverse=True)
    state_table = []
    for s in state_keys[:10]:
        best_a = max(Q[s].items(), key=lambda kv: kv[1])
        state_table.append({
            "signal":       s[0],
            "vol":          s[1],
            "trend":        s[2],
            "tail_regime":  s[3],
            "best_action":  best_a[0],
            "q_value":      round(best_a[1], 5),
            "visits":       int(visit.get(s, 0)),
        })

    avg_action_train = float(np.mean(train_actions))
    avg_action_test  = float(np.mean(test_actions))

    result = {
        "generated_at":           datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker":                 ticker,
        "lookback":               lookback,
        "horizon":                horizon,
        "lr":                     lr,
        "epsilon":                epsilon,
        "tc_bps":                 tc_bps,
        "n_epochs":               n_epochs,
        "lambda_cvar":            lambda_cvar,
        "var_95_threshold":       round(var_threshold, 5),
        "n_train":                int(len(train_state)),
        "n_test":                 int(len(test_state)),
        # Sharpe comparison (existing metric, kept for backwards compatibility)
        "rl_train_sharpe":        round(rl_train_sharpe, 3),
        "rl_test_sharpe":         round(rl_test_sharpe, 3),
        "baseline_train_sharpe":  round(base_train_sharpe, 3),
        "baseline_test_sharpe":   round(base_test_sharpe, 3),
        "test_lift_sharpe":       round(rl_test_sharpe - base_test_sharpe, 3),
        # CVaR-95 comparison (new primary risk metric)
        "rl_test_cvar95_pct":     round(rl_test_cvar * 100, 3),
        "baseline_test_cvar95_pct": round(base_test_cvar * 100, 3),
        "cvar_lift_pct":          round((base_test_cvar - rl_test_cvar) * 100, 3),
        # Position sizing behaviour
        "avg_action_train":       round(avg_action_train, 3),
        "avg_action_test":        round(avg_action_test, 3),
        "avg_action_fat_tail":    round(float(np.mean(fat_tail_actions)),  3) if fat_tail_actions  else None,
        "avg_action_normal_tail": round(float(np.mean(norm_tail_actions)), 3) if norm_tail_actions else None,
        "avg_action_low_tail":    round(float(np.mean(low_tail_actions)),  3) if low_tail_actions  else None,
        # Tail regime distribution on test set
        "test_tail_distribution": tail_dist,
        "n_visited_states":       int(len(Q)),
        "top_states":             state_table,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2))
    _print_report(result)
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_report(r: dict) -> None:
    print(f"\n{SEP}")
    print(f"  RL POSITION-SIZING AGENT (CVaR-aware tabular Q) -- {r['ticker']}")
    print(SEP)
    print(f"  Train obs:      {r['n_train']}")
    print(f"  Test obs:       {r['n_test']}")
    print(f"  Horizon:        {r['horizon']}d")
    print(f"  LR / ε:         {r['lr']:.3f} / {r['epsilon']:.3f}")
    print(f"  TC:             {r['tc_bps']:.1f} bps")
    print(f"  λ_cvar:         {r['lambda_cvar']:.2f}")
    print(f"  VaR-95 threshold: {r['var_95_threshold']:+.4f} ({r['var_95_threshold']*100:+.2f}%)")
    print(f"  Visited states: {r['n_visited_states']}")
    print()

    print(f"  SHARPE COMPARISON  (horizon-adjusted)")
    print(f"  {'─' * 50}")
    print(f"  {'set':<14s}  {'RL':>8s}  {'baseline':>10s}  {'lift':>8s}")
    print(f"  {'train':<14s}  {r['rl_train_sharpe']:>+8.3f}  "
          f"{r['baseline_train_sharpe']:>+10.3f}  "
          f"{r['rl_train_sharpe'] - r['baseline_train_sharpe']:>+8.3f}")
    print(f"  {'test':<14s}  {r['rl_test_sharpe']:>+8.3f}  "
          f"{r['baseline_test_sharpe']:>+10.3f}  "
          f"{r['test_lift_sharpe']:>+8.3f}")
    print()

    print(f"  CVaR-95 COMPARISON  (tail risk — lower is better)")
    print(f"  {'─' * 50}")
    print(f"  {'set':<14s}  {'RL':>8s}  {'baseline':>10s}  {'lift':>8s}")
    print(f"  {'test':<14s}  {r['rl_test_cvar95_pct']:>8.3f}%  "
          f"{r['baseline_test_cvar95_pct']:>9.3f}%  "
          f"{r['cvar_lift_pct']:>+7.3f}pp")
    print()

    td = r.get("test_tail_distribution", {})
    if td:
        print(f"  TAIL REGIME DISTRIBUTION (test set)")
        print(f"  Fat-tail: {td.get('fat_tail_pct', 0):.1f}%  "
              f"Normal: {td.get('normal_pct', 0):.1f}%  "
              f"Low-tail: {td.get('low_tail_pct', 0):.1f}%")
        if r.get("avg_action_fat_tail") is not None:
            print(f"  Avg position | fat-tail={r['avg_action_fat_tail']:.3f}  "
                  f"normal={r['avg_action_normal_tail']:.3f}  "
                  f"low-tail={r['avg_action_low_tail']:.3f}")
        print()

    print(f"  AVG ACTION  (position size)")
    print(f"  Train: {r['avg_action_train']:.3f}    Test: {r['avg_action_test']:.3f}")
    print()

    print(f"  TOP STATES BY VISIT COUNT")
    print(f"  {'─' * 66}")
    print(f"  {'sig':>4s}  {'vol':>4s}  {'trd':>4s}  {'tail':>4s}  "
          f"{'action':>8s}  {'Q':>10s}  {'visits':>7s}")
    for s in r["top_states"]:
        print(
            f"  {s['signal']:>+4d}  {s['vol']:>+4d}  {s['trend']:>+4d}  "
            f"{s.get('tail_regime', 0):>+4d}  "
            f"{s['best_action']:>8.2f}  {s['q_value']:>+10.5f}  {s['visits']:>7d}"
        )
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Position-Sizing Agent")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--lookback", default=DEFAULT_LOOKBACK)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--tc-bps", type=float, default=DEFAULT_TC_BPS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_N_EPOCHS)
    parser.add_argument("--lambda-cvar", type=float, default=DEFAULT_LAMBDA_CVAR,
                        help="CVaR-95 tail penalty weight (default 2.0). "
                             "Higher = more tail-averse.")
    args = parser.parse_args()
    run_rl_sizing(
        ticker=args.ticker,
        lookback=args.lookback,
        horizon=args.horizon,
        lr=args.lr,
        epsilon=args.epsilon,
        tc_bps=args.tc_bps,
        n_epochs=args.epochs,
        lambda_cvar=args.lambda_cvar,
    )
