#!/usr/bin/env python3
"""
Hierarchical Risk Parity (HRP) Allocator
=========================================
Implements López de Prado's HRP algorithm (2016) on a multi-asset universe.

Algorithm:
  1. Compute the correlation matrix of asset returns
  2. Convert to a distance matrix  d_ij = sqrt((1 - corr_ij) / 2)
  3. Build a single-linkage hierarchical cluster tree
  4. Quasi-diagonalize the correlation matrix using the cluster ordering
  5. Recursive bisection — split into top/bottom clusters and allocate
     weights inversely proportional to cluster variance

For comparison the engine also reports equal-weight and inverse-vol weights,
along with the realized portfolio Sharpe / vol / max drawdown for each.

Default universe: gold, silver, SPY, TLT, DXY (metals + risk + duration + USD)
override with --tickers GC=F,SI=F,SPY,TLT.

Output: data/hrp_allocator.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import yfinance as yf
except ImportError:
    yf = None

DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "hrp_allocator.json"

DEFAULT_TICKERS = ["GC=F", "SI=F", "SPY", "TLT", "DX-Y.NYB"]
DEFAULT_LOOKBACK = "5y"
SQ252 = float(np.sqrt(252))

LINE_W = 62
SEP = "━" * LINE_W


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _fetch_returns(tickers: list[str], lookback: str) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is required")
    raw = yf.download(
        tickers, period=lookback, interval="1d",
        progress=False, auto_adjust=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]]
        close.columns = tickers[:1]
    close = close.dropna(how="all")
    # Forward-fill small gaps (different exchanges have different holidays)
    close = close.ffill().dropna()
    returns = close.pct_change().dropna()
    return returns


# ---------------------------------------------------------------------------
# HRP core (López de Prado)
# ---------------------------------------------------------------------------
def _correl_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """d_ij = sqrt((1 - corr_ij) / 2). Diagonal forced to zero."""
    d = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
    np.fill_diagonal(d.values, 0.0)
    return d


def _quasi_diag(link: np.ndarray) -> list[int]:
    """Sort clustered items by single-linkage distance, recursively."""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = int(link[-1, 3])
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def _cluster_var(cov: pd.DataFrame, items: list) -> float:
    sub = cov.loc[items, items]
    inv_diag = 1.0 / np.diag(sub.values)
    w = inv_diag / inv_diag.sum()
    return float(w @ sub.values @ w)


def _recursive_bisection(cov: pd.DataFrame, sort_ix: list) -> pd.Series:
    w = pd.Series(1.0, index=sort_ix)
    cluster_items = [sort_ix]
    while len(cluster_items) > 0:
        next_clusters = []
        for c in cluster_items:
            if len(c) > 1:
                mid = len(c) // 2
                next_clusters.append(c[:mid])
                next_clusters.append(c[mid:])
        cluster_items = next_clusters
        for i in range(0, len(cluster_items), 2):
            c0 = cluster_items[i]
            c1 = cluster_items[i + 1]
            v0 = _cluster_var(cov, c0)
            v1 = _cluster_var(cov, c1)
            denom = v0 + v1 if (v0 + v1) > 1e-12 else 1.0
            alpha = 1.0 - v0 / denom
            w[c0] *= alpha
            w[c1] *= 1.0 - alpha
    return w


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    cov = returns.cov()
    corr = returns.corr()
    dist = _correl_distance(corr)
    # Convert to condensed form for scipy linkage
    condensed = squareform(dist.values, checks=False)
    link = linkage(condensed, method="single")
    sort_ix = _quasi_diag(link)
    sorted_tickers = [returns.columns[i] for i in sort_ix]
    weights = _recursive_bisection(cov, sorted_tickers)
    return weights.reindex(returns.columns).fillna(0.0)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
def equal_weights(returns: pd.DataFrame) -> pd.Series:
    n = returns.shape[1]
    return pd.Series(1.0 / n, index=returns.columns)


def inverse_vol_weights(returns: pd.DataFrame) -> pd.Series:
    vols = returns.std()
    inv = 1.0 / vols.replace(0, np.nan)
    inv = inv.fillna(0.0)
    if inv.sum() <= 0:
        return equal_weights(returns)
    return inv / inv.sum()


# ---------------------------------------------------------------------------
# Portfolio metrics
# ---------------------------------------------------------------------------
def _portfolio_metrics(returns: pd.DataFrame, weights: pd.Series) -> dict:
    port_ret = returns @ weights
    ann_ret = float(port_ret.mean() * 252)
    ann_vol = float(port_ret.std() * SQ252)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
    cum = (1 + port_ret).cumprod()
    max_dd = float((cum / cum.cummax() - 1).min())

    # Diversification ratio (Choueifaty-Coignard)
    asset_vols = returns.std() * SQ252
    weighted_vol_sum = float((weights * asset_vols).sum())
    div_ratio = weighted_vol_sum / ann_vol if ann_vol > 1e-12 else 0.0

    # Effective number of bets (Meucci) ≈ 1 / sum(w_i^2) for equal-correlation
    enb = float(1.0 / (weights ** 2).sum()) if (weights ** 2).sum() > 0 else 0.0

    return {
        "ann_return_pct":       round(ann_ret * 100, 3),
        "ann_vol_pct":          round(ann_vol * 100, 3),
        "sharpe":               round(sharpe, 3),
        "max_drawdown_pct":     round(max_dd * 100, 3),
        "diversification_ratio":round(div_ratio, 3),
        "effective_n_bets":     round(enb, 2),
        "weights":              {k: round(float(v), 4) for k, v in weights.items()},
    }


# ---------------------------------------------------------------------------
# Cluster tree (for diagnostics)
# ---------------------------------------------------------------------------
def _cluster_tree(returns: pd.DataFrame) -> list:
    corr = returns.corr()
    dist = _correl_distance(corr)
    condensed = squareform(dist.values, checks=False)
    link = linkage(condensed, method="single")
    n = len(returns.columns)
    nodes = list(returns.columns) + [f"merge_{i}" for i in range(n - 1)]
    out = []
    for i, row in enumerate(link):
        l, r, d, sz = int(row[0]), int(row[1]), float(row[2]), int(row[3])
        out.append({
            "merge_id":   f"merge_{i}",
            "left":       nodes[l],
            "right":      nodes[r],
            "distance":   round(d, 4),
            "size":       sz,
        })
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_hrp(
    tickers: list[str] = None,
    lookback: str = DEFAULT_LOOKBACK,
) -> dict:
    tickers = tickers or DEFAULT_TICKERS
    returns = _fetch_returns(tickers, lookback)

    # If yf dropped some tickers (delisted, no data) keep what we got
    available = list(returns.columns)
    if len(available) < 2:
        raise RuntimeError(f"Need at least 2 assets with data, got {available}")

    w_hrp = hrp_weights(returns)
    w_eq = equal_weights(returns)
    w_iv = inverse_vol_weights(returns)

    metrics = {
        "hrp":         _portfolio_metrics(returns, w_hrp),
        "equal_weight":_portfolio_metrics(returns, w_eq),
        "inverse_vol": _portfolio_metrics(returns, w_iv),
    }

    tree = _cluster_tree(returns)
    corr_dict = returns.corr().round(3).to_dict()

    result = {
        "generated_at":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tickers":       available,
        "lookback":      lookback,
        "n_obs":         int(len(returns)),
        "metrics":       metrics,
        "cluster_tree":  tree,
        "correlation":   corr_dict,
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
    print(f"  HIERARCHICAL RISK PARITY ALLOCATOR")
    print(SEP)
    print(f"  Universe:   {', '.join(r['tickers'])}")
    print(f"  Lookback:   {r['lookback']}  ({r['n_obs']} obs)")
    print()

    print(f"  WEIGHTS COMPARISON")
    print(f"  {'─' * 58}")
    header = f"  {'ticker':<10s}  {'HRP':>9s}  {'EqualW':>9s}  {'InvVol':>9s}"
    print(header)
    for tick in r["tickers"]:
        wh = r["metrics"]["hrp"]["weights"].get(tick, 0)
        we = r["metrics"]["equal_weight"]["weights"].get(tick, 0)
        wv = r["metrics"]["inverse_vol"]["weights"].get(tick, 0)
        print(f"  {tick:<10s}  {wh:>8.2%}  {we:>8.2%}  {wv:>8.2%}")
    print()

    print(f"  PORTFOLIO METRICS (in-sample)")
    print(f"  {'─' * 58}")
    cols = [("hrp", "HRP"), ("equal_weight", "EqualW"), ("inverse_vol", "InvVol")]
    print(f"  {'metric':<22s}  " + "  ".join(f"{name:>9s}" for _, name in cols))
    for fld, label in [
        ("ann_return_pct",       "Ann Return (%)"),
        ("ann_vol_pct",          "Ann Vol (%)"),
        ("sharpe",               "Sharpe"),
        ("max_drawdown_pct",     "Max DD (%)"),
        ("diversification_ratio","Div Ratio"),
        ("effective_n_bets",     "Eff # Bets"),
    ]:
        row = f"  {label:<22s}  " + "  ".join(
            f"{r['metrics'][k][fld]:>9.3f}" for k, _ in cols
        )
        print(row)
    print()

    print(f"  CLUSTER MERGES (single-linkage)")
    print(f"  {'─' * 58}")
    for m in r["cluster_tree"]:
        print(f"  {m['merge_id']:<10s}  d={m['distance']:.4f}  "
              f"left={m['left']:<10s}  right={m['right']:<10s}  size={m['size']}")
    print(SEP)
    print(f"  Saved: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical Risk Parity Allocator")
    parser.add_argument(
        "--tickers", default=",".join(DEFAULT_TICKERS),
        help=f"Comma-separated tickers (default: {','.join(DEFAULT_TICKERS)})",
    )
    parser.add_argument("--lookback", default=DEFAULT_LOOKBACK)
    args = parser.parse_args()
    tlist = [t.strip() for t in args.tickers.split(",") if t.strip()]
    run_hrp(tickers=tlist, lookback=args.lookback)
