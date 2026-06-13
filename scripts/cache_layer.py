#!/usr/bin/env python3
"""
Cache layer  (Phase XVI Stage 78)
==================================
Generic disk + in-memory cache with per-entry TTL and decorator API.

Why this exists
---------------
- The system makes 10s of yfinance / FRED / Perplexity / DeepSeek calls
  per pipeline run.  Many are redundant within a single day.
- The Next.js frontend polls /api/* every 15-30 s.  When the underlying
  JSON hasn't changed, we shouldn't re-parse 100KB of state.
- Perplexity and DeepSeek are paid APIs — cache misses are real money.

Design
------
- Namespace per data source so we can invalidate independently.
- TTL stored alongside each entry (the consumer picks the value).
- Disk layout:  data/cache/<namespace>/<sha256>.pkl
- In-memory dict on top of disk for hot-path lookups.
- Atomic write (tmp + rename) so an interrupted run never leaves a
  half-written cache file.
- A global stats counter is exposed via `cache_stats()` for telemetry.

Public API
----------
- `@cached(namespace, ttl_seconds)` — decorator
- `cache_get(namespace, key)` / `cache_set(namespace, key, value, ttl)` — manual
- `cache_invalidate(namespace=None)` — wipe a namespace (or all)
- `cache_stats()` — running counters

Threading
---------
This module is *thread-safe* for the simple put / get / invalidate ops
via a single module-level lock.  Decorated functions are NOT serialised
across calls — only the cache structure mutations are.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import re
import threading
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

ROOT = Path(__file__).parent.parent
CACHE_DIR = ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

STATS_FILE = CACHE_DIR / "_stats.json"

logger = logging.getLogger("cache_layer")

T = TypeVar("T")
_LOCK = threading.Lock()
_MEMORY: dict[tuple[str, str], tuple[float, float, Any]] = {}
# Approximate cost-savings model (USD per API call) — used for the UI panel
_CALL_COST_USD = {
    "perplexity":   0.0050,   # rough Sonar Pro pricing
    "deepseek":     0.0010,
    "yfinance":     0.0001,   # not paid, but they rate-limit — pretend it costs ε
    "fred":         0.0000,   # free
    "alpha_stacker_io": 0.0,  # in-process only
}


def _load_stats() -> dict:
    """Load persisted stats from disk so counters survive across processes."""
    if STATS_FILE.exists():
        try:
            return json.loads(STATS_FILE.read_text())
        except Exception:
            pass
    return {
        "hits":   {},
        "misses": {},
        "writes": {},
        "invalidations": {},
        "savings_usd_est": 0.0,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# Per-namespace counters — loaded from disk so they accumulate across runs
_STATS: dict = _load_stats()


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────
def _ns_dir(namespace: str) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", namespace):
        raise ValueError(f"invalid cache namespace: {namespace!r}")
    p = (CACHE_DIR / namespace).resolve()
    if not p.is_relative_to(CACHE_DIR.resolve()):
        raise ValueError(f"invalid cache namespace: {namespace!r}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _hash_key(*args, **kwargs) -> str:
    """Deterministic key from positional + keyword args."""
    blob = json.dumps(
        {"a": [_normalise(a) for a in args], "k": {k: _normalise(v) for k, v in sorted(kwargs.items())}},
        default=str, sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _normalise(x: Any) -> Any:
    """Make values JSON-serialisable for the key hash."""
    if isinstance(x, (list, tuple)):
        return [_normalise(v) for v in x]
    if isinstance(x, dict):
        return {k: _normalise(v) for k, v in sorted(x.items())}
    if isinstance(x, (str, int, float, bool, type(None))):
        return x
    # Fallback: rely on repr for typed objects
    return repr(x)


_STATS_DIRTY = False
_LAST_PERSIST = 0.0
_PERSIST_INTERVAL_S = 5.0   # debounce: don't write to disk on every bump


def _bump(counter: str, namespace: str) -> None:
    global _STATS_DIRTY
    bucket = _STATS.setdefault(counter, {})
    bucket[namespace] = bucket.get(namespace, 0) + 1
    _STATS_DIRTY = True
    _maybe_persist_stats()


def _persist_stats() -> None:
    """Atomic write of stats to disk."""
    global _STATS_DIRTY, _LAST_PERSIST
    try:
        tmp = STATS_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(_STATS, indent=2, default=str))
        os.replace(tmp, STATS_FILE)
        _STATS_DIRTY = False
        _LAST_PERSIST = time.time()
    except Exception:
        pass


def _maybe_persist_stats() -> None:
    """Debounced stats persistence — writes at most every _PERSIST_INTERVAL_S."""
    if not _STATS_DIRTY:
        return
    if time.time() - _LAST_PERSIST < _PERSIST_INTERVAL_S:
        return
    _persist_stats()


import atexit as _atexit
_atexit.register(_persist_stats)


# ──────────────────────────────────────────────────────────────────────────────
# Manual cache API
# ──────────────────────────────────────────────────────────────────────────────
def cache_get(namespace: str, key: str) -> tuple[bool, Any]:
    """Returns (hit, value).  ``hit`` is False on miss or expired entry."""
    now = time.time()
    mem_key = (namespace, key)
    with _LOCK:
        if mem_key in _MEMORY:
            created, ttl, value = _MEMORY[mem_key]
            if ttl == 0 or now - created < ttl:
                _bump("hits", namespace)
                return True, value
            del _MEMORY[mem_key]

        # Try disk
        path = _ns_dir(namespace) / f"{key}.pkl"
        if not path.exists():
            _bump("misses", namespace)
            return False, None
        try:
            with path.open("rb") as f:
                envelope = pickle.load(f)
            created = float(envelope["created"])
            ttl     = float(envelope["ttl"])
            value   = envelope["value"]
            if ttl != 0 and now - created >= ttl:
                # Expired — remove
                try:
                    path.unlink()
                except OSError:
                    pass
                _bump("misses", namespace)
                return False, None
            # Warm memory cache
            _MEMORY[mem_key] = (created, ttl, value)
            _bump("hits", namespace)
            return True, value
        except Exception as exc:
            logger.warning("cache read failed for %s/%s: %s", namespace, key, exc)
            _bump("misses", namespace)
            return False, None


def cache_set(namespace: str, key: str, value: Any, ttl_seconds: float = 0) -> None:
    """Persist a value.  ``ttl_seconds=0`` means never expires."""
    now = time.time()
    mem_key = (namespace, key)
    with _LOCK:
        _MEMORY[mem_key] = (now, ttl_seconds, value)
        path = _ns_dir(namespace) / f"{key}.pkl"
        tmp  = path.with_suffix(".pkl.tmp")
        try:
            with tmp.open("wb") as f:
                pickle.dump({"created": now, "ttl": ttl_seconds, "value": value}, f)
            os.replace(tmp, path)
            _bump("writes", namespace)
        except Exception as exc:
            logger.warning("cache write failed for %s/%s: %s", namespace, key, exc)
            try:
                tmp.unlink()
            except OSError:
                pass


def cache_invalidate(namespace: str | None = None) -> int:
    """
    Wipe a namespace (or all namespaces if ``namespace`` is None).
    Returns the number of entries removed.
    """
    n_removed = 0
    with _LOCK:
        targets = [namespace] if namespace else list({ns for ns, _ in _MEMORY.keys()})
        # Memory
        for key in list(_MEMORY.keys()):
            if namespace is None or key[0] == namespace:
                del _MEMORY[key]
                n_removed += 1
        # Disk
        if namespace is None:
            for ns_path in CACHE_DIR.iterdir():
                if ns_path.is_dir():
                    for f in ns_path.glob("*.pkl"):
                        try:
                            f.unlink()
                            n_removed += 1
                        except OSError:
                            pass
        else:
            ns_path = _ns_dir(namespace)
            for f in ns_path.glob("*.pkl"):
                try:
                    f.unlink()
                    n_removed += 1
                except OSError:
                    pass

        for ns in targets:
            if ns:
                _bump("invalidations", ns)
    return n_removed


# ──────────────────────────────────────────────────────────────────────────────
# Decorator
# ──────────────────────────────────────────────────────────────────────────────
def cached(namespace: str, ttl_seconds: float = 3600.0) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator: cache the return value of ``fn`` keyed on its args.

    Usage::

        @cached(namespace="yfinance", ttl_seconds=6 * 3600)
        def fetch_history(ticker: str, lookback: str) -> pd.DataFrame:
            ...

    Bypass with ``fn(..., _no_cache=True)``.
    Force a refresh (write but don't read) with ``fn(..., _refresh=True)``.
    """
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapped(*args, _no_cache: bool = False, _refresh: bool = False, **kwargs) -> T:
            if _no_cache:
                return fn(*args, **kwargs)

            key = _hash_key(fn.__name__, *args, **kwargs)

            if not _refresh:
                hit, value = cache_get(namespace, key)
                if hit:
                    # Estimate dollar savings
                    cost = _CALL_COST_USD.get(namespace, 0.0)
                    if cost > 0:
                        with _LOCK:
                            _STATS["savings_usd_est"] += cost
                    logger.debug("cache HIT %s/%s key=%s", namespace, fn.__name__, key[:8])
                    return value

            value = fn(*args, **kwargs)
            cache_set(namespace, key, value, ttl_seconds=ttl_seconds)
            logger.debug("cache STORE %s/%s key=%s ttl=%ss", namespace, fn.__name__, key[:8], ttl_seconds)
            return value

        # Cleaner repr for debuggability
        wrapped.__wrapped__ = fn  # type: ignore[attr-defined]

        # Attach helpers for callers that need direct control
        wrapped.invalidate = lambda: cache_invalidate(namespace)  # type: ignore[attr-defined]
        wrapped._namespace = namespace                            # type: ignore[attr-defined]
        return wrapped
    return decorator


# ──────────────────────────────────────────────────────────────────────────────
# Stats
# ──────────────────────────────────────────────────────────────────────────────
def cache_stats() -> dict:
    """Snapshot of running counters + disk footprint."""
    with _LOCK:
        snapshot = json.loads(json.dumps(_STATS, default=str))  # deep copy

    # Disk footprint per namespace
    sizes: dict[str, dict[str, int]] = {}
    for ns_dir in CACHE_DIR.iterdir():
        if not ns_dir.is_dir():
            continue
        n_files = 0
        n_bytes = 0
        for f in ns_dir.glob("*.pkl"):
            n_files += 1
            try:
                n_bytes += f.stat().st_size
            except OSError:
                pass
        sizes[ns_dir.name] = {"n_files": n_files, "size_bytes": n_bytes}

    # Hit-rate per namespace
    hit_rate = {}
    for ns in set(snapshot["hits"]) | set(snapshot["misses"]):
        h = snapshot["hits"].get(ns, 0)
        m = snapshot["misses"].get(ns, 0)
        total = h + m
        hit_rate[ns] = round(h / total, 3) if total else 0.0

    snapshot["hit_rate"] = hit_rate
    snapshot["disk"] = sizes
    snapshot["total_disk_bytes"] = sum(s.get("size_bytes", 0) for s in sizes.values())
    snapshot["snapshot_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    _persist_stats()
    return snapshot


# ──────────────────────────────────────────────────────────────────────────────
# CLI (debug / inspection)
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["stats", "invalidate", "list"], nargs="?", default="stats")
    ap.add_argument("--namespace", default=None)
    args = ap.parse_args()

    if args.command == "invalidate":
        n = cache_invalidate(args.namespace)
        print(f"invalidated {n} entries from {args.namespace or 'ALL'} namespace(s)")
        return 0

    if args.command == "list":
        target = args.namespace
        if target is None:
            for ns_dir in CACHE_DIR.iterdir():
                if ns_dir.is_dir():
                    print(f"{ns_dir.name}: {len(list(ns_dir.glob('*.pkl')))} entries")
        else:
            ns_path = CACHE_DIR / target
            for f in ns_path.glob("*.pkl"):
                stat = f.stat()
                age = time.time() - stat.st_mtime
                print(f"  {f.name}  {stat.st_size:>8} bytes  age={age:.0f}s")
        return 0

    # stats
    s = cache_stats()
    print(f"Cache stats (since {s['started_at']}):")
    print(f"  Estimated $ saved : ${s['savings_usd_est']:.4f}")
    print(f"  Total disk        : {s['total_disk_bytes'] / 1024:.1f} KB")
    print()
    print("  Per namespace:")
    namespaces = set(s["hits"]) | set(s["misses"]) | set(s.get("disk", {}))
    for ns in sorted(namespaces):
        h = s["hits"].get(ns, 0)
        m = s["misses"].get(ns, 0)
        w = s["writes"].get(ns, 0)
        rate = s["hit_rate"].get(ns, 0)
        disk = s.get("disk", {}).get(ns, {})
        print(f"    {ns:<14s} hits={h:>5d} misses={m:>5d} writes={w:>5d}  "
              f"hit-rate={rate:.1%}  disk={disk.get('n_files', 0)} files / "
              f"{disk.get('size_bytes', 0) / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
