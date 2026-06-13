"""Resilient local file I/O for pipeline scripts.

macOS (especially iCloud-synced Desktop folders) can raise
OSError errno 11 ("Resource deadlock avoided") on concurrent reads.
These helpers retry transient filesystem errors before surfacing failure.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

# EAGAIN on macOS often surfaces as "Resource deadlock avoided".
_TRANSIENT_ERRNOS = frozenset({11, 35})


def _is_transient_os_error(exc: OSError) -> bool:
    return exc.errno in _TRANSIENT_ERRNOS


def read_bytes(path: Path, *, retries: int = 5, backoff_s: float = 0.25) -> bytes:
    last_exc: OSError | None = None
    for attempt in range(retries):
        try:
            return path.read_bytes()
        except OSError as exc:
            if not _is_transient_os_error(exc) or attempt >= retries - 1:
                raise
            last_exc = exc
            time.sleep(backoff_s * (2 ** attempt))
    raise last_exc  # pragma: no cover


def read_text(path: Path, *, encoding: str = "utf-8",
              retries: int = 5, backoff_s: float = 0.25) -> str:
    last_exc: OSError | None = None
    for attempt in range(retries):
        try:
            return path.read_text(encoding=encoding)
        except OSError as exc:
            if not _is_transient_os_error(exc) or attempt >= retries - 1:
                raise
            last_exc = exc
            time.sleep(backoff_s * (2 ** attempt))
    raise last_exc  # pragma: no cover


def read_json(path: Path, **kwargs) -> object:
    return json.loads(read_text(path, **kwargs))


def write_bytes(path: Path, data: bytes, *, retries: int = 5,
                backoff_s: float = 0.25) -> None:
    last_exc: OSError | None = None
    for attempt in range(retries):
        try:
            path.write_bytes(data)
            return
        except OSError as exc:
            if not _is_transient_os_error(exc) or attempt >= retries - 1:
                raise
            last_exc = exc
            time.sleep(backoff_s * (2 ** attempt))
    raise last_exc  # pragma: no cover


def write_text(path: Path, text: str, *, encoding: str = "utf-8",
               retries: int = 5, backoff_s: float = 0.25) -> None:
    last_exc: OSError | None = None
    for attempt in range(retries):
        try:
            path.write_text(text, encoding=encoding)
            return
        except OSError as exc:
            if not _is_transient_os_error(exc) or attempt >= retries - 1:
                raise
            last_exc = exc
            time.sleep(backoff_s * (2 ** attempt))
    raise last_exc  # pragma: no cover


def atomic_write_json(path: Path, data: object, *, indent: int = 2) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_text(tmp, json.dumps(data, indent=indent))
    os.rename(tmp, path)
