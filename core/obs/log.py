"""Structured JSONL logging (FR-070): ring buffer, redaction, session scoping.

* Every event carries `sid` (or None for system events) so views/exports filter per session (ADR-012).
* The redactor masks registered secret values and key-shaped strings everywhere in an event.
* `span()` emits `<event>.start` / `<event>.end` with `dur_ms` for every pipeline stage.
* Events are mirrored to stdout as one JSON line (visible in the Community Cloud log viewer).
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from collections import deque
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from core.schemas import LogEvent

LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
MASK = "[REDACTED]"

# Key-shaped strings (NFR-SEC-SECRETS). Kept deliberately broad.
KEY_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),  # Google API keys
    re.compile(r"hf_[A-Za-z0-9]{20,}"),  # Hugging Face tokens
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),  # GitHub tokens
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),  # OpenAI-style keys
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{16,}"),
    re.compile(r"(?i)((?:api[_-]?key|token|secret|password)\s*[=:]\s*)[^\s,;\"']{8,}"),
]


class Redactor:
    def __init__(self) -> None:
        self._secrets: set[str] = set()
        self._lock = threading.Lock()

    def register(self, value: str | None) -> None:
        if value and len(value) >= 6:
            with self._lock:
                self._secrets.add(value)

    def unregister(self, value: str | None) -> None:
        if value:
            with self._lock:
                self._secrets.discard(value)

    def text(self, s: str) -> str:
        with self._lock:
            secrets = sorted(self._secrets, key=len, reverse=True)
        for value in secrets:
            if value in s:
                s = s.replace(value, MASK)
        for pat in KEY_PATTERNS:
            if pat.groups:
                s = pat.sub(lambda m: m.group(1) + MASK, s)
            else:
                s = pat.sub(MASK, s)
        return s

    def value(self, v: Any) -> Any:
        if isinstance(v, str):
            return self.text(v)
        if isinstance(v, Mapping):
            return {str(k): (MASK if _is_secret_key(str(k)) else self.value(x)) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [self.value(x) for x in v]
        if isinstance(v, (int, float, bool)) or v is None:
            return v
        return self.text(str(v))


_SECRET_KEY = re.compile(r"(^|[_\-])(api[_\-]?key|apikey|token|access[_\-]?token|secret|password|authorization|auth|bearer)$", re.I)


def _is_secret_key(k: str) -> bool:
    """`api_key`, `hf_token`, `Authorization` -> secret; `max_tokens`, `prompt_tokens` -> not."""
    return bool(_SECRET_KEY.search(k.strip()))


class LogRing:
    """Process-global bounded buffer (<= max_bytes of serialized events)."""

    def __init__(self, max_bytes: int = 5 * 1024 * 1024) -> None:
        self.max_bytes = max_bytes
        self._events: deque[tuple[int, LogEvent]] = deque()
        self._bytes = 0
        self._lock = threading.Lock()
        self._seq = 0

    def append(self, ev: LogEvent) -> int:
        size = len(ev.model_dump_json())
        with self._lock:
            self._seq += 1
            self._events.append((size, ev))
            self._bytes += size
            while self._bytes > self.max_bytes and self._events:
                old, _ = self._events.popleft()
                self._bytes -= old
            return self._seq

    @property
    def bytes(self) -> int:
        return self._bytes

    @property
    def seq(self) -> int:
        return self._seq

    def query(
        self,
        sid: str | None,
        *,
        min_level: str = "DEBUG",
        component: str | None = None,
        text: str | None = None,
        limit: int | None = None,
        include_system: bool = True,
    ) -> list[LogEvent]:
        """Events visible to session `sid`: its own + system events (never other sessions')."""
        floor = LEVELS.get(min_level, 0)
        needle = text.lower() if text else None
        with self._lock:
            snapshot = [e for _, e in self._events]
        out: list[LogEvent] = []
        for e in snapshot:
            if e.sid is not None and e.sid != sid:
                continue
            if e.sid is None and not include_system:
                continue
            if LEVELS[e.level] < floor:
                continue
            if component and e.component != component:
                continue
            if needle and needle not in (e.event + " " + json.dumps(e.fields, default=str)).lower():
                continue
            out.append(e)
        return out[-limit:] if limit else out


class Logger:
    def __init__(self, ring: LogRing | None = None, redactor: Redactor | None = None, level: str = "INFO", stdout: bool = True):
        self.ring = ring or LogRing()
        self.redactor = redactor or Redactor()
        self.level = level
        self.stdout = stdout

    def set_level(self, level: str) -> None:
        if level in LEVELS:
            self.level = level

    def event(
        self,
        component: str,
        event: str,
        *,
        level: str = "INFO",
        sid: str | None = None,
        dur_ms: float | None = None,
        **fields: Any,
    ) -> LogEvent | None:
        if LEVELS[level] < LEVELS[self.level]:
            return None
        ev = LogEvent(
            ts=time.time(),
            level=level,  # type: ignore[arg-type]
            component=component,
            event=self.redactor.text(event),
            sid=sid,
            session_scoped=sid is not None,
            dur_ms=round(dur_ms, 2) if dur_ms is not None else None,
            fields=self.redactor.value(fields),
        )
        self.ring.append(ev)
        if self.stdout:
            try:
                sys.stdout.write(ev.model_dump_json() + "\n")
                sys.stdout.flush()
            except Exception:
                pass
        return ev

    def debug(self, component: str, event: str, **kw: Any) -> None:
        self.event(component, event, level="DEBUG", **kw)

    def info(self, component: str, event: str, **kw: Any) -> None:
        self.event(component, event, level="INFO", **kw)

    def warning(self, component: str, event: str, **kw: Any) -> None:
        self.event(component, event, level="WARNING", **kw)

    def error(self, component: str, event: str, **kw: Any) -> None:
        self.event(component, event, level="ERROR", **kw)

    @contextmanager
    def span(self, component: str, event: str, *, sid: str | None = None, **fields: Any) -> Iterator[dict[str, Any]]:
        """Emit start/end events with duration; the yielded dict adds fields to the end event."""
        extra: dict[str, Any] = {}
        self.event(component, f"{event}.start", sid=sid, **fields)
        t0 = time.perf_counter()
        try:
            yield extra
        except Exception as exc:
            self.event(
                component,
                f"{event}.error",
                level="ERROR",
                sid=sid,
                dur_ms=(time.perf_counter() - t0) * 1000,
                error=type(exc).__name__,
                message=str(exc)[:500],
                **fields,
                **extra,
            )
            raise
        self.event(component, f"{event}.end", sid=sid, dur_ms=(time.perf_counter() - t0) * 1000, **fields, **extra)


def to_jsonl(events: list[LogEvent]) -> str:
    return "".join(e.model_dump_json() + "\n" for e in events)


# Process-wide default logger (the runtime singleton reuses it).
_DEFAULT = Logger()


def get_logger() -> Logger:
    return _DEFAULT
