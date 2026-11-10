"""Model governor (FR-005): FIFO work tickets + explicit model residency under a memory budget.

Rules (property-tested in tests/unit/test_governor.py):
  * one ticket holder at a time (one generation / one indexing job at a time); others queue FIFO
    with a visible position and can cancel while queued;
  * never two generators resident; `emb-mm` and a generator never co-reside;
  * a model with use count > 0 is never evicted;
  * before a load: need = measured peak (else estimate); evict idle models until it fits the hard budget;
  * after a load: if real usage exceeds the hard cap, unload it again and fail with the numbers;
  * every load/unload logs model, dur_ms, rss_before, rss_after.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol

from core.errors import CancelledError, DoesNotFitError, QueueTimeoutError
from core.obs.log import Logger, get_logger
from core.schemas import ModelPreset, ModelPresets

ALWAYS_RESIDENT = "emb-text"
MM_EMBEDDER = "emb-mm"


class RouterLike(Protocol):
    def load(self, model: str, timeout: float = ...) -> float: ...
    def unload(self, model: str, timeout: float = ...) -> float: ...
    def loaded(self) -> list[str]: ...


@dataclass
class Ticket:
    id: int
    sid: str | None
    reason: str
    created: float = field(default_factory=time.time)
    cancelled: bool = False
    granted: bool = False


@dataclass
class GovernorStatus:
    resident: list[str]
    in_use: dict[str, int]
    holder: Ticket | None
    queue: list[Ticket]
    loading: str | None


class Governor:
    def __init__(
        self,
        router: RouterLike,
        presets: ModelPresets,
        used_mb: Callable[[], float],
        *,
        hard_budget_mb: float,
        logger: Logger | None = None,
        queue_timeout_s: float = 120.0,
        load_timeout_s: float = 180.0,
    ) -> None:
        self.router = router
        self.presets = {p.id: p for p in presets.presets}
        self.used_mb = used_mb
        self.hard_budget_mb = hard_budget_mb
        self.log = logger or get_logger()
        self.queue_timeout_s = queue_timeout_s
        self.load_timeout_s = load_timeout_s
        self._cv = threading.Condition()
        self._queue: list[Ticket] = []
        self._holder: Ticket | None = None
        self._ids = itertools.count(1)
        self._model_lock = threading.RLock()  # single-flight load/unload (may be held for seconds)
        self._state = threading.Lock()  # guards _resident/_in_use for cheap non-blocking reads
        self._resident: set[str] = set()
        self._in_use: dict[str, int] = {}
        self._loading: str | None = None
        self.measured_peak: dict[str, float] = {}

    # ------------------------------------------------------------------ tickets (FIFO)
    def enqueue(self, sid: str | None, reason: str) -> Ticket:
        with self._cv:
            t = Ticket(next(self._ids), sid, reason)
            self._queue.append(t)
            self.log.info("governor", "ticket.queued", sid=sid, ticket=t.id, reason=reason, position=len(self._queue))
            self._cv.notify_all()
            return t

    def position(self, ticket: Ticket) -> int:
        """0 = running, N = N-th in line, -1 = gone."""
        with self._cv:
            if self._holder is ticket:
                return 0
            try:
                return self._queue.index(ticket) + 1
            except ValueError:
                return -1

    def cancel(self, ticket: Ticket) -> None:
        with self._cv:
            ticket.cancelled = True
            if ticket in self._queue:
                self._queue.remove(ticket)
                self.log.info("governor", "ticket.cancelled", sid=ticket.sid, ticket=ticket.id)
            self._cv.notify_all()

    def wait_turn(self, ticket: Ticket, timeout: float | None = None, on_wait: Callable[[int], None] | None = None) -> None:
        deadline = time.monotonic() + (self.queue_timeout_s if timeout is None else timeout)
        with self._cv:
            while True:
                if ticket.cancelled:
                    raise CancelledError("The request was cancelled.")
                if self._holder is None and self._queue and self._queue[0] is ticket:
                    self._queue.pop(0)
                    self._holder = ticket
                    ticket.granted = True
                    self.log.info("governor", "ticket.granted", sid=ticket.sid, ticket=ticket.id, waited_ms=(time.time() - ticket.created) * 1000)
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if ticket in self._queue:
                        self._queue.remove(ticket)
                    self._cv.notify_all()
                    raise QueueTimeoutError("The model is busy with another request. Please try again in a moment.")
                if on_wait is not None:
                    pos = self._queue.index(ticket) + 1 if ticket in self._queue else -1
                    self._cv.release()
                    try:
                        on_wait(pos)
                    finally:
                        self._cv.acquire()
                self._cv.wait(min(remaining, 0.5))

    def finish(self, ticket: Ticket) -> None:
        with self._cv:
            if self._holder is ticket:
                self._holder = None
                self.log.info("governor", "ticket.released", sid=ticket.sid, ticket=ticket.id, held_ms=(time.time() - ticket.created) * 1000)
            elif ticket in self._queue:
                self._queue.remove(ticket)
            self._cv.notify_all()

    @contextmanager
    def turn(self, sid: str | None, reason: str, on_wait: Callable[[int], None] | None = None, timeout: float | None = None) -> Iterator[Ticket]:
        t = self.enqueue(sid, reason)
        try:
            self.wait_turn(t, timeout=timeout, on_wait=on_wait)
            yield t
        finally:
            self.finish(t)

    # ------------------------------------------------------------------ residency
    def _preset(self, model: str) -> ModelPreset:
        return self.presets[model]

    def need_mb(self, model: str) -> float:
        p = self._preset(model)
        if model in self.measured_peak:
            return self.measured_peak[model]
        return float(p.measured_peak_mb or p.est_peak_mb or 1024)

    def resident(self) -> list[str]:
        with self._state:
            return sorted(self._resident)

    def is_generator(self, model: str) -> bool:
        return self._preset(model).role == "generator"

    def _conflicts(self, model: str) -> list[str]:
        """Models that must be unloaded before `model` by rule (not by memory)."""
        out = []
        for r in self._resident:
            if r == model:
                continue
            if (self.is_generator(model) and (self.is_generator(r) or r == MM_EMBEDDER)) or (model == MM_EMBEDDER and self.is_generator(r)):
                out.append(r)
        return out

    def _victims(self, model: str) -> list[str]:
        """Eviction order when memory is short: idle generators, idle emb-mm, then emb-text."""
        order = sorted(
            (r for r in self._resident if r != model and self._in_use.get(r, 0) == 0),
            key=lambda r: (0 if self.is_generator(r) else 1 if r == MM_EMBEDDER else 2),
        )
        return order

    def ensure(self, model: str, *, sid: str | None = None, progress: Callable[[str], None] | None = None) -> None:
        """Make `model` resident (caller should hold a ticket). Increments its use count."""
        with self._model_lock:
            if model in self._resident:
                with self._state:
                    self._in_use[model] = self._in_use.get(model, 0) + 1
                return
            for victim in self._conflicts(model):
                if self._in_use.get(victim, 0) > 0:
                    raise DoesNotFitError(f"'{model}' cannot be loaded while '{victim}' is in use.")
                if progress:
                    progress(f"unloading {victim}")
                self._unload(victim, sid=sid, why=f"rule:{model}")
            need = self.need_mb(model)
            while self.used_mb() + need > self.hard_budget_mb:
                victims = self._victims(model)
                if not victims:
                    used = self.used_mb()
                    self.log.warning("governor", "does_not_fit", sid=sid, model=model, need_mb=need, used_mb=used, budget_mb=self.hard_budget_mb)
                    raise DoesNotFitError(
                        f"'{self._preset(model).label}' does not fit in memory right now "
                        f"(needs ~{need:.0f} MB, {used:.0f} MB in use, budget {self.hard_budget_mb:.0f} MB)."
                    )
                if progress:
                    progress(f"unloading {victims[0]}")
                self._unload(victims[0], sid=sid, why=f"memory:{model}")
            if progress:
                progress(f"loading {model}")
            self._load(model, sid=sid)
            with self._state:
                self._in_use[model] = self._in_use.get(model, 0) + 1

    def release(self, model: str) -> None:
        with self._state:
            if self._in_use.get(model, 0) > 0:
                self._in_use[model] -= 1

    @contextmanager
    def using(self, model: str, *, sid: str | None = None, progress: Callable[[str], None] | None = None) -> Iterator[None]:
        self.ensure(model, sid=sid, progress=progress)
        try:
            yield
        finally:
            self.release(model)

    def _load(self, model: str, *, sid: str | None) -> None:
        before = self.used_mb()
        self._loading = model
        try:
            dur = self.router.load(model, timeout=self.load_timeout_s)
        finally:
            self._loading = None
        with self._state:
            self._resident.add(model)
        after = self.used_mb()
        delta = max(after - before, 0.0)
        self.measured_peak[model] = max(self.measured_peak.get(model, 0.0), delta)
        self.log.info("governor", "model.load", sid=sid, model=model, dur_ms=dur, rss_before=round(before), rss_after=round(after), delta_mb=round(delta))
        if after > self.hard_budget_mb:
            self._unload(model, sid=sid, why="over_hard_cap")
            raise DoesNotFitError(
                f"'{self._preset(model).label}' loaded but used {after:.0f} MB, above the {self.hard_budget_mb:.0f} MB limit; it was unloaded again."
            )

    def _unload(self, model: str, *, sid: str | None, why: str) -> None:
        if self._in_use.get(model, 0) > 0:
            raise DoesNotFitError(f"'{model}' is in use and cannot be unloaded.")
        before = self.used_mb()
        dur = self.router.unload(model)
        with self._state:
            self._resident.discard(model)
            self._in_use.pop(model, None)
        after = self.used_mb()
        self.log.info("governor", "model.unload", sid=sid, model=model, dur_ms=dur, rss_before=round(before), rss_after=round(after), why=why)

    def unload(self, model: str, *, sid: str | None = None) -> None:
        with self._model_lock:
            if model in self._resident:
                self._unload(model, sid=sid, why="explicit")

    def unload_all(self, *, sid: str | None = None, keep: tuple[str, ...] = ()) -> list[str]:
        with self._model_lock:
            done = []
            for m in sorted(self._resident):
                if m in keep or self._in_use.get(m, 0) > 0:
                    continue
                self._unload(m, sid=sid, why="unload_all")
                done.append(m)
            return done

    def sync(self) -> None:
        """Reconcile with the router's view (e.g. after a child crashed)."""
        with self._model_lock:
            actual = set(self.router.loaded())
            for gone in self._resident - actual:
                self.log.warning("governor", "model.vanished", model=gone)
            with self._state:
                self._resident = actual & set(self.presets)

    def status(self) -> GovernorStatus:
        with self._cv:
            queue = list(self._queue)
            holder = self._holder
        with self._state:
            return GovernorStatus(sorted(self._resident), dict(self._in_use), holder, queue, self._loading)
