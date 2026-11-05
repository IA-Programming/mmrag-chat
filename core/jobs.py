"""Background jobs (Rule R4): threads keyed by session id; survive reruns and page navigation.

The UI polls `Job.snapshot()` from a fragment. Cancel stops after the current item (FR-012 AC2).
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.errors import AppError, CancelledError
from core.obs.log import get_logger


@dataclass
class Step:
    name: str
    state: str = "pending"  # pending | running | done | skipped | failed
    detail: str = ""
    dur_ms: float | None = None


@dataclass
class Job:
    id: str
    sid: str
    kind: str
    state: str = "queued"  # queued | waiting | running | done | failed | cancelled
    steps: list[Step] = field(default_factory=list)
    done: int = 0
    total: int = 0
    message: str = ""
    error: str | None = None
    event_id: str | None = None
    result: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    queue_position: int = 0
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ---- called from the worker
    def step(self, name: str, state: str = "running", detail: str = "", dur_ms: float | None = None) -> None:
        with self.lock:
            for s in self.steps:
                if s.name == name:
                    s.state, s.detail, s.dur_ms = state, detail or s.detail, dur_ms if dur_ms is not None else s.dur_ms
                    return
            self.steps.append(Step(name, state, detail, dur_ms))

    def progress(self, done: int, total: int, message: str = "") -> None:
        with self.lock:
            self.done, self.total, self.message = done, total, message

    def check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise CancelledError("Indexing was cancelled.")

    # ---- called from the UI
    def cancel(self) -> None:
        self.cancel_event.set()

    @property
    def finished(self) -> bool:
        return self.state in ("done", "failed", "cancelled")

    def eta_s(self) -> float | None:
        if not self.done or not self.total or self.finished:
            return None
        elapsed = time.time() - self.started_at
        return elapsed / self.done * (self.total - self.done)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "state": self.state,
                "done": self.done,
                "total": self.total,
                "message": self.message,
                "error": self.error,
                "event_id": self.event_id,
                "queue_position": self.queue_position,
                "steps": [s.__dict__.copy() for s in self.steps],
                "result": dict(self.result),
                "elapsed_s": round((self.finished_at or time.time()) - self.started_at, 1),
                "eta_s": self.eta_s(),
            }


class JobRunner:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self.log = get_logger()

    def submit(self, sid: str, kind: str, fn: Callable[[Job], dict[str, Any] | None]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], sid=sid, kind=kind)
        with self._lock:
            self._jobs[job.id] = job

        def run() -> None:
            job.state = "running"
            try:
                res = fn(job)
                job.result.update(res or {})
                job.state = "cancelled" if job.cancel_event.is_set() else "done"
            except CancelledError:
                job.state = "cancelled"
            except AppError as exc:
                job.state, job.error, job.event_id = "failed", exc.user_message, exc.event_id
                self.log.error("jobs", "job.failed", sid=sid, job=job.id, message=exc.user_message, detail=exc.detail, event_id=exc.event_id)
            except Exception as exc:
                err = AppError("The background job failed unexpectedly.")
                job.state, job.error, job.event_id = "failed", err.user_message, err.event_id
                self.log.error(
                    "jobs", "job.crashed", sid=sid, job=job.id, error=f"{type(exc).__name__}: {exc}", tb=traceback.format_exc()[-1500:], event_id=err.event_id
                )
            finally:
                job.finished_at = time.time()

        threading.Thread(target=run, name=f"job-{kind}-{job.id}", daemon=True).start()
        return job

    def get(self, job_id: str | None) -> Job | None:
        if not job_id:
            return None
        with self._lock:
            return self._jobs.get(job_id)

    def for_session(self, sid: str) -> list[Job]:
        with self._lock:
            return [j for j in self._jobs.values() if j.sid == sid]

    def prune(self, max_age_s: float = 3600) -> None:
        now = time.time()
        with self._lock:
            for jid in [j.id for j in self._jobs.values() if j.finished and j.finished_at and now - j.finished_at > max_age_s]:
                self._jobs.pop(jid, None)
