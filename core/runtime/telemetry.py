"""Resource telemetry (FR-004): process-tree RSS (app + llama-server children) and cgroup usage."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

import psutil

from core.runtime.probe import cgroup_memory_current_bytes, cgroup_memory_limit_bytes


def tree_rss_bytes(root_pid: int | None = None) -> int:
    """RSS of `root_pid` (default: this process) plus all descendants."""
    try:
        root = psutil.Process(root_pid or os.getpid())
    except psutil.Error:
        return 0
    total = 0
    procs = [root]
    try:
        procs += root.children(recursive=True)
    except psutil.Error:
        pass
    for p in procs:
        try:
            total += p.memory_info().rss
        except psutil.Error:
            continue
    return total


@dataclass
class MemorySample:
    ts: float
    tree_mb: float
    cgroup_mb: float | None
    cap_mb: int

    @property
    def used_mb(self) -> float:
        """The figure the governor budgets against: process-tree RSS (conservative: counts mmap'd pages)."""
        return self.tree_mb

    @property
    def percent(self) -> float:
        return 100.0 * self.used_mb / self.cap_mb if self.cap_mb else 0.0

    @property
    def level(self) -> str:
        p = self.percent
        return "red" if p >= 95 else "amber" if p >= 80 else "ok"


class Telemetry:
    def __init__(self, cap_mb: int) -> None:
        self.cap_mb = cap_mb
        self._peak_mb = 0.0
        self._lock = threading.Lock()

    def sample(self) -> MemorySample:
        tree = tree_rss_bytes() / 1e6
        cg = cgroup_memory_current_bytes()
        limit = cgroup_memory_limit_bytes()
        s = MemorySample(time.time(), tree, (cg / 1e6) if (cg is not None and limit is not None) else None, self.cap_mb)
        with self._lock:
            self._peak_mb = max(self._peak_mb, s.used_mb)
        return s

    def used_mb(self) -> float:
        return self.sample().used_mb

    def peak_mb(self) -> float:
        with self._lock:
            return self._peak_mb

    def reset_peak(self) -> float:
        with self._lock:
            old, self._peak_mb = self._peak_mb, 0.0
        return old


class PeakSampler:
    """Background sampler that records the peak while a block runs (turn/index stats)."""

    def __init__(self, telemetry: Telemetry, interval: float = 0.25) -> None:
        self.t = telemetry
        self.interval = interval
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> PeakSampler:
        self.peak_mb = self.t.used_mb()
        self._thread = threading.Thread(target=self._run, daemon=True, name="peak-sampler")
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak_mb = max(self.peak_mb, self.t.used_mb())

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(2)
        self.peak_mb = max(self.peak_mb, self.t.used_mb())
