"""Memory / cap tests (make test-mem: run INSIDE docker/compose.cap.yml).

M-METER-thresholds, M-CAP-selftest (cgroup limit visible + 2 CPUs), M-BOOT-offline (after the
bootstrap the process tree holds no non-loopback sockets).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psutil
import pytest

from core.runtime.probe import cgroup_memory_limit_bytes
from core.runtime.telemetry import MemorySample

pytestmark = pytest.mark.memory
ROOT = Path(__file__).resolve().parents[2]


def test_meter_thresholds() -> None:
    cap = 2700
    assert MemorySample(0, 0.79 * cap, None, cap).level == "ok"
    assert MemorySample(0, 0.80 * cap, None, cap).level == "amber"
    assert MemorySample(0, 0.949 * cap, None, cap).level == "amber"
    assert MemorySample(0, 0.95 * cap, None, cap).level == "red"


def test_running_under_the_cap() -> None:
    limit = cgroup_memory_limit_bytes()
    if limit is None:
        pytest.skip("not inside the capped emulator")
    assert 2_600 <= limit / 2**20 <= 2_800
    assert len(os.sched_getaffinity(0)) == 2


def test_no_external_sockets_after_bootstrap(tmp_path: Path) -> None:
    """M-BOOT-offline with the fake runtime: the app process tree only talks to 127.0.0.1."""
    os.environ.update({"MMRAG_RUNTIME": "fake", "MMRAG_ALL_PRESETS": "1", "MMRAG_CACHE_DIR": str(tmp_path)})
    from core.config import load_config
    from core.obs.log import Logger, LogRing
    from core.runtime.service import Runtime

    rt = Runtime(load_config({}), Logger(LogRing(), stdout=False))
    rt.start()
    assert rt.wait_ready(30)
    try:
        procs = [psutil.Process(), *psutil.Process().children(recursive=True)]
        remote = []
        for p in procs:
            try:
                conns = p.net_connections(kind="inet")
            except (psutil.Error, AttributeError):
                continue
            remote += [c.raddr.ip for c in conns if c.raddr and c.raddr.ip not in ("127.0.0.1", "::1")]
        assert not remote, remote
    finally:
        rt.shutdown()
    _ = sys
