"""Environment probe (FR-003): one JSON document, identical schema on Cloud and in the emulator.

Every key is always present. A value that cannot be read is `null` with a `reason`; the probe
never raises.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.schemas import ProbeReport, ProbeValue

PROBE_KEYS = (
    "arch",
    "platform",
    "cpu_model",
    "cpu_flags",
    "cpu_count",
    "cpu_affinity",
    "cgroup_version",
    "cgroup_memory_max_bytes",
    "cgroup_memory_current_bytes",
    "cgroup_cpu_max",
    "mem_total_bytes",
    "glibc",
    "python",
    "streamlit",
    "disk_free_bytes",
    "writable_dirs",
    "can_spawn_subprocess",
    "can_bind_localhost",
    "ffmpeg",
    "llama_server_build",
    "bundle_id",
    "bundle_manifest_sha256",
)

INTERESTING_FLAGS = ("sse4_2", "avx", "f16c", "fma", "avx2", "bmi2", "avx512f", "avx512bw", "avx512_vnni", "avx512_bf16", "avx_vnni", "amx_tile")


def _read(path: str) -> str:
    return Path(path).read_text().strip()


def _cpuinfo() -> dict[str, str]:
    info: dict[str, str] = {}
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            info.setdefault(k.strip(), v.strip())
    return info


def _cgroup() -> dict[str, Any]:
    if Path("/sys/fs/cgroup/cgroup.controllers").exists():
        out: dict[str, Any] = {"version": 2}
        for key, fname in (("memory_max", "memory.max"), ("memory_current", "memory.current"), ("cpu_max", "cpu.max")):
            try:
                out[key] = _read(f"/sys/fs/cgroup/{fname}")
            except OSError as exc:
                out[key + "_reason"] = str(exc)
        return out
    out = {"version": 1}
    for key, fname in (
        ("memory_max", "memory/memory.limit_in_bytes"),
        ("memory_current", "memory/memory.usage_in_bytes"),
        ("cpu_quota", "cpu/cpu.cfs_quota_us"),
        ("cpu_period", "cpu/cpu.cfs_period_us"),
    ):
        try:
            out[key] = _read(f"/sys/fs/cgroup/{fname}")
        except OSError as exc:
            out[key + "_reason"] = str(exc)
    return out


def cgroup_memory_limit_bytes() -> int | None:
    """Effective cgroup memory limit, or None when unlimited/unreadable."""
    cg = _cgroup()
    raw = cg.get("memory_max")
    if not raw or raw == "max":
        return None
    try:
        val = int(raw)
    except ValueError:
        return None
    return val if val < (1 << 60) else None


def cgroup_memory_current_bytes() -> int | None:
    raw = _cgroup().get("memory_current")
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


def _can_spawn() -> bool:
    res = subprocess.run([sys.executable, "-c", "print(42)"], capture_output=True, timeout=10, check=False)
    return res.returncode == 0 and res.stdout.strip() == b"42"


def _can_bind() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _writable_dirs() -> dict[str, bool]:
    out: dict[str, bool] = {}
    for d in (tempfile.gettempdir(), str(Path.home()), str(Path.home() / ".cache"), os.getcwd()):
        out[d] = os.access(d, os.W_OK)
    return out


def _ffmpeg() -> dict[str, Any]:
    import imageio_ffmpeg

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    res = subprocess.run([exe, "-nostdin", "-version"], capture_output=True, timeout=15, check=False)
    first = res.stdout.decode(errors="replace").splitlines()[0] if res.stdout else ""
    return {"path": exe, "runs": res.returncode == 0, "version": first[:120]}


def run_probe(
    *,
    llama_server_build: Callable[[], Any] | None = None,
    bundle_id: str | None = None,
    manifest_sha256: str | None = None,
) -> ProbeReport:
    values: dict[str, ProbeValue] = {}

    def put(key: str, fn: Callable[[], Any]) -> None:
        try:
            values[key] = ProbeValue(value=fn())
        except Exception as exc:
            values[key] = ProbeValue(value=None, reason=f"{type(exc).__name__}: {exc}"[:300])

    cpu = {}
    try:
        cpu = _cpuinfo()
    except Exception:
        cpu = {}
    cg: dict[str, Any] = {}
    try:
        cg = _cgroup()
    except Exception:
        cg = {}

    put("arch", platform.machine)
    put("platform", platform.platform)
    put("cpu_model", lambda: cpu.get("model name") or (_ for _ in ()).throw(RuntimeError("no model name in /proc/cpuinfo")))
    put("cpu_flags", lambda: [f for f in INTERESTING_FLAGS if f in cpu.get("flags", "").split()])
    put("cpu_count", os.cpu_count)
    put("cpu_affinity", lambda: sorted(os.sched_getaffinity(0)))
    put("cgroup_version", lambda: cg["version"])
    put("cgroup_memory_max_bytes", lambda: cg["memory_max"])
    put("cgroup_memory_current_bytes", lambda: cg["memory_current"])
    put("cgroup_cpu_max", lambda: cg.get("cpu_max") or f"{cg['cpu_quota']} {cg['cpu_period']}")
    put("mem_total_bytes", lambda: os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    put("glibc", lambda: "-".join(platform.libc_ver()))
    put("python", lambda: sys.version.split()[0])
    put("streamlit", lambda: __import__("importlib.metadata").metadata.version("streamlit"))
    put("disk_free_bytes", lambda: shutil.disk_usage(Path.home()).free)
    put("writable_dirs", _writable_dirs)
    put("can_spawn_subprocess", _can_spawn)
    put("can_bind_localhost", lambda: {"ok": True, "port": _can_bind()})
    put("ffmpeg", _ffmpeg)
    if llama_server_build is not None:
        put("llama_server_build", llama_server_build)
    else:
        values["llama_server_build"] = ProbeValue(value=None, reason="runtime not started")
    values["bundle_id"] = ProbeValue(value=bundle_id, reason=None if bundle_id else "no manifest loaded")
    values["bundle_manifest_sha256"] = ProbeValue(value=manifest_sha256, reason=None if manifest_sha256 else "no manifest loaded")
    for key in PROBE_KEYS:  # schema completeness guarantee
        values.setdefault(key, ProbeValue(value=None, reason="not collected"))
    return ProbeReport(schema_=1, generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), values={k: values[k] for k in PROBE_KEYS})
