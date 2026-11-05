"""DEV ONLY (DR-008, M-CAP-selftest): prove the emulator caps bite before trusting any number.

Run inside the capped container: `docker compose -f docker/compose.cap.yml run --rm selftest`.
Checks: exactly 2 usable CPUs (affinity), cgroup memory limit ~2.7 GB, and a child that allocates
in 100 MB steps is killed between 2,400 and 2,800 MB. Writes the probe JSON as emulator reference.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

CHILD = r"""
import sys, time
blocks = []
for i in range(1, 60):
    b = bytearray(100 * 1024 * 1024)
    for j in range(0, len(b), 4096):
        b[j] = 1
    blocks.append(b)
    print(i * 100, flush=True)
    time.sleep(0.05)
"""


def main() -> int:
    from core.runtime.probe import cgroup_memory_limit_bytes, run_probe

    out: dict[str, object] = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out["cpu_count"] = os.cpu_count()
    out["affinity"] = sorted(os.sched_getaffinity(0))
    out["cgroup_limit_mb"] = (cgroup_memory_limit_bytes() or 0) / 1e6
    proc = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, timeout=300, check=False)
    reached = [int(x) for x in proc.stdout.split() if x.isdigit()]
    out["killed"] = proc.returncode != 0
    out["returncode"] = proc.returncode
    out["last_mb_allocated"] = reached[-1] if reached else 0
    ok_cpu = len(out["affinity"]) == 2  # type: ignore[arg-type]
    ok_mem = bool(out["killed"]) and 2300 <= int(out["last_mb_allocated"]) <= 2800  # type: ignore[call-overload]
    out["ok"] = ok_cpu and ok_mem
    out["probe"] = json.loads(run_probe().model_dump_json(by_alias=True))["values"]
    dest = Path("experiments") / "_raw"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "emulator_probe.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "probe"}, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
