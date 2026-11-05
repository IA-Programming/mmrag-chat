"""DEV ONLY (DR-005): append-only experiment ledger (experiments/ledger.jsonl). Never edit past lines."""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from core.schemas import ExperimentRecord

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"


def git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def append(exp_id: str, *, candidate_id: str | None = None, env: dict[str, Any] | None = None, params: dict[str, Any] | None = None,
           metrics: dict[str, Any] | None = None, gates: dict[str, str | None] | None = None, verdict: str = "", notes: str = "",
           bundle_id: str | None = None, ledger: Path = LEDGER) -> ExperimentRecord:  # fmt: skip
    rec = ExperimentRecord(
        run_id=uuid.uuid4().hex[:12],
        ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        exp_id=exp_id,
        candidate_id=candidate_id,
        bundle_id=bundle_id,
        git_sha=git_sha(),
        env=env or {},
        params=params or {},
        metrics=metrics or {},
        gates=gates or {},
        verdict=verdict,
        notes=notes,
    )
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as fh:  # append-only
        fh.write(rec.model_dump_json() + "\n")
    return rec


def read(ledger: Path = LEDGER) -> list[ExperimentRecord]:
    if not ledger.exists():
        return []
    return [ExperimentRecord.model_validate_json(line) for line in ledger.read_text().splitlines() if line.strip()]


def latest(exp_id: str, candidate_id: str | None = None, ledger: Path = LEDGER) -> ExperimentRecord | None:
    rows = [r for r in read(ledger) if r.exp_id == exp_id and (candidate_id is None or r.candidate_id == candidate_id)]
    return rows[-1] if rows else None


if __name__ == "__main__":
    for r in read():
        print(json.dumps({"ts": r.ts, "exp": r.exp_id, "candidate": r.candidate_id, "verdict": r.verdict}))
