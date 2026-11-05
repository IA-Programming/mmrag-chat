"""Field-report bundle (FR-072): a zip a tester/owner downloads from the deployed app so the dev
session can replay the situation in the emulator (DR-010). No media, no secrets.

Contents: report.json (config without secrets, versions, manifest hash, presets in use, session
settings, effective params of the last turns, metrics summary), probe.json, events.jsonl (the
last N events of THIS session + system events, already redacted).
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from dataclasses import asdict
from typing import Any

from core.config import AppConfig
from core.obs.log import Logger, to_jsonl
from core.schemas import ProbeReport

MAX_EVENTS = 2000


def _safe_config(cfg: AppConfig) -> dict[str, Any]:
    d = asdict(cfg)
    d["gemini"] = {"model": cfg.gemini.model, "base_url": cfg.gemini.base_url, "api_key_set": bool(cfg.gemini.api_key)}
    return {k: str(v) if not isinstance(v, (int, float, bool, dict, type(None))) else v for k, v in d.items()}


def build_field_report(
    *, cfg: AppConfig, logger: Logger, sid: str, probe: ProbeReport, bundle_id: str | None, manifest_sha256: str | None, session: dict[str, Any]
) -> bytes:
    events = logger.ring.query(sid, min_level="DEBUG", limit=MAX_EVENTS)
    report = {
        "schema": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bundle_id": bundle_id,
        "manifest_sha256": manifest_sha256,
        "config": _safe_config(cfg),
        "session": logger.redactor.value(session),
        "event_count": len(events),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("report.json", json.dumps(report, indent=2, default=str))
        z.writestr("probe.json", probe.model_dump_json(by_alias=True, indent=2))
        z.writestr("events.jsonl", to_jsonl(events))
    return buf.getvalue()


def read_field_report(data: bytes) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = set(z.namelist())
        if not {"report.json", "probe.json", "events.jsonl"} <= names or len(names) > 3:
            raise ValueError("not a field-report bundle")
        return {
            "report": json.loads(z.read("report.json")),
            "probe": json.loads(z.read("probe.json")),
            "events": [json.loads(x) for x in z.read("events.jsonl").decode().splitlines() if x.strip()],
        }
