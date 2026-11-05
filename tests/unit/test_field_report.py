"""U-FIELD-bundle (no media, no secrets), U-REPLAY (bundle reads back)."""

from __future__ import annotations

import io
import zipfile

from core.config import load_config
from core.obs.field_report import build_field_report, read_field_report
from core.obs.log import Logger, LogRing
from core.runtime.probe import run_probe

KEY = "AIza" + "FIELDcanary" * 3


def test_field_report_has_no_secrets_and_only_this_session() -> None:
    log = Logger(LogRing(), stdout=False)
    log.redactor.register(KEY)
    log.info("chat", "mine", sid="me", note=f"key {KEY}")
    log.info("chat", "other", sid="someone-else", prompt="private question")
    cfg = load_config({"gemini": {"api_key": KEY}}, env={})
    blob = build_field_report(cfg=cfg, logger=log, sid="me", probe=run_probe(), bundle_id="b1", manifest_sha256="0" * 64,
                              session={"gen_overrides": {"m": {"temperature": 0.3}}, "api_key": KEY})  # fmt: skip
    raw = b"".join(zipfile.ZipFile(io.BytesIO(blob)).read(n) for n in zipfile.ZipFile(io.BytesIO(blob)).namelist())
    assert KEY.encode() not in raw and b"private question" not in raw
    assert set(zipfile.ZipFile(io.BytesIO(blob)).namelist()) == {"report.json", "probe.json", "events.jsonl"}
    back = read_field_report(blob)
    assert back["report"]["bundle_id"] == "b1" and back["report"]["config"]["gemini"]["api_key_set"] is True
    assert [e["event"] for e in back["events"]] == ["mine"]
