"""U-LOG-schema, U-LOG-redact (canary), U-LOG-session-scope, U-LOG-events, U-SEC-redact, U-PROBE-schema."""

from __future__ import annotations

import json

import pytest

from core.obs.log import MASK, Logger, LogRing, to_jsonl
from core.runtime.probe import PROBE_KEYS, run_probe
from core.schemas import LogEvent

CANARY = "AIza" + "SyCANARYcanary" * 2 + "CANARY1"  # fake, key-shaped (built at runtime so the repo scan stays clean)


def make() -> Logger:
    return Logger(LogRing(max_bytes=200_000), stdout=False, level="DEBUG")


def test_event_schema_roundtrip() -> None:
    log = make()
    ev = log.event("chat", "turn.start", sid="s1", k=4)
    assert ev is not None
    again = LogEvent.model_validate_json(ev.model_dump_json())
    assert again.sid == "s1" and again.session_scoped and again.fields == {"k": 4}


def test_canary_never_appears() -> None:
    log = make()
    log.redactor.register(CANARY)
    log.info("chat", "prompt", sid="s1", text=f"my key is {CANARY}", nested={"api_key": "whatever-123456", "list": [CANARY]})
    log.info("chat", f"event with {CANARY}")
    log.info("x", "pattern-only", value="hf_" + "A" * 30, auth="Bearer abcdefghijklmnopqrstuvwxyz")
    dump = to_jsonl(log.ring.query(None) + log.ring.query("s1"))
    assert CANARY not in dump and "hf_AAAA" not in dump and "abcdefghijklmnop" not in dump
    assert MASK in dump


def test_session_scope() -> None:
    log = make()
    log.info("chat", "mine", sid="a")
    log.info("chat", "theirs", sid="b")
    log.info("runtime", "system")
    seen = {e.event for e in log.ring.query("a")}
    assert seen == {"mine", "system"}
    assert {e.event for e in log.ring.query("a", include_system=False)} == {"mine"}


def test_ring_is_bounded() -> None:
    ring = LogRing(max_bytes=5_000)
    log = Logger(ring, stdout=False)
    for i in range(500):
        log.info("x", "e", i=i, pad="y" * 50)
    assert ring.bytes <= 5_000 and len(ring.query(None)) < 500


def test_span_emits_start_end_with_duration() -> None:
    log = make()
    with log.span("index", "item", sid="s") as extra:
        extra["pieces"] = 3
    evs = log.ring.query("s")
    assert [e.event for e in evs] == ["item.start", "item.end"]
    assert evs[1].dur_ms is not None and evs[1].fields["pieces"] == 3


def test_span_error_event() -> None:
    log = make()
    with pytest.raises(ValueError), log.span("index", "item", sid="s"):
        raise ValueError("boom")
    assert log.ring.query("s")[-1].event == "item.error"


def test_level_filter() -> None:
    log = Logger(LogRing(), stdout=False, level="WARNING")
    assert log.event("x", "dropped") is None
    assert log.event("x", "kept", level="ERROR") is not None


def test_probe_schema_complete_and_never_raises() -> None:
    def broken() -> str:
        raise RuntimeError("no build")

    rep = run_probe(llama_server_build=broken)
    data = json.loads(rep.model_dump_json(by_alias=True))
    assert list(data["values"]) == list(PROBE_KEYS)
    assert data["values"]["llama_server_build"]["value"] is None and "no build" in data["values"]["llama_server_build"]["reason"]
    for v in data["values"].values():
        assert v["value"] is not None or v["reason"]
