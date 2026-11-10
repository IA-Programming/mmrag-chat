"""U-GEM-key-canary, I-GEM-fake-429, U-GEM-media-confirm, U-GP-capability (Gemini)."""

from __future__ import annotations

import json

import httpx
import pytest

from core import net
from core.errors import BackendError, RateLimitedError
from core.llm.backends import GeminiBackend
from core.obs.log import Logger, LogRing, to_jsonl
from core.rag.pipeline import Attachment, ChatPipeline, TurnRequest
from core.rag.sampling import request_fields, resolve
from core.runtime.presets import load_embed_params, load_sampling_presets
from core.schemas import Capabilities

KEY = "AIza" + "GEMcanary" * 4  # built at runtime: never a literal key in the repo
BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


@pytest.fixture(autouse=True)
def _gemini_allowed():  # type: ignore[no-untyped-def]
    net.configure(gemini_enabled=True)
    yield
    net.policy().gemini_enabled = False


def backend(handler, log: Logger) -> GeminiBackend:  # type: ignore[no-untyped-def]
    return GeminiBackend(KEY, "gemini-3.5-flash-lite", BASE, logger=log, transport=httpx.MockTransport(handler))


def test_key_sent_only_in_header_and_never_logged() -> None:
    log = Logger(LogRing(), stdout=False)
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers.get("authorization")
        seen["url"] = str(req.url)
        seen["body"] = req.content.decode()
        return httpx.Response(200, json={"choices": [{"message": {"content": f"echo {KEY}"}}]})

    out = backend(handler, log).complete([{"role": "user", "content": "hi"}], {"max_tokens": 8})
    assert seen["auth"] == f"Bearer {KEY}" and KEY not in seen["url"] and KEY not in seen["body"]
    log.info("chat", "answer", text=out, cfg={"api_key": KEY})
    assert KEY not in to_jsonl(log.ring.query(None))


def test_429_and_5xx_are_friendly() -> None:
    log = Logger(LogRing(), stdout=False)
    b429 = backend(lambda r: httpx.Response(429, json=[{"error": {"code": 429, "message": "quota"}}]), log)
    with pytest.raises(RateLimitedError) as ei:
        list(b429.stream([{"role": "user", "content": "x"}], {}))
    assert "rate limit" in ei.value.user_message and "Traceback" not in ei.value.user_message
    b503 = backend(lambda r: httpx.Response(503, json=[{"error": {"message": "high demand"}}]), log)
    with pytest.raises(BackendError, match="temporarily unavailable"):
        b503.complete([{"role": "user", "content": "x"}], {})


def test_streaming_parse() -> None:
    chunks = [
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 2}},
    ]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    b = backend(lambda r: httpx.Response(200, text=body, headers={"content-type": "text/event-stream"}), Logger(LogRing(), stdout=False))
    deltas = list(b.stream([{"role": "user", "content": "x"}], {}))
    assert "".join(d.content for d in deltas) == "Hello" and deltas[-1].finish == "stop"


def test_egress_blocked_when_fallback_disabled() -> None:
    net.policy().gemini_enabled = False
    with pytest.raises(Exception, match="allowlist"):
        GeminiBackend(KEY, "m", BASE, logger=Logger(LogRing(), stdout=False))


def test_capability_matrix_sends_only_supported_fields() -> None:
    e = resolve(load_sampling_presets(), "lfm3-vl-2b", backend="gemini", overrides={"seed": 5})
    body = request_fields(e, "gemini")
    assert set(body) == {"temperature", "top_p", "max_tokens"}


def test_media_requires_confirmation() -> None:
    pipe = ChatPipeline.__new__(ChatPipeline)
    pipe.params = load_embed_params()
    eff = resolve(load_sampling_presets(), "lfm3-vl-2b", backend="gemini")
    att = Attachment("image", b"\x89PNG\r\n\x1a\n", "image/png")
    req = TurnRequest(sid="s", query="q", history=[], preset=None, effective=eff, attachment=att, backend_name="gemini", allow_media_to_remote=False)
    parts, _ = pipe.query_media_for_generator(req, Capabilities(image=True, audio=True))
    assert parts and all(p["type"] == "text" for p in parts)  # nothing but a note is sent
