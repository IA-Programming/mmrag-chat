"""Generation backends (ADR-007): local llama-server and Gemini (OpenAI-compatible), one shape.

Each yields `Delta`s while streaming; closing the iterator closes the HTTP stream (Stop)."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from core import net
from core.errors import BackendError, RateLimitedError
from core.obs.log import Logger, get_logger
from core.runtime.router import RouterClient


@dataclass
class Delta:
    content: str = ""
    reasoning: str = ""
    usage: dict[str, Any] | None = None
    timings: dict[str, Any] | None = None
    finish: str | None = None


@dataclass
class StreamStats:
    ttft_ms: float | None = None
    total_ms: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, Any] = field(default_factory=dict)
    finish: str | None = None
    chunks: int = 0


class Backend(Protocol):
    name: str
    model: str

    def stream(self, messages: list[dict[str, Any]], body: dict[str, Any], cancel: threading.Event | None = None) -> Iterator[Delta]: ...
    def complete(self, messages: list[dict[str, Any]], body: dict[str, Any]) -> str: ...


def _delta(chunk: dict[str, Any]) -> Delta:
    d = Delta(usage=chunk.get("usage"), timings=chunk.get("timings"))
    for ch in chunk.get("choices") or []:
        delta = ch.get("delta") or {}
        d.content += delta.get("content") or ""
        d.reasoning += delta.get("reasoning_content") or ""
        if ch.get("finish_reason"):
            d.finish = ch["finish_reason"]
    return d


class LocalBackend:
    name = "local"

    def __init__(self, client: RouterClient, model: str) -> None:
        self.client = client
        self.model = model

    def stream(self, messages: list[dict[str, Any]], body: dict[str, Any], cancel: threading.Event | None = None) -> Iterator[Delta]:
        for chunk in self.client.chat_stream(self.model, messages, body, cancel):
            yield _delta(chunk)

    def complete(self, messages: list[dict[str, Any]], body: dict[str, Any]) -> str:
        out = self.client.chat(self.model, messages, body)
        msg = out["choices"][0]["message"]
        return str(msg.get("content") or "")


class GeminiBackend:
    """OpenAI-compatible Gemini endpoint. The key lives only in memory and is redacted from logs."""

    name = "gemini"

    def __init__(self, api_key: str, model: str, base_url: str, logger: Logger | None = None, timeout: float = 120.0) -> None:
        self.model = model
        self.log = logger or get_logger()
        self.log.redactor.register(api_key)
        self._http = net.client(base_url=base_url, timeout=httpx.Timeout(timeout, connect=10.0), headers={"Authorization": f"Bearer {api_key}"})

    def close(self) -> None:
        self._http.close()

    @staticmethod
    def _error(r: httpx.Response) -> BackendError:
        try:
            body = r.json()
            if isinstance(body, list):
                body = body[0]
            msg = (body.get("error") or {}).get("message", "") if isinstance(body, dict) else ""
        except Exception:
            msg = r.text[:300]
        if r.status_code == 429:
            return RateLimitedError(
                "Gemini's free-tier rate limit was reached. Wait a minute (or until the daily quota resets) and try again.",
                detail=f"HTTP 429: {msg[:300]}",
            )
        if r.status_code in (401, 403):
            return BackendError("Gemini rejected the API key (check that it is valid and enabled).", detail=f"HTTP {r.status_code}")
        if r.status_code >= 500:
            return BackendError("Gemini is temporarily unavailable. Please try again shortly.", detail=f"HTTP {r.status_code}: {msg[:200]}")
        return BackendError("Gemini rejected the request.", detail=f"HTTP {r.status_code}: {msg[:300]}")

    def stream(self, messages: list[dict[str, Any]], body: dict[str, Any], cancel: threading.Event | None = None) -> Iterator[Delta]:
        payload = {"model": self.model, "messages": messages, "stream": True, "stream_options": {"include_usage": True}, **body}
        try:
            with self._http.stream("POST", "chat/completions", json=payload) as r:
                if r.status_code >= 400:
                    r.read()
                    raise self._error(r)
                for line in r.iter_lines():
                    if cancel is not None and cancel.is_set():
                        break
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        yield _delta(json.loads(data))
                    except json.JSONDecodeError:
                        continue
        except httpx.HTTPError as exc:
            raise BackendError("Could not reach Gemini.", detail=type(exc).__name__) from exc

    def complete(self, messages: list[dict[str, Any]], body: dict[str, Any]) -> str:
        try:
            r = self._http.post("chat/completions", json={"model": self.model, "messages": messages, **body})
        except httpx.HTTPError as exc:
            raise BackendError("Could not reach Gemini.", detail=type(exc).__name__) from exc
        if r.status_code >= 400:
            raise self._error(r)
        return str(r.json()["choices"][0]["message"].get("content") or "")

    def test_connection(self) -> float:
        t0 = time.perf_counter()
        self.complete([{"role": "user", "content": "Reply with: ok"}], {"max_tokens": 8})
        return (time.perf_counter() - t0) * 1000
