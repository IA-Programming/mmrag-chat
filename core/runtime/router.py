"""llama-server router: child process lifecycle + HTTP client (ADR-003, ADR-007).

* Binds 127.0.0.1 on a random port with a per-boot API key.
* The child gets a sanitized environment (no secrets are inherited).
* Explicit /models/load and /models/unload; we never rely on idle sleep to free memory.
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import threading
import time
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

from core import net
from core.errors import BackendError, RuntimeNotReadyError
from core.obs.log import Logger, get_logger

SECRET_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def sanitized_env(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    for k in list(env):
        if any(m in k.upper() for m in SECRET_ENV_MARKERS) or k.upper().startswith("LLAMA_"):
            env.pop(k)
    return env


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class RouterProcess:
    def __init__(
        self,
        argv_prefix: list[str],
        preset_ini: Path,
        *,
        log_dir: Path,
        logger: Logger | None = None,
        models_max: int = 3,
        extra_args: list[str] | None = None,
    ) -> None:
        self.argv_prefix = argv_prefix
        self.preset_ini = preset_ini
        self.log_dir = log_dir
        self.log = logger or get_logger()
        self.models_max = models_max
        self.extra_args = extra_args or []
        self.port = 0
        self.api_key = ""
        self.proc: subprocess.Popen[bytes] | None = None
        self.tail: deque[str] = deque(maxlen=400)
        self._reader: threading.Thread | None = None

    @property
    def key_file(self) -> Path:
        return self.log_dir / ".api-key"

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def argv(self) -> list[str]:
        return [
            *self.argv_prefix,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--api-key-file",
            str(self.key_file),
            "--models-preset",
            str(self.preset_ini),
            "--models-max",
            str(self.models_max),
            "--no-models-autoload",
            "--no-webui",
            *self.extra_args,
        ]

    def start(self, timeout: float = 60.0) -> None:
        self.port = free_port()
        self.api_key = secrets.token_hex(16)
        self.log.redactor.register(self.api_key)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # per-boot key in a 0600 file (not in argv, which any local process can read)
        fd = os.open(self.key_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(self.api_key + "\n")
        argv = self.argv()
        with self.log.span("runtime", "router.start", port=self.port, argv=argv):
            self.proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=sanitized_env(),
                cwd=str(self.log_dir),
                start_new_session=True,
            )
            self._reader = threading.Thread(target=self._read_output, name="router-log", daemon=True)
            self._reader.start()
            self.wait_healthy(timeout)

    def _read_output(self) -> None:
        assert self.proc and self.proc.stdout
        for raw in iter(self.proc.stdout.readline, b""):
            line = raw.decode(errors="replace").rstrip()
            self.tail.append(line)
            if any(t in line for t in ("error", "ERROR", "failed", "warning")):
                self.log.debug("llama", "server.output", line=line[:400])

    def wait_healthy(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        with net.client(base_url=self.base_url, timeout=2.0) as http:
            while time.monotonic() < deadline:
                if self.proc and self.proc.poll() is not None:
                    raise RuntimeNotReadyError(
                        "The local model server exited during start-up.",
                        detail="\n".join(list(self.tail)[-20:]),
                    )
                try:
                    r = http.get("/health")
                    if r.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(0.2)
        raise RuntimeNotReadyError("The local model server did not become healthy in time.", detail="\n".join(list(self.tail)[-20:]))

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self, timeout: float = 10.0) -> None:
        if not self.proc:
            return
        with self.log.span("runtime", "router.stop"):
            self.proc.terminate()
            try:
                self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(5)
        self.log.redactor.unregister(self.api_key)


class RouterClient:
    """Thin client over the router HTTP API. Model ids are preset ids."""

    def __init__(self, base_url: str, api_key: str, *, logger: Logger | None = None) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.log = logger or get_logger()
        self._http = net.client(
            base_url=base_url,
            timeout=httpx.Timeout(600.0, connect=5.0),
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------------------ errors
    @staticmethod
    def _raise(r: httpx.Response, what: str) -> None:
        if r.status_code >= 400:
            try:
                body = r.json()
                msg = body.get("error", {}).get("message") if isinstance(body.get("error"), dict) else body.get("error")
            except Exception:
                msg = r.text[:300]
            raise BackendError(f"The local model server rejected the {what} request.", detail=f"HTTP {r.status_code}: {msg}")

    # ------------------------------------------------------------------ models
    def models(self) -> list[dict[str, Any]]:
        r = self._http.get("/models")
        self._raise(r, "models")
        data: list[dict[str, Any]] = r.json().get("data", [])
        return data

    def status(self, model: str) -> dict[str, Any]:
        for m in self.models():
            if m.get("id") == model:
                st: dict[str, Any] = m.get("status", {})
                return st
        return {"value": "missing"}

    def load(self, model: str, timeout: float = 120.0) -> float:
        t0 = time.perf_counter()
        r = self._http.post("/models/load", json={"model": model})
        self._raise(r, "load")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            st = self.status(model)
            if st.get("value") == "loaded":
                return (time.perf_counter() - t0) * 1000
            if st.get("failed") or (st.get("value") == "unloaded" and st.get("exit_code") is not None):
                raise BackendError(f"The model '{model}' failed to load.", detail=json.dumps(st)[:500])
            time.sleep(0.2)
        raise BackendError(f"Loading '{model}' timed out.", detail=f"timeout={timeout}s")

    def unload(self, model: str, timeout: float = 30.0) -> float:
        t0 = time.perf_counter()
        r = self._http.post("/models/unload", json={"model": model})
        if r.status_code >= 400 and self.status(model).get("value") in ("unloaded", "missing"):
            return 0.0
        self._raise(r, "unload")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.status(model).get("value") in ("unloaded", "missing"):
                return (time.perf_counter() - t0) * 1000
            time.sleep(0.2)
        raise BackendError(f"Unloading '{model}' timed out.")

    def loaded(self) -> list[str]:
        return [m["id"] for m in self.models() if m.get("status", {}).get("value") in ("loaded", "loading", "sleeping")]

    # ------------------------------------------------------------------ inference
    def embeddings(self, model: str, inputs: list[Any]) -> list[list[float]]:
        r = self._http.post("/v1/embeddings", json={"model": model, "input": inputs, "encoding_format": "float"})
        self._raise(r, "embedding")
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]

    def tokenize_count(self, model: str, text: str) -> int:
        r = self._http.post("/tokenize", json={"model": model, "content": text})
        self._raise(r, "tokenize")
        return len(r.json().get("tokens", []))

    def chat(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
        r = self._http.post("/v1/chat/completions", json={"model": model, "messages": messages, "stream": False, **params})
        self._raise(r, "chat")
        out: dict[str, Any] = r.json()
        return out

    def chat_stream(
        self, model: str, messages: list[dict[str, Any]], params: dict[str, Any], cancel: threading.Event | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yield parsed SSE chunks. Closing the iterator (or setting `cancel`) closes the HTTP
        stream, which makes llama-server stop decoding (FR-023 AC2)."""
        body = {"model": model, "messages": messages, "stream": True, "stream_options": {"include_usage": True}, **params}
        with self._http.stream("POST", "/v1/chat/completions", json=body) as r:
            if r.status_code >= 400:
                r.read()
                self._raise(r, "chat")
            for line in r.iter_lines():
                if cancel is not None and cancel.is_set():
                    break
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    continue
