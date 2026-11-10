"""Fake llama-server (router mode) for tests and `make run-fake`. Stdlib only.

Same CLI contract as the real router: --host --port --api-key --models-preset --models-max
--no-models-autoload --no-webui. Implements /health, /models, /models/load, /models/unload,
/v1/embeddings, /v1/chat/completions (stream + non-stream), /tokenize.

Environment knobs:
  FAKE_CAPTURE=<path>     append every request body (JSONL) for request-capture tests
  FAKE_MODE=<modes>       comma list: silent_ignore (identical media vectors), crash_on_load,
                          hang_load, slow_load, chat_error
  FAKE_LOAD_MS=<int>      simulated load latency (default 150)
  FAKE_TOKEN_MS=<int>     delay per streamed token (default 15)
  FAKE_MEM_MB=<int>       resident memory of each dummy model child (default 0 = none)
"""

from __future__ import annotations

import argparse
import base64
import configparser
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

DIM = 768
MODES = set(filter(None, os.environ.get("FAKE_MODE", "").split(",")))
LOAD_MS = int(os.environ.get("FAKE_LOAD_MS", "150"))
TOKEN_MS = int(os.environ.get("FAKE_TOKEN_MS", "15"))
MEM_MB = int(os.environ.get("FAKE_MEM_MB", "0"))
CAPTURE = os.environ.get("FAKE_CAPTURE")
_capture_lock = threading.Lock()
WORD = re.compile(r"[a-z0-9]+")


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def text_vector(text: str) -> list[float]:
    """Hashing bag-of-words: shared words -> high cosine (good enough for retrieval tests)."""
    v = [0.0] * DIM
    words = WORD.findall(text.lower())
    for w in words:
        if len(w) < 3 or w in {"the", "and", "task", "search", "result", "query", "title", "none", "text"}:
            continue
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        for k in range(3):
            v[(h >> (k * 10)) % DIM] += 1.0 if (h >> (40 + k)) & 1 else -1.0
    if not any(v):
        v[0] = 1.0
    return _unit(v)


def blob_vector(data: str) -> list[float]:
    rnd = random.Random(hashlib.sha256(data.encode()).hexdigest())
    return _unit([rnd.gauss(0, 1) for _ in range(DIM)])


class State:
    def __init__(self, presets: dict[str, dict[str, str]], api_key: str) -> None:
        self.presets = presets
        self.api_key = api_key
        self.status: dict[str, str] = {k: "unloaded" for k in presets}
        self.children: dict[str, subprocess.Popen[bytes]] = {}
        self.lock = threading.Lock()


STATE: State


def spawn_child(mb: int) -> subprocess.Popen[bytes]:
    code = f"import time\nb=bytearray({mb}*1024*1024)\nfor i in range(0,len(b),4096): b[i]=1\nwhile True: time.sleep(1)"
    return subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # quiet
        pass

    # ------------------------------------------------------------------ helpers
    def _json(self, code: int, body: Any) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _auth(self) -> bool:
        if self.path == "/health":
            return True
        if self.headers.get("Authorization") != f"Bearer {STATE.api_key}":
            self._json(401, {"error": {"message": "Invalid API Key", "code": 401}})
            return False
        return True

    def _body(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        body: dict[str, Any] = json.loads(self.rfile.read(n) or b"{}")
        if CAPTURE:
            with _capture_lock, open(CAPTURE, "a") as fh:
                fh.write(json.dumps({"path": self.path, "body": body}) + "\n")
        return body

    # ------------------------------------------------------------------ routes
    def do_GET(self) -> None:
        if not self._auth():
            return
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        elif self.path.startswith("/models"):
            with STATE.lock:
                data = [
                    {"id": k, "status": {"value": v}, "architecture": {"input_modalities": ["text", "image"] if "mmproj" in STATE.presets[k] else ["text"]}}
                    for k, v in STATE.status.items()
                ]
            self._json(200, {"data": data})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        if not self._auth():
            return
        body = self._body()
        path = self.path.split("?")[0]
        if path == "/models/load":
            self._load(body.get("model", ""))
        elif path == "/models/unload":
            self._unload(body.get("model", ""))
        elif path == "/v1/embeddings":
            self._embeddings(body)
        elif path == "/v1/chat/completions":
            self._chat(body)
        elif path == "/tokenize":
            self._json(200, {"tokens": list(range(len(WORD.findall(str(body.get("content", ""))))))})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def _load(self, model: str) -> None:
        if model not in STATE.presets:
            self._json(400, {"error": {"message": f"model {model} not found"}})
            return
        if "crash_on_load" in MODES:
            with STATE.lock:
                STATE.status[model] = "unloaded"
            self._json(200, {"success": True})
            return

        def work() -> None:
            with STATE.lock:
                STATE.status[model] = "loading"
            if "hang_load" in MODES:
                return
            time.sleep(LOAD_MS / 1000 * (10 if "slow_load" in MODES else 1))
            with STATE.lock:
                if MEM_MB:
                    STATE.children[model] = spawn_child(MEM_MB)
                STATE.status[model] = "loaded"

        threading.Thread(target=work, daemon=True).start()
        self._json(200, {"success": True})

    def _unload(self, model: str) -> None:
        with STATE.lock:
            child = STATE.children.pop(model, None)
            if child:
                child.kill()
                child.wait(5)
            if model in STATE.status:
                STATE.status[model] = "unloaded"
        self._json(200, {"success": True})

    def _require_loaded(self, model: str) -> bool:
        with STATE.lock:
            ok = STATE.status.get(model) == "loaded"
        if not ok:
            self._json(400, {"error": {"message": f"model {model} is not loaded"}})
        return ok

    def _embeddings(self, body: dict[str, Any]) -> None:
        model = body.get("model", "")
        if not self._require_loaded(model):
            return
        inputs = body.get("input", [])
        if isinstance(inputs, str):
            inputs = [inputs]
        out = []
        has_mm = "mmproj" in STATE.presets.get(model, {})
        for i, item in enumerate(inputs):
            if isinstance(item, str):
                vec = text_vector(item)
            elif isinstance(item, dict) and "content" in item:
                parts = item["content"]
                texts = [p.get("text", "") for p in parts if p.get("type") == "text"]
                media = [json.dumps(p, sort_keys=True) for p in parts if p.get("type") != "text"]
                if media and not has_mm:
                    self._json(400, {"error": {"message": "multimodal input requires --mmproj"}})
                    return
                if media and "silent_ignore" in MODES:
                    vec = text_vector(" ".join(texts) or "constant")
                elif media:
                    vec = blob_vector("".join(media))
                    if texts:
                        tv = text_vector(" ".join(texts))
                        vec = _unit([a + b for a, b in zip(vec, tv, strict=True)])
                else:
                    vec = text_vector(" ".join(texts))
            else:
                self._json(400, {"error": {"message": "invalid input element (flat parts are not supported)"}})
                return
            out.append({"object": "embedding", "index": i, "embedding": vec})
        self._json(200, {"object": "list", "data": out, "model": model, "usage": {"prompt_tokens": 0, "total_tokens": 0}})

    def _answer(self, body: dict[str, Any]) -> str:
        msgs = body.get("messages", [])
        user = next((m for m in reversed(msgs) if m.get("role") == "user"), {})
        content = user.get("content", "")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
        system = " ".join(str(m.get("content", "")) for m in msgs if m.get("role") == "system")
        if "[S1]" in system or "[S1]" in content:
            return "According to the indexed data [S1], here is a grounded answer to: " + content.strip()[-160:]
        return "That is not in the indexed data."

    def _chat(self, body: dict[str, Any]) -> None:
        model = body.get("model", "")
        if not self._require_loaded(model):
            return
        if "chat_error" in MODES:
            self._json(500, {"error": {"message": "simulated failure"}})
            return
        text = self._answer(body)
        max_tokens = int(body.get("max_tokens") or 256)
        tokens = re.findall(r"\S+\s*", text)[:max_tokens]
        prompt_n = sum(len(WORD.findall(json.dumps(m.get("content", "")))) for m in body.get("messages", []))
        timings = {
            "prompt_n": prompt_n,
            "prompt_ms": 5.0,
            "predicted_n": len(tokens),
            "predicted_ms": TOKEN_MS * len(tokens),
            "predicted_per_second": 1000.0 / max(TOKEN_MS, 1),
        }
        usage = {"prompt_tokens": prompt_n, "completion_tokens": len(tokens), "total_tokens": prompt_n + len(tokens)}
        if not body.get("stream"):
            self._json(
                200,
                {
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(tokens)}, "finish_reason": "stop"}],
                    "usage": usage,
                    "timings": timings,
                },
            )
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for tok in tokens:
                time.sleep(TOKEN_MS / 1000)
                chunk = {"choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            final = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage, "timings": timings}
            self.wfile.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # client closed the stream (Stop)
        self.close_connection = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--api-key-file", default=None)
    ap.add_argument("--models-preset", required=True)
    ap.add_argument("--models-max", default="3")
    ap.add_argument("--no-models-autoload", action="store_true")
    ap.add_argument("--no-webui", action="store_true")
    args, _ = ap.parse_known_args()
    if args.api_key_file:
        with open(args.api_key_file) as fh:
            args.api_key = fh.read().strip()
    cp = configparser.ConfigParser()
    with open(args.models_preset) as fh:
        cp.read_string("[__root__]\n" + fh.read())
    presets = {s: dict(cp[s]) for s in cp.sections() if s not in ("__root__", "*")}
    global STATE
    STATE = State(presets, args.api_key)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"fake llama-server listening on {args.host}:{args.port} with {len(presets)} presets", flush=True)
    try:
        srv.serve_forever()
    finally:
        for c in STATE.children.values():
            c.kill()


if __name__ == "__main__":
    _ = base64  # keep import for parity with media handling in tests
    main()
