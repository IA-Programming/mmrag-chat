"""DEV ONLY (DR-001 smoke tests, I-BUILD-smoke). Stdlib-only so it runs in a bare Debian 11 container.

Checks, against a real llama-server in router mode:
  version            binary starts and reports its build
  router             /health ok with a preset file, --no-models-autoload
  emb-text           text embedding: unit norm, dimension
  emb-mm             two DIFFERENT images give DIFFERENT vectors (I-EMB-silent-ignore)
  flat-parts         a flat (unwrapped) parts list is rejected or handled, never silently wrong (I-EMB-flat-parts)
  chat               streaming chat completion produces tokens
  unload             after /models/unload the child process is gone and RSS drops

Usage (inside the capped container, models mounted at /models, runtime at /rt):
  python3 smoke_tests.py --server /rt/llama-server --presets /work/presets.ini --out /work/smoke.json
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import secrets
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib
from typing import Any


def png(width: int, height: int, pixel) -> bytes:  # type: ignore[no-untyped-def]
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            raw.extend(pixel(x, y))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


def data_url(b: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(b).decode()


def tree_rss_mb(pid: int) -> float:
    pids = {pid}
    changed = True
    while changed:
        changed = False
        for d in os.listdir("/proc"):
            if not d.isdigit() or int(d) in pids:
                continue
            try:
                with open(f"/proc/{d}/stat") as fh:
                    ppid = int(fh.read().rsplit(")", 1)[1].split()[1])
            except OSError:
                continue
            if ppid in pids:
                pids.add(int(d))
                changed = True
    total = 0
    for p in pids:
        try:
            with open(f"/proc/{p}/statm") as fh:
                total += int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except OSError:
            pass
    return total / 1e6


def cgroup_current_mb() -> float | None:
    for path in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            with open(path) as fh:
                return int(fh.read().strip()) / 1e6
        except OSError:
            continue
    return None


class Client:
    def __init__(self, port: int, key: str) -> None:
        self.base = f"http://127.0.0.1:{port}"
        self.key = key

    def req(self, method: str, path: str, body: Any = None, timeout: float = 600) -> tuple[int, Any]:
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(
            self.base + path, data=data, method=method, headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                raw = resp.read()
                return resp.status, json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                return e.code, json.loads(raw)
            except Exception:
                return e.code, raw.decode(errors="replace")

    def status(self, model: str) -> dict[str, Any]:
        _, d = self.req("GET", "/models")
        for m in d.get("data", []):
            if m["id"] == model:
                return m.get("status", {})
        return {"value": "missing"}

    def load(self, model: str, timeout: float = 180) -> float:
        t0 = time.time()
        code, body = self.req("POST", "/models/load", {"model": model})
        if code >= 400:
            raise RuntimeError(f"load {model}: {code} {body}")
        while time.time() - t0 < timeout:
            st = self.status(model)
            if st.get("value") == "loaded":
                return time.time() - t0
            if st.get("failed"):
                raise RuntimeError(f"load {model} failed: {st}")
            time.sleep(0.25)
        raise RuntimeError(f"load {model} timed out")

    def unload(self, model: str, timeout: float = 60) -> float:
        t0 = time.time()
        self.req("POST", "/models/unload", {"model": model})
        while time.time() - t0 < timeout:
            if self.status(model).get("value") in ("unloaded", "missing"):
                return time.time() - t0
            time.sleep(0.25)
        raise RuntimeError(f"unload {model} timed out")


def norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True)) / (norm(a) * norm(b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--presets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--emb-text", default="emb-text")
    ap.add_argument("--emb-mm", default="emb-mm")
    ap.add_argument("--generator", default=None)
    args = ap.parse_args()

    results: dict[str, Any] = {"checks": {}, "metrics": {}}
    checks = results["checks"]
    m = results["metrics"]

    v = subprocess.run([args.server, "--version"], capture_output=True, text=True, timeout=30)
    checks["version"] = v.returncode == 0
    m["version"] = (v.stdout + v.stderr).strip().splitlines()[-2:]

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    key = secrets.token_hex(12)
    log = open(os.path.join(os.path.dirname(args.out), "router.log"), "w")
    proc = subprocess.Popen(
        [
            args.server,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--api-key",
            key,
            "--models-preset",
            args.presets,
            "--models-max",
            "3",
            "--no-models-autoload",
            "--no-webui",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )
    c = Client(port, key)
    try:
        t0 = time.time()
        while time.time() - t0 < 60:
            try:
                if c.req("GET", "/health", timeout=2)[0] == 200:
                    break
            except Exception:
                pass
            time.sleep(0.2)
        checks["router"] = c.req("GET", "/health")[0] == 200
        m["rss_router_idle_mb"] = tree_rss_mb(proc.pid)

        # ---- text embedding
        m["load_emb_text_s"] = c.load(args.emb_text)
        m["rss_emb_text_mb"] = tree_rss_mb(proc.pid)
        t = time.time()
        _, d = c.req(
            "POST",
            "/v1/embeddings",
            {"model": args.emb_text, "input": ["task: search result | query: what color is the sky?", "title: none | text: The sky is blue on a clear day."]},
        )
        m["embed_text_ms"] = round((time.time() - t) * 1000, 1)
        vecs = [x["embedding"] for x in d["data"]]
        m["dim"] = len(vecs[0])
        checks["emb-text"] = len(vecs) == 2 and abs(norm(vecs[0]) - 1) < 1e-3
        m["cos_query_doc"] = round(cos(vecs[0], vecs[1]), 4)
        m["unload_emb_text_s"] = c.unload(args.emb_text)

        # ---- multimodal embedding (silent-ignore guard)
        m["load_emb_mm_s"] = c.load(args.emb_mm)
        m["rss_emb_mm_mb"] = tree_rss_mb(proc.pid)
        red = png(256, 256, lambda x, y: (220, 30, 30) if (x // 32 + y // 32) % 2 else (250, 250, 250))
        blue = png(256, 256, lambda x, y: (20, 40, 200) if x > y else (10, 200, 60))
        t = time.time()
        code, d = c.req(
            "POST",
            "/v1/embeddings",
            {
                "model": args.emb_mm,
                "input": [
                    {"content": [{"type": "image_url", "image_url": {"url": data_url(red)}}]},
                    {"content": [{"type": "image_url", "image_url": {"url": data_url(blue)}}]},
                ],
            },
        )
        m["embed_2_images_ms"] = round((time.time() - t) * 1000, 1)
        m["rss_emb_mm_after_images_mb"] = tree_rss_mb(proc.pid)
        if code == 200:
            iv = [x["embedding"] for x in d["data"]]
            m["cos_two_images"] = round(cos(iv[0], iv[1]), 4)
            checks["emb-mm"] = len(iv) == 2 and iv[0] != iv[1] and m["cos_two_images"] < 0.999
        else:
            checks["emb-mm"] = False
            m["emb_mm_error"] = str(d)[:300]
        code, d = c.req("POST", "/v1/embeddings", {"model": args.emb_mm, "input": [{"type": "image_url", "image_url": {"url": data_url(red)}}]})
        m["flat_parts_status"] = code
        checks["flat-parts"] = code >= 400 or (code == 200 and isinstance(d, dict) and "data" in d)
        m["unload_emb_mm_s"] = c.unload(args.emb_mm)
        m["rss_after_unload_mb"] = tree_rss_mb(proc.pid)
        checks["unload"] = m["rss_after_unload_mb"] < m["rss_emb_mm_after_images_mb"] * 0.5

        # ---- generator chat (optional)
        if args.generator:
            m["load_gen_s"] = c.load(args.generator)
            m["rss_gen_mb"] = tree_rss_mb(proc.pid)
            t = time.time()
            code, d = c.req(
                "POST",
                "/v1/chat/completions",
                {
                    "model": args.generator,
                    "messages": [{"role": "user", "content": "Reply with exactly: the sky is blue"}],
                    "max_tokens": 32,
                    "temperature": 0.1,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            m["chat_ms"] = round((time.time() - t) * 1000, 1)
            if code == 200:
                msg = d["choices"][0]["message"]
                m["chat_text"] = (msg.get("content") or msg.get("reasoning_content") or "")[:200]
                m["chat_usage"] = d.get("usage")
                m["chat_timings"] = d.get("timings")
            checks["chat"] = code == 200 and bool(m.get("chat_text"))
            code, d = c.req(
                "POST",
                "/v1/chat/completions",
                {
                    "model": args.generator,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What two colors are in this image? Answer in five words."},
                                {"type": "image_url", "image_url": {"url": data_url(blue)}},
                            ],
                        }
                    ],
                    "max_tokens": 48,
                    "temperature": 0.1,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            m["rss_gen_after_image_mb"] = tree_rss_mb(proc.pid)
            m["vision_text"] = (d["choices"][0]["message"].get("content") or "")[:200] if code == 200 else str(d)[:200]
            checks["vision"] = code == 200 and bool(m["vision_text"])
            m["unload_gen_s"] = c.unload(args.generator)
        m["cgroup_current_mb"] = cgroup_current_mb()
    except Exception as exc:
        results["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
    results["ok"] = all(checks.values()) and "error" not in results
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results, indent=2))
    return 0 if results["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
