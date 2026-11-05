"""I-GP-request-capture + U-EMBED-request-capture against the fake llama-server (no models needed):
the request body sent to the server contains exactly the effective values / chosen knobs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from core.embed.client import EmbedService
from core.embed.preprocess import preprocess_image
from core.obs.log import Logger, LogRing
from core.rag.sampling import request_fields, resolve
from core.runtime.presets import load_embed_params, load_model_presets, load_sampling_presets, write_ini
from core.runtime.router import RouterClient, RouterProcess

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def fake(tmp_path: Path):  # type: ignore[no-untyped-def]
    cap = tmp_path / "capture.jsonl"
    os.environ["FAKE_CAPTURE"] = str(cap)
    os.environ["FAKE_TOKEN_MS"] = "1"
    try:
        ini = write_ini(tmp_path / "p.ini", load_model_presets().presets, lambda n: Path("/fake") / n)
        rp = RouterProcess([sys.executable, str(ROOT / "tests/fakes/fake_llama_server.py")], ini, log_dir=tmp_path, logger=Logger(LogRing(), stdout=False))
        rp.start(20)
        c = RouterClient(rp.base_url, rp.api_key, logger=Logger(LogRing(), stdout=False))
        yield c, cap
        c.close()
        rp.stop()
    finally:
        os.environ.pop("FAKE_CAPTURE", None)
        os.environ.pop("FAKE_TOKEN_MS", None)


def bodies(cap: Path, path: str) -> list[dict]:  # type: ignore[type-arg]
    return [json.loads(line)["body"] for line in cap.read_text().splitlines() if json.loads(line)["path"].startswith(path)]


def test_chat_body_equals_effective_params(fake) -> None:  # type: ignore[no-untyped-def]
    c, cap = fake
    gen = "lfm3-vl-2b-qad-q4_0"
    c.load(gen)
    eff = resolve(load_sampling_presets(), "lfm3-vl-2b", overrides={"temperature": 0.35, "top_k": 12, "seed": 9})
    body = request_fields(eff, "local", "chat_template_kwargs")
    list(c.chat_stream(gen, [{"role": "user", "content": "hi"}], body))
    sent = bodies(cap, "/v1/chat/completions")[-1]
    for k, v in body.items():
        assert sent[k] == v, k
    assert sent["temperature"] == 0.35 and sent["top_k"] == 12 and sent["seed"] == 9 and sent["chat_template_kwargs"] == {"enable_thinking": False}


def test_embedding_request_reflects_knobs(fake, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import io

    from PIL import Image

    c, cap = fake
    c.load("emb-mm")
    buf = io.BytesIO()
    Image.new("RGB", (1600, 800), "orange").save(buf, "PNG")
    p = load_embed_params()
    emb = EmbedService(c, p)
    for side in (256, 1024):
        p.image.max_side_px = side
        piece = preprocess_image(buf.getvalue(), p.image)[0]
        emb.embed_pieces([piece.content], "emb-mm")
    sent = bodies(cap, "/v1/embeddings")
    urls = [b["input"][0]["content"][0]["image_url"]["url"] for b in sent[-2:]]
    assert len(urls[0]) < len(urls[1])  # the knob changed what was sent
    q = emb.query_text("where?")
    emb.embed_query("where?", model="emb-mm")
    assert bodies(cap, "/v1/embeddings")[-1]["input"] == [q] and q.startswith("task: search result | query:")
