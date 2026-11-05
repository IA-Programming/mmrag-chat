"""Integration tests against the REAL llama-server build (make test-int, inside the emulator).

    docker run --rm --memory=2700m --memory-swap=2700m --cpus=2 -v $PWD:/app -w /app \\
      -v ~/.cache/mmrag/models:/models:ro -v build/llama/pkg:/rt:ro mmrag-runtime:bullseye \\
      pytest -m needs_runtime -q

Env: MMRAG_IT_SERVER (default /rt/llama-server), MMRAG_EVAL_MODELS (default /models).
I-EMB-silent-ignore, I-EMB-flat-parts, I-EMB-text-parity, I-GOV-unload-rss, I-GEN-stream,
I-GEN-stop, I-RETR-path, I-SWAP-sequence.
"""

from __future__ import annotations

import io
import os
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import pytest
from PIL import Image, ImageDraw

pytestmark = pytest.mark.needs_runtime

SERVER = Path(os.environ.get("MMRAG_IT_SERVER", "/rt/llama-server"))
if not SERVER.exists():
    pytest.skip("real llama-server not available (set MMRAG_IT_SERVER)", allow_module_level=True)

from core.embed.client import EmbedService, truncate_renorm  # noqa: E402
from core.embed.preprocess import image_part  # noqa: E402
from core.errors import BackendError  # noqa: E402
from core.index.store import IndexHandle  # noqa: E402
from core.rag.history import HistoryConfig  # noqa: E402
from core.rag.pipeline import Attachment, ChatPipeline, RetrievalConfig, TurnRequest  # noqa: E402
from core.rag.sampling import resolve  # noqa: E402
from core.runtime.governor import ALWAYS_RESIDENT, MM_EMBEDDER  # noqa: E402
from core.runtime.presets import load_embed_params, load_model_presets, load_sampling_presets  # noqa: E402
from tools.eval.run import Rig  # noqa: E402

GEN_ID = "lfm3-vl-2b-qad-q4_0"


def jpeg(color: str, shape: str) -> bytes:
    im = Image.new("RGB", (256, 256), "white")
    d = ImageDraw.Draw(im)
    if shape == "circle":
        d.ellipse([40, 40, 216, 216], fill=color)
    else:
        d.rectangle([30, 90, 226, 166], fill=color)
    buf = io.BytesIO()
    im.save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    from tools.eval.run import embedder_presets

    gen = load_model_presets().get(GEN_ID).model_copy(deep=True)
    gen.files = {k: f"{gen.source['repo'].replace('/', '__')}/{gen.source.get(k, v)}" for k, v in gen.files.items()}
    _ = embedder_presets
    r = Rig.start([gen], SERVER, tmp_path_factory.mktemp("rig"))
    yield r
    r.stop()


def test_media_not_silently_ignored(rig) -> None:  # type: ignore[no-untyped-def]
    with rig.governor.turn("it", "t"), rig.governor.using(MM_EMBEDDER):
        v = rig.client.embeddings(MM_EMBEDDER, [{"content": [image_part(jpeg("red", "circle"))]}, {"content": [image_part(jpeg("blue", "bar"))]}])
    a, b = np.array(v[0]), np.array(v[1])
    assert not np.allclose(a, b) and float(a @ b) < 0.999


def test_flat_parts_rejected_clearly(rig) -> None:  # type: ignore[no-untyped-def]
    with rig.governor.turn("it", "t"), rig.governor.using(MM_EMBEDDER), pytest.raises(BackendError) as ei:
        rig.client.embeddings(MM_EMBEDDER, [image_part(jpeg("red", "circle"))])
    assert "rejected" in ei.value.user_message


def test_text_vectors_identical_across_embedder_presets(rig) -> None:  # type: ignore[no-untyped-def]
    text = "title: none | text: The lighthouse has 412 steps."
    with rig.governor.turn("it", "t"):
        with rig.governor.using(ALWAYS_RESIDENT):
            a = np.array(rig.client.embeddings(ALWAYS_RESIDENT, [text])[0])
        with rig.governor.using(MM_EMBEDDER):
            b = np.array(rig.client.embeddings(MM_EMBEDDER, [text])[0])
        rig.governor.unload(MM_EMBEDDER)
    assert float(a @ b) > 0.9999


def test_unload_kills_child_and_frees_rss(rig) -> None:  # type: ignore[no-untyped-def]
    with rig.governor.turn("it", "t"):
        rig.governor.ensure(MM_EMBEDDER)
        rig.governor.release(MM_EMBEDDER)
        children_before = {p.pid for p in psutil.Process(rig.router.proc.pid).children(recursive=True)}
        before = rig.meter.used_mb()
        t0 = time.time()
        rig.governor.unload(MM_EMBEDDER)
    deadline = time.time() + 10
    while time.time() < deadline and {p.pid for p in psutil.Process(rig.router.proc.pid).children(recursive=True)} >= children_before:
        time.sleep(0.2)
    assert time.time() - t0 <= 5 + 5  # stop-timeout + 5 s
    assert rig.meter.used_mb() < before - 500


def _gen_ready(rig) -> None:  # type: ignore[no-untyped-def]
    with rig.governor.turn("it", "gen"):
        rig.governor.ensure(GEN_ID)
        rig.governor.release(GEN_ID)


def test_stream_is_incremental(rig) -> None:  # type: ignore[no-untyped-def]
    _gen_ready(rig)
    chunks = []
    for c in rig.client.chat_stream(
        GEN_ID, [{"role": "user", "content": "Count from one to ten in words."}], {"max_tokens": 40, "chat_template_kwargs": {"enable_thinking": False}}
    ):
        if c.get("choices") and c["choices"][0].get("delta", {}).get("content"):
            chunks.append(time.perf_counter())
    assert len(chunks) >= 5 and chunks[-1] > chunks[0]


def test_stop_ends_decoding_server_side(rig) -> None:  # type: ignore[no-untyped-def]
    _gen_ready(rig)
    cancel = threading.Event()
    got = 0
    for c in rig.client.chat_stream(
        GEN_ID,
        [{"role": "user", "content": "Write a very long story about the sea."}],
        {"max_tokens": 600, "chat_template_kwargs": {"enable_thinking": False}},
        cancel,
    ):
        if c.get("choices") and c["choices"][0].get("delta", {}).get("content"):
            got += 1
            if got == 5:
                cancel.set()  # the generator returns -> HTTP stream closed
    t0 = time.perf_counter()
    out = rig.client.chat(GEN_ID, [{"role": "user", "content": "Reply with: ok"}], {"max_tokens": 4, "chat_template_kwargs": {"enable_thinking": False}})
    # one slot only: if decoding had continued, this request would wait for ~600 tokens (> 30 s)
    assert time.perf_counter() - t0 < 2 + 3 and out["choices"]


def _index(rig) -> IndexHandle:  # type: ignore[no-untyped-def]
    params = load_embed_params()
    emb = EmbedService(rig.client, params)
    docs = ["The lighthouse on Kestrel Island has 412 steps.", "Glimmerfruit harvest reached 9400 crates."]
    with rig.governor.turn("it", "idx"), rig.governor.using(ALWAYS_RESIDENT):
        v = truncate_renorm(emb.embed_pieces([params.prefix.document.format(content=d) for d in docs], ALWAYS_RESIDENT), 768)
    h = IndexHandle(Path(os.environ.get("TMPDIR", "/tmp")) / f"it-index-{time.time_ns()}", fingerprint="it")
    h.add_source("s", "facts.md", "text", "a" * 64, 1)
    h.add_items("s", [{"modality": "text", "chunk_idx": i, "span": {"chunk": i}, "text": d, "tokens_est": 12} for i, d in enumerate(docs)], v, "p")
    return h


def _turn(rig, h: IndexHandle, q: str, att: Attachment | None = None):  # type: ignore[no-untyped-def]
    preset = load_model_presets().get(GEN_ID)
    eff = resolve(load_sampling_presets(), preset.sampling_preset, overrides={"max_tokens": 48, "reasoning": False})
    pipe = ChatPipeline(rig.client, rig.governor, EmbedService(rig.client, load_embed_params()), load_embed_params(), rig.telemetry, rig.log)
    t = pipe.turn(
        TurnRequest(sid="it", query=q, history=[], preset=preset, effective=eff, history_cfg=HistoryConfig(), retrieval=RetrievalConfig(k=2), attachment=att), h
    )
    answer = "".join(t.stream())
    return t, answer


def test_text_query_never_loads_multimodal_embedder(rig) -> None:  # type: ignore[no-untyped-def]
    h = _index(rig)
    n0 = len([e for e in rig.log.ring.query("it") if e.event == "model.load" and e.fields.get("model") == MM_EMBEDDER])
    t, answer = _turn(rig, h, "How many steps does the lighthouse have?")
    n1 = len([e for e in rig.log.ring.query("it") if e.event == "model.load" and e.fields.get("model") == MM_EMBEDDER])
    assert n1 == n0 and t.error is None and "412" in answer


def test_media_query_swap_sequence_restores_generator(rig) -> None:  # type: ignore[no-untyped-def]
    h = _index(rig)
    _gen_ready(rig)
    att = Attachment("image", jpeg("red", "circle"), "image/jpeg", "q.jpg")
    t, _ = _turn(rig, h, "What is shown here?", att)
    steps = [s["step"] for s in t.stats["swap_steps"]]
    assert steps[:3] == ["loading multimodal embedder", "embedding query", "unloading multimodal embedder"]
    assert "restoring generator" in steps and GEN_ID in rig.governor.resident() and MM_EMBEDDER not in rig.governor.resident()
