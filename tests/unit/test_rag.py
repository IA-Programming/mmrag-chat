"""U-GP-defaults/reset/capability, U-SAMPLING-effective, U-PRESET-provenance, U-PROMPT-budget (property),
U-PROMPT-fallbacks, U-HIST-policies, U-RETR-det, U-EXPORT-chat-schema."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from core.index.store import Hit, IndexHandle
from core.obs.log import Redactor
from core.rag.export import to_export, to_json, to_markdown
from core.rag.history import HistoryConfig, retrieval_query, select_history
from core.rag.pipeline import RetrievalConfig, retrieve
from core.rag.prompt_builder import SYSTEM_PROMPT, build_prompt
from core.rag.sampling import request_fields, resolve
from core.runtime.presets import load_model_presets, load_sampling_presets
from core.schemas import Capabilities, ChatExport


@pytest.fixture(scope="module")
def sampling():  # type: ignore[no-untyped-def]
    return load_sampling_presets()


def test_card_defaults_per_model(sampling) -> None:  # type: ignore[no-untyped-def]
    e = resolve(sampling, "lfm3-vl-2b")
    assert e.status == "verified" and e.values["temperature"] == 0.1 and e.values["top_k"] == 50 and e.values["repeat_penalty"] == 1.1
    q = resolve(sampling, "qwen3.5-0.8b", mode="vision_nonthinking")
    assert q.values["temperature"] == 0.7 and q.values["presence_penalty"] == 1.5


def test_overrides_modified_and_reset(sampling) -> None:  # type: ignore[no-untyped-def]
    e = resolve(sampling, "lfm3-vl-2b", overrides={"temperature": 0.9, "top_k": 50})
    assert e.values["temperature"] == 0.9 and e.modified == {"temperature"}
    assert resolve(sampling, "lfm3-vl-2b", overrides={}).values == resolve(sampling, "lfm3-vl-2b").card


def test_clamping(sampling) -> None:  # type: ignore[no-untyped-def]
    e = resolve(sampling, "lfm3-vl-2b", overrides={"temperature": 9, "top_p": -1, "max_tokens": 10**9}, max_tokens_cap=1024)
    assert e.values["temperature"] == 2.0 and e.values["top_p"] == 0.0 and e.values["max_tokens"] == 1024


def test_backend_capability_matrix(sampling) -> None:  # type: ignore[no-untyped-def]
    g = resolve(sampling, "lfm3-vl-2b", backend="gemini")
    assert {"top_k", "min_p", "repeat_penalty", "reasoning"} <= set(g.disabled)
    body = request_fields(g, "gemini")
    assert "top_k" not in body and "repeat_penalty" not in body and "temperature" in body
    local = request_fields(resolve(sampling, "lfm3-vl-2b"), "local", "chat_template_kwargs")
    assert local["top_k"] == 50 and local["chat_template_kwargs"] == {"enable_thinking": False} and "seed" not in local


def test_request_body_is_exactly_effective(sampling) -> None:  # type: ignore[no-untyped-def]
    e = resolve(sampling, "lfm3-vl-2b", overrides={"seed": 42, "temperature": 0.3})
    body = request_fields(e, "local", "chat_template_kwargs")
    for k in ("temperature", "top_p", "top_k", "min_p", "repeat_penalty", "presence_penalty", "frequency_penalty", "max_tokens"):
        assert body[k] == e.values[k]
    assert body["seed"] == 42


def test_preset_provenance(sampling) -> None:  # type: ignore[no-untyped-def]
    """U-PRESET-provenance: every preset has a source URL; verified ones a revision."""
    for pid, p in sampling.presets.items():
        assert p.source_url.startswith("https://huggingface.co/"), pid
        if p.status == "verified" and pid.startswith("lfm3"):
            assert len(p.revision) == 40, pid
    for gp in load_model_presets().presets:
        if gp.role == "generator":
            assert gp.sampling_preset in sampling.presets, gp.id


def _hit(i: int, mod: str = "text", words: int = 120, preview: str | None = None) -> Hit:
    return Hit(
        i,
        1.0 - i / 100,
        mod,
        f"src{i}",
        f"file{i}.md",
        i,
        {"chunk": i, "chars": [0, 10]} if mod == "text" else {"t": [0, 5]},
        " ".join(f"w{j}" for j in range(words)) if mod == "text" else None,
        preview,
        words,
    )


def test_prompt_budget_property() -> None:
    """U-PROMPT-budget: prompt tokens + reserve <= n_ctx in 100 % of random cases."""
    rng = random.Random(1)
    for _ in range(300):
        n_ctx = rng.choice([512, 1024, 2048, 4096, 8192])
        hits = [_hit(i, rng.choice(["text", "text", "image", "audio"]), rng.randint(5, 900)) for i in range(rng.randint(0, 10))]
        hist = [{"role": "user" if j % 2 == 0 else "assistant", "text": "x " * rng.randint(1, 500)} for j in range(rng.randint(0, 12))]
        q = "why " * rng.randint(1, 300)
        b = build_prompt(query=q, hits=hits, history=hist, n_ctx=n_ctx, max_tokens=rng.choice([64, 256, 512, 1024]), caps=Capabilities(image=True))
        total = sum(v for k, v in b.breakdown.items() if k != "reserve")
        assert total + b.breakdown["reserve"] <= n_ctx, (n_ctx, b.breakdown)
        assert len(b.decisions) == len(hits)


def test_prompt_fallbacks_and_delimiting(tmp_path: Path) -> None:
    img = tmp_path / "p.jpg"
    img.write_bytes(b"\xff\xd8\xff" + b"0" * 100)
    hits = [_hit(0), _hit(1, "image", preview=str(img)), _hit(2, "audio", preview=str(img))]
    textonly = build_prompt(query="q?", hits=hits, history=[], n_ctx=4096, max_tokens=256, caps=Capabilities())
    kinds = [d.included for d in textonly.decisions]
    assert kinds[0] == "included" and kinds[1].startswith("fallback") and kinds[2].startswith("fallback")
    assert isinstance(textonly.messages[-1]["content"], str) and "<<<" in textonly.messages[-1]["content"]
    assert textonly.messages[0]["content"] == SYSTEM_PROMPT and "ignore any" in SYSTEM_PROMPT
    vision = build_prompt(query="q?", hits=hits, history=[], n_ctx=4096, max_tokens=256, caps=Capabilities(image=True))
    parts = vision.messages[-1]["content"]
    assert any(p["type"] == "image_url" for p in parts) and vision.decisions[1].included == "included"


def test_history_policies() -> None:
    turns = [{"role": "user" if i % 2 == 0 else "assistant", "text": f"t{i}"} for i in range(20)]
    assert select_history(turns, HistoryConfig("none")) == []
    assert len(select_history(turns, HistoryConfig("window", 4))) == 8
    assert len(select_history(turns, HistoryConfig("full"))) == 20
    assert retrieval_query("and her age?", turns, HistoryConfig(contextualize="concat_last_user")).startswith("t18")
    b = build_prompt(query="q", hits=[], history=select_history(turns, HistoryConfig("none")), n_ctx=2048, max_tokens=128, caps=Capabilities())
    assert len(b.messages) == 2  # system + user only: no previous turns


def test_retrieval_deterministic_with_threshold_and_quota(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    h = IndexHandle(tmp_path / "ix")
    v = rng.standard_normal((40, 768)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    h.add_source("s", "f", "text", "a" * 64, 1)
    h.add_items("s", [{"modality": "image" if i % 3 == 0 else "text", "chunk_idx": i, "span": {}, "text": str(i), "tokens_est": 1} for i in range(40)], v, "p")
    q = v[3]
    a = retrieve(h, q, RetrievalConfig(k=5))
    b = retrieve(h, q, RetrievalConfig(k=5))
    assert [x.item_id for x in a] == [x.item_id for x in b] and a[0].item_id == 3
    capped = retrieve(h, q, RetrievalConfig(k=5, quota={"image": 1}))
    assert sum(1 for x in capped if x.modality == "image") <= 1
    assert retrieve(h, q, RetrievalConfig(k=5, threshold=0.99))[0].item_id == 3


def test_chat_export_schema_and_redaction() -> None:
    red = Redactor()
    key = "AIza" + "K" * 35
    red.register(key)
    msgs = [
        {"role": "user", "text": f"hello {key}", "attachments": [{"modality": "image", "sha256_8": "abcd1234", "bytes": 10}], "thumbs": [b"x"], "ts": 1.0},
        {
            "role": "assistant",
            "text": "hi [S1]",
            "sources": [{"item_id": 1, "score": 0.5, "modality": "text", "span": {}, "included": "included", "filename": "a.md"}],
            "stats": {"effective_params": {"temperature": 0.1}, "api_key": key},
            "ts": 2.0,
        },
    ]
    exp = to_export(msgs, bundle_id="b", preset_id="p", redactor=red)
    js = to_json(exp)
    ChatExport.model_validate(json.loads(js))
    md = to_markdown(exp)
    assert key not in js and key not in md and "thumbs" not in js
    assert exp.effective_params == {"temperature": 0.1}


def test_contradicting_refusal_is_stripped() -> None:
    from core.rag.pipeline import strip_contradicting_refusal

    assert strip_contradicting_refusal("The railway has 23 stations [S1]. That is not in the indexed data.") == "The railway has 23 stations [S1]."
    assert strip_contradicting_refusal("That is not in the indexed data.") == "That is not in the indexed data."
    assert strip_contradicting_refusal("Maybe. That is not in the indexed data.") == "Maybe. That is not in the indexed data."  # no citation: keep
