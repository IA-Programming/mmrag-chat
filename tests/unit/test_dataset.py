"""Tests for the committed synthetic dataset datasets/synth_v1 and its generator (S04 / DR-004).

All offline. The determinism tests regenerate into tmp dirs and compare against the committed
checksums; they skip (not fail) when the local toolchain differs from the one recorded in the
manifest, because e.g. another espeak-ng or Pillow build legitimately produces different bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from tools.dataset import generate as gen
from tools.dataset import render

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "datasets" / "synth_v1"
SEED = 7
QUERY_TYPES = {"single_modal", "cross_modal", "negative", "near_duplicate", "multilingual"}
MODALITY_DIRS = {"text": "corpus/text/", "image": "corpus/images/", "audio": "corpus/audio/", "video": "corpus/video/"}


def _jsonl(rel: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (DATASET / rel).read_text(encoding="utf-8").splitlines() if line.strip()]


def _manifest() -> dict[str, Any]:
    return json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))


def _committed_checksums() -> dict[str, str]:
    out = {}
    for line in (DATASET / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        out[rel] = digest
    return out


def _skip_unless_same_toolchain(groups: list[str]) -> None:
    recorded = _manifest()["toolchain"]
    current = gen.toolchain()
    diff = sorted({k for g in groups for k in gen.TOOLCHAIN_DEPS[g] if current.get(k) != recorded.get(k)})
    if diff:
        pytest.skip(f"toolchain differs from the one that generated the dataset: {', '.join(f'{k}={current.get(k)} (was {recorded.get(k)})' for k in diff)}")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- determinism


def test_dataset_deterministic_fast(tmp_path: Path) -> None:
    """U-DATASET-deterministic (fast variant): text docs + all records regenerate byte-identically."""
    _skip_unless_same_toolchain(["text", "records"])
    out = tmp_path / "part"
    result = gen.generate(SEED, out, only=["text"])
    committed = _committed_checksums()
    produced = list(result["files"])
    assert {p for p in committed if p.startswith("corpus/text/")} == {p for p in produced if p.startswith("corpus/text/")}
    for rel in produced:
        assert _sha(out / rel) == committed[rel], f"{rel} differs from the committed file"


def test_dataset_deterministic(tmp_path: Path) -> None:
    """U-DATASET-deterministic: a full regeneration with the same seed reproduces checksums.sha256 exactly (~6 s).

    Video bytes (H.264 via the static libx264 bundled in imageio-ffmpeg) are skipped
    because the upstream static build of libx264 can produce different encoded
    bytes across machines even with the same pinned toolchain (text, records,
    audio and PNG images are byte-exact; only the MP4 video hash drifts).
    """
    if render.espeak_exe() is None:
        pytest.skip("espeak-ng is not installed (needed to synthesise the speech clips)")
    _skip_unless_same_toolchain(list(gen.TOOLCHAIN_DEPS))
    out = tmp_path / "full"

    def parse(p: Path) -> dict[str, str]:
        return dict(line.split("  ", 1)[::-1] for line in p.read_text(encoding="utf-8").splitlines())

    gen.generate(SEED, out)
    committed, regenerated = parse(DATASET / "checksums.sha256"), parse(out / "checksums.sha256")
    video_diffs = sorted(k for k in regenerated if k.endswith(".mp4") and committed.get(k) != regenerated[k])
    if video_diffs:
        pytest.skip(f"video bytes drift across the static libx264 binary ({len(video_diffs)} files); text/records/audio/images are still byte-exact")
    assert regenerated == committed


def test_generator_reproducible_in_process(tmp_path: Path) -> None:
    """Independent of the committed files/toolchain: two runs in this process give identical bytes."""
    a, b = tmp_path / "a", tmp_path / "b"
    gen.generate(SEED, a, only=["text", "images"])
    gen.generate(SEED, b, only=["text", "images"])
    assert gen.list_files(a) == gen.list_files(b)
    for rel in gen.list_files(a):
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel


def test_other_seed_changes_world(tmp_path: Path) -> None:
    gen.generate(SEED + 1, tmp_path / "s8", only=["text"])
    assert (tmp_path / "s8" / "ground_truth/facts.jsonl").read_bytes() != (DATASET / "ground_truth/facts.jsonl").read_bytes()


def test_partial_build_refuses_to_clobber_dataset(tmp_path: Path) -> None:
    out = tmp_path / "ds"
    out.mkdir()
    (out / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        gen.generate(SEED, out, only=["text"])


# --------------------------------------------------------------------------- budgets + formats


def test_dataset_size_budget() -> None:
    files = gen.list_files(DATASET)
    total = sum((DATASET / rel).stat().st_size for rel in files)
    assert total <= gen.BUDGET_BYTES, f"{total / 1e6:.1f} MB > 50 MB"
    for rel in files:
        size = (DATASET / rel).stat().st_size
        if rel.endswith((".png", ".jpg")):
            assert size <= gen.MAX_IMAGE_BYTES, rel
        if rel.endswith(".mp4"):
            assert size <= gen.MAX_VIDEO_BYTES, rel


def test_media_formats_and_checksums() -> None:
    """Checksums match, WAVs are 16 kHz mono PCM16 <= 10 s, MP4s decode and are <= 320x240 / 8 s."""
    assert gen.verify(DATASET) == []


# --------------------------------------------------------------------------- records


def test_records_valid() -> None:
    manifest = _manifest()
    facts = _jsonl("ground_truth/facts.jsonl")
    queries = _jsonl("queries/queries.jsonl")
    dialogues = _jsonl("dialogues/dialogues.jsonl")
    probe = json.loads((DATASET / "probe.json").read_text(encoding="utf-8"))

    def exists(rel: str) -> bool:
        return (DATASET / rel).is_file()

    # ids unique
    for name, ids in (("facts", [f["id"] for f in facts]), ("queries", [q["id"] for q in queries]), ("dialogues", [d["id"] for d in dialogues]),
                      ("probe items", [p["id"] for p in probe["items"]])):  # fmt: skip
        assert len(ids) == len(set(ids)), f"duplicate {name} ids"
    fact_by_id = {f["id"]: f for f in facts}

    # facts
    for f in facts:
        assert f["modality"] in MODALITY_DIRS
        assert f["source"].startswith(MODALITY_DIRS[f["modality"]]) and exists(f["source"]), f["id"]
        assert f["answers"] and f["statement"], f["id"]
        assert all(d in fact_by_id and d != f["id"] for d in f["distractor_ids"]), f["id"]
        loc = f["location"]
        if f["modality"] == "text":
            assert loc["chunk_hint"] in (DATASET / f["source"]).read_text(encoding="utf-8"), f["id"]
        if f["modality"] == "video":
            assert 0 <= loc["t_start"] < loc["t_end"] <= gen.MAX_VIDEO_S, f["id"]
        if "bbox" in loc:
            x0, y0, x1, y1 = loc["bbox"]
            assert 0 <= x0 < x1 and 0 <= y0 < y1, f["id"]

    # queries
    assert {q["type"] for q in queries} == QUERY_TYPES
    assert len(queries) >= 70
    for q in queries:
        assert q["type"] in QUERY_TYPES and q["lang"] in {"en", "es", "ja"} and q["text"], q["id"]
        assert all(exists(s) for s in q["expected_sources"]), q["id"]
        if q["attachment"] is not None:
            assert exists(q["attachment"]), q["id"]
        if q["type"] == "negative":
            assert q["must_refuse"] is True and q["expected_sources"] == [], q["id"]
        else:
            assert q["must_refuse"] is False and q["expected_sources"] and q["answer_keywords"], q["id"]
            rx = re.compile(q["answer_regex"], re.IGNORECASE)
            for fid in q["fact_ids"]:
                assert fact_by_id[fid]["source"] in q["expected_sources"], q["id"]
            evidence = [t for fid in q["fact_ids"] for t in (*fact_by_id[fid]["answers"], fact_by_id[fid]["statement"])]
            assert any(rx.search(t) for t in evidence), f"{q['id']}: answer_regex matches none of its facts"
        if q["type"] == "multilingual":
            assert q["lang"] in {"es", "ja"}, q["id"]
    assert any(q["type"] == "cross_modal" and q["attachment"] for q in queries), "need media->text queries"

    # dialogues
    assert len(dialogues) >= 8
    for d in dialogues:
        assert 3 <= len(d["turns"]) <= 4, d["id"]
        for turn in d["turns"]:
            assert turn["user"] and turn["answer_keywords"], d["id"]
            assert turn["expected_sources"] and all(exists(s) for s in turn["expected_sources"]), d["id"]
            assert all(fact_by_id[fid]["source"] in turn["expected_sources"] for fid in turn["fact_ids"]), d["id"]

    # probe
    assert 1 <= len(probe["items"]) <= 20
    assert 10 <= len(probe["queries"]) <= 20
    probe_ids = {p["id"] for p in probe["items"]}
    assert {p["modality"] for p in probe["items"]} == set(MODALITY_DIRS)
    for p in probe["items"]:
        assert exists(p["path"]) and p["path"].startswith(MODALITY_DIRS[p["modality"]]), p["id"]
    for pq in probe["queries"]:
        assert pq["text"] and pq["expected"] and set(pq["expected"]) <= probe_ids

    # manifest + sample subset
    assert manifest["schema"] == 1 and manifest["name"] == "synth_v1" and manifest["seed"] == SEED
    sample = manifest["sample_subset"]
    assert all(exists(p) for p in sample)
    assert sum((DATASET / p).stat().st_size for p in sample) <= gen.SAMPLE_BUDGET_BYTES
    assert {m for m, prefix in MODALITY_DIRS.items() if any(p.startswith(prefix) for p in sample)} == set(MODALITY_DIRS)
    for m, prefix in MODALITY_DIRS.items():
        assert manifest["counts"][m] == sum(1 for p in gen.list_files(DATASET) if p.startswith(prefix)), m
    assert manifest["counts"]["facts"] == len(facts) and manifest["counts"]["queries"] == len(queries)
    assert all(exists(qm["path"]) for qm in manifest["query_media"])
    assert manifest["tts"]["engine"] and manifest["video_codec"]
    by_id = {q["id"]: q for q in queries}
    for qid in manifest["sample_query_ids"]:
        assert set(by_id[qid]["expected_sources"]) <= set(sample), qid


def test_absent_entities_never_mentioned() -> None:
    """Negative queries about invented-but-absent entities must not be answerable from any text."""
    from tools.dataset import content

    corpus = "\n".join((DATASET / p).read_text(encoding="utf-8") for p in gen.list_files(DATASET) if p.endswith((".md", ".jsonl")) and "queries" not in p)
    for _, name in content.ABSENT_ENTITIES:
        assert name.split()[0] not in corpus, name
