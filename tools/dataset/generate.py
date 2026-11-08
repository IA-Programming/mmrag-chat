"""Deterministic synthetic multimodal dataset generator (task S04 / DR-004).

    python -m tools.dataset.generate --seed 7 --out datasets/synth_v1            # full dataset (+ manifest, checksums, verify)
    python -m tools.dataset.generate --seed 7 --out /tmp/part --only text,images  # partial: chosen modalities + records only
    python -m tools.dataset.generate --verify --out datasets/synth_v1             # check an existing dataset

The same seed and toolchain (see manifest["toolchain"]) give byte-identical files and an identical
checksums.sha256. The world (Saltmere Archipelago) is entirely fictional; see world.py/content.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from tools.dataset import render
from tools.dataset.world import NAME, QUERY_MEDIA_DIR, World, build_world

SCHEMA = 1
MODALITIES = ("text", "images", "audio", "video")
BUDGET_BYTES = 50 * 1024 * 1024
SAMPLE_BUDGET_BYTES = 10 * 1024 * 1024
MAX_IMAGE_BYTES = 150 * 1024
MAX_AUDIO_S = 10.0
MAX_VIDEO_S = 8.0
MAX_VIDEO_BYTES = 400 * 1024
MAX_VIDEO_SIZE = (320, 240)
CHUNK_PARAMS = {"chunk_tokens": 256, "overlap_tokens": 32}  # core.schemas.TextParams defaults
# Which toolchain entries can change the bytes of which output group (used by the determinism tests).
TOOLCHAIN_DEPS = {
    "text": ("python",),
    "records": ("python", "pillow", "matplotlib"),  # slide bboxes come from font metrics
    "images": ("pillow", "matplotlib", "numpy"),
    "audio": ("espeak_ng", "ffmpeg", "ffmpeg_sha256"),
    "video": ("pillow", "ffmpeg", "ffmpeg_sha256", "video_encoder"),
}
RECORD_FILES = ("ground_truth/facts.jsonl", "queries/queries.jsonl", "dialogues/dialogues.jsonl", "probe.json")
SKIP_HASH = ("checksums.sha256",)
GITATTRIBUTES = b"# Keep every byte exactly as generated (checksums.sha256 covers all files): never convert line endings.\n* -text\n"


def _noop(_: str) -> None:
    return None


# --------------------------------------------------------------------------- helpers


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def list_files(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def checksums_text(root: Path) -> str:
    return "".join(f"{sha256((root / rel).read_bytes())}  {rel}\n" for rel in list_files(root) if rel not in SKIP_HASH)


def jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def toolchain() -> dict[str, str]:
    import matplotlib
    import numpy
    import PIL

    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        "matplotlib": matplotlib.__version__,
        "ffmpeg": render.ffmpeg_version(),
        "ffmpeg_sha256": render.ffmpeg_sha256(),
        "video_encoder": render.video_codec(),
        "espeak_ng": render.espeak_version(),
    }


def estimate_chunks(text: str, chunk_tokens: int = 256, overlap_tokens: int = 32) -> int:
    """Mirror of core.embed.preprocess.chunk_text's packing (paragraph-aware word windows), count only."""
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    budget = max(8, int(chunk_tokens / 1.25))
    overlap = min(int(overlap_tokens / 1.25), budget // 2)
    i = n = 0
    while i < len(words):
        j = min(len(words), i + budget)
        if j < len(words):
            for k in range(j, i + (2 * budget) // 3, -1):
                if "\n\n" in text[words[k - 1][1] : words[k][0]]:
                    j = k
                    break
        n += 1
        if j >= len(words):
            break
        i = max(i + 1, j - overlap)
    return n


# --------------------------------------------------------------------------- rendering


def render_files(world: World, sel: tuple[str, ...], log: Callable[[str], None]) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    t0 = time.monotonic()
    if "text" in sel:
        for d in world.docs:
            files[d.path] = d.markdown().encode("utf-8")
        log(f"text: {len(world.docs)} docs ({time.monotonic() - t0:.1f}s)")
    if "images" in sel:
        t = time.monotonic()
        for c in world.charts:
            files[c.path] = render.render_chart(c)
        for s in [*world.slides, *world.query_slides]:
            files[s.path] = render.render_slide(s)
        for p in world.photos:
            files[p.path] = render.render_photo(p)
        log(
            f"images: {len(world.charts)} charts, {len(world.slides)} slides, {len(world.photos)} photos, {len(world.query_slides)} query images ({time.monotonic() - t:.1f}s)"
        )
    if "audio" in sel:
        t = time.monotonic()
        for a in [*world.audios, *world.query_audios]:
            data = render.tts_wav(a.spoken, a.voice, a.speed)
            rate, ch, width, dur = render.wav_info(data)
            if (rate, ch, width) != (16000, 1, 2) or dur > MAX_AUDIO_S:
                raise RuntimeError(f"{a.path}: {rate} Hz, {ch} ch, {8 * width} bit, {dur:.2f} s violates 16 kHz mono PCM16 <= {MAX_AUDIO_S} s")
            files[a.path] = data
        log(
            f"audio: {len(world.audios)} clips + {len(world.query_audios)} spoken queries via espeak-ng {render.espeak_version()} ({time.monotonic() - t:.1f}s)"
        )
    if "video" in sel:
        t = time.monotonic()
        for v in world.videos:
            files[v.path] = render.render_video(v)
        log(f"video: {len(world.videos)} clips via {render.video_codec()} ({time.monotonic() - t:.1f}s)")
    return files


def record_files(world: World) -> dict[str, bytes]:
    return {
        "ground_truth/facts.jsonl": jsonl(f.record() for f in sorted(world.facts, key=lambda f: f.id)),
        "queries/queries.jsonl": jsonl(q.record() for q in world.queries),
        "dialogues/dialogues.jsonl": jsonl(world.dialogues),
        "probe.json": json_bytes(world.probe),
    }


def sample_stats(world: World, files: dict[str, bytes]) -> dict[str, Any]:
    """What the app's "Load sample data" has to embed (frames at the default 1 fps, max 16 per video)."""
    sample = set(world.sample_subset)
    audio_s = sum(render.wav_info(files[p])[3] for p in sample if p.endswith(".wav") and p in files)
    frames = sum(min(16, int(v.duration)) for v in world.videos if v.path in sample)
    chunks = sum(estimate_chunks(d.markdown(), **CHUNK_PARAMS) for d in world.docs if d.path in sample)
    images = sum(1 for p in sample if p.endswith((".png", ".jpg")))
    return {"text_chunks_estimate": chunks, "images": images, "video_frames_at_1fps": frames, "audio_seconds": round(audio_s, 2)}


def build_manifest(world: World, seed: int, files: dict[str, bytes]) -> dict[str, Any]:
    def modality(path: str) -> str:
        return path.split("/")[1] if path.startswith("corpus/") else ("query_media" if path.startswith(QUERY_MEDIA_DIR) else "records")

    sizes = {p: len(b) for p, b in files.items()}
    by_dir: Counter[str] = Counter()
    for p, n in sizes.items():
        by_dir[p.rsplit("/", 1)[0] if "/" in p else "."] += n
    sample = set(world.sample_subset)
    media_paths = {*sample, *(q.attachment for q in world.queries if q.attachment)}
    sample_qs = [q.id for q in world.queries if set(q.expected_sources) <= sample and (q.attachment is None or q.attachment in media_paths)]
    sample_ds = [d["id"] for d in world.dialogues if all(set(t["expected_sources"]) <= sample for t in d["turns"])]
    text = [d.markdown() for d in world.docs]
    return {
        "schema": SCHEMA,
        "name": NAME,
        "seed": seed,
        "generator": "tools/dataset/generate.py",
        "command": f"python -m tools.dataset.generate --seed {seed} --out datasets/{NAME}",
        "description": "Fictional 'Saltmere Archipelago' world: every fact is planted in exactly one item (plus labelled distractors), so retrieval and answers are machine-checkable.",
        "license": "CC0-1.0 (synthetic, generated by this repo)",
        "counts": {
            "text": len(world.docs),
            "image": len(world.charts) + len(world.slides) + len(world.photos),
            "audio": len(world.audios),
            "video": len(world.videos),
            "query_media": len(world.query_slides) + len(world.query_audios),
            "facts": len(world.facts),
            "queries": len(world.queries),
            "dialogues": len(world.dialogues),
            "probe_items": len(world.probe["items"]),
            "probe_queries": len(world.probe["queries"]),
        },
        "counts_detail": {
            "images": {"charts": len(world.charts), "slides": len(world.slides), "photos": len(world.photos)},
            "facts_by_modality": dict(sorted(Counter(f.modality for f in world.facts).items())),
            "queries_by_type": dict(Counter(q.type for q in world.queries)),
            "queries_by_lang": dict(Counter(q.lang for q in world.queries)),
            "text_words": sum(len(t.split()) for t in text),
            "text_chunks_estimate": sum(estimate_chunks(t, **CHUNK_PARAMS) for t in text),
            "text_chunk_params": CHUNK_PARAMS,
        },
        "bytes_total": sum(sizes.values()),
        "bytes_total_scope": "every file except manifest.json and checksums.sha256",
        "bytes_by_dir": dict(sorted(by_dir.items())),
        "layout": {
            "corpus": ["corpus/text", "corpus/images", "corpus/audio", "corpus/video"],
            "query_media": QUERY_MEDIA_DIR,
            "query_media_note": "attachments of image->text / audio->text queries; NOT part of the corpus, do not index",
            "records": list(RECORD_FILES),
        },
        "tts": {
            "engine": "espeak-ng",
            "version": render.espeak_version(),
            "license": "GPL-3.0-or-later (engine only; the generated speech is released under the dataset license)",
            "speech": True,
            "voices": sorted({a.voice for a in [*world.audios, *world.query_audios]}),
            "format": "WAV PCM16, 16000 Hz, mono (resampled with ffmpeg, bitexact)",
        },
        "video_codec": f"{render.video_codec()} (yuv420p, {render.VIDEO_SIZE[0]}x{render.VIDEO_SIZE[1]}, {world.videos[0].fps} fps, mp4, threads=1, bitexact)",
        "image_formats": "PNG (charts: matplotlib Agg, slides: Pillow) and JPEG (photo-like); fonts: DejaVu Sans",
        "toolchain": toolchain(),
        "query_media": [
            *({"path": s.path, "modality": "image", "shows": " / ".join(ln.text for ln in s.lines)} for s in world.query_slides),
            *({"path": a.path, "modality": "audio", "transcript": a.text} for a in world.query_audios),
        ],
        "sample_subset": world.sample_subset,
        "sample_bytes": sum(sizes[p] for p in world.sample_subset if p in sizes),
        "sample_counts": dict(sorted(Counter(modality(p) for p in world.sample_subset).items())),
        "sample_media_stats": sample_stats(world, files),
        "sample_query_ids": sample_qs,
        "sample_dialogue_ids": sample_ds,
        "record_semantics": {
            "answer_keywords": "canonical substrings; every keyword is expected (case-insensitive) in a correct answer",
            "answer_regex": "tolerant Python regex (re.search, re.IGNORECASE) covering separators, decimal commas and aliases; prefer it over keywords",
            "location": "text: chunk_hint = first words of the paragraph holding the fact (+ section); image slides: bbox [x0,y0,x1,y1] of the text; "
            "video: t_start/t_end seconds where the fact is visible (end exclusive); audio and photo-like images: whole item ({})",
            "distractor_ids": "facts about the same entity/family with a different, clearly labelled attribute (edition, year, owner, colour/shape)",
            "near_duplicate": "a *_2019_edition.md doc paraphrases its 2024 sibling with one changed number; queries name the edition",
            "negative": "plausible questions about this world whose answer is in no item: must_refuse=true, expected_sources=[]",
            "fact_ids": "queries and dialogue turns carry the ids of the facts that answer them (empty for negatives)",
        },
    }


# --------------------------------------------------------------------------- main entry points


def _prepare_out(out: Path, partial: bool, force: bool) -> None:
    if out.exists() and any(out.iterdir()):
        is_dataset = (out / "manifest.json").exists() or (out / "checksums.sha256").exists()
        if partial and is_dataset and not force:
            raise SystemExit(f"refusing to overwrite the full dataset in {out} with a partial (--only) build; use another --out or --force")
        if not is_dataset and not force:
            raise SystemExit(f"{out} is not empty and does not look like a generated dataset; use --force to replace it")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def generate(seed: int, out: Path, only: Iterable[str] | None = None, *, force: bool = False, log: Callable[[str], None] = _noop) -> dict[str, Any]:
    """Generate the dataset into `out`. With `only`, render just those modalities (+ records) and skip manifest/checksums."""
    sel = MODALITIES if only is None else tuple(only)
    bad = [m for m in sel if m not in MODALITIES]
    if bad:
        raise ValueError(f"unknown modalities {bad}; choose from {MODALITIES}")
    partial = set(sel) != set(MODALITIES)
    t0 = time.monotonic()
    world = build_world(seed)
    for s in world.slides:
        for f in s.facts:
            f.location = {"bbox": render.slide_bbox(s)}
    files = render_files(world, sel, log)
    files.update(record_files(world))
    files[".gitattributes"] = GITATTRIBUTES
    _prepare_out(out, partial, force)
    manifest: dict[str, Any] | None = None
    if not partial:
        manifest = build_manifest(world, seed, files)
        files["manifest.json"] = json_bytes(manifest)
    for rel, data in sorted(files.items()):
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    if not partial:
        (out / "checksums.sha256").write_text(checksums_text(out), encoding="utf-8", newline="\n")
    elapsed = time.monotonic() - t0
    log(f"wrote {len(files)} files, {sum(len(b) for b in files.values()) / 1e6:.2f} MB to {out} in {elapsed:.1f}s")
    return {"files": sorted(files), "bytes": sum(len(b) for b in files.values()), "seconds": elapsed, "manifest": manifest}


def verify(root: Path) -> list[str]:
    """Check checksums, size budgets and media formats of a generated dataset. Returns a list of problems."""
    problems: list[str] = []
    listed: dict[str, str] = {}
    for line in (root / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        listed[rel] = digest
    actual = [p for p in list_files(root) if p not in SKIP_HASH]
    if sorted(listed) != actual:
        problems.append(f"checksums.sha256 lists {len(listed)} files but {len(actual)} exist: {sorted(set(listed) ^ set(actual))[:10]}")
    total = 0
    for rel in actual:
        data = (root / rel).read_bytes()
        total += len(data)
        if listed.get(rel) not in (None, sha256(data)):
            problems.append(f"{rel}: checksum mismatch")
        if rel.endswith((".png", ".jpg")) and len(data) > MAX_IMAGE_BYTES:
            problems.append(f"{rel}: {len(data)} bytes > {MAX_IMAGE_BYTES}")
        if rel.endswith(".wav"):
            rate, ch, width, dur = render.wav_info(data)
            if (rate, ch, width) != (16000, 1, 2) or dur > MAX_AUDIO_S:
                problems.append(f"{rel}: {rate} Hz / {ch} ch / {8 * width} bit / {dur:.2f} s")
        if rel.endswith(".mp4"):
            if len(data) > MAX_VIDEO_BYTES:
                problems.append(f"{rel}: {len(data)} bytes > {MAX_VIDEO_BYTES}")
            err = render.decode_check(root / rel)
            if err:
                problems.append(f"{rel}: decode error {err}")
            w, h, dur = render.video_probe(root / rel)
            if w > MAX_VIDEO_SIZE[0] or h > MAX_VIDEO_SIZE[1] or dur > MAX_VIDEO_S:
                problems.append(f"{rel}: {w}x{h}, {dur:.2f} s")
    total += (root / "checksums.sha256").stat().st_size
    if total > BUDGET_BYTES:
        problems.append(f"dataset is {total} bytes > {BUDGET_BYTES}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    sample = sum((root / p).stat().st_size for p in manifest["sample_subset"])
    if sample > SAMPLE_BUDGET_BYTES:
        problems.append(f"sample_subset is {sample} bytes > {SAMPLE_BUDGET_BYTES}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.dataset.generate", description=__doc__.split("\n\n")[0] if __doc__ else None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("datasets") / NAME)
    ap.add_argument("--only", default="", help=f"comma-separated subset of {','.join(MODALITIES)} (records are always written; no manifest/checksums)")
    ap.add_argument("--force", action="store_true", help="replace a non-dataset output directory")
    ap.add_argument("--verify", action="store_true", help="only verify an existing dataset in --out")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    log: Callable[[str], None] = _noop if args.quiet else (lambda m: print(m, flush=True))
    if args.verify:
        problems = verify(args.out)
        for p in problems:
            print(f"PROBLEM: {p}", file=sys.stderr)
        log("verify: OK" if not problems else f"verify: {len(problems)} problem(s)")
        return 1 if problems else 0
    only = [m.strip() for m in args.only.split(",") if m.strip()] or None
    result = generate(args.seed, args.out, only, force=args.force, log=log)
    if result["manifest"] is not None:
        problems = verify(args.out)
        for p in problems:
            print(f"PROBLEM: {p}", file=sys.stderr)
        m = result["manifest"]
        log(
            f"counts: {json.dumps(m['counts'])}; bytes_total={m['bytes_total']}; sample={m['sample_bytes']} bytes; verify: {'OK' if not problems else 'FAILED'}"
        )
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
