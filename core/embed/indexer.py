"""Indexing pipeline (FR-012): load embedder -> per item (sniff, preprocess, embed, normalize,
add) -> persist -> unload emb-mm. Step-level progress; one bad item never aborts the batch."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.embed.client import DuplicateGuard, EmbedService, params_hash, truncate_renorm
from core.embed.preprocess import Piece, Sniffed, fit_ubatch, preprocess, sha256_8, validate_upload
from core.errors import AppError, CancelledError
from core.index.store import IndexHandle
from core.jobs import Job
from core.obs.log import Logger
from core.runtime.governor import MM_EMBEDDER, Governor
from core.runtime.telemetry import PeakSampler, Telemetry
from core.schemas import EmbedParams

BATCH_TEXT = 8


@dataclass
class UploadItem:
    name: str
    path: Path  # randomized name inside the session dir
    kind: Sniffed
    nbytes: int
    sha256: str


def stage_upload(session_dir: Path, name: str, data: bytes, *, used_bytes: int, quota_mb: int, max_pixels: int) -> UploadItem:
    """Validate (magic bytes, size, pixels) and write under a random name (FR-010 AC4)."""
    kind = validate_upload(name, data, used_bytes=used_bytes, session_quota_mb=quota_mb, max_pixels=max_pixels)
    up = session_dir / "uploads"
    up.mkdir(parents=True, exist_ok=True)
    path = up / (uuid.uuid4().hex + kind.ext)
    path.write_bytes(data)
    return UploadItem(name=Path(name).name[:200], path=path, kind=kind, nbytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def run_index_job(
    job: Job,
    *,
    items: list[UploadItem],
    index: IndexHandle,
    governor: Governor,
    embed: EmbedService,
    params: EmbedParams,
    telemetry: Telemetry,
    log: Logger,
    preview_dir: Path,
) -> dict[str, Any]:
    sid = job.sid
    ph = params_hash(params)
    preview_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"ok": [], "skipped": [], "failed": []}
    job.total = len(items)
    t_job = time.perf_counter()

    def on_wait(pos: int) -> None:
        job.state, job.queue_position = "waiting", pos

    with PeakSampler(telemetry) as peak, governor.turn(sid, "index", on_wait=on_wait) as _ticket:
        job.state, job.queue_position = "running", 0
        job.step("load embedder")
        t0 = time.perf_counter()
        with governor.using(MM_EMBEDDER, sid=sid, progress=lambda m: job.step("load embedder", detail=m)):
            job.step("load embedder", "done", dur_ms=(time.perf_counter() - t0) * 1000)
            guard = DuplicateGuard()
            for n, it in enumerate(items, start=1):
                label = f"item {n}/{len(items)} {it.name}"
                try:
                    job.check_cancel()
                except CancelledError:
                    job.step(label, "skipped", "cancelled")
                    summary["skipped"].append({"name": it.name, "reason": "cancelled"})
                    for rest in items[n:]:
                        summary["skipped"].append({"name": rest.name, "reason": "cancelled"})
                    break
                t_item = time.perf_counter()
                job.step(label, "running", "preprocess")
                try:
                    with log.span("index", "item", sid=sid, modality=it.kind.modality, sha256_8=it.sha256[:8], bytes=it.nbytes) as extra:
                        pieces: list[Piece] = []
                        for p in preprocess(it.path, it.kind, params):
                            pieces.extend(fit_ubatch(p, params.server.ubatch))
                        if not pieces:
                            raise AppError(f"'{it.name}' produced nothing to index (empty file?).")
                        job.step(label, "running", f"embed {len(pieces)} piece(s)")
                        vecs = _embed_pieces(embed, pieces, guard, it)
                        vecs = truncate_renorm(vecs, params.dims.index_dim)
                        rows = []
                        for i, p in enumerate(pieces):
                            prev = None
                            if p.preview:
                                prev_path = preview_dir / f"{uuid.uuid4().hex}{p.preview_ext}"
                                prev_path.write_bytes(p.preview)
                                prev = str(prev_path)
                            rows.append(
                                {"modality": p.modality, "chunk_idx": i, "span": p.span, "text": p.text, "tokens_est": p.tokens_est, "preview_path": prev}
                            )
                        meta = pieces[0].meta
                        source_id = it.sha256[:16]
                        index.add_source(
                            source_id,
                            it.name,
                            it.kind.modality,
                            it.sha256,
                            it.nbytes,
                            duration_s=meta.get("duration_s"),
                            width=meta.get("width"),
                            height=meta.get("height"),
                        )
                        index.add_items(source_id, rows, vecs, ph)
                        extra["pieces"] = len(pieces)
                    dur = (time.perf_counter() - t_item) * 1000
                    job.step(label, "done", f"{len(pieces)} piece(s)", dur_ms=dur)
                    summary["ok"].append({"name": it.name, "pieces": len(pieces), "ms": round(dur)})
                except AppError as exc:
                    job.step(label, "failed", exc.user_message)
                    summary["failed"].append({"name": it.name, "reason": exc.user_message})
                    log.warning("index", "item.failed", sid=sid, name_sha=sha256_8(it.name.encode()), reason=exc.user_message, detail=exc.detail)
                except Exception as exc:  # one failing item does not abort the batch (AC3)
                    job.step(label, "failed", "unexpected error")
                    summary["failed"].append({"name": it.name, "reason": f"unexpected error ({type(exc).__name__})"})
                    log.error("index", "item.crashed", sid=sid, error=f"{type(exc).__name__}: {exc}")
                job.progress(n, len(items), label)
            job.step("persist", "running")
            t0 = time.perf_counter()
            index.save()
            job.step("persist", "done", dur_ms=(time.perf_counter() - t0) * 1000)
        job.step("unload multimodal embedder", "running")
        t0 = time.perf_counter()
        governor.unload(MM_EMBEDDER, sid=sid)
        job.step("unload multimodal embedder", "done", dur_ms=(time.perf_counter() - t0) * 1000)
    index.build_ms += (time.perf_counter() - t_job) * 1000
    index.peak_mb = max(index.peak_mb, peak.peak_mb)
    summary["peak_mb"] = round(peak.peak_mb, 1)
    summary["index"] = index.stats()
    return summary


def _embed_pieces(embed: EmbedService, pieces: list[Piece], guard: DuplicateGuard, it: UploadItem) -> np.ndarray:
    out = []
    if pieces[0].modality == "text":
        for i in range(0, len(pieces), BATCH_TEXT):
            out.append(embed.embed_pieces([p.content for p in pieces[i : i + BATCH_TEXT]], MM_EMBEDDER))
    else:
        for p in pieces:
            v = embed.embed_pieces([p.content], MM_EMBEDDER)
            # key on the media bytes: identical frames may share a vector, different media may not
            guard.check(v[0], hashlib.sha256(json.dumps(p.content, sort_keys=True).encode()).hexdigest()[:16])
            out.append(v)
    return np.vstack(out)
