"""Probe-set runner (FR-041/FR-042): re-embed <= 20 dataset items with given params and report
hit@k, latency and peak RSS. Used by the Playground and by the dev tuning harness."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from core.analysis import hit_at_k, mrr
from core.embed.client import EmbedService, truncate_renorm
from core.embed.preprocess import fit_ubatch, preprocess, sniff
from core.runtime.governor import MM_EMBEDDER, Governor
from core.runtime.telemetry import PeakSampler, Telemetry
from core.schemas import EmbedParams

MAX_PROBE_ITEMS = 20


def load_probe(dataset_dir: Path) -> dict[str, Any] | None:
    p = dataset_dir / "probe.json"
    if not p.exists():
        return None
    data: dict[str, Any] = json.loads(p.read_text())
    data["items"] = data["items"][:MAX_PROBE_ITEMS]
    return data


def run_probe(
    dataset_dir: Path,
    params: EmbedParams,
    *,
    governor: Governor,
    embed: EmbedService,
    telemetry: Telemetry,
    sid: str | None,
    k: int = 5,
    prefix_query: str | None = None,
    prefix_document: str | None = None,
) -> dict[str, Any]:
    probe = load_probe(dataset_dir)
    if probe is None:
        return {"error": "No probe set in this build (datasets/synth_v1/probe.json)."}
    p = params.model_copy(deep=True)
    if prefix_query is not None:
        p.prefix.query = prefix_query
    if prefix_document is not None:
        p.prefix.document = prefix_document
    svc = EmbedService(embed.backend, p)
    t0 = time.perf_counter()
    owners: list[str] = []
    vecs: list[np.ndarray] = []
    with PeakSampler(telemetry) as peak, governor.turn(sid, "probe"), governor.using(MM_EMBEDDER, sid=sid):
        for it in probe["items"]:
            path = dataset_dir / it["path"]
            kind = sniff(path.read_bytes()[:4096] if it.get("modality") != "text" else path.read_bytes())
            if kind is None:
                continue
            pieces = [q for piece in preprocess(path, kind, p) for q in fit_ubatch(piece, p.server.ubatch)]
            v = svc.embed_pieces([x.content for x in pieces], MM_EMBEDDER)
            vecs.append(truncate_renorm(v, p.dims.index_dim))
            owners += [it["id"]] * len(pieces)
        embed_ms = (time.perf_counter() - t0) * 1000
        mat = np.vstack(vecs) if vecs else np.zeros((0, p.dims.index_dim), np.float32)
        hits, mrrs, q_ms = [], [], []
        for q in probe["queries"]:
            t1 = time.perf_counter()
            qv = svc.embed_query(q["text"], model=MM_EMBEDDER)[: p.dims.index_dim]
            qv = qv / (np.linalg.norm(qv) or 1.0)
            q_ms.append((time.perf_counter() - t1) * 1000)
            order = np.argsort(-(mat @ qv), kind="stable")
            ranked: list[str] = []
            for i in order:
                if owners[i] not in ranked:
                    ranked.append(owners[i])
            rel = set(q["expected"])
            ids = {name: n for n, name in enumerate(dict.fromkeys(owners))}
            hits.append(hit_at_k([ids[r] for r in ranked], {ids[r] for r in rel if r in ids}, k))
            mrrs.append(mrr([ids[r] for r in ranked], {ids[r] for r in rel if r in ids}))
    governor.unload(MM_EMBEDDER, sid=sid)
    return {
        "items": len(probe["items"]),
        "pieces": int(mat.shape[0]),
        "queries": len(probe["queries"]),
        f"hit@{k}": round(float(np.mean(hits)) if hits else 0.0, 3),
        "mrr": round(float(np.mean(mrrs)) if mrrs else 0.0, 3),
        "embed_ms": round(embed_ms, 1),
        "query_ms_p50": round(float(np.median(q_ms)) if q_ms else 0.0, 1),
        "peak_mb": round(peak.peak_mb, 1),
    }


def scratch_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="mmrag-probe-"))
