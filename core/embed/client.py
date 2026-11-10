"""Embedding service: prefixes + Matryoshka truncation are client-side (ADR-011).

Text uses `emb-text` (always resident) unless `emb-mm` is already resident (indexing jobs);
both share the same weights, so text vectors are identical (checked by I-EMB-text-parity).
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Protocol

import numpy as np

from core.errors import BackendError
from core.schemas import EmbedParams


class EmbeddingsBackend(Protocol):
    def embeddings(self, model: str, inputs: list[Any]) -> list[list[float]]: ...


def truncate_renorm(vecs: np.ndarray, dim: int) -> np.ndarray:
    """Matryoshka: keep the first `dim` coordinates and re-normalize to unit L2."""
    v = np.asarray(vecs, dtype=np.float32)
    if v.ndim == 1:
        v = v[None, :]
    v = v[:, :dim]
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    out: np.ndarray = (v / n).astype(np.float32)
    return out


def params_hash(params: EmbedParams) -> str:
    return hashlib.sha256(json.dumps(params.model_dump(by_alias=True), sort_keys=True).encode()).hexdigest()[:12]


class EmbedService:
    def __init__(self, backend: EmbeddingsBackend, params: EmbedParams) -> None:
        self.backend = backend
        self.params = params
        self.last_ms = 0.0

    def query_text(self, query: str) -> str:
        return self.params.prefix.query.format(query=query)

    def _call(self, model: str, inputs: list[Any]) -> np.ndarray:
        t0 = time.perf_counter()
        raw = self.backend.embeddings(model, inputs)
        self.last_ms = (time.perf_counter() - t0) * 1000
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != len(inputs):
            raise BackendError("The embedder returned an unexpected shape.", detail=f"{arr.shape} for {len(inputs)} inputs")
        if not np.isfinite(arr).all():
            raise BackendError("The embedder returned invalid numbers (NaN/inf).")
        return arr

    def embed_query(self, query: str, model: str = "emb-text") -> np.ndarray:
        """Full-dimension (768) unit vector for a text query."""
        return truncate_renorm(self._call(model, [self.query_text(query)]), 10_000)[0]

    def embed_query_media(self, parts: list[dict[str, Any]], text: str | None = None, model: str = "emb-mm") -> np.ndarray:
        content = list(parts)
        if text:
            content.append({"type": "text", "text": self.query_text(text)})
        return truncate_renorm(self._call(model, [{"content": content}]), 10_000)[0]

    def embed_pieces(self, contents: list[Any], model: str) -> np.ndarray:
        """Document pieces: strings (already prefixed) or typed-part lists -> wrapped elements."""
        inputs = [c if isinstance(c, str) else {"content": c} for c in contents]
        return truncate_renorm(self._call(model, inputs), 10_000)


class DuplicateGuard:
    """Runtime guard for the silent-media-ignore failure mode (R-09): different media must
    not produce identical vectors."""

    def __init__(self) -> None:
        self._seen: dict[bytes, str] = {}

    def check(self, vec: np.ndarray, content_key: str) -> None:
        key = np.round(vec[:64], 5).tobytes()
        prev = self._seen.get(key)
        if prev is not None and prev != content_key:
            raise BackendError(
                "The embedder returned identical vectors for different media; the media input is probably being ignored.",
                detail=f"{prev} vs {content_key}",
            )
        self._seen[key] = content_key
