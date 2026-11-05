"""Index store (FR-013): TurboVec (2/4-bit) behind `VectorStore`, SQLite sidecar, float32 vectors.

Invariant (U-STORE-invariant): count(item) == store.count == rows(vectors) at all times.
Item ids == TurboVec slot ids (append-only; `clear` resets everything).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

SCHEMA_VERSION = 1

DDL = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS source(id TEXT PRIMARY KEY, filename TEXT, modality TEXT NOT NULL, sha256 TEXT NOT NULL,
                    bytes INTEGER NOT NULL, duration_s REAL, width INTEGER, height INTEGER, added_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS item(id INTEGER PRIMARY KEY,
                  source_id TEXT NOT NULL REFERENCES source(id),
                  modality TEXT NOT NULL CHECK(modality IN ('text','image','audio','video','mixed')),
                  chunk_idx INTEGER NOT NULL,
                  span TEXT NOT NULL,
                  text TEXT,
                  tokens_est INTEGER NOT NULL,
                  params_hash TEXT NOT NULL,
                  preview_path TEXT);
CREATE INDEX IF NOT EXISTS item_source ON item(source_id);
"""


class VectorStore(Protocol):
    dim: int

    def add(self, vecs: np.ndarray) -> None: ...
    def search(self, q: np.ndarray, k: int, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]: ...
    def count(self) -> int: ...
    def nbytes(self) -> int: ...
    def save(self, path: Path) -> None: ...


class NumpyStore:
    """Brute-force float32 twin (parity tests, emergency fallback)."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.vecs: np.ndarray = np.zeros((0, dim), dtype=np.float32)

    def add(self, vecs: np.ndarray) -> None:
        self.vecs = np.vstack([self.vecs, np.asarray(vecs, dtype=np.float32)])

    def search(self, q: np.ndarray, k: int, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        q = np.atleast_2d(q).astype(np.float32)
        scores = q @ self.vecs.T
        if mask is not None:
            scores[:, ~mask] = -np.inf
        k = min(k, int(mask.sum()) if mask is not None else self.vecs.shape[0])
        if k <= 0:
            return np.zeros((q.shape[0], 0), np.float32), np.zeros((q.shape[0], 0), np.int64)
        idx = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        return np.take_along_axis(scores, idx, 1), idx

    def count(self) -> int:
        return int(self.vecs.shape[0])

    def nbytes(self) -> int:
        return int(self.vecs.nbytes)

    def save(self, path: Path) -> None:
        np.save(path, self.vecs, allow_pickle=False)


class TurboVecStore:
    def __init__(self, dim: int, bits: int) -> None:
        import turbovec

        self.dim = dim
        self.bits = bits
        self.index = turbovec.TurboQuantIndex(dim, bits)

    @classmethod
    def from_bytes(cls, data: bytes, dim: int, bits: int) -> TurboVecStore:
        import turbovec

        obj = cls.__new__(cls)
        obj.dim, obj.bits = dim, bits
        obj.index = turbovec.TurboQuantIndex.from_bytes(data)
        return obj

    def add(self, vecs: np.ndarray) -> None:
        self.index.add(np.ascontiguousarray(vecs, dtype=np.float32))

    def search(self, q: np.ndarray, k: int, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(np.atleast_2d(q), dtype=np.float32)
        n = self.count()
        k = min(k, n if mask is None else int(mask.sum()))
        if k <= 0 or n == 0:
            return np.zeros((q.shape[0], 0), np.float32), np.zeros((q.shape[0], 0), np.int64)
        if mask is not None:
            scores, ids = self.index.search(q, k, mask=np.ascontiguousarray(mask, dtype=bool))
        else:
            scores, ids = self.index.search(q, k)
        return np.asarray(scores, np.float32), np.asarray(ids, np.int64)

    def count(self) -> int:
        return len(self.index)

    def to_bytes(self) -> bytes:
        return bytes(self.index.to_bytes())

    def nbytes(self) -> int:
        return len(self.to_bytes()) if self.count() else 0

    def save(self, path: Path) -> None:
        path.write_bytes(self.to_bytes())


@dataclass
class Hit:
    item_id: int
    score: float
    modality: str
    source_id: str
    filename: str
    chunk_idx: int
    span: dict[str, Any]
    text: str | None
    preview_path: str | None
    tokens_est: int


class IndexHandle:
    """One session's index: TurboVec + sidecar + optional float32 vectors (in a session dir)."""

    def __init__(self, root: Path, *, dim: int = 768, bits: int = 4, keep_float32: bool = True, fingerprint: str = "") -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.bits = bits
        self.keep_float32 = keep_float32
        self.store = TurboVecStore(dim, bits)
        self.vectors: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(str(root / "sidecar.sqlite"), check_same_thread=False)
        self.db.executescript(DDL)
        self.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.build_ms = 0.0
        self.peak_mb = 0.0
        if not self.meta().get("schema_version"):
            self.set_meta(schema_version=str(SCHEMA_VERSION), embedder_fingerprint=fingerprint, dim=str(dim), bits=str(bits), created_at=self.created_at)

    @classmethod
    def open(cls, root: Path) -> IndexHandle:
        """Re-open a persisted index (eval cache / replay). Vectors load with allow_pickle=False."""
        db = sqlite3.connect(str(root / "sidecar.sqlite"))
        meta = dict(db.execute("SELECT key, value FROM meta").fetchall())
        db.close()
        h = cls(
            root, dim=int(meta["dim"]), bits=int(meta["bits"]), keep_float32=(root / "vectors.npy").exists(), fingerprint=meta.get("embedder_fingerprint", "")
        )
        h.store = TurboVecStore.from_bytes((root / "index.tv").read_bytes(), h.dim, h.bits)
        if h.keep_float32:
            h.vectors = np.load(root / "vectors.npy", allow_pickle=False).astype(np.float32)
        return h

    # ------------------------------------------------------------------ meta
    def set_meta(self, **kv: str) -> None:
        with self._lock:
            self.db.executemany("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", list(kv.items()))
            self.db.commit()

    def meta(self) -> dict[str, str]:
        with self._lock:
            return dict(self.db.execute("SELECT key, value FROM meta").fetchall())

    # ------------------------------------------------------------------ writes
    def add_source(
        self,
        source_id: str,
        filename: str,
        modality: str,
        sha256: str,
        nbytes: int,
        *,
        duration_s: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        with self._lock:
            self.db.execute(
                "INSERT OR REPLACE INTO source(id, filename, modality, sha256, bytes, duration_s, width, height, added_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (source_id, filename, modality, sha256, nbytes, duration_s, width, height, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
            )
            self.db.commit()

    def add_items(self, source_id: str, items: list[dict[str, Any]], vecs: np.ndarray, params_hash: str) -> list[int]:
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.shape != (len(items), self.dim):
            raise ValueError(f"vector shape {vecs.shape} != ({len(items)}, {self.dim})")
        with self._lock:
            start = self.store.count()
            ids = list(range(start, start + len(items)))
            rows = [
                (
                    i,
                    source_id,
                    it["modality"],
                    it["chunk_idx"],
                    json.dumps(it["span"]),
                    it.get("text"),
                    int(it.get("tokens_est", 0)),
                    params_hash,
                    it.get("preview_path"),
                )
                for i, it in zip(ids, items, strict=True)
            ]
            try:
                self.db.executemany(
                    "INSERT INTO item(id, source_id, modality, chunk_idx, span, text, tokens_est, params_hash, preview_path) VALUES (?,?,?,?,?,?,?,?,?)", rows
                )
                self.store.add(vecs)
                if self.keep_float32:
                    self.vectors = np.vstack([self.vectors, vecs])
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise
            return ids

    # ------------------------------------------------------------------ reads
    def count(self) -> int:
        return self.store.count()

    def sidecar_count(self) -> int:
        with self._lock:
            return int(self.db.execute("SELECT COUNT(*) FROM item").fetchone()[0])

    def modalities(self) -> np.ndarray:
        with self._lock:
            rows = self.db.execute("SELECT modality FROM item ORDER BY id").fetchall()
        return np.array([r[0] for r in rows], dtype=object)

    def search(self, q: np.ndarray, k: int, *, modalities: set[str] | None = None) -> list[Hit]:
        mask = None
        if modalities is not None:
            mods = self.modalities()
            mask = np.isin(mods, list(modalities))
            if not mask.any():
                return []
        scores, ids = self.store.search(q[: self.dim] if q.shape[-1] != self.dim else q, k, mask)
        return self.hits(ids[0].tolist(), scores[0].tolist())

    def hits(self, ids: list[int], scores: list[float]) -> list[Hit]:
        if not ids:
            return []
        with self._lock:
            qmarks = ",".join("?" * len(ids))
            rows = self.db.execute(
                f"SELECT i.id, i.modality, i.source_id, s.filename, i.chunk_idx, i.span, i.text, i.preview_path, i.tokens_est "  # noqa: S608 - placeholders only
                f"FROM item i JOIN source s ON s.id = i.source_id WHERE i.id IN ({qmarks})",
                ids,
            ).fetchall()
        by_id = {r[0]: r for r in rows}
        out = []
        for i, sc in zip(ids, scores, strict=True):
            r = by_id.get(int(i))
            if r is None:
                continue
            out.append(Hit(int(r[0]), float(sc), r[1], r[2], r[3], int(r[4]), json.loads(r[5]), r[6], r[7], int(r[8])))
        return out

    def item(self, item_id: int) -> Hit | None:
        hs = self.hits([item_id], [1.0])
        return hs[0] if hs else None

    def stats(self) -> dict[str, Any]:
        with self._lock:
            per_mod = dict(self.db.execute("SELECT modality, COUNT(*) FROM item GROUP BY modality").fetchall())
            sources = int(self.db.execute("SELECT COUNT(*) FROM source").fetchone()[0])
            src_bytes = int(self.db.execute("SELECT COALESCE(SUM(bytes),0) FROM source").fetchone()[0])
        idx_bytes = self.store.nbytes()
        n = self.count()
        return {
            "items": n,
            "sidecar_items": self.sidecar_count(),
            "vector_rows": int(self.vectors.shape[0]),
            "sources": sources,
            "per_modality": per_mod,
            "source_bytes": src_bytes,
            "index_bytes": idx_bytes,
            "bytes_per_vector": round(idx_bytes / n, 1) if n else 0,
            "float32_bytes": int(self.vectors.nbytes),
            "dim": self.dim,
            "bits": self.bits,
            "build_ms": round(self.build_ms, 1),
            "peak_mb": round(self.peak_mb, 1),
        }

    def recall_at_k(self, k: int = 10, n_queries: int = 50, seed: int = 0) -> float | None:
        """FR-013 AC3: TurboVec top-k vs float32 brute force, queries sampled from the index itself."""
        n = self.count()
        if not self.keep_float32 or n < 2:
            return None
        rng = np.random.default_rng(seed)
        q_idx = rng.choice(n, size=min(n_queries, n), replace=False)
        qs = self.vectors[q_idx]
        exact = NumpyStore(self.dim)
        exact.add(self.vectors)
        kk = min(k, n)
        _, ids_exact = exact.search(qs, kk)
        _, ids_tv = self.store.search(qs, kk)
        hits = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(ids_exact, ids_tv, strict=True))
        return round(hits / (len(q_idx) * kk), 4)

    def content_hash(self) -> str:
        import hashlib

        h = hashlib.sha256()
        h.update(str(self.count()).encode())
        h.update(self.vectors[:: max(1, len(self.vectors) // 64)].tobytes())
        return h.hexdigest()[:16]

    def save(self) -> None:
        with self._lock:
            self.store.save(self.root / "index.tv")
            if self.keep_float32:
                np.save(self.root / "vectors.npy", self.vectors, allow_pickle=False)
            self.db.commit()

    def close(self) -> None:
        with self._lock:
            self.db.close()
