"""Analysis helpers: PCA (FR-050), Matryoshka comparison (FR-040), retrieval metrics."""

from __future__ import annotations

import numpy as np

from core.embed.client import truncate_renorm


def pca_2d(vecs: np.ndarray, max_points: int = 5000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic 2-D PCA via SVD. Returns (coords[n,2], kept_indices, explained_ratio[2])."""
    v = np.asarray(vecs, dtype=np.float32)
    n = v.shape[0]
    idx: np.ndarray = np.arange(n)
    if n > max_points:
        idx = np.sort(np.random.default_rng(seed).choice(n, size=max_points, replace=False))
        v = v[idx]
    if v.shape[0] < 3:
        return np.zeros((v.shape[0], 2), np.float32), idx, np.zeros(2)
    mu = v.mean(axis=0, keepdims=True)
    x = v - mu
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    comps = vt[:2]
    # sign convention: largest-magnitude loading positive (determinism across BLAS builds)
    for i in range(2):
        j = int(np.argmax(np.abs(comps[i])))
        if comps[i, j] < 0:
            comps[i] = -comps[i]
    coords = x @ comps.T
    var = s**2
    ratio = var[:2] / var.sum() if var.sum() else np.zeros(2)
    return coords.astype(np.float32), idx, ratio


def project(vec: np.ndarray, mean: np.ndarray, comps: np.ndarray) -> np.ndarray:
    out: np.ndarray = (np.atleast_2d(vec) - mean) @ comps.T
    return out


def overlap_at_k(a: list[int], b: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(a[:k]) & set(b[:k])) / float(k)


def spearman(a: list[int], b: list[int]) -> float | None:
    """Rank correlation on the items present in both lists (None if < 2 shared)."""
    common = [x for x in a if x in b]
    if len(common) < 2:
        return None
    ra = np.array([a.index(x) for x in common], dtype=float)
    rb = np.array([b.index(x) for x in common], dtype=float)
    ra = ra.argsort().argsort().astype(float)
    rb = rb.argsort().argsort().astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return 1.0
    return float(np.corrcoef(ra, rb)[0, 1])


def topk_exact(vectors: np.ndarray, q: np.ndarray, k: int) -> tuple[list[int], list[float]]:
    scores = vectors @ q
    idx = np.argsort(-scores, kind="stable")[:k]
    return idx.tolist(), scores[idx].tolist()


def matryoshka_compare(vectors: np.ndarray, q: np.ndarray, dims: list[int], k: int) -> dict[int, dict[str, object]]:
    """Results per dimension (truncate + L2 renorm both sides) vs the full-dim reference."""
    full = int(vectors.shape[1])
    ref_ids, _ = topk_exact(truncate_renorm(vectors, full), truncate_renorm(q, full)[0], k)
    out: dict[int, dict[str, object]] = {}
    for d in sorted(set(dims), reverse=True):
        d = min(d, full)
        ids, scores = topk_exact(truncate_renorm(vectors, d), truncate_renorm(q, d)[0], k)
        out[d] = {
            "ids": ids,
            "scores": scores,
            "overlap": overlap_at_k(ref_ids, ids, k),
            "spearman": spearman(ref_ids, ids),
            "bytes_float32": d * 4,
        }
    return out


def turbovec_bytes_per_vector(dim: int, bits: int, n: int = 256, seed: int = 0) -> float:
    """Marginal bytes/vector of a real TurboVec index file (fixed overhead removed)."""
    from core.index.store import TurboVecStore

    rng = np.random.default_rng(seed)
    x = truncate_renorm(rng.standard_normal((n + 1, dim)).astype(np.float32), dim)
    a = TurboVecStore(dim, bits)
    a.add(x[:1])
    b = TurboVecStore(dim, bits)
    b.add(x)
    return (len(b.to_bytes()) - len(a.to_bytes())) / n


def hit_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    return 1.0 if any(r in relevant for r in ranked[:k]) else 0.0


def mrr(ranked: list[int], relevant: set[int]) -> float:
    for i, r in enumerate(ranked, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0
