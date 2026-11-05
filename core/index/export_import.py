"""Index export/import (FR-014): zip with manifest, TurboVec bytes, sidecar, optional vectors.

Import never unpickles (NPZ/NPY with allow_pickle=False), rejects absolute/`..` paths, caps the
uncompressed size, and refuses a mismatched embedder fingerprint, dimension or schema version.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sqlite3
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from core.errors import ValidationError
from core.index.store import SCHEMA_VERSION, IndexHandle, TurboVecStore

MAX_UNCOMPRESSED = 512 * 1024 * 1024
MAX_MEMBERS = 16
ALLOWED_MEMBERS = {"manifest.json", "index.tv", "sidecar.sqlite", "vectors.npy"}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def export_index(h: IndexHandle) -> bytes:
    h.save()
    members: dict[str, bytes] = {"index.tv": h.store.to_bytes()}
    # consistent sqlite snapshot
    buf = io.BytesIO()
    tmp = h.root / "_export.sqlite"
    tmp.unlink(missing_ok=True)
    with sqlite3.connect(str(tmp)) as dst:
        h.db.backup(dst)
    members["sidecar.sqlite"] = tmp.read_bytes()
    tmp.unlink(missing_ok=True)
    if h.keep_float32:
        vb = io.BytesIO()
        np.save(vb, h.vectors, allow_pickle=False)
        members["vectors.npy"] = vb.getvalue()
    meta = h.meta()
    manifest = {
        "schema": SCHEMA_VERSION,
        "embedder_fingerprint": meta.get("embedder_fingerprint", ""),
        "dim": h.dim,
        "bits": h.bits,
        "counts": {"items": h.count(), "sidecar": h.sidecar_count(), "vectors": int(h.vectors.shape[0])},
        "sha256": {k: _sha(v) for k, v in members.items()},
    }
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
        for k, v in members.items():
            z.writestr(k, v)
    return buf.getvalue()


def _check_members(z: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos = z.infolist()
    if len(infos) > MAX_MEMBERS:
        raise ValidationError("The index archive has too many entries.")
    total = 0
    out: dict[str, zipfile.ZipInfo] = {}
    for info in infos:
        name = info.filename
        pp = PurePosixPath(name)
        if pp.is_absolute() or ".." in pp.parts or "\\" in name or name.startswith("/"):
            raise ValidationError(f"The index archive contains an unsafe path: {name!r}.")
        if name not in ALLOWED_MEMBERS:
            raise ValidationError(f"Unexpected entry in the index archive: {name!r}.")
        total += info.file_size
        if total > MAX_UNCOMPRESSED or (info.compress_size and info.file_size / max(info.compress_size, 1) > 200):
            raise ValidationError("The index archive is too large once uncompressed (possible zip bomb).")
        out[name] = info
    for need in ("manifest.json", "index.tv", "sidecar.sqlite"):
        if need not in out:
            raise ValidationError(f"The index archive is missing {need}.")
    return out


def import_index(data: bytes, root: Path, *, expected_fingerprint: str, expected_dim: int) -> IndexHandle:
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValidationError("That file is not a valid index archive.") from exc
    with z:
        _check_members(z)
        manifest: dict[str, Any] = json.loads(z.read("manifest.json"))
        if manifest.get("schema") != SCHEMA_VERSION:
            raise ValidationError(f"Index schema {manifest.get('schema')} is not supported (expected {SCHEMA_VERSION}).")
        if manifest.get("embedder_fingerprint") != expected_fingerprint:
            raise ValidationError("This index was built with a different embedder build or parameters; re-index your files instead.")
        if int(manifest.get("dim", -1)) != expected_dim:
            raise ValidationError(f"Index dimension {manifest.get('dim')} does not match the current {expected_dim}.")
        blobs = {k: z.read(k) for k in ("index.tv", "sidecar.sqlite", "vectors.npy") if k in z.namelist()}
    for k, b in blobs.items():
        if manifest.get("sha256", {}).get(k) != _sha(b):
            raise ValidationError(f"Checksum mismatch for {k} in the index archive.")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    (root / "sidecar.sqlite").write_bytes(blobs["sidecar.sqlite"])
    bits = int(manifest["bits"])
    h = IndexHandle(root, dim=expected_dim, bits=bits, keep_float32="vectors.npy" in blobs, fingerprint=expected_fingerprint)
    h.store = TurboVecStore.from_bytes(blobs["index.tv"], expected_dim, bits)
    if "vectors.npy" in blobs:
        h.vectors = np.load(io.BytesIO(blobs["vectors.npy"]), allow_pickle=False).astype(np.float32)
    if not (h.count() == h.sidecar_count() and (not h.keep_float32 or h.vectors.shape[0] == h.count())):
        raise ValidationError("The index archive is inconsistent (counts differ).")
    return h
