"""U-BUNDLE-verify (hash/size), retry-once then visible error, warm cache (no re-download), safe tar."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from core.bundle.manager import BundleError, BundleManager
from core.obs.log import Logger, LogRing


def _manifest(tmp: Path, assets: dict[str, bytes], runtime: bytes | None = None) -> Path:
    entries = []
    for name, data in assets.items():
        entries.append({"name": name, "kind": "model", "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)})
    if runtime is not None:
        entries.append(
            {"name": "rt.tar.gz", "kind": "runtime", "sha256": hashlib.sha256(runtime).hexdigest(), "size_bytes": len(runtime), "extract_to": "runtime/"}
        )
    m = {
        "schema": 1,
        "bundle_id": "test-b1",
        "base_url": "https://github.com/o/r/releases/download/t/",
        "llama_cpp": {"base": "b0", "commit": "0" * 40, "target": "debian11", "max_glibc_required": "2.31"},
        "assets": entries,
    }
    p = tmp / "manifest.json"
    p.write_text(json.dumps(m))
    return p


def _runtime_tar(extra: dict[str, bytes] | None = None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in {"pkg/llama-server": b"#!/bin/sh\necho hi\n", **(extra or {})}.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def _mgr(tmp: Path, manifest: Path, src: Path) -> BundleManager:
    return BundleManager(manifest, tmp / "cache", source_dir=src, logger=Logger(LogRing(), stdout=False))


def test_verify_extract_and_warm_start(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    assets = {"a.gguf": b"A" * 1000, "b.gguf": b"B" * 2000}
    rt = _runtime_tar()
    for k, v in {**assets, "rt.tar.gz": rt}.items():
        (src / k).write_bytes(v)
    man = _manifest(tmp_path, assets, rt)
    m = _mgr(tmp_path, man, src)
    m.ensure()
    assert m.all_cached() and m.llama_server.exists() and m.llama_server.stat().st_mode & 0o100
    # AC4: second run with intact cache performs no copies/downloads
    for k in list(assets) + ["rt.tar.gz"]:
        (src / k).unlink()
    m2 = _mgr(tmp_path, man, src)
    m2.ensure()
    assert all(a["state"] == "cached" for a in m2.state.snapshot()["assets"])  # type: ignore[union-attr]


def test_corrupt_asset_rejected_after_retry(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    good = b"G" * 512
    man = _manifest(tmp_path, {"m.gguf": good})
    (src / "m.gguf").write_bytes(b"X" * 512)  # same size, wrong content
    m = _mgr(tmp_path, man, src)
    with pytest.raises(BundleError) as ei:
        m.ensure()
    msg = ei.value.user_message
    assert "m.gguf" in msg and hashlib.sha256(good).hexdigest()[:16] in msg
    assert not (m.assets_dir / "m.gguf").exists()


def test_tampered_cache_is_redownloaded(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    data = b"D" * 4096
    (src / "m.gguf").write_bytes(data)
    man = _manifest(tmp_path, {"m.gguf": data})
    m = _mgr(tmp_path, man, src)
    m.ensure()
    p = m.assets_dir / "m.gguf"
    p.write_bytes(b"E" * 4096)  # tamper (mtime changes -> marker invalid)
    m2 = _mgr(tmp_path, man, src)
    m2.ensure()
    assert p.read_bytes() == data


def test_unsafe_tar_member_rejected(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    rt = _runtime_tar({"../evil": b"x"})
    (src / "rt.tar.gz").write_bytes(rt)
    man = _manifest(tmp_path, {}, rt)
    m = _mgr(tmp_path, man, src)
    with pytest.raises(BundleError):
        m.ensure()
    assert not (tmp_path / "cache" / "evil").exists()


def test_manifest_rejects_path_names(tmp_path: Path) -> None:
    from pydantic import ValidationError

    from core.schemas import BundleAsset

    with pytest.raises(ValidationError):
        BundleAsset(name="../x.gguf", kind="model", sha256="0" * 64, size_bytes=1)
