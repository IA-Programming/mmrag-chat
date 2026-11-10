"""Frozen bundle bootstrap (FR-001/FR-002): download, verify SHA-256 + size, cache, extract.

Cache layout: <cache>/bundle/<bundle_id>/assets/<name>   (+ runtime/ extracted from the tarball)
A `<name>.verified` marker (size, mtime, sha256) lets a warm start skip re-hashing large files.
Bootstrap runs once per process in a background thread; every session reads the same progress.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tarfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from core import net
from core.errors import AppError
from core.obs.log import Logger, get_logger
from core.schemas import BundleAsset, Manifest

CHUNK = 1024 * 1024


def load_manifest(path: Path) -> Manifest:
    return Manifest.model_validate_json(path.read_text())


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class AssetProgress:
    name: str
    size_bytes: int
    done_bytes: int = 0
    state: str = "pending"  # pending | cached | downloading | verifying | ready | failed
    error: str | None = None


@dataclass
class BootstrapState:
    state: str = "idle"  # idle | running | ready | failed
    assets: dict[str, AssetProgress] = field(default_factory=dict)
    error: str | None = None
    event_id: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            total = sum(a.size_bytes for a in self.assets.values()) or 1
            done = sum(a.size_bytes if a.state in ("ready", "cached") else a.done_bytes for a in self.assets.values())
            return {
                "state": self.state,
                "error": self.error,
                "event_id": self.event_id,
                "percent": round(100.0 * done / total, 1),
                "assets": [
                    {"name": a.name, "size_mb": a.size_bytes / 1e6, "done_mb": a.done_bytes / 1e6, "state": a.state, "error": a.error}
                    for a in self.assets.values()
                ],
                "elapsed_s": round((self.finished_at or time.time()) - self.started_at, 1) if self.started_at else 0.0,
            }


class BundleError(AppError):
    code = "bundle"


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if not str(target).startswith(str(dest) + os.sep) and target != dest:
            raise BundleError("The runtime archive contains an unsafe path.", detail=member.name)
        if member.issym() or member.islnk():
            link_target = (target.parent / member.linkname).resolve()
            if not str(link_target).startswith(str(dest) + os.sep):
                raise BundleError("The runtime archive contains an unsafe link.", detail=member.name)
        if member.isdev():
            raise BundleError("The runtime archive contains a device file.", detail=member.name)
    tar.extractall(dest, filter="tar")


class BundleManager:
    def __init__(
        self,
        manifest_path: Path,
        cache_root: Path,
        *,
        base_url_override: str | None = None,
        source_dir: Path | None = None,
        logger: Logger | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.manifest = load_manifest(manifest_path)
        self.manifest_sha = manifest_sha256(manifest_path)
        self.base_url = base_url_override or self.manifest.base_url
        self.source_dir = source_dir  # dev/emulator: copy from a local dir instead of downloading
        self.root = cache_root / self.manifest.bundle_id
        self.assets_dir = self.root / "assets"
        self.runtime_dir = self.root / "runtime"
        self.log = logger or get_logger()
        self.state = BootstrapState(assets={a.name: AssetProgress(a.name, a.size_bytes) for a in self.manifest.assets})
        self._run_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        if base_url_override:
            host = urlsplit(base_url_override).hostname
            if host:
                net.configure(extra_hosts=[host])

    # ------------------------------------------------------------------ paths
    def asset_path(self, name: str) -> Path:
        return self.assets_dir / name

    def model_path(self, name: str) -> Path:
        return self.asset_path(name)

    @property
    def llama_server(self) -> Path:
        return self.runtime_dir / "llama-server"

    # ------------------------------------------------------------------ cache checks
    def _marker(self, asset: BundleAsset) -> Path:
        return self.assets_dir / f".{asset.name}.verified"

    def is_cached(self, asset: BundleAsset) -> bool:
        p = self.asset_path(asset.name)
        m = self._marker(asset)
        if not (p.exists() and m.exists()):
            return False
        try:
            info = json.loads(m.read_text())
            st = p.stat()
            # I-BUNDLE-fingerprint: a cheap SHA-256 of first 4 KiB + last 4 KiB + size, written by
            # _verify alongside the manifest sha256. Catches in-place corruption (same size,
            # different bytes) that the marker-vs-manifest sha256 + mtime_ns check would miss on
            # filesystems with coarse mtime granularity (tmpfs, overlayfs, snapshot-restore).
            return bool(
                info["sha256"] == asset.sha256
                and info["size"] == st.st_size == asset.size_bytes
                and info["mtime_ns"] == st.st_mtime_ns
                and info.get("fingerprint") == self._fingerprint(asset)
            )
        except (OSError, ValueError, KeyError):
            return False

    def _fingerprint(self, asset: BundleAsset) -> str:
        """Cheap content fingerprint (8 KiB head+tail + size), stable across in-place rewrites."""
        p = self.asset_path(asset.name)
        h = hashlib.sha256()
        with p.open("rb") as fh:
            head = fh.read(4096)
            h.update(head)
            try:
                fh.seek(-4096, 2)
                tail = fh.read(4096)
                h.update(tail)
            except OSError:
                pass  # file smaller than 8 KiB; head is enough
        h.update(str(p.stat().st_size).encode())
        return h.hexdigest()

    def all_cached(self) -> bool:
        return all(self.is_cached(a) for a in self.manifest.assets) and self._runtime_extracted()

    def _runtime_extracted(self) -> bool:
        return all(not a.extract_to or (self.runtime_dir / ".extracted").exists() for a in self.manifest.assets if a.kind == "runtime")

    # ------------------------------------------------------------------ bootstrap
    def start(self) -> None:
        """Start bootstrap in the background (idempotent)."""
        with self._run_lock:
            if self.state.state in ("running", "ready"):
                return
            self.state.state = "running"
            self.state.error = None
            self.state.started_at = time.time()
            self._thread = threading.Thread(target=self._run, name="bundle-bootstrap", daemon=True)
            self._thread.start()

    def wait(self, timeout: float | None = None) -> str:
        if self._thread:
            self._thread.join(timeout)
        return self.state.state

    def ensure(self, progress: Callable[[], None] | None = None) -> None:
        """Synchronous bootstrap (tools/tests)."""
        self.state.state = "running"
        self.state.started_at = time.time()
        self._run()
        if self.state.state != "ready":
            raise BundleError(self.state.error or "Bootstrap failed.")

    def _run(self) -> None:
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        try:
            with self.log.span("bundle", "bootstrap", bundle_id=self.manifest.bundle_id):
                for asset in self.manifest.assets:
                    self._ensure_asset(asset)
                for asset in self.manifest.assets:
                    if asset.kind == "runtime" and asset.extract_to:
                        self._extract_runtime(asset)
            with self.state.lock:
                self.state.state = "ready"
                self.state.finished_at = time.time()
        except AppError as exc:
            self.log.error("bundle", "bootstrap.failed", message=exc.user_message, detail=exc.detail, event_id=exc.event_id)
            with self.state.lock:
                self.state.state = "failed"
                self.state.error = exc.user_message
                self.state.event_id = exc.event_id
                self.state.finished_at = time.time()
        except Exception as exc:  # never let the thread die silently
            err = BundleError("Preparing the runtime failed unexpectedly.", detail=f"{type(exc).__name__}: {exc}")
            self.log.error("bundle", "bootstrap.failed", detail=err.detail, event_id=err.event_id)
            with self.state.lock:
                self.state.state = "failed"
                self.state.error = err.user_message
                self.state.event_id = err.event_id
                self.state.finished_at = time.time()

    def _set(self, name: str, **kw: object) -> None:
        with self.state.lock:
            ap = self.state.assets[name]
            for k, v in kw.items():
                setattr(ap, k, v)

    def _ensure_asset(self, asset: BundleAsset) -> None:
        if self.is_cached(asset):
            self._set(asset.name, state="cached", done_bytes=asset.size_bytes)
            self.log.info("bundle", "asset.cached", asset=asset.name)
            return
        last_error = ""
        for attempt in (1, 2):  # AC2: retry once, then a visible error naming asset + hash
            try:
                self._fetch(asset)
                self._verify(asset)
                return
            except BundleError as exc:
                last_error = exc.detail or exc.user_message
                self.log.warning("bundle", "asset.retry", asset=asset.name, attempt=attempt, detail=last_error)
                self.asset_path(asset.name).unlink(missing_ok=True)
                self._part(asset).unlink(missing_ok=True)
        self._set(asset.name, state="failed", error=last_error)
        raise BundleError(
            f"Could not prepare '{asset.name}' (expected SHA-256 {asset.sha256[:16]}…). Retry, or check the release assets.",
            detail=last_error,
        )

    def _part(self, asset: BundleAsset) -> Path:
        return self.assets_dir / f"{asset.name}.part"

    def _fetch(self, asset: BundleAsset) -> None:
        part = self._part(asset)
        self._set(asset.name, state="downloading", error=None)
        if self.source_dir is not None:
            src = self.source_dir / asset.name
            if not src.exists():
                raise BundleError("A bundle asset is missing from the local source.", detail=str(src))
            with self.log.span("bundle", "asset.copy", asset=asset.name):
                shutil.copyfile(src, part)
            self._set(asset.name, done_bytes=asset.size_bytes)
        else:
            url = asset.url or (self.base_url.rstrip("/") + "/" + asset.name)
            have = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={have}-"} if 0 < have < asset.size_bytes else {}
            if have >= asset.size_bytes:
                part.unlink()
                have, headers = 0, {}
            with self.log.span("bundle", "asset.download", asset=asset.name, resume_from=have) as extra:
                with net.client(timeout=60.0, follow_redirects=True) as http, http.stream("GET", url, headers=headers) as resp:
                    if resp.status_code not in (200, 206):
                        raise BundleError("Download failed.", detail=f"HTTP {resp.status_code} for {asset.name}")
                    if resp.status_code == 200:
                        have = 0  # server ignored Range: restart
                    extra["final_host"] = resp.request.url.host
                    with part.open("ab" if have else "wb") as fh:
                        done = have
                        for chunk in resp.iter_bytes(CHUNK):
                            fh.write(chunk)
                            done += len(chunk)
                            self._set(asset.name, done_bytes=done)
                            if done > asset.size_bytes:
                                raise BundleError("Download larger than expected.", detail=asset.name)
        part.replace(self.asset_path(asset.name))

    def _verify(self, asset: BundleAsset) -> None:
        p = self.asset_path(asset.name)
        self._set(asset.name, state="verifying")
        size = p.stat().st_size
        if size != asset.size_bytes:
            raise BundleError("Size mismatch.", detail=f"{asset.name}: {size} != {asset.size_bytes}")
        h = hashlib.sha256()
        with self.log.span("bundle", "asset.verify", asset=asset.name), p.open("rb") as fh:
            while block := fh.read(8 * CHUNK):
                h.update(block)
        if h.hexdigest() != asset.sha256:
            raise BundleError("Checksum mismatch.", detail=f"{asset.name}: got {h.hexdigest()[:16]} expected {asset.sha256[:16]}")
        st = p.stat()
        self._marker(asset).write_text(
            json.dumps(
                {
                    "sha256": asset.sha256,
                    "size": st.st_size,
                    "mtime_ns": st.st_mtime_ns,
                    "fingerprint": self._fingerprint(asset),
                }
            )
        )
        self._set(asset.name, state="ready", done_bytes=asset.size_bytes)

    def _extract_runtime(self, asset: BundleAsset) -> None:
        marker = self.runtime_dir / ".extracted"
        if marker.exists() and marker.read_text().strip() == asset.sha256:
            return
        tmp = self.root / "runtime.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        with self.log.span("bundle", "runtime.extract", asset=asset.name), tarfile.open(self.asset_path(asset.name), "r:gz") as tar:
            _safe_extract(tar, tmp)
        # Accept archives with or without a single top-level directory.
        entries = [e for e in tmp.iterdir()]
        src = entries[0] if len(entries) == 1 and entries[0].is_dir() else tmp
        exe = src / "llama-server"
        if not exe.exists():
            raise BundleError("The runtime archive does not contain llama-server.", detail=asset.name)
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        shutil.rmtree(self.runtime_dir, ignore_errors=True)
        src.replace(self.runtime_dir) if src != tmp else tmp.replace(self.runtime_dir)
        shutil.rmtree(tmp, ignore_errors=True)
        (self.runtime_dir / ".extracted").write_text(asset.sha256)

    def retry(self) -> None:
        with self._run_lock:
            if self.state.state == "running":
                return
            self.state.state = "idle"
        self.start()
