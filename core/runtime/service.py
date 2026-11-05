"""Process-wide runtime: bundle bootstrap -> router child -> governor (+ emb-text resident).

`app` wraps `Runtime` in st.cache_resource (one per process, shared by all sessions).
Real mode serves the frozen bundle; fake mode (MMRAG_RUNTIME=fake) runs
tests/fakes/fake_llama_server.py with the same presets, so the whole UI works without models.
"""

from __future__ import annotations

import atexit
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from core import net
from core.bundle.manager import BundleManager
from core.config import BUNDLE_DIR, REPO_ROOT, AppConfig
from core.errors import AppError, RuntimeNotReadyError
from core.obs.log import Logger, get_logger
from core.runtime.governor import ALWAYS_RESIDENT, Governor
from core.runtime.presets import load_embed_params, load_model_presets, load_sampling_presets, write_ini
from core.runtime.probe import run_probe
from core.runtime.router import RouterClient, RouterProcess
from core.runtime.telemetry import Telemetry
from core.schemas import ModelPreset, ProbeReport

FAKE_SERVER = REPO_ROOT / "tests" / "fakes" / "fake_llama_server.py"


class Runtime:
    def __init__(self, cfg: AppConfig, logger: Logger | None = None) -> None:
        self.cfg = cfg
        self.log = logger or get_logger()
        self.log.set_level(cfg.log_level)
        if cfg.gemini.api_key:
            self.log.redactor.register(cfg.gemini.api_key)
        net.configure(gemini_enabled=cfg.enable_gemini_fallback)
        self.presets = load_model_presets()
        self.sampling = load_sampling_presets()
        self.embed_defaults = load_embed_params()
        self.telemetry = Telemetry(cfg.memory_cap_mb)
        self.bundle: BundleManager | None = None
        self.router: RouterProcess | None = None
        self.client: RouterClient | None = None
        self.governor: Governor | None = None
        self.state = "idle"  # idle | bootstrapping | starting | ready | failed
        self.error: str | None = None
        self.event_id: str | None = None
        self.started_at: float | None = None
        self.ready_at: float | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        manifest = Path(os.environ.get("MMRAG_MANIFEST") or (BUNDLE_DIR / "manifest.json"))
        if cfg.runtime_mode != "fake" and manifest.exists():
            src = os.environ.get("MMRAG_BUNDLE_SOURCE_DIR")
            self.bundle = BundleManager(
                manifest, cfg.bundle_cache, base_url_override=cfg.bundle_base_url, source_dir=Path(src) if src else None, logger=self.log
            )
        atexit.register(self.shutdown)

    # ------------------------------------------------------------------ presets served
    def served_presets(self) -> list[ModelPreset]:
        include_all = os.environ.get("MMRAG_ALL_PRESETS") == "1"
        return [p for p in self.presets.presets if p.role == "embedder" or p.shipped or include_all]

    def generators(self) -> list[ModelPreset]:
        return [p for p in self.served_presets() if p.role == "generator"]

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        with self._lock:
            if self.state in ("bootstrapping", "starting", "ready"):
                return
            self.state = "bootstrapping"
            self.error = None
            self.started_at = time.time()
            self._thread = threading.Thread(target=self._boot, name="runtime-boot", daemon=True)
            self._thread.start()

    def wait_ready(self, timeout: float = 300.0) -> bool:
        if self._thread:
            self._thread.join(timeout)
        return self.state == "ready"

    def _boot(self) -> None:
        try:
            with self.log.span("runtime", "boot", mode=self.cfg.runtime_mode):
                if self.cfg.runtime_mode == "fake":
                    argv = [sys.executable, str(FAKE_SERVER)]

                    def resolve(name: str) -> Path:
                        return Path("/fake-models") / name
                else:
                    if self.bundle is None:
                        raise RuntimeNotReadyError(
                            "No frozen bundle is configured (bundle/manifest.json is missing). Publish a bundle release or run with MMRAG_RUNTIME=fake."
                        )
                    self.bundle.ensure()
                    argv = [str(self.bundle.llama_server)]
                    resolve = self.bundle.model_path
                self.state = "starting"
                ini = write_ini(self.cfg.cache_dir / "router" / "presets.ini", self.served_presets(), resolve, threads=self.cfg.threads)
                self.router = RouterProcess(argv, ini, log_dir=self.cfg.cache_dir / "router", logger=self.log)
                self.router.start()
                self.client = RouterClient(self.router.base_url, self.router.api_key, logger=self.log)
                self.governor = Governor(self.client, self.presets, self.telemetry.used_mb, hard_budget_mb=self.cfg.budget_hard_mb, logger=self.log)
                with self.governor.turn(None, "boot"):
                    self.governor.ensure(ALWAYS_RESIDENT)
                    self.governor.release(ALWAYS_RESIDENT)
            self.state = "ready"
            self.ready_at = time.time()
        except AppError as exc:
            self.state, self.error, self.event_id = "failed", exc.user_message, exc.event_id
            self.log.error("runtime", "boot.failed", message=exc.user_message, detail=exc.detail, event_id=exc.event_id)
        except Exception as exc:
            err = RuntimeNotReadyError("Starting the local runtime failed unexpectedly.", detail=f"{type(exc).__name__}: {exc}")
            self.state, self.error, self.event_id = "failed", err.user_message, err.event_id
            self.log.error("runtime", "boot.failed", detail=err.detail, event_id=err.event_id)

    def retry(self) -> None:
        with self._lock:
            if self.state in ("bootstrapping", "starting"):
                return
            self.state = "idle"
        if self.router:
            self.router.stop()
        if self.bundle and self.bundle.state.state == "failed":
            self.bundle.state.state = "idle"
        self.start()

    def shutdown(self) -> None:
        if self.client:
            self.client.close()
        if self.router:
            self.router.stop()

    # ------------------------------------------------------------------ views
    @property
    def ready(self) -> bool:
        return self.state == "ready" and self.router is not None and self.router.alive()

    def require_ready(self) -> tuple[RouterClient, Governor]:
        if not self.ready or self.client is None or self.governor is None:
            raise RuntimeNotReadyError("The local runtime is still warming up.")
        return self.client, self.governor

    def snapshot(self) -> dict[str, Any]:
        bundle = self.bundle.state.snapshot() if self.bundle else None
        if self.state == "ready" and self.router is not None and not self.router.alive():
            return {"state": "failed", "error": "The local model server stopped unexpectedly.", "bundle": bundle, "event_id": None}
        return {
            "state": self.state,
            "error": self.error,
            "event_id": self.event_id,
            "bundle": bundle,
            "mode": self.cfg.runtime_mode,
            "elapsed_s": round((self.ready_at or time.time()) - self.started_at, 1) if self.started_at else 0.0,
        }

    def build_id(self) -> str:
        if self.cfg.runtime_mode == "fake":
            return "fake-llama-server"
        if self.bundle:
            info = self.bundle.manifest.llama_cpp
            return f"{info.base} {info.commit[:10]} ({info.target})"
        return "unknown"

    def probe(self) -> ProbeReport:
        return run_probe(
            llama_server_build=self.build_id,
            bundle_id=self.bundle.manifest.bundle_id if self.bundle else None,
            manifest_sha256=self.bundle.manifest_sha if self.bundle else None,
        )

    def bundle_id(self) -> str | None:
        return self.bundle.manifest.bundle_id if self.bundle else ("fake" if self.cfg.runtime_mode == "fake" else None)
