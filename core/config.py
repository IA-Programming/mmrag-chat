"""App configuration resolved from Streamlit secrets (passed in by `app`) + environment.

Precedence: secrets value (non-empty) > environment variable > documented default.
`core` never imports streamlit; the app layer passes `st.secrets` as a plain mapping.
Runtime secrets are all optional (the core app needs none).
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = REPO_ROOT / "bundle"
DATASETS_DIR = REPO_ROOT / "datasets"

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# Verified 2026-11-05 with a free-tier key (experiments/ledger.jsonl, EXP-GEMINI-01): Flash-Lite is the
# free-tier family with the most generous limits, and 3.5-flash-lite accepts text, image, audio and
# video (as a data URL) on the OpenAI-compatible endpoint. 3.1-flash-lite returned 503 (overloaded);
# gemma-4 rejects audio. Testers can type another model id in Settings.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def _get(secrets: Mapping[str, Any], section: str, key: str) -> str | None:
    try:
        sec = secrets.get(section)  # type: ignore[union-attr]
    except Exception:  # st.secrets raises when no secrets file exists
        return None
    if not sec:
        return None
    try:
        value = sec.get(key)
    except Exception:
        return None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _bool(text: str | None, default: bool) -> bool:
    if text is None:
        return default
    return text.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str | None = field(default=None, repr=False)  # never printed
    model: str = DEFAULT_GEMINI_MODEL
    base_url: str = DEFAULT_GEMINI_BASE_URL


@dataclass(frozen=True)
class AppConfig:
    memory_cap_mb: int = 2700
    log_level: str = "INFO"
    enable_gemini_fallback: bool = False
    bundle_base_url: str | None = None
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    runtime_mode: str = "real"  # real | fake (fake = tests/fakes/fake_llama_server.py)
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "mmrag")
    session_root: Path = field(default_factory=lambda: Path(tempfile.gettempdir()) / "mmrag")
    session_quota_mb: int = 200
    session_dir_cap_mb: int = 300
    session_ttl_s: int = 2 * 3600
    threads: int = 2

    @property
    def bundle_cache(self) -> Path:
        return self.cache_dir / "bundle"

    @property
    def budget_hard_mb(self) -> float:
        return self.memory_cap_mb * 0.95

    @property
    def budget_steady_mb(self) -> float:
        return self.memory_cap_mb * 0.80


def load_config(secrets: Mapping[str, Any] | None = None, env: Mapping[str, str] | None = None) -> AppConfig:
    secrets = secrets or {}
    env = os.environ if env is None else env

    def pick(section: str, key: str, env_key: str | None) -> str | None:
        return _get(secrets, section, key) or ((env.get(env_key) or None) if env_key else None)

    cap_text = pick("app", "memory_cap_mb", "MMRAG_MEMORY_CAP_MB")
    try:
        cap = int(cap_text) if cap_text else 2700
    except ValueError:
        cap = 2700
    cap = max(256, min(cap, 65536))

    level = (pick("app", "log_level", "MMRAG_LOG_LEVEL") or "INFO").upper()
    if level not in LOG_LEVELS:
        level = "INFO"

    gemini = GeminiConfig(
        api_key=pick("gemini", "api_key", "GEMINI_API_KEY"),
        model=pick("gemini", "model", "MMRAG_GEMINI_MODEL") or DEFAULT_GEMINI_MODEL,
        base_url=pick("gemini", "base_url", None) or DEFAULT_GEMINI_BASE_URL,
    )
    threads_text = env.get("MMRAG_THREADS")
    cache_dir = env.get("MMRAG_CACHE_DIR")
    session_root = env.get("MMRAG_SESSION_ROOT")
    return AppConfig(
        memory_cap_mb=cap,
        log_level=level,
        enable_gemini_fallback=_bool(pick("app", "enable_gemini_fallback", "MMRAG_ENABLE_GEMINI"), False),
        bundle_base_url=pick("app", "bundle_base_url", "MMRAG_BUNDLE_BASE_URL"),
        gemini=gemini,
        runtime_mode=(env.get("MMRAG_RUNTIME") or "real").lower(),
        cache_dir=Path(cache_dir) if cache_dir else Path.home() / ".cache" / "mmrag",
        session_root=Path(session_root) if session_root else Path(tempfile.gettempdir()) / "mmrag",
        threads=int(threads_text) if threads_text and threads_text.isdigit() else 2,
    )
