"""Process-wide singletons (st.cache_resource) and per-session service objects for pages."""

from __future__ import annotations

from typing import Any

import streamlit as st

from app import state
from core.config import AppConfig, load_config
from core.embed.client import EmbedService
from core.index.store import IndexHandle
from core.jobs import JobRunner
from core.llm.backends import GeminiBackend
from core.obs.log import Logger, get_logger
from core.rag.pipeline import ChatPipeline
from core.runtime.service import Runtime
from core.schemas import EmbedParams


def _secrets() -> dict[str, Any]:
    try:
        return {k: dict(v) if hasattr(v, "items") else v for k, v in st.secrets.items()}
    except Exception:  # no secrets.toml: every secret is optional
        return {}


@st.cache_resource(show_spinner=False)
def get_config() -> AppConfig:
    return load_config(_secrets())


@st.cache_resource(show_spinner=False)
def get_runtime() -> Runtime:
    rt = Runtime(get_config(), get_logger())
    rt.start()
    state.cleanup_stale_sessions(rt.cfg.session_root, rt.cfg.session_ttl_s, rt.cfg.session_dir_cap_mb)
    return rt


@st.cache_resource(show_spinner=False)
def get_jobs() -> JobRunner:
    return JobRunner()


def log() -> Logger:
    return get_logger()


def embed_params() -> EmbedParams:
    p = state.get("embed.params")
    if p is None:
        p = get_runtime().embed_defaults.model_copy(deep=True)
        state.set("embed.params", p)
    return p  # type: ignore[no-any-return]


def embed_service() -> EmbedService:
    client, _ = get_runtime().require_ready()
    return EmbedService(client, embed_params())


def fingerprint() -> str:
    """U-19: model + mmproj + params + llama build id; imported indexes must match it."""
    import hashlib
    import json

    rt = get_runtime()
    emb = rt.presets.get("emb-mm")
    p = embed_params()
    blob = json.dumps({"files": emb.files, "build": rt.build_id(), "prefix": p.prefix.model_dump(), "dim": p.dims.index_dim}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def index_handle(create: bool = False) -> IndexHandle | None:
    h = state.get("index.handle")
    if h is None and create:
        cfg = get_config()
        h = IndexHandle(
            state.session_dir(cfg.session_root) / "index",
            dim=embed_params().dims.index_dim,
            bits=int(state.get("index.bits")),
            keep_float32=bool(state.get("index.keep_vectors")),
            fingerprint=fingerprint(),
        )
        state.set("index.handle", h)
    return h  # type: ignore[no-any-return]


def remote_backend() -> GeminiBackend | None:
    cfg = get_config()
    fb = state.get("settings.fallback")
    key = fb.get("api_key") or cfg.gemini.api_key
    if not (cfg.enable_gemini_fallback or fb.get("enabled")) or not key or not fb.get("consent"):
        return None
    from core import net

    net.configure(gemini_enabled=True)
    return GeminiBackend(key, fb.get("model") or cfg.gemini.model, cfg.gemini.base_url, get_logger())


def pipeline() -> ChatPipeline:
    rt = get_runtime()
    client, gov = rt.require_ready()
    return ChatPipeline(client, gov, EmbedService(client, embed_params()), embed_params(), rt.telemetry, get_logger(), remote=remote_backend())
