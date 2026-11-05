"""The ONLY accessor of st.session_state keys (Rule R5). Widget keys are w_<page>_<name>."""

from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import streamlit as st

DEFAULTS: dict[str, Any] = {
    "chat.messages": list,
    "chat.attachment": lambda: None,
    "settings.generator_id": lambda: None,
    "settings.gen_overrides": dict,
    "settings.gen_mode": dict,
    "settings.history_policy": lambda: "window",
    "settings.history_window": lambda: 4,
    "settings.contextualize": lambda: "off",
    "settings.retrieval": lambda: {"k": 4, "threshold": 0.0, "quota": {}},
    "settings.fallback": lambda: {"enabled": False, "api_key": None, "model": None, "consent": False, "media_ok": False},
    "index.handle": lambda: None,
    "index.job_id": lambda: None,
    "index.uploaded_bytes": lambda: 0,
    "index.bits": lambda: 4,
    "index.keep_vectors": lambda: True,
    "embed.params": lambda: None,
    "playground.section": lambda: "Matryoshka",
    "playground.last_run": lambda: None,
    "explorer.selection": lambda: None,
    "logs.filter": lambda: {"level": "INFO", "component": None, "text": ""},
    "logs.paused": lambda: False,
    "logs.debug_prompts": lambda: False,
    "index.export_blob": lambda: None,
}


def _init() -> None:
    ss = st.session_state
    if "sid" not in ss:
        ss["sid"] = uuid.uuid4().hex[:12]
        ss["session_started"] = time.time()
    for k, factory in DEFAULTS.items():
        if k not in ss:
            ss[k] = factory()


def get(key: str) -> Any:
    _init()
    if key not in DEFAULTS and key not in ("sid", "session_started"):
        raise KeyError(f"unknown session key {key}")
    return st.session_state[key]


def set(key: str, value: Any) -> None:
    _init()
    if key not in DEFAULTS:
        raise KeyError(f"unknown session key {key}")
    st.session_state[key] = value


def sid() -> str:
    _init()
    return str(st.session_state["sid"])


def session_dir(root: Path) -> Path:
    d = root / sid()
    d.mkdir(parents=True, exist_ok=True)
    return d


def confirm_flag(action: str) -> bool:
    return bool(st.session_state.get(f"ui.confirm.{action}", False))


def set_confirm(action: str, value: bool) -> None:
    st.session_state[f"ui.confirm.{action}"] = value


def reset_session(root: Path) -> None:
    """Danger zone: drop chat, index, uploads, overrides and keys (keeps the sid)."""
    handle = st.session_state.get("index.handle")
    if handle is not None:
        try:
            handle.close()
        except Exception:
            pass
    shutil.rmtree(root / sid(), ignore_errors=True)
    for k, factory in DEFAULTS.items():
        st.session_state[k] = factory()


def cleanup_stale_sessions(root: Path, ttl_s: int, cap_mb: int) -> None:
    """TTL cleanup of /tmp/mmrag/<sid> (DATA doc section 11)."""
    if not root.exists():
        return
    now = time.time()
    for d in root.iterdir():
        try:
            if d.is_dir() and now - d.stat().st_mtime > ttl_s:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            continue
    _ = cap_mb


def clear_widget_keys(prefix: str) -> None:
    """Drop widget values (e.g. after 'Reset to model default') so widgets re-initialise."""
    for k in [k for k in st.session_state if str(k).startswith(prefix)]:
        del st.session_state[k]


def widget_value(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)
