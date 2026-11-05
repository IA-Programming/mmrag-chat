"""Config resolution, U-SEC-config (config.toml hardening), U-SEC-egress (runtime allowlist)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from core import net
from core.config import DEFAULT_GEMINI_MODEL, load_config
from core.errors import EgressDeniedError

ROOT = Path(__file__).resolve().parents[2]


def test_defaults_without_secrets() -> None:
    cfg = load_config({}, env={})
    assert cfg.memory_cap_mb == 2700 and cfg.ui_mode == "navigation" and not cfg.enable_gemini_fallback
    assert cfg.gemini.api_key is None and cfg.gemini.model == DEFAULT_GEMINI_MODEL
    assert cfg.budget_hard_mb == pytest.approx(2565) and cfg.budget_steady_mb == pytest.approx(2160)


def test_secret_beats_env_and_invalid_values_fall_back() -> None:
    cfg = load_config({"app": {"memory_cap_mb": "1500", "ui_mode": "bogus", "log_level": "loud"}}, env={"MMRAG_MEMORY_CAP_MB": "900"})
    assert cfg.memory_cap_mb == 1500 and cfg.ui_mode == "navigation" and cfg.log_level == "INFO"
    assert load_config({}, env={"MMRAG_MEMORY_CAP_MB": "900"}).memory_cap_mb == 900
    assert load_config({"app": {"memory_cap_mb": "x"}}, env={}).memory_cap_mb == 2700


def test_api_key_not_in_repr() -> None:
    cfg = load_config({"gemini": {"api_key": "AIza" + "F" * 35}}, env={})
    assert "AIza" not in repr(cfg)


def test_streamlit_config_hardening() -> None:
    conf = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    assert conf["server"]["enableXsrfProtection"] is True
    assert conf["server"]["maxUploadSize"] == 50
    assert conf["client"]["showErrorDetails"] in ("none", "type", False)
    assert conf["browser"]["gatherUsageStats"] is False
    assert conf["theme"]["primaryColor"].upper() == "#4FD1C5"


def test_egress_allowlist() -> None:
    p = net.EgressPolicy()
    p.check("http://127.0.0.1:8080/health")
    p.check("https://github.com/x/y/releases/download/a/b")
    p.check("https://release-assets.githubusercontent.com/abc")
    for bad in ("https://huggingface.co/x", "https://evil.example.com/", "http://github.com/x", "https://generativelanguage.googleapis.com/v1beta/openai/"):
        with pytest.raises(EgressDeniedError):
            p.check(bad)
    p.gemini_enabled = True
    p.check("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")


def test_client_blocks_redirect_to_foreign_host(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "github.com":
            return httpx.Response(302, headers={"Location": "https://evil.example.com/x"})
        return httpx.Response(200, text="pwned")

    c = net.client(follow_redirects=True, transport=httpx.MockTransport(handler))
    with pytest.raises(EgressDeniedError):
        c.get("https://github.com/a/b")
    with pytest.raises(EgressDeniedError):
        c.get("https://huggingface.co/x")  # blocked before leaving the process
