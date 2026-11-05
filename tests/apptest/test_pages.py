"""Page flows with AppTest against the fake llama-server (MMRAG_RUNTIME=fake).

A-WARMUP, A-DIAG, A-CHAT-input, A-SOURCES, A-STATS, A-GENPARAMS, A-SETTINGS-models,
A-SETTINGS-validation, A-DANGER, A-LOGS, A-SHELL-modes, A-INDEX-validate.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
APP = str(ROOT / "streamlit_app.py")


@pytest.fixture(scope="module", autouse=True)
def fake_env(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    old = dict(os.environ)
    cache = tmp_path_factory.mktemp("cache")
    os.environ.update({"MMRAG_RUNTIME": "fake", "MMRAG_ALL_PRESETS": "1", "MMRAG_CACHE_DIR": str(cache), "MMRAG_SESSION_ROOT": str(cache / "sessions")})
    st.cache_resource.clear()
    yield
    from app import services

    try:
        services.get_runtime().shutdown()
    except Exception:
        pass
    st.cache_resource.clear()
    os.environ.clear()
    os.environ.update(old)


def boot(timeout: float = 30) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=timeout)
    at.run()
    from app import services

    services.get_runtime().wait_ready(30)
    at.run()
    return at


def titles(at: AppTest) -> list[str]:
    return [t.value for t in at.title]


def no_exceptions(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


def _index_some_text() -> None:
    """Index two fake documents straight through core (AppTest cannot drive background jobs well)."""
    from app import services

    rt = services.get_runtime()
    client, gov = rt.require_ready()
    from core.embed.client import EmbedService, truncate_renorm
    from core.index.store import IndexHandle

    docs = ["The lighthouse on Kestrel Island has 412 steps and is kept by Mira Talvane.", "Glimmerfruit harvest reached 9400 crates in Northreach."]
    emb = EmbedService(client, rt.embed_defaults)
    with gov.turn("test", "index"), gov.using("emb-text"):
        v = truncate_renorm(emb.embed_pieces([rt.embed_defaults.prefix.document.format(content=d) for d in docs], "emb-text"), 768)
    root = Path(os.environ["MMRAG_SESSION_ROOT"]) / "apptest-index"
    h = IndexHandle(root, fingerprint="x")
    h.add_source("s1", "facts.md", "text", "a" * 64, 100)
    h.add_items(
        "s1", [{"modality": "text", "chunk_idx": i, "span": {"chunk": i, "chars": [0, 10]}, "text": d, "tokens_est": 20} for i, d in enumerate(docs)], v, "p"
    )
    _HANDLE["h"] = h


_HANDLE: dict[str, object] = {}


def test_warmup_then_ready_chat_empty_state() -> None:
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    no_exceptions(at)
    assert titles(at)[0] in ("Chat", "Warming up") or "Warming up" in titles(at)
    from app import services

    assert services.get_runtime().wait_ready(30)
    at.run()
    no_exceptions(at)
    assert "Chat" in titles(at)
    assert any("No data indexed yet" in i.value for i in at.info)
    assert at.chat_input[0].disabled  # input disabled until something is indexed


def test_chat_text_turn_with_sources_and_stats() -> None:
    _index_some_text()
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["index.handle"] = _HANDLE["h"]
    at.run()
    no_exceptions(at)
    assert not at.chat_input[0].disabled
    at.chat_input[0].set_value("How many steps does the lighthouse have?").run()
    no_exceptions(at)
    msgs = at.session_state["chat.messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    a = msgs[-1]
    assert a["text"] and not a.get("error"), a
    assert a["sources"] and a["sources"][0]["filename"] == "facts.md"
    st_ = a["stats"]
    for k in ("embed_ms", "retrieve_ms", "ttft_ms", "prompt_breakdown", "effective_params", "peak_mb", "model"):
        assert k in st_, k
    labels = [e.label for e in at.expander]
    assert any(lbl.startswith("Sources (") for lbl in labels) and "Turn stats" in labels and "Effective parameters" in labels
    # FR-021 AC1: a text query never loaded the multimodal embedder
    from app import services

    loads = [e.fields.get("model") for e in services.log().ring.query(None, include_system=True) if e.event == "model.load"]
    loads += [e.fields.get("model") for e in services.log().ring.query(at.session_state["sid"]) if e.event == "model.load"]
    assert "emb-mm" not in loads


def test_history_none_sends_no_previous_turns() -> None:
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["index.handle"] = _HANDLE.get("h")
    at.session_state["settings.history_policy"] = "none"
    at.session_state["chat.messages"] = [{"role": "user", "text": "earlier question", "ts": 0}, {"role": "assistant", "text": "earlier answer", "ts": 0}]
    at.run()
    at.chat_input[0].set_value("Who keeps the lighthouse?").run()
    no_exceptions(at)
    assert at.session_state["chat.messages"][-1]["stats"]["prompt_breakdown"]["history"] == 0


def test_genparams_defaults_modified_and_reset() -> None:
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    temp = next(s for s in at.slider if s.label.startswith("temperature"))
    assert temp.value == pytest.approx(0.1)  # LFM3-VL card default
    temp.set_value(0.8).run()
    no_exceptions(at)
    ov = at.session_state["settings.gen_overrides"]
    assert next(iter(ov.values()))["temperature"] == pytest.approx(0.8)
    next(b for b in at.button if b.label == "Reset to model default").click().run()
    assert at.session_state["settings.gen_overrides"] == {}
    assert next(s for s in at.slider if s.label.startswith("temperature")).value == pytest.approx(0.1)


def _section(name: str) -> AppTest:
    at = AppTest.from_function(_settings_page, default_timeout=30)
    at.session_state["w_settings_section"] = name
    at.run()
    no_exceptions(at)
    return at


def test_settings_diagnostics() -> None:
    at = _section("Diagnostics")
    assert any("Environment probe" in m.value for m in at.markdown)
    assert any(b.label == "Download diagnostics JSON" for b in at.get("download_button"))


def test_settings_retrieval_validation_and_apply() -> None:
    at = _section("Retrieval & history")
    at.number_input(key="w_settings_k").set_value(7)
    at.selectbox(key="w_settings_ctx").set_value("llm_rewrite")
    next(b for b in at.button if b.label == "Apply").click().run()
    no_exceptions(at)
    assert at.session_state["settings.retrieval"]["k"] == 7 and at.session_state["settings.contextualize"] == "llm_rewrite"
    assert any("Applied" in x.value for x in at.success)


def test_settings_models_table_and_panel() -> None:
    at = _section("Models & generation")
    assert any(s.label.startswith("temperature") for s in at.slider) or at.info


def test_settings_fallback_requires_consent() -> None:
    at = _section("Fallback (Gemini)")
    assert any("leave this app" in m.value for m in at.markdown)
    test_btn = next(b for b in at.button if b.label == "Test connection")
    assert test_btn.disabled  # nothing can be sent before enabling + consent


def test_danger_zone_actions_need_confirmation() -> None:
    at = _section("Danger zone")
    at.session_state["chat.messages"] = [{"role": "user", "text": "keep me", "ts": 0}]
    next(b for b in at.button if b.label == "Reset session").click().run()
    no_exceptions(at)
    assert at.session_state["chat.messages"]  # a click alone does not reset: a confirm dialog opens


def _settings_page() -> None:
    from app.pages import settings

    settings.render()


def _logs_page() -> None:
    from app.pages import logs

    logs.render()


def test_logs_page_session_scoped() -> None:
    at = AppTest.from_function(_logs_page, default_timeout=30)
    at.run()
    no_exceptions(at)
    assert "Logs" in titles(at)
    assert at.dataframe  # tail rendered


def _index_page() -> None:
    from app.pages import index

    index.render()


def test_index_page_states() -> None:
    at = AppTest.from_function(_index_page, default_timeout=30)
    at.run()
    no_exceptions(at)
    assert "Index" in titles(at)
    start = next(b for b in at.button if b.label == "Start indexing")
    assert start.disabled  # nothing chosen yet


def test_failed_runtime_shows_error_not_traceback(tmp_path: Path) -> None:
    os.environ["MMRAG_RUNTIME"] = "real"
    os.environ["MMRAG_MANIFEST"] = str(tmp_path / "missing.json")
    st.cache_resource.clear()
    try:
        at = AppTest.from_file(APP, default_timeout=30)
        at.run()
        from app import services

        services.get_runtime().wait_ready(10)
        at.run()
        no_exceptions(at)
        assert "Warming up" in titles(at)
        assert any("No frozen bundle" in e.value for e in at.error)
        assert any(b.label == "Retry" for b in at.button)
    finally:
        os.environ["MMRAG_RUNTIME"] = "fake"
        os.environ.pop("MMRAG_MANIFEST", None)
        st.cache_resource.clear()
