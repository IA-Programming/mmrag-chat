"""Settings page (UX section 7): models & generation, retrieval & history, Gemini fallback,
diagnostics, danger zone."""

from __future__ import annotations

import json

import streamlit as st

from app import services, state
from app.components import genparams
from core import net
from core.errors import AppError
from core.rag.history import CONTEXT_MODES, POLICIES

SECTIONS = ["Models & generation", "Retrieval & history", "Fallback (Gemini)", "Diagnostics", "Danger zone"]


def _models() -> None:
    rt = services.get_runtime()
    rows = []
    for p in rt.presets.presets:
        if p.role != "generator" or not p.shipped:
            continue
        caps = ", ".join(k for k in ("image", "audio") if getattr(p.capabilities, k)) or "text"
        if p.capabilities.video:
            caps += ", video (frames)"
        rows.append(
            {
                "model": p.label,
                "id": p.id,
                "n_ctx (fixed)": p.n_ctx,
                "files": " + ".join(p.files.values()),
                "measured peak MB": p.measured_peak_mb,
                "inputs": caps,
                "gates": " ".join(f"{k}:{v}" for k, v in p.gates.items() if v),
            }
        )
    st.subheader("Shipped models")
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.info("No generator is marked as shipped in bundle/model_presets.json.")
    st.subheader("Generation parameters")
    from app.pages.chat import _generator_options, selected_generator

    opts, labels = _generator_options()
    gen = selected_generator()
    if not opts:
        return
    pick = st.selectbox(
        "Model",
        opts,
        index=opts.index(gen) if gen in opts else 0,
        format_func=lambda o: labels.get(o, o),
        key="w_settings_model",
        on_change=lambda: state.set("settings.generator_id", state.widget_value("w_settings_model")),
        help="Selecting a model never loads it; the swap happens on the next generation.",
    )
    preset = rt.presets.get(pick) if pick != "gemini" else None
    genparams.panel(
        "settings",
        pick,
        label=labels.get(pick, pick),
        backend="gemini" if pick == "gemini" else "local",
        family=preset.family if preset else None,
        reasoning_supported=bool(preset and preset.capabilities.reasoning_control == "chat_template_kwargs"),
        n_ctx=preset.n_ctx if preset else None,
    )


def _retrieval() -> None:
    r = dict(state.get("settings.retrieval"))
    c1, c2 = st.columns(2)
    k = c1.number_input("Top-k sources", 1, 10, int(r["k"]), key="w_settings_k")
    thr = c2.slider("Score threshold", 0.0, 1.0, float(r["threshold"]), 0.01, key="w_settings_threshold", help="Sources below this cosine score are ignored.")
    st.markdown("**Per-modality quota** (0 = no cap)")
    q = dict(r.get("quota") or {})
    qc = st.columns(4)
    quota = {m: int(qc[i].number_input(m, 0, 10, int(q.get(m, 0)), key=f"w_settings_quota_{m}")) for i, m in enumerate(("text", "image", "audio", "video"))}
    st.divider()
    c3, c4, c5 = st.columns(3)
    pol = c3.selectbox("History policy", POLICIES, index=POLICIES.index(state.get("settings.history_policy")), key="w_settings_history")
    win = c4.number_input("Window N (exchanges)", 1, 16, int(state.get("settings.history_window")), key="w_settings_window", disabled=pol != "window")
    ctx = c5.selectbox(
        "Contextualization",
        CONTEXT_MODES,
        index=CONTEXT_MODES.index(state.get("settings.contextualize")),
        key="w_settings_ctx",
        help="llm_rewrite adds exactly one extra generation call per turn.",
    )
    if st.button("Apply", key="w_settings_apply", type="primary"):
        errors = []
        if not (1 <= int(k) <= 10):
            errors.append("Top-k must be between 1 and 10.")
        if not (0.0 <= float(thr) <= 1.0):
            errors.append("Threshold must be between 0 and 1.")
        if errors:
            for e in errors:
                st.error(e)
            return
        state.set("settings.retrieval", {"k": int(k), "threshold": float(thr), "quota": {m: v for m, v in quota.items() if v}})
        state.set("settings.history_policy", pol)
        state.set("settings.history_window", int(win))
        state.set("settings.contextualize", ctx)
        services.log().info("ui", "settings.retrieval", sid=state.sid(), k=int(k), threshold=float(thr), policy=pol, window=int(win), contextualize=ctx)
        st.success("Applied from the next turn.")


def _fallback() -> None:
    cfg = services.get_config()
    fb = dict(state.get("settings.fallback"))
    st.markdown(
        "Optional cloud generator through Google's OpenAI-compatible Gemini endpoint. **Prompts, retrieved text and any media you "
        "confirm leave this app and are processed by Google under your account's terms; on the free tier Google may use them to "
        "improve its products.** Local retrieval (embeddings, index) always stays on this server."
    )
    enabled = st.toggle("Enable Gemini fallback", value=bool(fb.get("enabled") or cfg.enable_gemini_fallback), key="w_settings_gem_enabled")
    consent = st.checkbox("I understand that prompts are sent to Google", value=bool(fb.get("consent")), key="w_settings_gem_consent", disabled=not enabled)
    has_secret = bool(cfg.gemini.api_key)
    key = st.text_input(
        "Gemini API key (kept in this browser session only)",
        type="password",
        key="w_settings_gem_key",
        disabled=not enabled,
        placeholder="configured by the app owner" if has_secret else "paste a Google AI Studio key",
    )
    model = st.text_input(
        "Model id",
        value=fb.get("model") or cfg.gemini.model,
        key="w_settings_gem_model",
        disabled=not enabled,
        help="Default chosen for the free tier: multimodal input with the most generous limits.",
    )
    media_ok = st.checkbox(
        "Also send attached media (images/audio) to Gemini", value=bool(fb.get("media_ok")), key="w_settings_gem_media", disabled=not enabled
    )
    c1, c2, c3 = st.columns(3)
    if c1.button("Save", key="w_settings_gem_save", type="primary"):
        fb.update(enabled=enabled, consent=consent, model=model.strip() or None, media_ok=media_ok)
        if key.strip():
            fb["api_key"] = key.strip()
            services.log().redactor.register(key.strip())
        state.set("settings.fallback", fb)
        net.configure(gemini_enabled=enabled)
        st.success("Saved for this session." + ("" if consent or not enabled else " Consent is required before Gemini can be used."))
    if c2.button("Test connection", key="w_settings_gem_test", disabled=not (enabled and consent)):
        backend = services.remote_backend()
        if backend is None:
            st.error("Save an API key and give consent first.")
        else:
            try:
                ms = backend.test_connection()
                st.success(f"Gemini answered in {ms:.0f} ms.")
            except AppError as exc:
                st.error(exc.user_message)
    if c3.button("Forget key", key="w_settings_gem_forget"):
        services.log().redactor.unregister(fb.get("api_key"))
        fb["api_key"] = None
        state.set("settings.fallback", fb)
        st.info("The session key was removed.")
    st.caption(f"Key in use: {'session key' if fb.get('api_key') else ('app secret' if has_secret else 'none')}")


def _diagnostics() -> None:
    rt = services.get_runtime()
    report = rt.probe()
    data = report.model_dump(by_alias=True)
    st.download_button(
        "Download diagnostics JSON",
        json.dumps(data, indent=2),
        file_name="mmrag-diagnostics.json",
        mime="application/json",
        key="w_settings_diag_download",
        on_click="ignore",
        icon=":material/download:",
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Runtime**")
        st.json(
            {
                "state": rt.snapshot()["state"],
                "build": rt.build_id(),
                "bundle_id": rt.bundle_id(),
                "manifest_sha256": rt.bundle.manifest_sha if rt.bundle else None,
                "memory_cap_mb": rt.cfg.memory_cap_mb,
                "resident": rt.governor.resident() if rt.governor else [],
            }
        )
        st.markdown("**Egress allowlist**")
        st.code("\n".join(sorted(net.policy().allowed_hosts())))
    with c2:
        st.markdown("**Environment probe (FR-003)**")
        st.json(data["values"], expanded=False)
    if rt.router is not None:
        with st.expander("llama-server output (tail)"):
            st.code("\n".join(list(rt.router.tail)[-60:]) or "(empty)")


@st.dialog("Please confirm")
def _confirm(action: str) -> None:
    text = {"clear": "Clear this session's index?", "reset": "Reset this session (chat, index, uploads, keys)?", "unload": "Unload all idle models?"}[action]
    st.write(text)
    c1, c2 = st.columns(2)
    if c1.button("Confirm", type="primary", key=f"w_settings_danger_yes_{action}"):
        _do(action)
        st.rerun()
    if c2.button("Cancel", key=f"w_settings_danger_no_{action}"):
        st.rerun()


def _do(action: str) -> None:
    rt = services.get_runtime()
    log = services.log()
    if action == "clear":
        from app.pages.index_io import clear_index

        clear_index()
    elif action == "reset":
        state.reset_session(services.get_config().session_root)
    elif action == "unload" and rt.governor is not None:
        done = rt.governor.unload_all(sid=state.sid(), keep=("emb-text",))
        log.info("ui", "danger.unload_all", sid=state.sid(), unloaded=done)
    log.info("ui", f"danger.{action}", sid=state.sid())


def _danger() -> None:
    st.warning("These actions cannot be undone.", icon=":material/warning:")
    c1, c2, c3 = st.columns(3)
    if c1.button("Clear index", key="w_settings_danger_clear", icon=":material/delete:"):
        _confirm("clear")
    if c2.button("Reset session", key="w_settings_danger_reset", icon=":material/restart_alt:"):
        _confirm("reset")
    if c3.button("Unload all models", key="w_settings_danger_unload", icon=":material/eject:", disabled=services.get_runtime().governor is None):
        _confirm("unload")


def render() -> None:
    st.title("Settings")
    section = st.segmented_control("Section", SECTIONS, default=SECTIONS[0], key="w_settings_section", required=True, label_visibility="collapsed")
    ready = services.get_runtime().ready
    if section in ("Models & generation", "Danger zone") and not ready and section != "Danger zone":
        st.info("Model settings become available once the runtime is ready (see Diagnostics).")
        return
    {"Models & generation": _models, "Retrieval & history": _retrieval, "Fallback (Gemini)": _fallback, "Diagnostics": _diagnostics, "Danger zone": _danger}[
        section or SECTIONS[0]
    ]()
