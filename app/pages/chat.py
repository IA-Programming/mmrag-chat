"""Chat page (UX section 3): multimodal RAG chat with sources, stats, export, queue + Stop."""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from app import services, state
from app.components import chrome, genparams, source_card, warmup
from core.embed.preprocess import thumbnail, validate_upload
from core.errors import AppError
from core.rag.export import to_export, to_json, to_markdown
from core.rag.history import POLICIES, HistoryConfig
from core.rag.pipeline import Attachment, RetrievalConfig, TurnRequest
from core.schemas import Capabilities

GEMINI_ID = "gemini"
CHAT_FILE_TYPES = ["png", "jpg", "jpeg", "webp", "wav", "mp3"]


def _generator_options() -> tuple[list[str], dict[str, str]]:
    rt = services.get_runtime()
    opts = [p.id for p in rt.generators()]
    labels = {p.id: p.label for p in rt.generators()}
    if services.remote_backend() is not None:
        opts.append(GEMINI_ID)
        cfg = services.get_config()
        labels[GEMINI_ID] = f"Gemini ({state.get('settings.fallback').get('model') or cfg.gemini.model})"
    return opts, labels


def selected_generator() -> str | None:
    opts, _ = _generator_options()
    cur = state.get("settings.generator_id")
    if cur not in opts:
        cur = opts[0] if opts else None
        state.set("settings.generator_id", cur)
    return cur  # type: ignore[no-any-return]


def export_buttons(location: str) -> None:
    msgs = state.get("chat.messages")
    rt = services.get_runtime()
    exp = to_export(msgs, bundle_id=rt.bundle_id(), preset_id=state.get("settings.generator_id"), redactor=services.log().redactor)
    c1, c2 = st.columns(2)
    c1.download_button(
        "Chat .md",
        to_markdown(exp),
        file_name="mmrag-chat.md",
        mime="text/markdown",
        key=f"w_chat_export_md_{location}",
        on_click="ignore",
        disabled=not msgs,
        icon=":material/download:",
    )
    c2.download_button(
        "Chat .json",
        to_json(exp),
        file_name="mmrag-chat.json",
        mime="application/json",
        key=f"w_chat_export_json_{location}",
        on_click="ignore",
        disabled=not msgs,
        icon=":material/data_object:",
    )


def _render_message(i: int, m: dict[str, Any]) -> None:
    with st.chat_message(m["role"]):
        for th in m.get("thumbs", []):
            st.image(th, width=160)
        for a in m.get("attachments", []):
            if a["modality"] == "audio":
                st.caption(f":material/graphic_eq: audio attachment ({a['bytes'] / 1000:.0f} kB)")
        if m.get("error"):
            st.error(m["error"])
            if m.get("event_id"):
                st.caption(f"Event id `{m['event_id']}` - details on the Logs page.")
        if m.get("text"):
            st.markdown(m["text"])  # unsafe_allow_html stays False (NFR-SEC-LLM)
        if m["role"] != "assistant":
            return
        if m.get("reasoning"):
            with st.expander("Reasoning"):
                st.text(m["reasoning"][:4000])
        srcs = m.get("sources", [])
        with st.expander(f"Sources ({len(srcs)})"):
            if not srcs:
                st.caption("No sources were retrieved for this turn.")
            for j, s in enumerate(srcs):
                source_card.source_card(s, key=f"src_{i}_{j}")
        with st.expander("Turn stats"):
            _stats_table(m.get("stats", {}))
        with st.expander("Effective parameters"):
            st.json(m.get("stats", {}).get("effective_params", {}), expanded=True)


def _stats_table(stats: dict[str, Any]) -> None:
    if not stats:
        st.caption("No stats.")
        return
    top = {
        k: stats.get(k)
        for k in (
            "model",
            "backend",
            "queue_ms",
            "embed_ms",
            "retrieve_ms",
            "swap_ms",
            "ttft_ms",
            "decode_tok_s",
            "prefill_tok_s",
            "prompt_tokens",
            "completion_tokens",
            "peak_mb",
            "total_ms",
            "extra_generation_calls",
            "finish",
            "stopped",
        )
    }
    st.dataframe([{"metric": k, "value": "" if v is None else str(v)} for k, v in top.items() if v is not None], hide_index=True, width="stretch")
    if stats.get("prompt_breakdown"):
        st.caption("Prompt budget (tokens)")
        st.dataframe([{"part": k, "tokens": v} for k, v in stats["prompt_breakdown"].items()], hide_index=True, width="stretch")
    if stats.get("swap_steps"):
        st.caption("Model swap / pipeline steps")
        st.dataframe(stats["swap_steps"], hide_index=True, width="stretch")
    if stats.get("rewritten_query"):
        st.caption(f"Rewritten query: {stats['rewritten_query']}")


def _attachment_from(value: Any) -> tuple[Attachment | None, list[bytes], str | None]:
    """Turn chat_input files/audio into one validated Attachment (FR-020 AC2)."""
    files = list(getattr(value, "files", None) or [])
    audio = getattr(value, "audio", None)
    if audio is not None:
        files.append(audio)
    if not files:
        return None, [], None
    f = files[0]
    data = f.getvalue()
    name = getattr(f, "name", None) or "recording.wav"
    try:
        kind = validate_upload(name if "." in name else name + ".wav", data, max_pixels=services.embed_params().image.max_pixels)
    except AppError as exc:
        return None, [], exc.user_message
    if kind.modality not in ("image", "audio"):
        return None, [], "Chat attachments must be an image or an audio clip (index documents and videos on the Index page)."
    thumbs = [thumbnail(data, 320)] if kind.modality == "image" else []
    note = "Only the first attachment is used." if len(files) > 1 else None
    return Attachment(kind.modality, data, kind.mime, name), thumbs, note


def _empty_state() -> None:
    from app import shell
    from app.pages import index as index_page

    st.info("No data indexed yet. Upload your own files on the Index page, or load the bundled sample data.", icon=":material/info:")
    c1, c2 = st.columns(2)
    shell.link(c1, "index", "Go to Index", ":material/database:")
    if c2.button("Load sample data", key="w_chat_sample", icon=":material/dataset:", disabled=not index_page.sample_available()):
        index_page.start_sample_job()
        shell.switch("index")


def render() -> None:
    st.title("Chat")
    if not warmup.runtime_ready():
        warmup.warmup_screen()
        return
    rt = services.get_runtime()
    opts, labels = _generator_options()
    gen_id = selected_generator()
    h = services.index_handle()
    n_items = h.count() if h else 0

    c1, c2, c3, c4 = st.columns([3, 2, 2, 1.3], vertical_alignment="bottom")
    with c1:
        if opts:
            picked = st.selectbox("Model", opts, index=opts.index(gen_id) if gen_id in opts else 0, format_func=lambda o: labels.get(o, o), key="w_chat_model")
            if picked != gen_id:  # selecting never loads a model (FR-060 AC2); the swap is lazy
                state.set("settings.generator_id", picked)
                gen_id = picked
        else:
            st.warning("No generator is shipped in this bundle.")
    backend = "gemini" if gen_id == GEMINI_ID else "local"
    preset = rt.presets.get(gen_id) if gen_id and gen_id != GEMINI_ID else None
    with c2, st.popover("Generation parameters", icon=":material/tune:", width="stretch"):
        if gen_id:
            genparams.panel(
                "chat",
                gen_id,
                label=labels.get(gen_id, gen_id),
                backend=backend,
                family=preset.family if preset else None,
                reasoning_supported=bool(preset and preset.capabilities.reasoning_control == "chat_template_kwargs"),
                n_ctx=preset.n_ctx if preset else None,
                compact=True,
            )
    with c3:
        pol = state.get("settings.history_policy")
        new_pol = st.selectbox(
            "History",
            POLICIES,
            index=POLICIES.index(pol),
            key="w_chat_history",
            format_func=lambda p: f"window ({state.get('settings.history_window')})" if p == "window" else p,
        )
        if new_pol != pol:
            state.set("settings.history_policy", new_pol)
    with c4:
        r = dict(state.get("settings.retrieval"))
        k = int(st.number_input("Sources k", 1, 10, int(r["k"]), key="w_chat_k"))
        if k != int(r["k"]):
            r["k"] = k
            state.set("settings.retrieval", r)

    if n_items == 0:
        _empty_state()

    msgs: list[dict[str, Any]] = state.get("chat.messages")
    for i, m in enumerate(msgs):
        _render_message(i, m)

    value = st.chat_input(
        "Ask about your indexed data (attach an image or record audio)",
        key="w_chat_input",
        accept_file=True,
        file_type=CHAT_FILE_TYPES,
        accept_audio=True,
        submit_mode="stop",
        disabled=(n_items == 0 or gen_id is None),
    )
    if value is None:
        return
    text = (getattr(value, "text", None) if not isinstance(value, str) else value) or ""
    att, thumbs, note = _attachment_from(value) if not isinstance(value, str) else (None, [], None)
    if not text.strip() and att is None:
        st.toast("Type a question or attach an image/audio clip.")
        if note:
            st.error(note)
        return
    if note and att is None:
        st.error(note)
        return
    user_msg = {"role": "user", "text": text, "attachments": [att.meta()] if att else [], "thumbs": thumbs, "ts": time.time()}
    history = [dict(m) for m in msgs]
    msgs.append(user_msg)
    _render_message(len(msgs) - 1, user_msg)
    if note:
        st.caption(note)
    _run_turn(text, att, history, gen_id, preset, backend)


def _run_turn(text: str, att: Attachment | None, history: list[dict[str, Any]], gen_id: str | None, preset: Any, backend: str) -> None:
    fb = state.get("settings.fallback")
    eff = genparams.effective_for(
        gen_id or "",
        backend=backend,
        family=preset.family if preset else None,
        reasoning_supported=bool(preset and preset.capabilities.reasoning_control == "chat_template_kwargs"),
        n_ctx=preset.n_ctx if preset else None,
    )
    r = state.get("settings.retrieval")
    req = TurnRequest(
        sid=state.sid(),
        query=text,
        history=history,
        preset=preset,
        effective=eff,
        history_cfg=HistoryConfig(state.get("settings.history_policy"), int(state.get("settings.history_window")), state.get("settings.contextualize")),
        retrieval=RetrievalConfig(k=int(r["k"]), threshold=float(r["threshold"]), quota=dict(r.get("quota") or {})),
        attachment=att,
        backend_name=backend,
        gemini_caps=Capabilities(image=True, audio=True),
        allow_media_to_remote=bool(fb.get("media_ok")),
    )
    record: dict[str, Any] = {"role": "assistant", "text": "", "sources": [], "stats": {}, "ts": time.time()}
    try:
        pipe = services.pipeline()
    except AppError as exc:
        record.update(text="", error=exc.user_message, event_id=exc.event_id)
        state.get("chat.messages").append(record)
        st.rerun()
    with st.chat_message("assistant"):
        status = st.status("Working…", expanded=False)

        def on_status(msg: str) -> None:
            status.update(label=msg[:120].capitalize())
            chrome.live_update(msg, force=True)

        def live(chunks: Any) -> Any:
            for c in chunks:
                chrome.live_update("generating")
                yield c

        turn = pipe.turn(req, services.index_handle(), on_status=on_status)
        try:
            st.write_stream(live(turn.stream()))
        finally:
            # Runs on success, on Stop (script stopped) and on errors: the turn is always recorded.
            record["text"] = turn.answer
            record["reasoning"] = turn.reasoning
            record["sources"] = turn.source_rows()
            record["stats"] = turn.stats or {"stopped": True}
            if not turn.finished:
                record["stats"]["stopped"] = True
            if turn.error:
                record["error"] = turn.error.user_message
                record["event_id"] = turn.error.event_id
            state.get("chat.messages").append(record)
        status.update(label="Done" if not turn.error else "Failed", state="complete" if not turn.error else "error")
    st.rerun()
