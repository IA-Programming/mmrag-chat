"""Playground (UX section 5): Matryoshka comparison, embedding parameters (+ probe set), prefix A/B."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import streamlit as st

from app import services, state
from app.components import warmup
from core.analysis import matryoshka_compare, turbovec_bytes_per_vector
from core.errors import AppError
from core.runtime.governor import ALWAYS_RESIDENT

SECTIONS = ["Matryoshka", "Embedding parameters", "Prefix A/B"]


@st.cache_data(ttl=3600, max_entries=16, show_spinner=False)
def _bytes_table(dims: tuple[int, ...]) -> list[dict[str, Any]]:
    return [
        {
            "dim": d,
            "float32 B/vec": d * 4,
            "TurboVec 4-bit B/vec": round(turbovec_bytes_per_vector(d, 4), 1),
            "TurboVec 2-bit B/vec": round(turbovec_bytes_per_vector(d, 2), 1),
        }
        for d in dims
    ]


def _embed_query(text: str) -> np.ndarray:
    rt = services.get_runtime()
    _, gov = rt.require_ready()
    emb = services.embed_service()
    with gov.turn(state.sid(), "playground"), gov.using(ALWAYS_RESIDENT, sid=state.sid()):
        return emb.embed_query(text, model=ALWAYS_RESIDENT)


def _matryoshka() -> None:
    h = services.index_handle()
    if h is None or h.count() < 2 or not h.keep_float32:
        st.info("Index at least two items (with float32 vectors kept) to compare Matryoshka dimensions.")
        return
    c1, c2, c3 = st.columns([3, 2, 1], vertical_alignment="bottom")
    q = c1.text_input("Query", key="w_pg_query", placeholder="e.g. Which planet has the tallest volcano?")
    dims = c2.pills("Dimensions", [768, 512, 256, 128], selection_mode="multi", default=[768, 512, 256, 128], key="w_pg_dims")
    k = c3.number_input("k", 1, 20, 5, key="w_pg_k")
    run = st.button("Run comparison", key="w_pg_run", type="primary")
    if run and not q.strip():
        st.warning("Type a query first.")
    if run and q.strip():
        try:
            with st.status("Embedding the query…") as s:
                t0 = time.perf_counter()
                qv = _embed_query(q)
                emb_ms = (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
                res = matryoshka_compare(h.vectors, qv[: h.dim], sorted(set(dims or [768]) | {768}, reverse=True), int(k))
                s.update(label=f"Done (embed {emb_ms:.0f} ms, search {(time.perf_counter() - t0) * 1000:.1f} ms)", state="complete")
            state.set("playground.last_run", {"query": q, "res": res, "k": int(k), "emb_ms": emb_ms})
        except AppError as exc:
            st.error(exc.user_message)
    run = state.get("playground.last_run")
    if not run:
        return
    res, k = run["res"], run["k"]
    cols = st.columns(len(res))
    for col, (d, r) in zip(cols, res.items(), strict=True):
        with col:
            head = "reference" if d == max(res) else f"overlap@{k} {r['overlap']:.2f}"
            st.markdown(f"**{d} dims** - {head}")
            if r["spearman"] is not None and d != max(res):
                st.caption(f"rank correlation {r['spearman']:.2f}")
            for rank, (iid, sc) in enumerate(zip(r["ids"], r["scores"], strict=True), start=1):
                hit = h.item(int(iid))
                label = f"{hit.modality}: {hit.filename} #{hit.chunk_idx}" if hit else str(iid)
                st.caption(f"{rank}. {label[:48]} ({sc:.3f})")
    st.markdown("**Storage per vector** (real TurboVec index files, fixed overhead removed)")
    st.dataframe(_bytes_table(tuple(sorted(res, reverse=True))), hide_index=True, width="stretch")


def _params() -> None:
    p = services.embed_params()
    st.caption("Changes apply to the next indexing job in this session. The defaults come from bundle/embed_params.json.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Text**")
        chunk = st.slider("chunk tokens", 64, 1024, p.text.chunk_tokens, 32, key="w_pg_text_chunk")
        overlap = st.slider("overlap tokens", 0, 256, p.text.overlap_tokens, 8, key="w_pg_text_overlap")
        st.markdown("**Image**")
        side = st.select_slider(
            "max side (px)",
            [256, 384, 512, 768, 1024],
            value=p.image.max_side_px if p.image.max_side_px in (256, 384, 512, 768, 1024) else 768,
            key="w_pg_img_max_side",
        )
    with c2:
        st.markdown("**Audio**")
        win = st.select_slider(
            "window (s)", [10.0, 15.0, 30.0, 60.0], value=p.audio.window_s if p.audio.window_s in (10.0, 15.0, 30.0, 60.0) else 30.0, key="w_pg_audio_window_s"
        )
        aov = st.select_slider(
            "overlap (s)", [0.0, 2.0, 5.0], value=p.audio.overlap_s if p.audio.overlap_s in (0.0, 2.0, 5.0) else 2.0, key="w_pg_audio_overlap_s"
        )
        sr = st.selectbox(
            "sample rate",
            [8000, 16000, 22050],
            index=[8000, 16000, 22050].index(p.audio.sample_rate_hz) if p.audio.sample_rate_hz in (8000, 16000, 22050) else 1,
            key="w_pg_audio_sr",
        )
    with c3:
        st.markdown("**Video**")
        fps = st.select_slider("fps", [0.5, 1.0, 2.0], value=p.video.fps if p.video.fps in (0.5, 1.0, 2.0) else 1.0, key="w_pg_video_fps")
        frames = st.select_slider("max frames", [8, 16, 32], value=p.video.max_frames if p.video.max_frames in (8, 16, 32) else 16, key="w_pg_video_max_frames")
        fside = st.select_slider(
            "frame side (px)",
            [224, 320, 448],
            value=p.video.frame_max_side_px if p.video.frame_max_side_px in (224, 320, 448) else 448,
            key="w_pg_video_frame_side",
        )
    st.caption(f"ubatch {p.server.ubatch} (server-side; changing it needs an embedder reload and is fixed in this bundle)")
    if overlap >= chunk:
        st.error("Overlap must be smaller than the chunk size.")
        return
    if st.button("Apply to this session", key="w_pg_apply", type="primary"):
        new = p.model_copy(deep=True)
        new.text.chunk_tokens, new.text.overlap_tokens = int(chunk), int(overlap)
        new.image.max_side_px = int(side)
        new.audio.window_s, new.audio.overlap_s, new.audio.sample_rate_hz = float(win), float(aov), int(sr)
        new.video.fps, new.video.max_frames, new.video.frame_max_side_px = float(fps), int(frames), int(fside)
        state.set("embed.params", new)
        services.log().info("ui", "embed.params.changed", sid=state.sid(), params=new.model_dump(by_alias=True))
        st.success("Applied. Re-index files to use the new parameters.")
    if st.button("Reset to bundle defaults", key="w_pg_reset"):
        state.set("embed.params", None)
        for k in [k for k in st.session_state if str(k).startswith("w_pg_")]:
            del st.session_state[k]
        st.rerun()
    st.divider()
    from app.pages import probe

    probe.render_probe()


def _prefix_ab() -> None:
    from app.pages import probe

    probe.render_prefix_ab()


def render() -> None:
    st.title("Playground")
    if not warmup.runtime_ready():
        warmup.warmup_screen()
        return
    section = st.segmented_control(
        "Section", SECTIONS, default=state.get("playground.section"), key="w_pg_section", required=True, label_visibility="collapsed"
    )
    state.set("playground.section", section)
    {"Matryoshka": _matryoshka, "Embedding parameters": _params, "Prefix A/B": _prefix_ab}[section or SECTIONS[0]]()
