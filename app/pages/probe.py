"""Probe-set controls used by the Playground (FR-041 'Apply to probe set', FR-042 prefix A/B)."""

from __future__ import annotations

import streamlit as st

from app import services, state
from core.config import DATASETS_DIR
from core.embed.probe import load_probe, run_probe
from core.errors import AppError

PROBE_DIR = DATASETS_DIR / "synth_v1"
PREFIX_VARIANTS = {
    "model card": ("task: search result | query: {query}", "title: none | text: {content}"),
    "none": ("{query}", "{content}"),
    "question/answer": ("question: {query}", "answer: {content}"),
}


def _run(params, **kw):  # type: ignore[no-untyped-def]
    rt = services.get_runtime()
    _, gov = rt.require_ready()
    return run_probe(PROBE_DIR, params, governor=gov, embed=services.embed_service(), telemetry=rt.telemetry, sid=state.sid(), **kw)


def render_probe() -> None:
    st.markdown("**Apply to probe set** - re-embeds ≤ 20 bundled items and compares against the saved reference parameters.")
    if load_probe(PROBE_DIR) is None:
        st.caption("No probe set in this build.")
        return
    if st.button("Run probe (current vs reference)", key="w_pg_apply_probe", help="Loads the multimodal embedder; takes a minute on 2 cores."):
        try:
            with st.status("Running the probe set…", expanded=False) as s:
                cur = _run(services.embed_params())
                ref = _run(services.get_runtime().embed_defaults)
                s.update(label="Probe finished", state="complete")
            rows = [{"params": "current (session)", **cur}, {"params": "reference (bundle)", **ref}]
            st.dataframe(rows, hide_index=True, width="stretch")
            services.log().info("ui", "probe.run", sid=state.sid(), current=cur, reference=ref)
        except AppError as exc:
            st.error(exc.user_message)


def render_prefix_ab() -> None:
    st.markdown("Compare retrieval with the model-card prefixes against alternatives on the same probe set.")
    if load_probe(PROBE_DIR) is None:
        st.caption("No probe set in this build.")
        return
    picks = st.multiselect("Variants", list(PREFIX_VARIANTS), default=["model card", "none"], key="w_pg_prefix")
    if st.button("Run prefix A/B", key="w_pg_prefix_run", type="primary", disabled=len(picks) < 2):
        rows = []
        try:
            with st.status("Running variants…") as s:
                for name in picks:
                    qp, dp = PREFIX_VARIANTS[name]
                    s.update(label=f"Variant: {name}")
                    rows.append({"variant": name, **_run(services.embed_params(), prefix_query=qp, prefix_document=dp)})
                s.update(label="Done", state="complete")
            st.dataframe(rows, hide_index=True, width="stretch")
        except AppError as exc:
            st.error(exc.user_message)
