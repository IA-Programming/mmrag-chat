"""Explorer (UX section 6): 2-D PCA of the index, colored by modality; selection -> item preview."""

from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st

from app import services, state
from app.components import source_card, warmup
from core.analysis import pca_2d
from core.errors import AppError
from core.runtime.governor import ALWAYS_RESIDENT

MOD_COLORS = {"text": "#4FD1C5", "image": "#F2A65A", "audio": "#B392F0", "video": "#FF7B72", "query": "#E8EEF4"}


@st.cache_data(ttl=3600, max_entries=4, show_spinner=False)
def _project(content_hash: str, _vectors: np.ndarray, max_points: int, seed: int) -> dict[str, Any]:
    """Cached by index content hash (FR-050 AC3); the array itself is not hashed."""
    v = np.asarray(_vectors, dtype=np.float32)
    coords, idx, ratio = pca_2d(v, max_points=max_points, seed=seed)
    sub = v[idx] if len(idx) else v
    mu = sub.mean(axis=0, keepdims=True) if len(sub) else np.zeros((1, v.shape[1]), np.float32)
    _, _, vt = np.linalg.svd(sub - mu, full_matrices=False) if len(sub) >= 3 else (None, None, np.eye(2, v.shape[1], dtype=np.float32))
    comps = vt[:2]
    for i in range(min(2, comps.shape[0])):
        j = int(np.argmax(np.abs(comps[i])))
        if comps[i, j] < 0:
            comps[i] = -comps[i]
    return {"coords": coords, "idx": idx, "ratio": ratio, "mean": mu, "comps": comps}


def render() -> None:
    st.title("Explorer")
    if not warmup.runtime_ready():
        warmup.warmup_screen()
        return
    h = services.index_handle()
    if h is None or h.count() < 3 or not h.keep_float32:
        st.info("Index at least three items (with float32 vectors kept) to explore the embedding space.")
        return
    import altair as alt
    import pandas as pd

    c1, c2, c3 = st.columns([1, 1, 2], vertical_alignment="bottom")
    max_points = int(c1.number_input("Max points", 100, 5000, 2000, 100, key="w_explorer_max_points"))
    overlay = c2.toggle("Query overlay", key="w_explorer_overlay")
    q = c3.text_input("Overlay query", key="w_explorer_query", disabled=not overlay)
    proj = _project(h.content_hash(), h.vectors, max_points, 0)
    mods = h.modalities()
    idx = proj["idx"]
    df = pd.DataFrame({"x": proj["coords"][:, 0], "y": proj["coords"][:, 1], "item_id": idx.astype(int), "modality": [str(mods[i]) for i in idx]})
    neighbors: list[int] = []
    if overlay and q.strip():
        try:
            rt = services.get_runtime()
            _, gov = rt.require_ready()
            with gov.turn(state.sid(), "explorer"), gov.using(ALWAYS_RESIDENT, sid=state.sid()):
                qv = services.embed_service().embed_query(q, model=ALWAYS_RESIDENT)[: h.dim]
            qv = qv / (np.linalg.norm(qv) or 1.0)
            qp = ((qv[None, :] - proj["mean"]) @ proj["comps"].T)[0]
            hits = h.search(qv, 5)
            neighbors = [x.item_id for x in hits]
            df = pd.concat([df, pd.DataFrame({"x": [qp[0]], "y": [qp[1]], "item_id": [-1], "modality": ["query"]})], ignore_index=True)
        except AppError as exc:
            st.error(exc.user_message)
    df["neighbor"] = df["item_id"].isin(neighbors)
    sel = alt.selection_point(fields=["item_id"], name="pick", on="click")
    chart = (
        alt.Chart(df)
        .mark_circle()
        .encode(
            x=alt.X("x", title=f"PC1 ({proj['ratio'][0] * 100:.1f} %)"),
            y=alt.Y("y", title=f"PC2 ({proj['ratio'][1] * 100:.1f} %)"),
            color=alt.Color("modality", scale=alt.Scale(domain=list(MOD_COLORS), range=list(MOD_COLORS.values()))),
            size=alt.condition("datum.neighbor || datum.modality == 'query'", alt.value(180), alt.value(50)),
            tooltip=["item_id", "modality"],
            opacity=alt.condition(sel, alt.value(0.95), alt.value(0.45)),
        )
        .add_params(sel)
        .properties(height=520)
    )
    left, right = st.columns([2, 1])
    with left:
        event = st.altair_chart(chart, key="w_explorer_chart", on_select="rerun", selection_mode="pick", width="stretch")
    picked = None
    try:
        pts = event.selection.get("pick") if event else None  # type: ignore[union-attr]
        if pts:
            picked = int(pts[0]["item_id"])
    except Exception:
        picked = None
    if picked is not None and picked >= 0:
        state.set("explorer.selection", picked)
    with right:
        sel_id = state.get("explorer.selection")
        if sel_id is None:
            st.caption("Click a point to see the item and its nearest neighbors.")
        else:
            it = h.item(int(sel_id))
            if it is None:
                st.caption("Item not found (index changed).")
            else:
                row = {
                    "label": "selected",
                    "item_id": it.item_id,
                    "score": 1.0,
                    "modality": it.modality,
                    "span": it.span,
                    "filename": it.filename,
                    "text": it.text or "",
                    "preview_path": it.preview_path,
                    "included": "selected",
                }
                source_card.source_card(row, key="explorer_sel")
                st.markdown("**Nearest neighbors**")
                for nb in h.search(h.vectors[it.item_id], 6)[1:]:
                    st.caption(f"{nb.modality}: {nb.filename} ({nb.score:.3f})")
        if neighbors:
            st.markdown("**Query neighbors (top 5)**")
            for nb in h.hits(neighbors, [0.0] * len(neighbors)):
                st.caption(f"{nb.modality}: {nb.filename}")
