"""Index export / import / clear controls (FR-014, FR-063) shown under the Index page."""

from __future__ import annotations

import streamlit as st

from app import services, state
from core.errors import AppError
from core.index.export_import import export_index, import_index


@st.dialog("Clear the index?")
def _confirm_clear() -> None:
    st.write("This removes every indexed item of this session. Your chat history stays.")
    c1, c2 = st.columns(2)
    if c1.button("Clear index", type="primary", key="w_index_clear_yes"):
        clear_index()
        st.rerun()
    if c2.button("Cancel", key="w_index_clear_no"):
        st.rerun()


def clear_index() -> None:
    h = services.index_handle()
    if h is not None:
        h.close()
    import shutil

    cfg = services.get_config()
    shutil.rmtree(state.session_dir(cfg.session_root) / "index", ignore_errors=True)
    state.set("index.handle", None)
    services.log().info("ui", "index.cleared", sid=state.sid())


def render_io() -> None:
    h = services.index_handle()
    c1, c2, c3 = st.columns(3)
    with c1:
        if h is not None and h.count() > 0:
            if st.button("Prepare export", key="w_index_export_prepare", icon=":material/archive:"):
                state.set("index.export_blob", export_index(h))
            blob = state.get("index.export_blob")
            if blob:
                st.download_button("Download index .zip", blob, file_name="mmrag-index.zip", mime="application/zip", key="w_index_export", on_click="ignore")
        else:
            st.button("Prepare export", key="w_index_export_prepare", disabled=True, icon=":material/archive:")
    with c2, st.popover("Import index", icon=":material/unarchive:"):
        up = st.file_uploader("Index archive (.zip)", type=["zip"], key="w_index_import")
        if up is not None and st.button("Import", key="w_index_import_go", type="primary"):
            try:
                cfg = services.get_config()
                if h is not None:
                    h.close()
                new = import_index(
                    up.getvalue(),
                    state.session_dir(cfg.session_root) / "index",
                    expected_fingerprint=services.fingerprint(),
                    expected_dim=services.embed_params().dims.index_dim,
                )
                state.set("index.handle", new)
                services.log().info("ui", "index.imported", sid=state.sid(), items=new.count())
                st.success(f"Imported {new.count()} items.")
            except AppError as exc:
                st.error(exc.user_message)
    with c3:
        if st.button("Clear index", key="w_index_clear", disabled=h is None or h.count() == 0, icon=":material/delete:"):
            _confirm_clear()
