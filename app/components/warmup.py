"""Warm-up screen (FR-002): shown on model pages until the runtime is ready; never a traceback."""

from __future__ import annotations

import streamlit as st

from app import services

STATE_ICON = {"pending": "⏳", "downloading": "⬇️", "verifying": "🔎", "ready": "✅", "cached": "✅", "failed": "❌"}


def runtime_ready() -> bool:
    return services.get_runtime().ready


@st.fragment(run_every=1)
def _progress() -> None:
    rt = services.get_runtime()
    snap = rt.snapshot()
    if rt.ready:
        st.rerun(scope="app")
    bundle = snap.get("bundle")
    if snap["state"] == "failed":
        st.error(snap["error"] or "The runtime failed to start.")
        if snap.get("event_id"):
            st.caption(f"Event id: `{snap['event_id']}` (see the Logs page)")
        if st.button("Retry", key="w_warmup_retry", icon=":material/refresh:"):
            rt.retry()
            st.rerun()
        return
    phase = {"idle": "Starting…", "bootstrapping": "Preparing the frozen runtime…", "starting": "Starting the local model server…"}.get(
        snap["state"], snap["state"]
    )
    st.markdown(f"**{phase}** &nbsp; ({snap.get('elapsed_s', 0):.0f} s)")
    if bundle:
        st.progress(min(bundle["percent"] / 100.0, 1.0), text=f"Overall {bundle['percent']:.0f} %")
        for a in bundle["assets"]:
            frac = 1.0 if a["state"] in ("ready", "cached") else (a["done_mb"] / a["size_mb"] if a["size_mb"] else 0.0)
            st.progress(min(frac, 1.0), text=f"{STATE_ICON.get(a['state'], '•')} {a['name']} - {a['done_mb']:.0f}/{a['size_mb']:.0f} MB ({a['state']})")
            if a.get("error"):
                st.caption(f":red[{a['error']}]")


def warmup_screen() -> None:
    st.title("Warming up")
    st.write("Preparing the frozen runtime; the first visit after the app has been idle can take a few minutes.")
    _progress()
    st.caption("Settings → Diagnostics and the Logs page stay available in the navigation while this runs.")
