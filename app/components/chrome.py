"""Sidebar chrome (UX section 2): memory meter, model chips, queue, index count, exports."""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from app import services, state

LEVEL_ICON = {"ok": ":material/check_circle:", "amber": ":material/warning:", "red": ":material/error:"}
_LIVE: dict[str, Any] = {"slot": None, "last": 0.0}


def live_update(note: str = "", *, force: bool = False) -> None:
    """Fragments do not rerun while the script itself is busy (e.g. streaming a turn), so long
    actions call this to keep the sidebar memory reading live (FR-004 AC3)."""
    slot = _LIVE.get("slot")
    now = time.monotonic()
    if slot is None or (not force and now - _LIVE["last"] < 1.0):
        return
    _LIVE["last"] = now
    s = services.get_runtime().telemetry.sample()
    color = {"ok": "green", "amber": "orange", "red": "red"}[s.level]
    slot.markdown(f":{color}[● live] {s.used_mb / 1000:.2f} GB ({s.percent:.0f} %) · {note[:60]}")


def _chip(label: str, value: str, color: str) -> None:
    st.markdown(f"**{label}** &nbsp; :{color}-badge[{value}]")


@st.fragment(run_every=2)
def resource_meter() -> None:
    """Refreshes alone every 2 s (FR-004), also while a generation streams."""
    rt = services.get_runtime()
    s = rt.telemetry.sample()
    pct = min(s.percent / 100.0, 1.0)
    st.progress(pct, text=f"{LEVEL_ICON[s.level]} Memory {s.used_mb / 1000:.2f} / {s.cap_mb / 1000:.2f} GB ({s.percent:.0f} %)")
    if s.level == "amber":
        st.caption(":orange[Above 80 % of the memory cap]")
    elif s.level == "red":
        st.caption(":red[Above 95 % of the memory cap]")
    snap = rt.snapshot()
    if rt.governor is None:
        _chip("Runtime", snap["state"], "blue" if snap["state"] != "failed" else "red")
        return
    gs = rt.governor.status()
    if "emb-mm" in gs.resident:
        emb = ("multimodal", "violet")
    elif "emb-text" in gs.resident:
        emb = ("text", "green")
    else:
        emb = ("unloaded", "gray")
    if gs.loading and gs.loading.startswith("emb"):
        emb = ("loading", "blue")
    _chip("Embedder", emb[0], emb[1])
    gens = [r for r in gs.resident if rt.governor.is_generator(r)]
    if gs.loading and not gs.loading.startswith("emb"):
        _chip("Generator", f"loading {gs.loading}", "blue")
    elif gens:
        busy = gs.in_use.get(gens[0], 0) > 0
        _chip("Generator", f"{'generating' if busy else 'ready'} - {rt.presets.get(gens[0]).label}", "orange" if busy else "green")
    else:
        _chip("Generator", "unloaded", "gray")
    mine = [t for t in gs.queue if t.sid == state.sid()]
    if mine:
        _chip("Queue", f"position {gs.queue.index(mine[0]) + 1}", "orange")
    elif gs.holder is not None:
        _chip("Queue", f"busy ({gs.holder.reason})", "blue")
    else:
        _chip("Queue", "idle", "gray")
    h = services.index_handle()
    _chip("Index", f"{h.count() if h else 0} items", "blue")


def sidebar() -> None:
    with st.sidebar:
        st.markdown("### mmrag-lab")
        st.caption("Multimodal RAG on small local models")
        resource_meter()
        _LIVE["slot"] = st.empty()
        from app.pages import chat as chat_page

        chat_page.export_buttons(location="sidebar")
