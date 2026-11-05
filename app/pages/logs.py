"""Logs page (FR-071): live session-scoped tail with filters, pause and download."""

from __future__ import annotations

import time

import streamlit as st

from app import services, state
from core.obs.log import LEVELS, to_jsonl

COMPONENTS = ["(all)", "runtime", "governor", "bundle", "index", "chat", "jobs", "ui", "llama"]


def _events(limit: int | None = 300) -> list:  # type: ignore[type-arg]
    f = state.get("logs.filter")
    comp = None if f.get("component") in (None, "(all)") else f["component"]
    return services.log().ring.query(state.sid(), min_level=f.get("level", "INFO"), component=comp, text=f.get("text") or None, limit=limit)


@st.fragment(run_every=2)
def _tail() -> None:
    if state.get("logs.paused"):
        st.caption("Paused.")
        return
    evs = _events()
    st.caption(f"{len(evs)} event(s) - updated {time.strftime('%H:%M:%S')}")
    rows = [
        {
            "time": time.strftime("%H:%M:%S", time.localtime(e.ts)),
            "level": e.level,
            "component": e.component,
            "event": e.event,
            "ms": e.dur_ms,
            "fields": str(e.fields)[:300],
            "scope": "session" if e.session_scoped else "system",
        }
        for e in reversed(evs)
    ]
    st.dataframe(rows, hide_index=True, width="stretch", height=520)


def render() -> None:
    st.title("Logs")
    st.caption("Only this session's events plus system events are shown (never other testers' prompts).")
    f = dict(state.get("logs.filter"))
    c1, c2, c3, c4 = st.columns([1, 1, 2, 1], vertical_alignment="bottom")
    lvl = c1.selectbox("Level", list(LEVELS), index=list(LEVELS).index(f.get("level", "INFO")), key="w_logs_level")
    comp = c2.selectbox("Component", COMPONENTS, index=COMPONENTS.index(f.get("component") or "(all)"), key="w_logs_component")
    text = c3.text_input("Contains", value=f.get("text", ""), key="w_logs_text")
    paused = c4.toggle("Pause", value=bool(state.get("logs.paused")), key="w_logs_pause")
    state.set("logs.filter", {"level": lvl, "component": comp, "text": text})
    state.set("logs.paused", paused)
    _tail()
    st.download_button(
        "Download my events (.jsonl)",
        to_jsonl(_events(limit=None)),
        file_name="mmrag-session-events.jsonl",
        mime="application/json",
        key="w_logs_download",
        on_click="ignore",
        icon=":material/download:",
    )
