"""Entrypoint: page config + runtime bootstrap (background) + shell. No logic here."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="mmrag-lab", page_icon=":material/hub:", layout="wide", initial_sidebar_state="expanded")

from app import services, shell, state  # noqa: E402

state.sid()  # initialise session keys before any widget
_runtime = services.get_runtime()  # starts the bundle bootstrap once per process (non-blocking)
shell.run(services.get_config().ui_mode)
