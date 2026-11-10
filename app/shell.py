"""UI shell (FR-082): st.navigation with one st.Page per render() function.

EXP-UI-01 (experiments/ledger.jsonl) compared st.navigation with st.tabs built from the same
render() functions: navigation executes 1 page body per rerun instead of 6, has a 24 ms vs 51 ms
p95 rerun latency, keeps state and gives per-page URLs. The tabs shell was deleted (FR-082 AC2).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st

from app.components import chrome

PageFn = Callable[[], None]
_PAGES: dict[str, Any] = {}


def specs() -> list[tuple[str, str, str, PageFn, bool]]:
    from app.pages import chat, explorer, index, logs, playground, settings

    return [
        ("chat", "Chat", ":material/chat:", chat.render, True),
        ("index", "Index", ":material/database:", index.render, False),
        ("playground", "Playground", ":material/tune:", playground.render, False),
        ("explorer", "Explorer", ":material/scatter_plot:", explorer.render, False),
        ("settings", "Settings", ":material/settings:", settings.render, False),
        ("logs", "Logs", ":material/terminal:", logs.render, False),
    ]


def run() -> None:
    _PAGES.clear()
    pages = []
    for pid, title, icon, fn, default in specs():
        page = st.Page(fn, title=title, icon=icon, url_path=pid, default=default)
        _PAGES[pid] = page
        pages.append(page)
    current = st.navigation(pages, position="sidebar")
    chrome.sidebar()
    current.run()


def link(container: Any, page_id: str, label: str, icon: str | None = None) -> None:
    if page_id in _PAGES:
        container.page_link(_PAGES[page_id], label=label, icon=icon)


def switch(page_id: str) -> None:
    if page_id in _PAGES:
        st.switch_page(_PAGES[page_id])
    st.rerun()
