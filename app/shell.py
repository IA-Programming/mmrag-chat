"""UI shell (FR-082): st.navigation (default) or st.tabs, built from the same page render() functions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st

from app.components import chrome

PageFn = Callable[[], None]
_PAGES: dict[str, Any] = {}
_MODE = {"value": "navigation"}


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


def run(ui_mode: str = "navigation") -> None:
    _MODE["value"] = ui_mode
    if ui_mode == "tabs":
        chrome.sidebar()
        items = specs()
        tabs = st.tabs([f"{title}" for _, title, _, _, _ in items])
        for tab, (_, _, _, fn, _) in zip(tabs, items, strict=True):
            with tab:
                fn()
        return
    pages = []
    _PAGES.clear()
    for pid, title, icon, fn, default in specs():
        page = st.Page(fn, title=title, icon=icon, url_path=pid, default=default)
        _PAGES[pid] = page
        pages.append(page)
    current = st.navigation(pages, position="sidebar")
    chrome.sidebar()
    current.run()


def link(container: Any, page_id: str, label: str, icon: str | None = None) -> None:
    if _MODE["value"] == "navigation" and page_id in _PAGES:
        container.page_link(_PAGES[page_id], label=label, icon=icon)
    else:
        container.caption(f"Open the **{label.removeprefix('Go to ').removeprefix('Open ')}** tab.")


def switch(page_id: str) -> None:
    if _MODE["value"] == "navigation" and page_id in _PAGES:
        st.switch_page(_PAGES[page_id])
    st.rerun()
