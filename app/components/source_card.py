"""Source cards (FR-024): modality, score, location, preview and prompt inclusion status."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

MOD_ICON = {"text": ":material/description:", "image": ":material/image:", "audio": ":material/graphic_eq:", "video": ":material/movie:"}
INCLUDED_COLOR = {"included": "green", "truncated": "orange"}


def location(span: dict[str, Any]) -> str:
    if "t" in span:
        a, b = span["t"]
        return f"{a:.1f}-{b:.1f} s" if a != b else f"at {a:.1f} s"
    if "chars" in span:
        return f"chunk {span.get('chunk', '?')} (chars {span['chars'][0]}-{span['chars'][1]})"
    if "bbox" in span:
        return "whole image"
    return ""


def source_card(row: dict[str, Any], key: str) -> None:
    with st.container(border=True):
        inc = row["included"]
        color = INCLUDED_COLOR.get(inc, "gray" if inc.startswith("dropped") else "violet")
        st.markdown(
            f"{MOD_ICON.get(row['modality'], '')} **[{row['label']}] {row['filename']}** &nbsp; :blue-badge[score {row['score']:.3f}] :{color}-badge[{inc}]"
        )
        st.caption(location(row.get("span", {})))
        prev = row.get("preview_path")
        if row["modality"] == "text":
            st.text(row.get("text", "")[:600])
        elif prev and Path(prev).exists():
            if row["modality"] in ("image", "video"):
                st.image(prev, width=220)
            elif row["modality"] == "audio":
                st.audio(prev, format="audio/wav")
        else:
            st.caption("(preview not available)")
