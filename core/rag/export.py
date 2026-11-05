"""Chat export (FR-028): Markdown + JSON (ChatExport schema). Media bytes and keys never included."""

from __future__ import annotations

import json
from typing import Any

from core.obs.log import Redactor
from core.schemas import ChatExport, ExportAttachment, ExportSource, ExportTurn

APP_VERSION = "0.1.0"


def to_export(messages: list[dict[str, Any]], *, bundle_id: str | None, preset_id: str | None, redactor: Redactor) -> ChatExport:
    turns = []
    last_params: dict[str, Any] = {}
    for m in messages:
        stats = redactor.value(m.get("stats") or {})
        if stats.get("effective_params"):
            last_params = stats["effective_params"]
        turns.append(
            ExportTurn(
                role=m["role"],
                text=redactor.text(m.get("text") or ""),
                attachments=[ExportAttachment(**a) for a in m.get("attachments", [])],
                sources=[
                    ExportSource(
                        item_id=s["item_id"],
                        score=float(s["score"]),
                        modality=s["modality"],
                        span=s.get("span", {}),
                        included=s["included"],
                        source_name=s.get("filename"),
                    )
                    for s in m.get("sources", [])
                ],
                stats=stats,
                ts=float(m.get("ts", 0.0)),
            )
        )
    return ChatExport(schema_=1, app_version=APP_VERSION, bundle_id=bundle_id, preset_id=preset_id, effective_params=last_params, turns=turns)


def to_json(export: ChatExport) -> str:
    return json.dumps(export.model_dump(by_alias=True), indent=2, default=str)


def to_markdown(export: ChatExport) -> str:
    lines = ["# mmrag-lab chat export", "", f"- app version: {export.app_version}", f"- bundle: {export.bundle_id}", f"- model preset: {export.preset_id}", ""]
    for t in export.turns:
        who = "You" if t.role == "user" else "Assistant"
        lines.append(f"## {who}")
        lines.append("")
        if t.attachments:
            lines.append("_Attachments: " + ", ".join(f"{a.modality} ({a.bytes} bytes, sha256 {a.sha256_8})" for a in t.attachments) + "_")
            lines.append("")
        lines.append(t.text or "")
        lines.append("")
        if t.sources:
            lines.append("**Sources**")
            lines.append("")
            for i, s in enumerate(t.sources, start=1):
                lines.append(f"- [S{i}] {s.source_name} ({s.modality}, score {s.score:.3f}, {s.included}) span={json.dumps(s.span)}")
            lines.append("")
        if t.stats:
            keep = {k: v for k, v in t.stats.items() if k not in ("swap_steps",)}
            lines.append("<details><summary>Turn stats</summary>")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(keep, indent=2, default=str))
            lines.append("```")
            lines.append("</details>")
            lines.append("")
    return "\n".join(lines)
