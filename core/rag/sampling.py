"""Generation-parameter resolution (FR-061, ADR-013): card defaults -> user overrides -> backend
capability filter -> clamp -> request fields. Sampling values are data, never hard-coded in UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.schemas import SamplingPresets, SamplingValues

# name -> (min, max, step, default-if-card-silent)
RANGES: dict[str, tuple[float, float, float, float]] = {
    "temperature": (0.0, 2.0, 0.05, 0.7),
    "top_p": (0.0, 1.0, 0.01, 1.0),
    "top_k": (0, 200, 1, 0),
    "min_p": (0.0, 1.0, 0.01, 0.0),
    "repeat_penalty": (0.8, 2.0, 0.01, 1.0),
    "presence_penalty": (-2.0, 2.0, 0.05, 0.0),
    "frequency_penalty": (-2.0, 2.0, 0.05, 0.0),
    "max_tokens": (16, 4096, 16, 512),
    "seed": (-1, 2**31 - 1, 1, -1),
}
SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "repeat_penalty", "presence_penalty", "frequency_penalty")
ALL_FIELDS = (*SAMPLING_FIELDS, "max_tokens", "seed", "reasoning")
INT_FIELDS = {"top_k", "max_tokens", "seed"}

# Backend capability matrix: fields a backend honors (others are disabled in the UI and never sent).
BACKEND_FIELDS: dict[str, set[str]] = {
    "local": set(ALL_FIELDS),
    # Verified on gemini-3.5-flash-lite (EXP-GEMINI-01): seed, top_k, presence/frequency_penalty -> HTTP 400.
    "gemini": {"temperature", "top_p", "max_tokens"},
}
DISABLED_TOOLTIP = {
    "gemini": "Not supported by the Gemini OpenAI-compatible endpoint; not sent.",
}


@dataclass
class Effective:
    values: dict[str, Any]
    preset_id: str
    mode: str
    status: str
    source_url: str = ""
    revision: str = ""
    checked_at: str = ""
    modified: set[str] = field(default_factory=set)
    disabled: dict[str, str] = field(default_factory=dict)
    card: dict[str, Any] = field(default_factory=dict)


def modes_for(sampling: SamplingPresets, preset_id: str | None) -> list[str]:
    if preset_id and preset_id in sampling.presets:
        return list(sampling.presets[preset_id].modes)
    return ["default"]


def default_mode(sampling: SamplingPresets, preset_id: str | None) -> str:
    if preset_id and preset_id in sampling.presets:
        return sampling.presets[preset_id].default_mode
    return "default"


def card_values(sampling: SamplingPresets, preset_id: str | None, mode: str | None, family: str | None = None) -> tuple[dict[str, Any], str]:
    """Card values for a mode (status `verified`/`partial`), or the family fallback (`to_verify`)."""
    sp = sampling.presets.get(preset_id or "")
    values: SamplingValues | None = None
    status = "to_verify"
    if sp is not None:
        m = mode if mode in sp.modes else sp.default_mode
        values = sp.modes.get(m)
        status = sp.status
    if values is None or status == "to_verify":
        fb = sampling.family_fallback.get(family or "")
        if fb is not None:
            values = fb
            status = "to_verify"
    out: dict[str, Any] = {}
    for f in SAMPLING_FIELDS:
        v = getattr(values, f) if values is not None else None
        out[f] = v if v is not None else RANGES[f][3]
    out["max_tokens"] = RANGES["max_tokens"][3]
    out["seed"] = -1
    out["reasoning"] = bool(sp.reasoning.on_by_default) if sp is not None else False
    return out, status


def clamp(name: str, value: Any) -> Any:
    if name == "reasoning":
        return bool(value)
    lo, hi, _, _ = RANGES[name]
    v = min(max(float(value), lo), hi)
    return int(round(v)) if name in INT_FIELDS else round(v, 4)


def resolve(
    sampling: SamplingPresets,
    preset_id: str | None,
    *,
    mode: str | None = None,
    overrides: dict[str, Any] | None = None,
    backend: str = "local",
    family: str | None = None,
    reasoning_supported: bool = True,
    max_tokens_cap: int | None = None,
) -> Effective:
    sp = sampling.presets.get(preset_id or "")
    mode = mode or (sp.default_mode if sp else "default")
    card, status = card_values(sampling, preset_id, mode, family)
    values = dict(card)
    modified: set[str] = set()
    for k, v in (overrides or {}).items():
        if k in values and v is not None:
            cv = clamp(k, v)
            values[k] = cv
            if cv != card.get(k):
                modified.add(k)
    for k in list(values):
        if k != "reasoning":
            values[k] = clamp(k, values[k])
    if max_tokens_cap is not None:
        values["max_tokens"] = min(values["max_tokens"], max_tokens_cap)
    allowed = BACKEND_FIELDS.get(backend, set(ALL_FIELDS))
    disabled = {f: DISABLED_TOOLTIP.get(backend, "Not supported by this backend.") for f in ALL_FIELDS if f not in allowed}
    if not reasoning_supported:
        disabled["reasoning"] = "This model has no verified reasoning on/off control."
    return Effective(
        values=values,
        preset_id=preset_id or "",
        mode=mode,
        status=status,
        source_url=sp.source_url if sp else "",
        revision=sp.revision if sp else "",
        checked_at=sp.checked_at if sp else "",
        modified=modified,
        disabled=disabled,
        card=card,
    )


def request_fields(eff: Effective, backend: str = "local", reasoning_control: str = "none") -> dict[str, Any]:
    """Exactly the effective values the backend supports (I-GP-request-capture)."""
    allowed = BACKEND_FIELDS.get(backend, set(ALL_FIELDS)) - set(eff.disabled)
    body: dict[str, Any] = {}
    for f in (*SAMPLING_FIELDS, "max_tokens"):
        if f in allowed:
            body[f] = eff.values[f]
    if "seed" in allowed and eff.values.get("seed", -1) >= 0:
        body["seed"] = eff.values["seed"]
    if backend == "local" and reasoning_control == "chat_template_kwargs" and "reasoning" not in eff.disabled:
        body["chat_template_kwargs"] = {"enable_thinking": bool(eff.values["reasoning"])}
    return body
