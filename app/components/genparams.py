"""Generation-parameters panel (FR-061, UX section 9): quick panel (Chat) + full view (Settings).

Defaults per model = the releaser's recommended values (bundle/sampling_presets.json).
Edits are remembered per model for the session; nothing here triggers a model load.
"""

from __future__ import annotations

import streamlit as st

from app import services, state
from core.rag.sampling import RANGES, Effective, default_mode, modes_for, resolve

LABELS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k (0 = off)",
    "min_p": "min_p",
    "repeat_penalty": "repeat_penalty",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "max_tokens": "max_tokens",
    "seed": "seed (-1 = random)",
}
HELP = {"repeat_penalty": "llama.cpp name; model cards call it repetition_penalty."}


def sampling_id(model_id: str) -> str:
    """Model preset id -> its sampling preset id (bundle/model_presets.json `sampling_preset`)."""
    rt = services.get_runtime()
    try:
        return rt.presets.get(model_id).sampling_preset or model_id
    except KeyError:
        return model_id


def current_mode(preset_id: str) -> str:
    modes = state.get("settings.gen_mode")
    return str(modes.get(preset_id) or default_mode(services.get_runtime().sampling, sampling_id(preset_id)))


def effective_for(
    preset_id: str, *, backend: str = "local", family: str | None = None, reasoning_supported: bool = True, n_ctx: int | None = None
) -> Effective:
    rt = services.get_runtime()
    overrides = state.get("settings.gen_overrides").get(preset_id, {})
    cap = max(16, (n_ctx or 8192) // 2)
    eff = resolve(
        rt.sampling,
        sampling_id(preset_id),
        mode=current_mode(preset_id),
        overrides=overrides,
        backend=backend,
        family=family,
        reasoning_supported=reasoning_supported,
        max_tokens_cap=cap,
    )
    eff.preset_id = preset_id
    return eff


def _on_change(preset_id: str, field: str, key: str) -> None:
    ov = state.get("settings.gen_overrides")
    ov.setdefault(preset_id, {})[field] = state.widget_value(key)
    state.set("settings.gen_overrides", ov)


def _on_mode(preset_id: str, key: str) -> None:
    modes = state.get("settings.gen_mode")
    modes[preset_id] = state.widget_value(key)
    state.set("settings.gen_mode", modes)
    ov = state.get("settings.gen_overrides")
    ov.pop(preset_id, None)  # selecting a mode replaces all values with that mode's card values
    state.set("settings.gen_overrides", ov)
    state.clear_widget_keys(f"w_gp_v_{preset_id}_")


def _on_reset(preset_id: str) -> None:
    ov = state.get("settings.gen_overrides")
    ov.pop(preset_id, None)
    state.set("settings.gen_overrides", ov)
    state.clear_widget_keys(f"w_gp_v_{preset_id}_")


def panel(
    loc: str,
    preset_id: str,
    *,
    label: str,
    backend: str = "local",
    family: str | None = None,
    reasoning_supported: bool = True,
    n_ctx: int | None = None,
    compact: bool = False,
) -> Effective:
    rt = services.get_runtime()
    eff = effective_for(preset_id, backend=backend, family=family, reasoning_supported=reasoning_supported, n_ctx=n_ctx)
    badge = {"verified": ":green-badge[card values]", "partial": ":orange-badge[partial card values]"}.get(
        eff.status, ":red-badge[unverified: family fallback]"
    )
    st.markdown(f"**{label}** &nbsp; preset `{eff.mode}` {badge}")
    if eff.source_url:
        rev = f"rev `{eff.revision[:10]}`" if eff.revision else "rev n/a"
        st.caption(f"Source: [{eff.source_url.removeprefix('https://')}]({eff.source_url}) - {rev}, checked {eff.checked_at}")
    modes = modes_for(rt.sampling, sampling_id(preset_id))
    if len(modes) > 1:
        mkey = f"w_gp_{loc}_mode_{preset_id}"
        if mkey not in st.session_state:
            st.session_state[mkey] = eff.mode
        st.segmented_control("Mode", modes, key=mkey, on_change=_on_mode, args=(preset_id, mkey), required=True)
    cols = st.columns(2 if compact else 3)
    for i, field in enumerate(("temperature", "top_p", "top_k", "min_p", "repeat_penalty", "presence_penalty", "frequency_penalty", "max_tokens", "seed")):
        lo, hi, step, _ = RANGES[field]
        if field == "max_tokens" and n_ctx:
            hi = max(lo, n_ctx // 2)
        key = f"w_gp_v_{preset_id}_{loc}_{field}"
        value = eff.values[field]
        disabled = field in eff.disabled
        help_ = eff.disabled.get(field) or HELP.get(field)
        mark = " •" if field in eff.modified else ""
        with cols[i % len(cols)]:
            if field in ("top_k", "max_tokens", "seed"):
                st.number_input(
                    LABELS[field] + mark,
                    min_value=int(lo),
                    max_value=int(hi),
                    value=int(value),
                    step=int(step),
                    key=key,
                    disabled=disabled,
                    help=help_,
                    on_change=_on_change,
                    args=(preset_id, field, key),
                )
            else:
                st.slider(
                    LABELS[field] + mark,
                    min_value=float(lo),
                    max_value=float(hi),
                    value=float(value),
                    step=float(step),
                    key=key,
                    disabled=disabled,
                    help=help_,
                    on_change=_on_change,
                    args=(preset_id, field, key),
                )
    rkey = f"w_gp_v_{preset_id}_{loc}_reasoning"
    st.toggle(
        "Reasoning (thinking)",
        value=bool(eff.values["reasoning"]),
        key=rkey,
        disabled="reasoning" in eff.disabled,
        help=eff.disabled.get("reasoning") or "Off answers faster; on lets the model think first.",
        on_change=_on_change,
        args=(preset_id, "reasoning", rkey),
    )
    c1, c2 = st.columns([1, 2])
    c1.button("Reset to model default", key=f"w_gp_{loc}_reset_{preset_id}", on_click=_on_reset, args=(preset_id,), icon=":material/restart_alt:")
    if eff.modified:
        c2.markdown(":orange-badge[modified] " + ", ".join(sorted(eff.modified)))
    return eff
