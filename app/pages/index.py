"""Index page (UX section 4): upload + validation, background indexing with step progress, stats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from app import services, state
from app.components import warmup
from core.config import DATASETS_DIR
from core.embed.indexer import UploadItem, run_index_job, stage_upload
from core.embed.preprocess import ALLOWED_EXT, validate_upload
from core.errors import AppError

SAMPLE_DIR = DATASETS_DIR / "synth_v1"
STEP_ICON = {"pending": "⏳", "running": "🔄", "done": "✅", "skipped": "⏭️", "failed": "❌"}


def sample_available() -> bool:
    return (SAMPLE_DIR / "manifest.json").exists()


def sample_files() -> list[Path]:
    man = json.loads((SAMPLE_DIR / "manifest.json").read_text())
    return [SAMPLE_DIR / p for p in man.get("sample_subset", [])]


def _job() -> Any:
    return services.get_jobs().get(state.get("index.job_id"))


def job_running() -> bool:
    j = _job()
    return j is not None and not j.finished


def start_job(items: list[UploadItem], label: str = "index") -> None:
    rt = services.get_runtime()
    _, gov = rt.require_ready()
    handle = services.index_handle(create=True)
    assert handle is not None
    embed = services.embed_service()
    params = services.embed_params().model_copy(deep=True)
    cfg = services.get_config()
    preview_dir = state.session_dir(cfg.session_root) / "previews"
    log = services.log()

    def work(job: Any) -> dict[str, Any]:
        return run_index_job(job, items=items, index=handle, governor=gov, embed=embed, params=params, telemetry=rt.telemetry, log=log, preview_dir=preview_dir)

    job = services.get_jobs().submit(state.sid(), label, work)
    state.set("index.job_id", job.id)
    log.info("index", "job.submitted", sid=state.sid(), job=job.id, items=len(items))


def start_sample_job() -> None:
    if job_running() or not sample_available():
        return
    cfg = services.get_config()
    items = []
    used = int(state.get("index.uploaded_bytes"))
    for p in sample_files():
        data = p.read_bytes()
        items.append(
            stage_upload(state.session_dir(cfg.session_root), p.name, data, used_bytes=0, quota_mb=10_000, max_pixels=services.embed_params().image.max_pixels)
        )
    state.set("index.uploaded_bytes", used)
    start_job(items, "sample")


def _validate_files(files: list[Any]) -> tuple[list[tuple[str, bytes]], list[tuple[str, str]]]:
    ok, bad = [], []
    used = int(state.get("index.uploaded_bytes"))
    quota = services.get_config().session_quota_mb
    for f in files:
        data = f.getvalue()
        try:
            validate_upload(f.name, data, used_bytes=used, session_quota_mb=quota, max_pixels=services.embed_params().image.max_pixels)
            ok.append((f.name, data))
            used += len(data)
        except AppError as exc:
            bad.append((f.name, exc.user_message))
    return ok, bad


def _on_start(uploader_key: str) -> None:
    files = state.widget_value(uploader_key) or []
    ok, _ = _validate_files(files)
    if not ok or job_running():
        return
    cfg = services.get_config()
    items = []
    used = int(state.get("index.uploaded_bytes"))
    for name, data in ok:
        items.append(
            stage_upload(
                state.session_dir(cfg.session_root),
                name,
                data,
                used_bytes=used,
                quota_mb=cfg.session_quota_mb,
                max_pixels=services.embed_params().image.max_pixels,
            )
        )
        used += len(data)
    state.set("index.uploaded_bytes", used)
    start_job(items)
    st.session_state["w_index_uploader_nonce"] = st.session_state.get("w_index_uploader_nonce", 0) + 1  # clears the uploader


def _on_cancel() -> None:
    j = _job()
    if j is not None:
        j.cancel()


@st.fragment(run_every=1)
def job_panel() -> None:
    j = _job()
    if j is None:
        st.caption("Job: idle. Choose files and press **Start indexing**.")
        return
    snap = j.snapshot()
    label = {
        "queued": "queued",
        "waiting": f"waiting for the model (queue position {snap['queue_position']})",
        "running": "running",
        "done": "done",
        "failed": "failed",
        "cancelled": "cancelled",
    }[snap["state"]]
    st.markdown(f"**Job:** {label} &nbsp; ({snap['elapsed_s']:.0f} s)")
    total = snap["total"] or 1
    eta = f" - ETA {snap['eta_s']:.0f} s" if snap.get("eta_s") else ""
    st.progress(min(snap["done"] / total, 1.0), text=f"{snap['done']}/{snap['total']} items{eta}")
    st.dataframe(
        [
            {
                "": STEP_ICON.get(s["state"], ""),
                "step": s["name"],
                "state": s["state"],
                "detail": s["detail"],
                "ms": round(s["dur_ms"]) if s["dur_ms"] else None,
            }
            for s in snap["steps"][-60:]
        ],
        hide_index=True,
        width="stretch",
        height=min(420, 38 + 35 * min(len(snap["steps"]), 11)),
    )
    if snap["state"] == "failed":
        st.error(snap["error"] or "Indexing failed.")
        if snap.get("event_id"):
            st.caption(f"Event id `{snap['event_id']}`")
    res = snap.get("result") or {}
    if j.finished and res:
        ok, skipped, failed = res.get("ok", []), res.get("skipped", []), res.get("failed", [])
        st.success(f"Indexed {len(ok)} file(s); skipped {len(skipped)}; failed {len(failed)}. Peak memory {res.get('peak_mb', 0):.0f} MB.")
        if skipped or failed:
            with st.expander("Skipped / failed items"):
                st.dataframe(
                    [{"file": x["name"], "status": "skipped", "reason": x["reason"]} for x in skipped]
                    + [{"file": x["name"], "status": "failed", "reason": x["reason"]} for x in failed],
                    hide_index=True,
                    width="stretch",
                )
        if not st.session_state.get(f"_job_seen_{j.id}"):
            st.session_state[f"_job_seen_{j.id}"] = True
            st.rerun(scope="app")


def stats_panel() -> None:
    h = services.index_handle()
    st.subheader("Index stats")
    if h is None or h.count() == 0:
        st.caption("The index is empty.")
        return
    s = h.stats()
    c = st.columns(5)
    c[0].metric("Items", s["items"])
    c[1].metric("Sources", s["sources"])
    c[2].metric("Index size", f"{s['index_bytes'] / 1e6:.2f} MB")
    c[3].metric("Bits / dim", f"{s['bits']} / {s['dim']}")
    c[4].metric("Peak memory", f"{s['peak_mb']:.0f} MB")
    st.caption("Per modality: " + ", ".join(f"{k}: {v}" for k, v in sorted(s["per_modality"].items())) + f" - build time {s['build_ms'] / 1000:.1f} s")
    if st.button("Compute recall@10 vs float32", key="w_index_recall", disabled=not h.keep_float32):
        r = h.recall_at_k(10)
        st.info(f"recall@10 (TurboVec {s['bits']}-bit vs float32 brute force): **{r:.3f}**" if r is not None else "Needs at least 2 items and float32 vectors.")


def render() -> None:
    st.title("Index")
    if not warmup.runtime_ready():
        warmup.warmup_screen()
        return
    p = services.embed_params()
    h = services.index_handle()
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        nonce = st.session_state.get("w_index_uploader_nonce", 0)
        files = st.file_uploader(
            "Upload files",
            type=[e.lstrip(".") for e in sorted(ALLOWED_EXT)],
            accept_multiple_files=True,
            key=f"w_index_uploader_{nonce}",
            help="Text, Markdown, images, WAV/MP3 audio, MP4/WebM video. ≤ 50 MB per file, ≤ 200 MB per session.",
        )
        ok, bad = _validate_files(files or [])
        for _name, why in bad:
            st.error(why, icon=":material/block:")
        if ok:
            st.caption(f"{len(ok)} file(s) ready: " + ", ".join(n for n, _ in ok[:8]) + (" …" if len(ok) > 8 else ""))
        with st.container(border=True):
            st.markdown("**Preprocessing** (edit in Playground → Embedding parameters)")
            st.caption(
                f"text {p.text.chunk_tokens}/{p.text.overlap_tokens} tokens · image ≤ {p.image.max_side_px}px · "
                f"audio {p.audio.window_s:.0f} s windows @ {p.audio.sample_rate_hz / 1000:.0f} kHz · "
                f"video {p.video.fps:g} fps × ≤ {p.video.max_frames} frames @ {p.video.frame_max_side_px}px"
            )
        locked = h is not None and h.count() > 0
        c1, c2 = st.columns(2)
        bits = c1.radio(
            "Index bits", [2, 4], index=[2, 4].index(int(state.get("index.bits"))), horizontal=True, key="w_index_bits", disabled=locked,
            help="Clear the index to change.",
        )  # fmt: skip
        keep = c2.checkbox("Keep float32 vectors", value=bool(state.get("index.keep_vectors")), key="w_index_keep_vectors", disabled=locked,
                           help="Needed by the Playground and Explorer.")  # fmt: skip
        if not locked:
            state.set("index.bits", int(bits or 4))
            state.set("index.keep_vectors", bool(keep))
        running = job_running()
        st.button(
            "Start indexing",
            width="stretch",
            key="w_index_start",
            type="primary",
            on_click=_on_start,
            args=(f"w_index_uploader_{nonce}",),
            disabled=running or not ok,
            icon=":material/play_arrow:",
        )
        b2, b3 = st.columns(2)
        b2.button("Cancel", key="w_index_cancel", on_click=_on_cancel, disabled=not running, icon=":material/stop:", width="stretch")
        b3.button(
            "Load sample data",
            width="stretch",
            key="w_index_sample",
            on_click=start_sample_job,
            disabled=running or not sample_available(),
            icon=":material/dataset:",
            help="Synthetic mini-set bundled with the repo" if sample_available() else "Sample dataset not present in this build.",
        )
    with right:
        job_panel()
    st.divider()
    stats_panel()
    from app.pages import index_io

    index_io.render_io()
