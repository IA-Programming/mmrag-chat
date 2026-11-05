"""Chat turn orchestration (FR-020..FR-029): validate -> ticket -> embed query (swap for media)
-> retrieve -> history/contextualize -> prompt budget -> ensure generator -> stream -> stats.

`Turn.stream()` yields text chunks (for st.write_stream); stats/sources are on the Turn after it
finishes. Closing the generator (Stop / rerun) closes the HTTP stream so decoding stops.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.embed.client import EmbedService
from core.embed.preprocess import audio_part, image_part, image_to_jpeg, pcm_to_wav, sha256_8
from core.errors import AppError
from core.index.store import Hit, IndexHandle
from core.llm.backends import Backend, LocalBackend
from core.obs.log import Logger
from core.rag.history import HistoryConfig, retrieval_query, rewrite_messages, select_history
from core.rag.prompt_builder import BuiltPrompt, build_prompt
from core.rag.sampling import Effective, request_fields
from core.runtime.governor import ALWAYS_RESIDENT, MM_EMBEDDER, Governor
from core.runtime.router import RouterClient
from core.runtime.telemetry import PeakSampler, Telemetry
from core.schemas import Capabilities, EmbedParams, ModelPreset


@dataclass
class Attachment:
    modality: str  # image | audio
    data: bytes
    mime: str
    name: str = "attachment"

    def meta(self) -> dict[str, Any]:
        return {"modality": self.modality, "sha256_8": sha256_8(self.data), "bytes": len(self.data)}


@dataclass
class RetrievalConfig:
    k: int = 4
    threshold: float = 0.0
    quota: dict[str, int] = field(default_factory=dict)  # modality -> max hits (0/absent = no cap)


@dataclass
class TurnRequest:
    sid: str
    query: str
    history: list[dict[str, Any]]
    preset: ModelPreset | None  # local generator (None when backend is gemini)
    effective: Effective
    history_cfg: HistoryConfig = field(default_factory=HistoryConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    attachment: Attachment | None = None
    backend_name: str = "local"
    gemini_caps: Capabilities | None = None
    allow_media_to_remote: bool = False


def retrieve(index: IndexHandle | None, qvec: np.ndarray | None, cfg: RetrievalConfig) -> list[Hit]:
    """Top-k with score threshold and per-modality quota (FR-021). Deterministic for fixed inputs."""
    if index is None or qvec is None or index.count() == 0:
        return []
    cand = index.search(qvec, max(cfg.k * 4, cfg.k))
    out: list[Hit] = []
    per: dict[str, int] = {}
    for h in cand:
        if h.score < cfg.threshold:
            continue
        cap = cfg.quota.get(h.modality, 0)
        if cap and per.get(h.modality, 0) >= cap:
            continue
        per[h.modality] = per.get(h.modality, 0) + 1
        out.append(h)
        if len(out) >= cfg.k:
            break
    return out


class Turn:
    def __init__(self, pipeline: ChatPipeline, req: TurnRequest, index: IndexHandle | None, on_status: Callable[[str], None]) -> None:
        self.p = pipeline
        self.req = req
        self.index = index
        self.on_status = on_status
        self.cancel = threading.Event()
        self.answer = ""
        self.reasoning = ""
        self.hits: list[Hit] = []
        self.prompt: BuiltPrompt | None = None
        self.stats: dict[str, Any] = {}
        self.error: AppError | None = None
        self.swap_steps: list[dict[str, Any]] = []
        self.finished = False

    # ------------------------------------------------------------------ helpers
    def _status(self, msg: str) -> None:
        try:
            self.on_status(msg)
        except Exception:
            pass

    def _step(self, name: str, fn: Callable[[], Any]) -> Any:
        t0 = time.perf_counter()
        self._status(name)
        out = fn()
        self.swap_steps.append({"step": name, "ms": round((time.perf_counter() - t0) * 1000, 1)})
        return out

    def _embed_query(self, gov: Governor, query_text: str) -> tuple[np.ndarray | None, float]:
        req, embed, sid = self.req, self.p.embed, self.req.sid
        if self.index is None or self.index.count() == 0:
            return None, 0.0
        t0 = time.perf_counter()
        if req.attachment is None:
            with gov.using(ALWAYS_RESIDENT, sid=sid):  # FR-021 AC1: text never loads emb-mm
                vec = embed.embed_query(query_text, model=ALWAYS_RESIDENT)
            return vec, (time.perf_counter() - t0) * 1000
        # ---- media query: swap sequence (FR-021 AC2)
        parts = self.p.media_parts(req.attachment, self.p.params)
        self._step("loading multimodal embedder", lambda: gov.ensure(MM_EMBEDDER, sid=sid, progress=self._status))
        try:
            vec = self._step("embedding query", lambda: embed.embed_query_media(parts, query_text or None, model=MM_EMBEDDER))
        finally:
            gov.release(MM_EMBEDDER)
        self._step("unloading multimodal embedder", lambda: gov.unload(MM_EMBEDDER, sid=sid))
        return vec, (time.perf_counter() - t0) * 1000

    # ------------------------------------------------------------------ main
    def stream(self) -> Iterator[str]:
        req, p = self.req, self.p
        log, sid = p.log, req.sid
        gov = p.governor
        t_turn = time.perf_counter()
        stats: dict[str, Any] = {"model": req.preset.id if req.preset else f"gemini:{getattr(p.remote, 'model', '?')}", "backend": req.backend_name}
        stats["preset_id"] = req.effective.preset_id
        stats["mode"] = req.effective.mode
        extra_calls = 0
        try:
            with PeakSampler(p.telemetry) as peak, log.span("chat", "turn", sid=sid, has_media=req.attachment is not None) as span:
                q_wait = time.perf_counter()
                with gov.turn(sid, "chat", on_wait=lambda pos: self._status(f"Waiting for the model - position {pos}")):
                    stats["queue_ms"] = round((time.perf_counter() - q_wait) * 1000, 1)
                    caps = req.preset.capabilities if req.preset else (req.gemini_caps or Capabilities(image=True, audio=True))
                    backend: Backend = p.remote if req.backend_name == "gemini" and p.remote else LocalBackend(p.client, req.preset.id if req.preset else "")
                    query = req.query.strip()

                    # ---- contextualization
                    rq = retrieval_query(query, req.history, req.history_cfg)
                    if req.history_cfg.contextualize == "llm_rewrite" and req.history and query:
                        self._status("rewriting the question")
                        if req.backend_name == "local" and req.preset:
                            gov.ensure(req.preset.id, sid=sid, progress=self._status)
                            gov.release(req.preset.id)
                        t0 = time.perf_counter()
                        body: dict[str, Any] = {"max_tokens": 64, "temperature": 0.0}
                        if req.backend_name == "local" and req.preset and req.preset.capabilities.reasoning_control == "chat_template_kwargs":
                            body["chat_template_kwargs"] = {"enable_thinking": False}
                        rq = backend.complete(rewrite_messages(query, req.history, req.history_cfg), body).strip() or query
                        extra_calls += 1
                        stats["rewrite_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                        stats["rewritten_query"] = rq[:300]

                    # ---- query embedding + retrieval
                    self._status("searching the index")
                    qvec, embed_ms = self._embed_query(gov, rq)
                    stats["embed_ms"] = round(embed_ms, 1)
                    t0 = time.perf_counter()
                    if qvec is not None:
                        qvec = qvec[: self.index.dim] if self.index else qvec
                        qvec = qvec / (np.linalg.norm(qvec) or 1.0)
                    self.hits = retrieve(self.index, qvec, req.retrieval)
                    stats["retrieve_ms"] = round((time.perf_counter() - t0) * 1000, 1)

                    # ---- prompt
                    n_ctx = req.preset.n_ctx if req.preset else 32768
                    max_tokens = int(req.effective.values["max_tokens"])
                    hist = select_history(req.history, req.history_cfg)
                    qmedia, qmedia_tokens = self.p.query_media_for_generator(req, caps)
                    self.prompt = build_prompt(
                        query=query,
                        hits=self.hits,
                        history=hist,
                        n_ctx=n_ctx,
                        max_tokens=max_tokens,
                        caps=caps,
                        query_media=qmedia,
                        query_media_tokens=qmedia_tokens,
                    )
                    stats["prompt_breakdown"] = self.prompt.breakdown
                    stats["prompt_notes"] = self.prompt.notes
                    log.info("chat", "prompt.built", sid=sid, breakdown=self.prompt.breakdown, sources=[d.included for d in self.prompt.decisions])

                    # ---- generator
                    if req.backend_name == "local" and req.preset:
                        t0 = time.perf_counter()
                        gen_id = req.preset.id
                        if gen_id not in gov.resident():
                            self._step(
                                "restoring generator" if req.attachment else "loading generator",
                                lambda: gov.ensure(gen_id, sid=sid, progress=self._status),
                            )
                        else:
                            gov.ensure(gen_id, sid=sid)
                        stats["swap_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                    body = request_fields(req.effective, req.backend_name, req.preset.capabilities.reasoning_control if req.preset else "none")
                    stats["effective_params"] = body
                    log.info("chat", "generate.params", sid=sid, params=body, model=stats["model"])
                    self._status("generating")
                    t_gen = time.perf_counter()
                    first: float | None = None
                    usage: dict[str, Any] = {}
                    timings: dict[str, Any] = {}
                    try:
                        for d in backend.stream(self.prompt.messages, body, self.cancel):
                            if d.reasoning:
                                self.reasoning += d.reasoning
                                if first is None:
                                    first = time.perf_counter()
                            if d.content:
                                if first is None:
                                    first = time.perf_counter()
                                self.answer += d.content
                                yield d.content
                            if d.usage:
                                usage = d.usage
                            if d.timings:
                                timings = d.timings
                            if d.finish:
                                stats["finish"] = d.finish
                    finally:
                        if req.backend_name == "local" and req.preset:
                            gov.release(req.preset.id)
                        gen_ms = (time.perf_counter() - t_gen) * 1000
                        stats["ttft_ms"] = round((first - t_gen) * 1000, 1) if first else None
                        stats["gen_ms"] = round(gen_ms, 1)
                        stats["prompt_tokens"] = usage.get("prompt_tokens")
                        stats["completion_tokens"] = usage.get("completion_tokens")
                        if timings.get("predicted_per_second"):
                            stats["decode_tok_s"] = round(float(timings["predicted_per_second"]), 2)
                        elif usage.get("completion_tokens") and first:
                            stats["decode_tok_s"] = round(usage["completion_tokens"] / max(time.perf_counter() - first, 1e-3), 2)
                        if timings.get("prompt_per_second"):
                            stats["prefill_tok_s"] = round(float(timings["prompt_per_second"]), 2)
                span["extra_calls"] = extra_calls
            stats["peak_mb"] = round(peak.peak_mb, 1)
        except AppError as exc:
            self.error = exc
            log.error("chat", "turn.failed", sid=sid, message=exc.user_message, detail=exc.detail, event_id=exc.event_id)
        except GeneratorExit:
            stats["stopped"] = True
            log.info("chat", "turn.stopped", sid=sid)
            raise
        finally:
            stats["extra_generation_calls"] = extra_calls
            stats["swap_steps"] = self.swap_steps
            stats["total_ms"] = round((time.perf_counter() - t_turn) * 1000, 1)
            stats.setdefault("peak_mb", round(p.telemetry.peak_mb(), 1))
            self.stats = stats
            self.finished = True
        if not self.answer and self.error is None and self.reasoning:
            self.answer = "(The model only produced reasoning and no final answer; try a larger max_tokens or turn reasoning off.)"
            yield self.answer

    def source_rows(self) -> list[dict[str, Any]]:
        rows = []
        decided = {d.hit.item_id: d for d in (self.prompt.decisions if self.prompt else [])}
        for i, h in enumerate(self.hits, start=1):
            d = decided.get(h.item_id)
            rows.append(
                {
                    "label": f"S{i}",
                    "item_id": h.item_id,
                    "score": round(h.score, 4),
                    "modality": h.modality,
                    "span": h.span,
                    "filename": h.filename,
                    "text": (h.text or "")[:600],
                    "preview_path": h.preview_path,
                    "included": d.included if d else "dropped:not_considered",
                    "tokens": d.tokens if d else 0,
                }
            )
        return rows


class ChatPipeline:
    def __init__(
        self,
        client: RouterClient,
        governor: Governor,
        embed: EmbedService,
        params: EmbedParams,
        telemetry: Telemetry,
        log: Logger,
        remote: Backend | None = None,
    ) -> None:
        self.client = client
        self.governor = governor
        self.embed = embed
        self.params = params
        self.telemetry = telemetry
        self.log = log
        self.remote = remote

    def turn(self, req: TurnRequest, index: IndexHandle | None, on_status: Callable[[str], None] = lambda _m: None) -> Turn:
        return Turn(self, req, index, on_status)

    @staticmethod
    def media_parts(att: Attachment, params: EmbedParams) -> list[dict[str, Any]]:
        if att.modality == "image":
            jpg, _ = image_to_jpeg(att.data, params.image)
            return [image_part(jpg)]
        if att.modality == "audio":
            return [audio_part(ChatPipeline.audio_wav(att, params))]
        raise AppError("Only image or audio attachments are supported in chat.")

    @staticmethod
    def audio_wav(att: Attachment, params: EmbedParams) -> bytes:
        import tempfile
        from pathlib import Path

        from core.embed.preprocess import decode_audio_pcm

        a = params.audio.model_copy(update={"max_total_s": min(params.audio.window_s, params.audio.max_total_s)})
        with tempfile.NamedTemporaryFile(suffix=".bin") as fh:
            fh.write(att.data)
            fh.flush()
            pcm = decode_audio_pcm(Path(fh.name), a)
        return pcm_to_wav(pcm, a.sample_rate_hz)

    def query_media_for_generator(self, req: TurnRequest, caps: Capabilities) -> tuple[list[dict[str, Any]] | None, int]:
        att = req.attachment
        if att is None:
            return None, 0
        if req.backend_name == "gemini" and not req.allow_media_to_remote:
            return [{"type": "text", "text": "(The user attached media, but did not confirm sending it to the remote model.)"}], 16
        if att.modality == "image" and caps.image:
            return self.media_parts(att, self.params), 300
        if att.modality == "audio" and caps.audio:
            return self.media_parts(att, self.params), int(self.params.audio.window_s * 25)
        return [{"type": "text", "text": f"(The user attached {att.modality}; this model cannot read it directly - it was used for retrieval only.)"}], 24
