"""Pydantic schemas: the code form of every file/record the app reads or writes."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SHA256_RE = r"^[0-9a-f]{64}$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- bundle
class PatchEntry(_Strict):
    pr: int
    head_sha: str
    reason: str
    status_when_pinned: str


class LlamaBuildInfo(_Strict):
    base: str
    commit: str
    target: str
    max_glibc_required: str
    compiler: str = ""
    patch_queue: list[PatchEntry] = Field(default_factory=list)
    build_flags: list[str] = Field(default_factory=list)


class BundleAsset(_Strict):
    name: str
    kind: Literal["runtime", "model"]
    role: str | None = None
    preset: str | None = None
    sha256: str = Field(pattern=SHA256_RE)
    size_bytes: int = Field(gt=0)
    extract_to: str | None = None
    url: str | None = None  # absolute override (default: base_url + name)

    @field_validator("name")
    @classmethod
    def _safe_name(cls, v: str) -> str:
        if "/" in v or "\\" in v or v.startswith(".") or not v:
            raise ValueError("asset names must be plain file names")
        return v


class Manifest(_Strict):
    schema_: int = Field(alias="schema")
    bundle_id: str
    base_url: str
    llama_cpp: LlamaBuildInfo
    assets: list[BundleAsset]

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("assets")
    @classmethod
    def _unique(cls, v: list[BundleAsset]) -> list[BundleAsset]:
        names = [a.name for a in v]
        if len(names) != len(set(names)):
            raise ValueError("asset names must be unique across the bundle")
        return v


# --------------------------------------------------------------------------- presets
class KVConfig(_Strict):
    k: str = "q8_0"
    v: str = "q8_0"


class Capabilities(_Strict):
    image: bool = False
    audio: bool = False
    video: Literal["frames", "native", False] = False  # type: ignore[valid-type]
    reasoning_control: str = "none"  # none | chat_template_kwargs | unverified


class ModelPreset(_Strict):
    id: str
    label: str
    role: Literal["generator", "embedder"]
    family: str
    files: dict[str, str]  # {"model": ..., "mmproj": ...}
    n_ctx: int = Field(gt=0)
    kv: KVConfig = Field(default_factory=KVConfig)
    flash_attn: Literal["on", "off", "auto"] = "auto"
    threads: int = 2
    batch: int = 512
    ubatch: int = 512
    capabilities: Capabilities = Field(default_factory=Capabilities)
    sampling_preset: str | None = None
    est_peak_mb: int | None = None
    measured_peak_mb: int | None = None
    gates: dict[str, str | None] = Field(default_factory=dict)
    shipped: bool = False
    extra_args: dict[str, str | int | float | bool] = Field(default_factory=dict)
    source: dict[str, str] = Field(default_factory=dict)  # repo, revision
    notes: str = ""


class ModelPresets(_Strict):
    schema_: int = Field(alias="schema")
    presets: list[ModelPreset]
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    def get(self, preset_id: str) -> ModelPreset:
        for p in self.presets:
            if p.id == preset_id:
                return p
        raise KeyError(preset_id)

    def shipped_generators(self) -> list[ModelPreset]:
        return [p for p in self.presets if p.role == "generator" and p.shipped]


class SamplingValues(_Strict):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None


class ReasoningInfo(_Strict):
    on_by_default: bool = False
    control: str = "none"


class SamplingPreset(_Strict):
    status: Literal["verified", "partial", "to_verify", "n/a"]
    source_url: str
    revision: str
    checked_at: str
    default_mode: str
    modes: dict[str, SamplingValues]
    reasoning: ReasoningInfo = Field(default_factory=ReasoningInfo)
    notes: str = ""


class SamplingPresets(_Strict):
    schema_: int = Field(alias="schema")
    family_fallback: dict[str, SamplingValues] = Field(default_factory=dict)
    presets: dict[str, SamplingPreset]
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# --------------------------------------------------------------------------- embedding params
class TextParams(_Strict):
    chunk_tokens: int = Field(256, ge=32, le=2048)
    overlap_tokens: int = Field(32, ge=0, le=512)


class ImageParams(_Strict):
    max_side_px: int = Field(768, ge=64, le=4096)
    max_pixels: int = Field(40_000_000, ge=10_000)
    token_cap_min: int | None = None
    token_cap_max: int | None = None
    format: Literal["jpeg", "png"] = "jpeg"


class AudioParams(_Strict):
    sample_rate_hz: int = Field(16000, ge=8000, le=48000)
    channels: int = 1
    window_s: float = Field(30.0, gt=0, le=300)
    overlap_s: float = Field(2.0, ge=0, le=60)
    max_total_s: float = Field(300.0, gt=0, le=3600)


class VideoParams(_Strict):
    max_total_s: float = Field(60.0, gt=0, le=600)
    fps: float = Field(1.0, gt=0, le=10)
    max_frames: int = Field(16, ge=1, le=128)
    frame_max_side_px: int = Field(448, ge=64, le=1920)


class DimsParams(_Strict):
    index_dim: int = 768
    allowed: list[int] = Field(default_factory=lambda: [768, 512, 256, 128])


class IndexParams(_Strict):
    bits: Literal[2, 4] = 4
    keep_float32: bool = True


class PrefixParams(_Strict):
    query: str = "task: search result | query: {query}"
    document: str = "title: none | text: {content}"


class ServerParams(_Strict):
    ubatch: int = 2048
    ctx: int = 2048
    pooling: str = "default"


class EmbedParams(_Strict):
    schema_: int = Field(1, alias="schema")
    text: TextParams = Field(default_factory=lambda: TextParams())
    image: ImageParams = Field(default_factory=lambda: ImageParams())
    audio: AudioParams = Field(default_factory=lambda: AudioParams())
    video: VideoParams = Field(default_factory=lambda: VideoParams())
    dims: DimsParams = Field(default_factory=lambda: DimsParams())
    index: IndexParams = Field(default_factory=lambda: IndexParams())
    prefix: PrefixParams = Field(default_factory=lambda: PrefixParams())
    server: ServerParams = Field(default_factory=lambda: ServerParams())
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# --------------------------------------------------------------------------- logs, probe, ledger
class LogEvent(_Strict):
    ts: float
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    component: str
    event: str
    sid: str | None = None
    session_scoped: bool = False
    dur_ms: float | None = None
    fields: dict[str, Any] = Field(default_factory=dict)


class ProbeValue(_Strict):
    value: Any = None
    reason: str | None = None


class ProbeReport(_Strict):
    schema_: int = Field(1, alias="schema")
    generated_at: str
    values: dict[str, ProbeValue]
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ExperimentRecord(_Strict):
    run_id: str
    ts: str
    exp_id: str
    candidate_id: str | None = None
    bundle_id: str | None = None
    git_sha: str | None = None
    env: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    gates: dict[str, str | None] = Field(default_factory=dict)
    verdict: str = ""
    notes: str = ""


# --------------------------------------------------------------------------- chat export
class ExportSource(_Strict):
    item_id: int
    score: float
    modality: str
    span: dict[str, Any] = Field(default_factory=dict)
    included: str  # included | truncated | dropped:<reason> | fallback:<kind>
    source_name: str | None = None


class ExportAttachment(_Strict):
    modality: str
    sha256_8: str
    bytes: int


class ExportTurn(_Strict):
    role: Literal["user", "assistant"]
    text: str
    attachments: list[ExportAttachment] = Field(default_factory=list)
    sources: list[ExportSource] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
    ts: float


class ChatExport(_Strict):
    schema_: int = Field(1, alias="schema")
    app_version: str
    bundle_id: str | None = None
    preset_id: str | None = None
    effective_params: dict[str, Any] = Field(default_factory=dict)
    turns: list[ExportTurn]
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
