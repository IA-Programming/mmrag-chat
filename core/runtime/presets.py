"""Model presets -> llama-server router `--models-preset` INI (the only code that knows flag names).

Flag names verified against `llama-server --help` of the pinned build (b12119).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from core.config import BUNDLE_DIR
from core.schemas import EmbedParams, ModelPreset, ModelPresets, SamplingPresets


@lru_cache(maxsize=4)
def _load_presets(path: str, mtime: float) -> ModelPresets:
    return ModelPresets.model_validate_json(Path(path).read_text())


def load_model_presets(path: Path = BUNDLE_DIR / "model_presets.json") -> ModelPresets:
    return _load_presets(str(path), path.stat().st_mtime)


def load_sampling_presets(path: Path = BUNDLE_DIR / "sampling_presets.json") -> SamplingPresets:
    return SamplingPresets.model_validate_json(path.read_text())


def load_embed_params(path: Path = BUNDLE_DIR / "embed_params.json") -> EmbedParams:
    return EmbedParams.model_validate_json(path.read_text())


def _fmt(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def preset_args(p: ModelPreset, resolve: Callable[[str], Path], *, threads: int | None = None) -> dict[str, str]:
    """Key/value pairs for one INI section (keys = long CLI flag names without dashes)."""
    args: dict[str, str] = {
        "model": str(resolve(p.files["model"])),
        "ctx-size": str(p.n_ctx),
        "threads": str(threads or p.threads),
        "threads-batch": str(threads or p.threads),
        "batch-size": str(p.batch),
        "ubatch-size": str(p.ubatch),
        "cache-type-k": p.kv.k,
        "cache-type-v": p.kv.v,
        "flash-attn": p.flash_attn,
        "parallel": "1",  # one sequence: KV sized for exactly one conversation
        "cache-ram": "0",  # no host-RAM prompt cache (would grow outside our budget)
        "stop-timeout": "5",
        "load-on-startup": "false",
    }
    if p.files.get("mmproj"):
        args["mmproj"] = str(resolve(p.files["mmproj"]))
    else:
        args["no-mmproj"] = "true"
    if p.role == "embedder":
        args["embeddings"] = "true"
    for k, v in p.extra_args.items():
        args[k] = _fmt(v)
    return args


def render_ini(presets: list[ModelPreset], resolve: Callable[[str], Path], *, threads: int | None = None) -> str:
    lines = ["version = 1", ""]
    for p in presets:
        lines.append(f"[{p.id}]")
        for k, v in preset_args(p, resolve, threads=threads).items():
            lines.append(f"{k} = {v}")
        lines.append("")
    return "\n".join(lines)


def write_ini(path: Path, presets: list[ModelPreset], resolve: Callable[[str], Path], *, threads: int | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_ini(presets, resolve, threads=threads))
    return path
