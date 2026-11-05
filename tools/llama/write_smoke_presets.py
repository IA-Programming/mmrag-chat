"""DEV ONLY: write a router presets INI for smoke tests/evals that points at the fetch cache.

python -m tools.llama.write_smoke_presets --out build/smoke/presets.ini --generator lfm3-vl-2b-qad-q4_0 [--models-root /models]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.runtime.presets import load_model_presets, render_ini


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--generator", nargs="*", default=[], help="generator preset ids (space separated)")
    ap.add_argument("--models-root", default="/models", help="where the fetch cache is mounted inside the container")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)
    gens = [g for item in args.generator for g in item.split()]
    presets = load_model_presets()
    chosen = [p for p in presets.presets if p.role == "embedder" or p.id in gens]
    by_name: dict[str, Path] = {}
    for p in chosen:
        for role, name in p.files.items():
            upstream = p.source.get(role) or name
            by_name[name] = Path(args.models_root) / p.source["repo"].replace("/", "__") / upstream
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_ini(chosen, lambda n: by_name[n], threads=args.threads))
    print(args.out.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
