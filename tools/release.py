"""DEV ONLY (DR-002): assemble a frozen bundle and its manifest.

    python -m tools.release --bundle-id bundle-2026-11-r1 --runtime-dir build/llama/pkg --out dist/bundle

* runtime tarball: deterministic (sorted, mtime 0, uid/gid 0) `llama-runtime-<tag>-debian11-x86_64.tar.gz`
* model assets: every file of the served presets (embedders + shipped generators, or --presets),
  copied from the fetch cache under the UNIQUE bundle name in model_presets.json, after checking
  the SHA-256 recorded by tools/fetch_models.py (provenance)
* manifest.json: sizes + SHA-256 of every asset, llama.cpp build info; written to --out and,
  with --write-repo-manifest, to bundle/manifest.json. Each asset must be < 2 GiB.
Publishing (gh release upload) happens in .github/workflows/bundle-release.yml.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

from core.runtime.presets import load_model_presets
from tools.fetch_models import MODELS_DIR

ROOT = Path(__file__).resolve().parents[1]
REPO = "IA-Programming/mmrag-chat"
MAX_ASSET = 2 * 1024**3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while b := fh.read(8 << 20):
            h.update(b)
    return h.hexdigest()


def deterministic_tar(src: Path, dest: Path, top: str) -> None:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for p in sorted(src.iterdir()):
            if p.name.startswith("."):
                continue
            ti = tar.gettarinfo(str(p), arcname=f"{top}/{p.name}")
            ti.mtime = 0
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            if p.is_symlink():
                tar.addfile(ti)
            else:
                with p.open("rb") as fh:
                    tar.addfile(ti, fh)
    with dest.open("wb") as out, gzip.GzipFile(filename="", fileobj=out, mode="wb", mtime=0, compresslevel=9) as gz:
        gz.write(raw.getvalue())


def model_source(preset_files: dict[str, str], source: dict[str, str], role: str) -> Path:
    repo = source["repo"]
    upstream = source.get(role) or preset_files[role]
    return MODELS_DIR / repo.replace("/", "__") / upstream


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-id", required=True)
    ap.add_argument("--runtime-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--presets", nargs="*", help="generator preset ids to include (default: shipped)")
    ap.add_argument("--link", action="store_true", help="hard-link model files instead of copying (local dev)")
    ap.add_argument("--write-repo-manifest", action="store_true")
    ap.add_argument("--manifest-path", type=Path, default=None)
    args = ap.parse_args(argv)

    info = json.loads((args.runtime_dir / "build-info.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    assets: list[dict[str, Any]] = []

    rt_name = f"llama-runtime-{info['base']}-debian11-x86_64.tar.gz"
    deterministic_tar(args.runtime_dir, args.out / rt_name, f"llama-runtime-{info['base']}")
    assets.append(
        {"name": rt_name, "kind": "runtime", "sha256": sha256(args.out / rt_name), "size_bytes": (args.out / rt_name).stat().st_size, "extract_to": "runtime/"}
    )

    presets = load_model_presets()
    chosen = [p for p in presets.presets if p.role == "embedder" or (p.id in args.presets if args.presets else p.shipped)]
    seen: set[str] = set()
    for p in chosen:
        for role, name in p.files.items():
            if name in seen:
                continue
            seen.add(name)
            src = model_source(p.files, p.source, role)
            prov_path = src.with_name(src.name + ".provenance.json")
            if not src.exists() or not prov_path.exists():
                raise SystemExit(f"missing {src} - run: python -m tools.fetch_models --only <candidate for {p.id}>")
            prov = json.loads(prov_path.read_text())
            dest = args.out / name
            if dest.exists():
                dest.unlink()
            if args.link:
                dest.hardlink_to(src)
            else:
                shutil.copyfile(src, dest)
            digest = sha256(dest)
            if digest != prov["sha256"]:
                raise SystemExit(f"{name}: sha256 differs from provenance")
            size = dest.stat().st_size
            if size >= MAX_ASSET:
                raise SystemExit(f"{name} is {size} bytes (>= 2 GiB release asset limit)")
            role_name = {
                "model": "embedder_text" if p.role == "embedder" else "generator",
                "mmproj": "embedder_mmproj" if p.role == "embedder" else "generator_mmproj",
            }[role]
            assets.append({"name": name, "kind": "model", "role": role_name, "preset": p.id, "sha256": digest, "size_bytes": size})

    manifest = {
        "schema": 1,
        "bundle_id": args.bundle_id,
        "base_url": f"https://github.com/{REPO}/releases/download/{args.bundle_id}/",
        "llama_cpp": {k: info[k] for k in ("base", "commit", "target", "max_glibc_required", "compiler", "patch_queue", "build_flags")},
        "assets": assets,
    }
    text = json.dumps(manifest, indent=2) + "\n"
    (args.out / "manifest.json").write_text(text)
    if args.manifest_path:
        args.manifest_path.write_text(text)
    if args.write_repo_manifest:
        (ROOT / "bundle" / "manifest.json").write_text(text)
    total = sum(a["size_bytes"] for a in assets)
    print(f"bundle {args.bundle_id}: {len(assets)} assets, {total / 1e9:.2f} GB -> {args.out}")
    for a in assets:
        print(f"  {a['name']:50s} {a['size_bytes'] / 1e6:9.1f} MB  {a['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
