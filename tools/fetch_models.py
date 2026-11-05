"""DEV ONLY (DR-003): fetch candidate GGUF files pinned by HF revision, verify SHA-256.

Usage:
    python -m tools.fetch_models --only embeddinggemma-2 lfm3-vl-2b-qad-q4_0
    python -m tools.fetch_models --all            # every candidate (large!)
    python -m tools.fetch_models --list

HF_TOKEN is read from the environment by huggingface_hub; it is never printed or
written anywhere by this module. Re-running is idempotent: files whose recorded
provenance (repo, revision, sha256, size) matches are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "tools" / "eval" / "candidates.yaml"
MODELS_DIR = Path(os.environ.get("MMRAG_MODELS_DIR", Path.home() / ".cache" / "mmrag" / "models"))


@dataclass(frozen=True)
class FileSpec:
    candidate: str
    repo: str
    revision: str
    filename: str

    @property
    def local_path(self) -> Path:
        return MODELS_DIR / self.repo.replace("/", "__") / self.filename

    @property
    def provenance_path(self) -> Path:
        return self.local_path.with_name(self.local_path.name + ".provenance.json")


def load_candidates(path: Path = CANDIDATES) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(path.read_text())
    return data


def file_specs(data: dict[str, Any], include_reference: bool = False, include_ladder: bool = False) -> list[FileSpec]:
    specs: list[FileSpec] = []
    emb = data["embedder"]
    for f in emb["files"].values():
        specs.append(FileSpec(emb["id"], emb["repo"], emb["revision"], f))
    if include_reference:
        for f in emb.get("reference", {}).values():
            specs.append(FileSpec(emb["id"] + "-reference", emb["repo"], emb["revision"], f))
    for g in data["generators"]:
        for f in g["files"].values():
            specs.append(FileSpec(g["id"], g["repo"], g["revision"], f))
        ladder = g.get("quant_ladder")
        if include_ladder and ladder:
            for f in ladder["files"]:
                specs.append(FileSpec(g["id"] + "-ladder", ladder["repo"], ladder["revision"], f))
    return specs


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def remote_info(spec: FileSpec) -> tuple[str, int]:
    from huggingface_hub import HfApi

    infos = HfApi().get_paths_info(spec.repo, [spec.filename], revision=spec.revision)
    if not infos or getattr(infos[0], "lfs", None) is None:
        raise RuntimeError(f"{spec.repo}@{spec.revision[:8]}/{spec.filename}: not an LFS file on the Hub")
    lfs = infos[0].lfs
    return str(lfs.sha256), int(lfs.size)


def is_current(spec: FileSpec) -> bool:
    if not (spec.local_path.exists() and spec.provenance_path.exists()):
        return False
    prov = json.loads(spec.provenance_path.read_text())
    return prov.get("repo") == spec.repo and prov.get("revision") == spec.revision and prov.get("size_bytes") == spec.local_path.stat().st_size


def fetch(spec: FileSpec, verify_existing: bool = False) -> dict[str, Any]:
    if is_current(spec) and not verify_existing:
        prov: dict[str, Any] = json.loads(spec.provenance_path.read_text())
        print(f"skip   {spec.filename} (present, {prov['size_bytes'] / 1e6:.1f} MB)")
        return prov
    from huggingface_hub import hf_hub_download

    expected_sha, expected_size = remote_info(spec)
    spec.local_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    print(f"fetch  {spec.repo}@{spec.revision[:8]}/{spec.filename} ({expected_size / 1e6:.1f} MB)", flush=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=spec.repo,
            filename=spec.filename,
            revision=spec.revision,
            local_dir=spec.local_path.parent,
        )
    )
    if downloaded.resolve() != spec.local_path.resolve():
        downloaded.replace(spec.local_path)
    size = spec.local_path.stat().st_size
    digest = sha256_file(spec.local_path)
    if size != expected_size or digest != expected_sha:
        spec.local_path.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {spec.filename}: got {digest[:12]}.. size {size}")
    prov = {
        "candidate": spec.candidate,
        "repo": spec.repo,
        "revision": spec.revision,
        "filename": spec.filename,
        "sha256": digest,
        "size_bytes": size,
        "source_url": f"https://huggingface.co/{spec.repo}/blob/{spec.revision}/{spec.filename}",
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.monotonic() - t0, 1),
    }
    spec.provenance_path.write_text(json.dumps(prov, indent=2))
    print(f"ok     {spec.filename} sha256={digest[:12]}.. in {prov['seconds']} s", flush=True)
    return prov


def fetch_license(repo: str, revision: str) -> Path | None:
    """Model licenses travel with redistributed weights (release assets)."""
    from huggingface_hub import HfApi, hf_hub_download

    names = [f for f in HfApi().list_repo_files(repo, revision=revision) if f.upper().startswith(("LICENSE", "NOTICE"))]
    if not names:
        return None
    dest = MODELS_DIR / repo.replace("/", "__")
    dest.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo_id=repo, filename=names[0], revision=revision, local_dir=dest))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None, help="candidate ids to fetch")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--reference", action="store_true", help="also fetch the BF16 embedder reference")
    ap.add_argument("--ladder", action="store_true", help="also fetch quant ladders")
    ap.add_argument("--verify", action="store_true", help="re-hash files that are already present")
    args = ap.parse_args(argv)

    specs = file_specs(load_candidates(), include_reference=args.reference, include_ladder=args.ladder)
    if args.list:
        for s in specs:
            print(f"{s.candidate:32s} {s.repo}/{s.filename}  {'present' if is_current(s) else '-'}")
        return 0
    if not args.all and not args.only:
        ap.error("pass --only <ids...> or --all")
    wanted = specs if args.all else [s for s in specs if s.candidate.removesuffix("-ladder").removesuffix("-reference") in set(args.only or [])]
    failures = 0
    for spec in wanted:
        try:
            fetch(spec, verify_existing=args.verify)
        except Exception as exc:  # report and continue with the next file
            failures += 1
            print(f"FAIL   {spec.filename}: {exc}", file=sys.stderr)
    for repo, rev in sorted({(s.repo, s.revision) for s in wanted}):
        try:
            lic = fetch_license(repo, rev)
            print(f"license {repo}: {lic.name if lic else 'none in repo (see model card)'}")
        except Exception as exc:
            print(f"license {repo}: not fetched ({type(exc).__name__})", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
