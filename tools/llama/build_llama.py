"""DEV ONLY (DR-001): reproducible llama.cpp build for Debian 11 (glibc 2.31).

    python -m tools.llama.build_llama --tag b12119 --src /path/to/llama.cpp --out build/llama

Steps: (optional) clone the pinned tag -> apply the allowlisted patch queue -> build inside
docker/Dockerfile.build (clang 18 on bullseye, static libstdc++/libgcc, runtime CPU variants,
no OpenMP) -> ELF checks (max GLIBC <= 2.31, no GLIBCXX, only system libs) -> build-info.json.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLCHAIN_IMAGE = "mmrag-llama-build:bullseye"
MAX_GLIBC = (2, 31)
SYSTEM_LIBS = {"libc.so.6", "libm.so.6", "libpthread.so.0", "libdl.so.2", "ld-linux-x86-64.so.2", "librt.so.1"}

# Allowlisted patch queue (ARCHITECTURE section 8). Each entry needs a requirement id and a
# test that fails without it. Triage at b12119:
#   PR 22284 (router unload/reload deadlock + --models-memory-max): NOT applied. The pinned base
#   already has --models-memory-margin, and 20 load/unload cycles (tools/llama/smoke_tests.py,
#   tests/integration) pass without it, so there is no failing test that justifies it.
PATCH_QUEUE: list[dict[str, str]] = []


def run(argv: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(argv), flush=True)
    return subprocess.run(argv, check=True, text=True, **kw)  # type: ignore[call-overload,no-any-return]


def elf_report(pkg: Path) -> dict[str, object]:
    """I-BUILD-glibc: max GLIBC symbol version and non-system NEEDED libs over every ELF file."""
    glibc: set[tuple[int, ...]] = set()
    glibcxx = 0
    needed: set[str] = set()
    own = {p.name for p in pkg.iterdir()}
    for f in sorted(pkg.iterdir()):
        if f.is_symlink() or not f.is_file() or f.read_bytes()[:4] != b"\x7fELF":
            continue
        syms = subprocess.run(["objdump", "-T", str(f)], capture_output=True, text=True, check=True).stdout
        glibc |= {tuple(int(x) for x in m.split(".")) for m in re.findall(r"GLIBC_([0-9.]+)", syms)}
        glibcxx += syms.count("GLIBCXX_")
        dyn = subprocess.run(["readelf", "-d", str(f)], capture_output=True, text=True, check=True).stdout
        needed |= set(re.findall(r"Shared library: \[([^\]]+)\]", dyn))
    foreign = sorted(n for n in needed if n not in SYSTEM_LIBS and n not in own)
    max_glibc = max(glibc) if glibc else (0,)
    return {
        "max_glibc": ".".join(map(str, max_glibc)),
        "glibc_ok": max_glibc <= MAX_GLIBC,
        "glibcxx_refs": glibcxx,
        "foreign_needed": foreign,
        "ok": max_glibc <= MAX_GLIBC and glibcxx == 0 and not foreign,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="b12119")
    ap.add_argument("--src", type=Path, default=ROOT / "build" / "llama.cpp")
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "llama")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--image", default=TOOLCHAIN_IMAGE)
    ap.add_argument("--build-image", action="store_true", help="(re)build the toolchain image from docker/Dockerfile.build")
    args = ap.parse_args(argv)

    if not (args.src / "CMakeLists.txt").exists():
        run(["git", "clone", "--depth", "1", "--branch", args.tag, "https://github.com/ggml-org/llama.cpp", str(args.src)])
    commit = run(["git", "-C", str(args.src), "rev-parse", "HEAD"], capture_output=True).stdout.strip()
    for p in PATCH_QUEUE:
        run(["git", "-C", str(args.src), "apply", str(ROOT / "tools" / "llama" / "patches" / p["file"])])
    if args.build_image:
        run(["docker", "build", "-f", str(ROOT / "docker" / "Dockerfile.build"), "-t", args.image, str(ROOT / "docker")])
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{args.src.resolve()}:/src/llama.cpp:ro",
            "-v",
            f"{args.out.resolve()}:/out",
            "-v",
            f"{ROOT / 'tools' / 'llama' / 'build.sh'}:/build.sh:ro",
            "-e",
            f"JOBS={args.jobs}",
            "-e",
            "STATIC_STDLIB=1",
            "-e",
            f"LLAMA_BUILD_NUMBER={args.tag.lstrip('b')}",
            "-e",
            f"LLAMA_BUILD_COMMIT={commit[:10]}",
            args.image,
            "bash",
            "/build.sh",
        ]
    )
    pkg = args.out / "pkg"
    report = elf_report(pkg)
    toolchain = dict(line.split("=", 1) for line in (pkg / ".toolchain").read_text().splitlines() if "=" in line)
    info = {
        "base": args.tag,
        "commit": commit,
        "target": "debian11-glibc2.31-x86_64",
        "max_glibc_required": report["max_glibc"],
        "compiler": toolchain.get("compiler", ""),
        "patch_queue": PATCH_QUEUE,
        "build_flags": [
            "-DGGML_NATIVE=OFF",
            "-DGGML_BACKEND_DL=ON",
            "-DGGML_CPU_ALL_VARIANTS=ON",
            "-DGGML_OPENMP=OFF",
            "-DLLAMA_OPENSSL=OFF",
            "-DLLAMA_BUILD_UI=OFF",
            "-static-libstdc++",
            "-static-libgcc",
            "-fuse-ld=lld",
        ],
        "elf": report,
        "build_seconds": round(time.time() - t0),
    }
    (pkg / "build-info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    if not report["ok"]:
        print("ELF check FAILED (I-BUILD-glibc)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
