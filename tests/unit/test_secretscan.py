"""U-SEC-secretscan: no key-shaped strings or secret files are tracked in git."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    re.compile(r"hf_[A-Za-z0-9]{30,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{36}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{50,}"),
    re.compile(r"sk-[A-Za-z0-9]{32,}"),
    re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]
FORBIDDEN_FILES = {".env", ".streamlit/secrets.toml", "secrets.toml"}
# Test fixtures that intentionally contain fake key-shaped canaries.
ALLOW: set[str] = set()  # canaries in tests are built at runtime, so nothing needs an exception


def tracked_files() -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts and ".venv" not in p.parts]
    return [line for line in out.splitlines() if line]


def test_no_secret_files_tracked() -> None:
    files = set(tracked_files())
    assert not (files & FORBIDDEN_FILES), files & FORBIDDEN_FILES
    assert not [f for f in files if f.endswith(".gguf")], "never commit GGUF files"


def test_no_key_shaped_strings() -> None:
    hits = []
    for rel in tracked_files():
        if rel in ALLOW:
            continue
        p = ROOT / rel
        if not p.is_file() or p.stat().st_size > 2_000_000:
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        for pat in PATTERNS:
            m = pat.search(text)
            if m:
                hits.append(f"{rel}: {m.group(0)[:8]}…")
    assert not hits, hits


def test_secrets_example_has_no_values() -> None:
    text = (ROOT / ".streamlit" / "secrets.toml.example").read_text()
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line and not line.startswith("["):
            assert line.split("=", 1)[1].strip() in ('""', "''"), f"example must not contain values: {line}"


def test_gitignore_covers_secrets() -> None:
    gi = (ROOT / ".gitignore").read_text()
    for pat in (".streamlit/secrets.toml", ".env", "*.gguf"):
        assert pat in gi, pat
