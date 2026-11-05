"""U-ARCH-imports + U-SEC-egress + U-SEC-render + U-SEC-nopickle + U-SEC-argv (static checks)."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIRS = [ROOT / "core", ROOT / "app"]
HTTP_MODULES = {"httpx", "requests", "urllib.request", "urllib3", "aiohttp", "http.client"}


def _py(dirs: list[Path]) -> list[Path]:
    return sorted(p for d in dirs for p in d.rglob("*.py"))


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def test_core_never_imports_streamlit() -> None:
    bad = [str(p.relative_to(ROOT)) for p in _py([ROOT / "core"]) if any(m == "streamlit" or m.startswith("streamlit.") for m in _imports(p))]
    assert not bad, f"core must be Streamlit-free: {bad}"


def test_runtime_never_imports_tools_or_tests() -> None:
    bad = [str(p.relative_to(ROOT)) for p in _py(RUNTIME_DIRS) if any(m.split(".")[0] in ("tools", "tests") for m in _imports(p))]
    assert not bad, f"app/core must not import tools/ or tests/: {bad}"


def test_http_clients_only_in_net() -> None:
    allowed = {ROOT / "core" / "net.py"}
    offenders = []
    for p in _py(RUNTIME_DIRS):
        if p in allowed:
            continue
        mods = _imports(p)
        if any(m in HTTP_MODULES or m.split(".")[0] in {"requests", "urllib3", "aiohttp"} for m in mods):
            # type-only uses of httpx (exceptions, Timeout) are fine; client construction is not
            src = p.read_text()
            if "httpx.Client(" in src or "httpx.AsyncClient(" in src or "requests." in src or "urlopen(" in src:
                offenders.append(str(p.relative_to(ROOT)))
    assert not offenders, f"HTTP clients must be created in core/net.py only: {offenders}"


def test_app_starts_no_processes_or_sockets() -> None:
    bad = []
    for p in _py([ROOT / "app"]):
        mods = _imports(p)
        if {"subprocess", "socket", "multiprocessing"} & {m.split(".")[0] for m in mods}:
            bad.append(str(p.relative_to(ROOT)))
    assert not bad, f"app/ must not start processes or open sockets: {bad}"


def test_no_unsafe_html() -> None:
    """U-SEC-render: model output is never rendered as HTML."""
    hits = [str(p.relative_to(ROOT)) for p in _py(RUNTIME_DIRS) if "unsafe_allow_html=True" in p.read_text()]
    assert not hits, hits


def test_no_pickle_loads() -> None:
    """U-SEC-nopickle: no pickle/marshal deserialization, numpy loads always allow_pickle=False."""
    for p in _py(RUNTIME_DIRS):
        src = p.read_text()
        assert "pickle.load" not in src and "marshal.load" not in src, p
        for line in src.splitlines():
            if "np.load(" in line:
                assert "allow_pickle=False" in line, f"{p}: {line.strip()}"


def test_subprocess_uses_argv_lists() -> None:
    """U-SEC-argv: never shell=True."""
    for p in _py(RUNTIME_DIRS + [ROOT / "tools"]):
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        raise AssertionError(f"shell=True in {p}")


def test_page_modules_are_import_safe() -> None:
    """Rule 5: page modules expose render() and do no work at import time."""
    import importlib

    for name in ("chat", "index", "playground", "explorer", "settings", "logs"):
        mod = importlib.import_module(f"app.pages.{name}")
        assert callable(getattr(mod, "render", None)), name


def test_sql_is_parameterized() -> None:
    """U-SEC-sql: no f-string/format SQL except the placeholder list in IndexHandle.hits."""
    src = (ROOT / "core" / "index" / "store.py").read_text()
    for line in src.splitlines():
        if "execute(" in line and ('f"' in line or ".format(" in line):
            raise AssertionError(line)
