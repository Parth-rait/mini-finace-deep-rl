"""The dependency rule from PLAN_SHIPPABLE.md section 1.2, enforced.

    interfaces -> system -> engine -> core
                    |          ^
                    +-> adapters -> aip

Parses every module's imports with `ast` (no importing, so it runs in any
environment). The point: the finance core stays the centre. It never
learns about the LLM, the web layer or the wiring, and only adapters touch
aip, so replacing aip later touches adapters/ only.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "minifinrl"

WEB = {"fastapi", "starlette", "uvicorn", "sse_starlette"}
LLM = {"aip", "litellm", "openai", "anthropic"}
ENGINE_LAYER = {"minifinrl.engine", "minifinrl.capabilities", "minifinrl.ports", "minifinrl.schemas"}


def _imports(path: Path) -> set[str]:
    """Absolute module names imported anywhere in the file (incl. lazily)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module)
            # `from minifinrl import system` names a module too
            out.update(f"{node.module}.{a.name}" for a in node.names if node.module == "minifinrl")
    return out


def _top(mod: str) -> str:
    return mod.split(".")[0]


def _is(mod: str, prefix: str) -> bool:
    return mod == prefix or mod.startswith(prefix + ".")


def _files(sub: str) -> list[Path]:
    base = PKG / sub
    return sorted(base.rglob("*.py")) if base.is_dir() else ([base] if base.exists() else [])


def _label(f: Path) -> str:
    return str(f.relative_to(PKG.parent)) if f.is_relative_to(PKG.parent) else f.name


def _violations(files: list[Path], allowed_minifinrl: list[str], banned_top: set[str]) -> list[str]:
    bad = []
    for f in files:
        for mod in sorted(_imports(f)):
            if _top(mod) in banned_top:
                bad.append(f"{_label(f)} imports {mod}")
            elif _top(mod) == "minifinrl" and mod != "minifinrl":
                if not any(_is(mod, a) for a in allowed_minifinrl):
                    bad.append(f"{_label(f)} imports {mod}")
    return bad


def test_package_exists():
    assert (PKG / "core").is_dir()


def test_core_is_self_contained():
    """core/ = finance logic only: no LLM, no web, nothing from outer layers."""
    bad = _violations(_files("core"), ["minifinrl.core"], WEB | LLM | {"pydantic"})
    assert not bad, "\n".join(bad)


def test_engine_layer_never_sees_adapters_web_or_llm():
    files = [f for name in ("engine.py", "capabilities.py", "ports.py", "schemas.py") for f in _files(name)]
    bad = _violations(files, ["minifinrl.core", *ENGINE_LAYER], WEB | LLM)
    assert not bad, "\n".join(bad)


def test_adapters_implement_ports_only():
    """adapters may use aip, but not the engine, the wiring or the web."""
    bad = _violations(_files("adapters"), ["minifinrl.core", "minifinrl.ports", "minifinrl.schemas"], WEB)
    assert not bad, "\n".join(bad)


def test_only_adapters_import_aip():
    offenders = [
        f"{f.relative_to(PKG.parent)} imports {m}"
        for f in PKG.rglob("*.py")
        if "adapters" not in f.relative_to(PKG).parts
        for m in _imports(f)
        if _top(m) in LLM
    ]
    assert not offenders, "\n".join(offenders)


def test_system_is_not_a_web_app():
    bad = _violations(_files("system.py"), ["minifinrl"], WEB)
    assert not bad, "\n".join(bad)


def test_interfaces_only_go_through_system():
    """interfaces/ reach the engine through build_system() and the capability
    registry, never by constructing core objects themselves."""
    files = _files("interfaces") + _files("__main__.py")
    bad = _violations(files, ["minifinrl.system", "minifinrl.capabilities", "minifinrl.interfaces"], LLM)
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("script", sorted((PKG.parent / "scripts").glob("*.py")), ids=lambda p: p.name)
def test_scripts_are_thin_shells(script):
    """scripts/ are kept for muscle memory; all logic is behind the CLI."""
    mods = {m for m in _imports(script) if _top(m) == "minifinrl"}
    assert mods <= {"minifinrl.interfaces.cli"}, f"{script.name} imports {sorted(mods)}"


def test_rule_detects_a_violation(tmp_path):
    """The checker itself works: a core-style file importing aip, fastapi
    and an adapter is caught three times, including a lazy import."""
    f = tmp_path / "bad.py"
    f.write_text("import aip\nfrom minifinrl.adapters.x import y\ndef g():\n    import fastapi\n")
    assert len(_violations([f], ["minifinrl.core"], WEB | LLM)) == 3
