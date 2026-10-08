"""The package layout, enforced (see ARCHITECTURE.md).

    interfaces -> system -> engine -> feature services
                                         |
    features:  market  regime  research  sentiment  review  orders
    shared:    platform (settings, logging, errors, the capability registry)

Parses every module's imports with `ast` (no importing, so it runs in any
environment). The rules keep features independent, so a new feature is a
new package rather than edits across old ones, and no file grows into a
god module again.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "minifinrl"

WEB = {"fastapi", "starlette", "uvicorn", "sse_starlette"}
LLM = {"aip", "litellm", "openai", "anthropic"}
TRAINING = {"torch", "stable_baselines3", "gymnasium"}
MAX_LINES = 400  # split a module before it passes this

# feature -> the features it may import (besides itself and platform). Acyclic.
FEATURES = {
    "market": set(),
    "regime": {"market"},
    "research": {"market", "regime"},
    "sentiment": set(),
    "review": {"market", "regime"},
    "orders": set(),
}


def _imports(path: Path) -> set[str]:
    """Absolute module names imported anywhere in the file (incl. lazily)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module)
            # `from minifinrl.market import symbols` names a module too
            out.update(f"{node.module}.{a.name}" for a in node.names if node.module.count(".") <= 1)
    return out


def _top(mod: str) -> str:
    return mod.split(".")[0]


def _part(mod: str, i: int) -> str | None:
    parts = mod.split(".")
    return parts[i] if len(parts) > i else None


def _files(sub: str) -> list[Path]:
    base = PKG / sub
    return sorted(base.rglob("*.py")) if base.is_dir() else ([base] if base.exists() else [])


def _all() -> list[Path]:
    return sorted(f for f in PKG.rglob("*.py") if "__pycache__" not in f.parts)


def _label(f: Path) -> str:
    return str(f.relative_to(PKG.parent)) if f.is_relative_to(PKG.parent) else f.name


def _feature_violations(feature: str, files: list[Path]) -> list[str]:
    allowed = {feature, "platform"} | FEATURES.get(feature, set())
    bad = []
    for f in files:
        for mod in sorted(_imports(f)):
            if _top(mod) != "minifinrl" or mod == "minifinrl":
                continue
            other = _part(mod, 1)
            if other not in allowed:
                bad.append(f"{_label(f)} imports {mod} ({feature} may use {sorted(allowed)})")
            elif other != feature and _part(mod, 2) in ("service", "adapters"):
                bad.append(f"{_label(f)} imports {mod}: services and adapters are wired by engine/system only")
    return bad


def test_every_feature_is_known():
    """A new feature package gets an entry in FEATURES (its allowed dependencies)."""
    packages = {p.name for p in PKG.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    assert packages - {"platform", "interfaces"} == set(FEATURES)


@pytest.mark.parametrize("feature", sorted(FEATURES))
def test_features_depend_only_downwards(feature):
    bad = _feature_violations(feature, _files(feature))
    assert not bad, "\n".join(bad)


def test_platform_depends_on_no_feature():
    bad = [f"{_label(f)} imports {m}" for f in _files("platform") for m in _imports(f)
           if _top(m) == "minifinrl" and m != "minifinrl" and _part(m, 1) != "platform"]
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("feature", sorted(FEATURES))
def test_feature_layout(feature):
    """Every feature has the same shape: capabilities in service.py, their
    input and output models in schemas.py."""
    assert (PKG / feature / "service.py").exists() and (PKG / feature / "schemas.py").exists()


def test_engine_registers_every_feature_service():
    from minifinrl.engine import Engine, EngineConfig

    engine = Engine(EngineConfig(profile="test"))
    registered = {type(s).__module__.split(".")[1] for s in engine.services}
    assert registered == set(FEATURES)


def test_only_adapters_import_llm_libraries():
    bad = [f"{_label(f)} imports {m}" for f in _all() if "adapters" not in f.relative_to(PKG).parts
           for m in _imports(f) if _top(m) in LLM]
    assert not bad, "\n".join(bad)


def test_only_interfaces_import_web_libraries():
    bad = [f"{_label(f)} imports {m}" for f in _all() if f.relative_to(PKG).parts[0] != "interfaces"
           for m in _imports(f) if _top(m) in WEB]
    assert not bad, "\n".join(bad)


def test_only_research_imports_the_training_stack():
    bad = [f"{_label(f)} imports {m}" for f in _all() if f.relative_to(PKG).parts[0] != "research"
           for m in _imports(f) if _top(m) in TRAINING]
    assert not bad, "\n".join(bad)


def test_serving_never_loads_the_training_stack():
    """The API and website start without torch: about half the memory, so
    they fit a small server. Training code is imported inside batch methods."""
    code = ("import sys\nfrom minifinrl.interfaces.api import create_app\nfrom minifinrl.system import build_system\n"
            "create_app(build_system('ci'))\nprint(sorted(m for m in sys.modules if m.split('.')[0] in %r))" % (TRAINING,))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=PKG.parent, check=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_interfaces_only_go_through_system():
    """interfaces/ reach features through build_system() and the capability
    registry, never by constructing feature objects themselves."""
    allowed = ("minifinrl.system", "minifinrl.platform.capabilities", "minifinrl.interfaces")
    files = _files("interfaces") + _files("__main__.py")
    bad = [f"{_label(f)} imports {m}" for f in files for m in _imports(f)
           if (_top(m) == "minifinrl" and m != "minifinrl" and not m.startswith(allowed)) or _top(m) in LLM]
    assert not bad, "\n".join(bad)


def test_engine_and_system_are_not_web_apps():
    bad = [f"{_label(f)} imports {m}" for name in ("engine.py", "system.py") for f in _files(name)
           for m in _imports(f) if _top(m) in WEB]
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("path", _all(), ids=lambda p: _label(p))
def test_no_module_grows_too_big(path):
    n = len(path.read_text().splitlines())
    assert n <= MAX_LINES, f"{_label(path)} has {n} lines; split it by responsibility (limit {MAX_LINES})"


@pytest.mark.parametrize("script", sorted((PKG.parent / "scripts").glob("*.py")), ids=lambda p: p.name)
def test_scripts_are_thin_shells(script):
    """scripts/ are kept for muscle memory; all logic is behind the CLI."""
    mods = {m for m in _imports(script) if _top(m) == "minifinrl"}
    assert mods <= {"minifinrl.interfaces.cli"}, f"{script.name} imports {sorted(mods)}"


def test_rule_detects_a_violation(tmp_path):
    """The checker itself works: a market module importing review, another
    feature's service, and (lazily) the training stack is caught."""
    f = tmp_path / "bad.py"
    f.write_text("from minifinrl.review.intake import merge\nfrom minifinrl.regime.service import RegimeService\n"
                 "def g():\n    import torch\n")
    assert len(_feature_violations("market", [f])) == 2
    assert len(_feature_violations("research", [tmp_path / "bad.py"])) == 2  # review not allowed; a service import
    assert "torch" in _imports(f)
