"""The composition root: profiles, and the aip import-order guard."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from minifinrl.system import SystemConfig, build_system, configure_aip_env


@pytest.fixture(autouse=True)
def clean_aip_env(monkeypatch):
    for k in ("AIP_CACHE_DIR", "AIP_TRACE_DIR", "AIP_OFFLINE"):
        monkeypatch.delenv(k, raising=False)


def test_ci_profile_is_offline():
    s = build_system("ci")
    import os

    assert s.config.engine.spec.refresh is False
    assert os.environ["AIP_OFFLINE"] == "1"
    assert s.engine.bias is not None and s.engine.bias.name == "rules-v1"
    assert "place_order" in s.capabilities.names()


def test_aip_dirs_default_to_this_repo_and_respect_env(monkeypatch, tmp_path):
    build_system("research")
    import os

    assert Path(os.environ["AIP_CACHE_DIR"]).parent == SystemConfig.from_profile("research").aip_cache_dir.parent
    monkeypatch.setenv("AIP_CACHE_DIR", str(tmp_path / "mine"))
    build_system("research")
    assert os.environ["AIP_CACHE_DIR"] == str(tmp_path / "mine")


def test_aip_imported_too_early_is_an_error(monkeypatch, tmp_path):
    fake = types.ModuleType("aip.config")
    fake.settings = types.SimpleNamespace(cache_dir=tmp_path / "site-packages" / ".aip_cache")
    monkeypatch.setitem(sys.modules, "aip.config", fake)
    with pytest.raises(RuntimeError, match="imported before build_system"):
        configure_aip_env(SystemConfig.from_profile("research"))


def test_unknown_profile_and_backend():
    with pytest.raises(ValueError, match="unknown profile"):
        build_system("prod")
    with pytest.raises(ValueError, match="unknown bias backend"):
        build_system("research", bias_backend="gpt")


def test_no_bias_backend():
    assert build_system("research", bias_backend=None).engine.bias is None


def test_universe_and_workspace_isolate_a_run():
    s = build_system("research", universe="cross_asset12", workspace="E99")
    eng = s.engine.cfg
    assert len(eng.spec.tickers) == 12 and "TLT" in eng.spec.tickers
    for p in (eng.paths.model_dir, eng.paths.synthetic_dir, eng.paths.results_path):
        assert "experiments/E99/workspace" in str(p)
    assert "experiments" not in str(build_system("research").engine.cfg.paths.model_dir)
    with pytest.raises(ValueError, match="unknown universe"):
        build_system("research", universe="nasdaq100")
