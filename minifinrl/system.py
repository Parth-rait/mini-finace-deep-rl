"""
The composition root: the one place the whole system is assembled.

    from minifinrl.system import build_system
    system = build_system("research")          # research | live | ci
    system.capabilities.invoke("rank_stability", {"app": "trading"})

Every interface (CLI, API, agent, gate) starts with build_system() and
talks to `system.capabilities`, so this file is where to look to see how
everything fits: which data source, which store, which bias classifier,
which aip settings. Nothing constructs core objects or adapters anywhere
else (tests/test_architecture.py enforces that for interfaces/).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from minifinrl.capabilities import CapabilityRegistry
from minifinrl.core.configs.logging_config import get_logger
from minifinrl.core import pipeline
from minifinrl.core.configs.settings import RESULTS, ROOT
from minifinrl.core.configs.tickers import UNIVERSES
from minifinrl.core.meta.panel import DataSpec
from minifinrl.engine import Engine, EngineConfig
from minifinrl.ports import BiasClassifier

log = get_logger(__name__)


def _rules_bias() -> BiasClassifier:
    from minifinrl.adapters.rules_bias import RulesBiasClassifier

    return RulesBiasClassifier()


# Phase F2 adds "aip": minifinrl.adapters.aip_bias.AipBiasClassifier
BIAS_BACKENDS: dict[str, Callable[[], BiasClassifier]] = {"rules": _rules_bias}


@dataclass(frozen=True)
class SystemConfig:
    """Everything a profile decides. `engine` is what the finance side sees;
    the rest is wiring (adapters, aip) the Engine never learns about."""

    profile: str
    engine: EngineConfig
    bias_backend: str | None = "rules"
    aip_offline: bool = False
    aip_cache_dir: Path = ROOT / ".aip_cache"
    aip_trace_dir: Path = ROOT / ".aip_traces"

    @classmethod
    def from_profile(cls, profile: str = "research", *, universe: str | None = None, workspace: str | None = None,
                     **overrides) -> "SystemConfig":
        """`universe`: a name in configs/tickers.UNIVERSES. `workspace`: an
        experiment id; models, synthetic paths and results then live under
        results/experiments/<workspace>/, so the run can't touch anything else."""
        if profile not in PROFILES:
            raise ValueError(f"unknown profile '{profile}', expected one of {sorted(PROFILES)}")
        cfg = PROFILES[profile]()
        eng = cfg.engine
        if universe is not None:
            if universe not in UNIVERSES:
                raise ValueError(f"unknown universe '{universe}', expected one of {sorted(UNIVERSES)}")
            eng = replace(eng, spec=replace(eng.spec, tickers=tuple(UNIVERSES[universe])))
        if workspace is not None:
            ws = RESULTS / "experiments" / workspace / "workspace"
            eng = replace(eng, paths=pipeline.Paths(model_dir=ws / "models", synthetic_dir=ws / "synthetic",
                                                    processed_dir=ws / "processed",
                                                    results_path=ws / "backtest_results.csv"))
        return replace(cfg, engine=eng, **overrides)


def _research() -> SystemConfig:
    # pinned dates, Yahoo, the store refreshes on demand
    return SystemConfig(profile="research", engine=EngineConfig(profile="research", mode="research"))


def _live() -> SystemConfig:
    # up to the last close; freshness is checked; still Yahoo (decision 30 Sep)
    return SystemConfig(profile="live", engine=EngineConfig(profile="live", mode="live"))


def _ci() -> SystemConfig:
    # never touches the network: store read-only, aip replays its cache only.
    # G1 will point spec.store_root/paths at a committed fixture.
    return SystemConfig(
        profile="ci",
        engine=EngineConfig(profile="ci", mode="research", spec=DataSpec(refresh=False)),
        bias_backend="rules",
        aip_offline=True,
    )


PROFILES: dict[str, Callable[[], SystemConfig]] = {"research": _research, "live": _live, "ci": _ci}


def configure_aip_env(cfg: SystemConfig) -> None:
    """aip resolves its cache/trace directories from the environment when it
    is first imported; an installed aip left alone would write into
    site-packages (verified 30 Sep). So set them before any adapter imports
    aip. Values already in the environment win, except that `ci` always
    forces offline. If aip was imported earlier with different paths, the
    import order is wrong: fail rather than write to the wrong place."""
    os.environ.setdefault("AIP_CACHE_DIR", str(cfg.aip_cache_dir))
    os.environ.setdefault("AIP_TRACE_DIR", str(cfg.aip_trace_dir))
    if cfg.aip_offline:
        os.environ["AIP_OFFLINE"] = "1"
    aip_config = sys.modules.get("aip.config")
    if aip_config is not None:
        actual = Path(aip_config.settings.cache_dir).resolve()
        if actual != Path(os.environ["AIP_CACHE_DIR"]).resolve():
            raise RuntimeError(
                f"aip was imported before build_system() and uses cache {actual}; "
                "import aip only through minifinrl.adapters"
            )


@dataclass(frozen=True)
class System:
    config: SystemConfig
    engine: Engine
    capabilities: CapabilityRegistry = field(repr=False)


def build_system(profile: str = "research", *, universe: str | None = None, workspace: str | None = None,
                 **overrides) -> System:
    cfg = SystemConfig.from_profile(profile, universe=universe, workspace=workspace, **overrides)
    configure_aip_env(cfg)
    bias = None
    if cfg.bias_backend is not None:
        if cfg.bias_backend not in BIAS_BACKENDS:
            raise ValueError(f"unknown bias backend '{cfg.bias_backend}', expected one of {sorted(BIAS_BACKENDS)}")
        bias = BIAS_BACKENDS[cfg.bias_backend]()
    engine = Engine(cfg.engine, bias=bias)
    system = System(config=cfg, engine=engine, capabilities=CapabilityRegistry.from_engine(engine))
    log.info(
        "system built: profile=%s mode=%s universe=%d tickers workspace=%s bias=%s capabilities=%d",
        cfg.profile, cfg.engine.mode, len(cfg.engine.spec.tickers), workspace or "-",
        bias.name if bias else None, len(system.capabilities.names()),
    )
    return system


__all__ = ["System", "SystemConfig", "build_system", "PROFILES"]
