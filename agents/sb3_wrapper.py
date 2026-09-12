"""
Unified wrapper around Stable-Baselines3 algorithms.

PROVENANCE: original to this project — FinRL has an analogous `DRLAgent`
class in `finrl/agents/stablebaselines3/models.py`; this is a much
smaller version that only supports what this project trains (PPO, SAC),
so that scripts/eval code never has to branch on which algorithm is
underneath.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv

from configs.logging_config import get_logger

log = get_logger(__name__)

_ALGOS: dict[str, type[BaseAlgorithm]] = {"ppo": PPO, "sac": SAC}


class DRLAgentWrapper:
    """Same train()/predict()/save()/load() surface regardless of which
    SB3 algorithm is underneath. Add a new algorithm by adding one entry
    to `_ALGOS` — nothing else in the pipeline changes."""

    def __init__(self, model_name: str, env: gym.Env, params: dict[str, Any], seed: int):
        self.model_name = model_name.lower()
        if self.model_name not in _ALGOS:
            raise ValueError(f"unknown model '{model_name}', expected one of {list(_ALGOS)}")

        self.seed = seed
        self.env = DummyVecEnv([lambda: env])
        algo_cls = _ALGOS[self.model_name]
        self.model: BaseAlgorithm = algo_cls("MlpPolicy", self.env, verbose=0, seed=seed, **params)
        log.info("initialized %s agent, seed=%d, params=%s", self.model_name.upper(), seed, params)

    def train(self, total_timesteps: int) -> "DRLAgentWrapper":
        log.info("training %s (seed=%s) for %d timesteps", self.model_name.upper(), self.seed, total_timesteps)
        start = time.monotonic()
        self.model.learn(total_timesteps=total_timesteps)
        elapsed = time.monotonic() - start
        log.info("finished training %s in %.1fs (%.0f steps/s)", self.model_name.upper(), elapsed, total_timesteps / elapsed if elapsed > 0 else float("inf"))
        return self

    def predict(self, obs, deterministic: bool = True):
        return self.model.predict(obs, deterministic=deterministic)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        log.info("saved %s model to %s", self.model_name.upper(), path)

    @classmethod
    def load(cls, model_name: str, path: str | Path, env: gym.Env) -> "DRLAgentWrapper":
        """Load a saved policy for evaluation against a (possibly
        different — historical vs. synthetic) environment instance. The
        env only needs matching observation/action space shapes, not the
        same one used for training."""
        algo_cls = _ALGOS[model_name.lower()]
        obj = cls.__new__(cls)
        obj.model_name = model_name.lower()
        obj.seed = None
        obj.env = DummyVecEnv([lambda: env])
        obj.model = algo_cls.load(str(path), env=obj.env)
        log.info("loaded %s model from %s", obj.model_name.upper(), path)
        return obj
