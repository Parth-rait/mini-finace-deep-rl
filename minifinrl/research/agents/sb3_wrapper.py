"""
Unified wrapper around Stable-Baselines3 algorithms.

PROVENANCE: original to this project - FinRL has an analogous `DRLAgent`
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
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv

from minifinrl.research.agents.registry import ModelCard, ModelSpec, verify, write_card
from minifinrl.platform.log import get_logger

log = get_logger(__name__)

_ALGOS: dict[str, type[BaseAlgorithm]] = {"ppo": PPO, "sac": SAC, "td3": TD3}


class DRLAgentWrapper:
    """Same train()/predict()/save()/load() surface regardless of which
    SB3 algorithm is underneath. Add a new algorithm by adding one entry
    to `_ALGOS` - nothing else in the pipeline changes."""

    def __init__(self, model_name: str, env: gym.Env, params: dict[str, Any], seed: int):
        self.model_name = model_name.lower()
        if self.model_name not in _ALGOS:
            raise ValueError(f"unknown model '{model_name}', expected one of {list(_ALGOS)}")

        self.seed = seed
        self.env = DummyVecEnv([lambda: env])
        algo_cls = _ALGOS[self.model_name]
        params = dict(params)
        if self.model_name == "td3" and "action_noise" not in params:
            from minifinrl.research.settings import TD3_ACTION_NOISE

            half_range = (env.action_space.high - env.action_space.low) / 2.0
            params["action_noise"] = NormalActionNoise(np.zeros_like(half_range), TD3_ACTION_NOISE * half_range)
        self.model: BaseAlgorithm = algo_cls("MlpPolicy", self.env, verbose=0, seed=seed, **params)
        log.info("initialized %s agent, seed=%d, params=%s", self.model_name.upper(), seed, params)

    def train(self, total_timesteps: int) -> "DRLAgentWrapper":
        log.info("training %s (seed=%s) for %d timesteps", self.model_name.upper(), self.seed, total_timesteps)
        start = time.monotonic()
        self.model.learn(total_timesteps=total_timesteps)
        elapsed = time.monotonic() - start
        self.train_seconds = elapsed
        log.info("finished training %s in %.1fs (%.0f steps/s)", self.model_name.upper(), elapsed, total_timesteps / elapsed if elapsed > 0 else float("inf"))
        return self

    def predict(self, obs, deterministic: bool = True):
        return self.model.predict(obs, deterministic=deterministic)

    def save(self, path: str | Path, card: ModelCard | None = None) -> None:
        """Save the policy, and its metadata card next to it (M1). Without a
        card the model can only be loaded with expect=None."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        if card is not None:
            write_card(path, card)
        log.info("saved %s model to %s%s", self.model_name.upper(), path, " (+ card)" if card else "")

    @classmethod
    def load(
        cls,
        model_name: str,
        path: str | Path,
        env: gym.Env,
        *,
        expect: ModelSpec | None = None,
        data_hash: str | None = None,
        strict_data: bool = False,
    ) -> "DRLAgentWrapper":
        """Load a saved policy for evaluation against a (possibly
        different - historical vs. synthetic) environment instance.

        With `expect` (M2), the policy's card must match what this env feeds
        it (tickers, features, scaling, shapes), or ModelMismatch is raised
        before SB3 is touched. `data_hash` is compared too: a warning, or an
        error with strict_data=True."""
        if expect is not None:
            verify(path, expect, model_name.lower(), data_hash=data_hash, strict_data=strict_data)
        algo_cls = _ALGOS[model_name.lower()]
        obj = cls.__new__(cls)
        obj.model_name = model_name.lower()
        obj.seed = None
        obj.env = DummyVecEnv([lambda: env])
        obj.model = algo_cls.load(str(path), env=obj.env)
        log.info("loaded %s model from %s", obj.model_name.upper(), path)
        return obj
