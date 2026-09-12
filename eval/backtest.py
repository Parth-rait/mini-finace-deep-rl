"""
Single-episode backtest runner: step an agent through a Gym environment
end-to-end and collect the portfolio value trajectory.

Doesn't care if the agent is a DRLAgentWrapper, a
BuyAndHoldAgent/EqualWeightAgent, or something else entirely, as long as
it has `.predict(obs, deterministic=True) -> (action, _)`.
"""

from __future__ import annotations

from typing import Any, Protocol

import gymnasium as gym
import numpy as np

from configs.logging_config import get_logger
from eval.metrics import summarize

log = get_logger(__name__)


class Predictor(Protocol):
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]: ...


def run_episode(agent: Predictor, env: gym.Env) -> np.ndarray:
    """Runs one full episode, returns the portfolio value at every step
    (length T+1, including the initial value before any trades)."""
    obs, _ = env.reset()
    values = [env.unwrapped.portfolio_value]
    done = False
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        values.append(info["portfolio_value"])
    return np.array(values, dtype=np.float64)


def backtest(agent: Predictor, env: gym.Env, label: str) -> dict:
    """Runs one episode and returns {label, equity_curve, **summary metrics}."""
    values = run_episode(agent, env)
    stats = summarize(values)
    log.info(
        "backtest[%s]: final_value=%.0f sharpe=%.3f max_drawdown=%.3f",
        label, stats["final_value"], stats["sharpe"], stats["max_drawdown"],
    )
    return {"label": label, "equity_curve": values, **stats}
