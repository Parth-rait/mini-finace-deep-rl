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

from minifinrl.platform.log import get_logger
from minifinrl.research.evaluation.metrics import summarize

log = get_logger(__name__)


class Predictor(Protocol):
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]: ...


def run_episode_with_actions(agent: Predictor, env: gym.Env) -> tuple[np.ndarray, np.ndarray]:
    """Runs one full episode. Returns the portfolio value at every step
    (length T+1, including the initial value before any trades) and the
    actions taken (T, n_assets)."""
    obs, _ = env.reset()
    values = [env.unwrapped.portfolio_value]
    actions, turnover, costs = [], [], []
    done = False
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        actions.append(np.asarray(action, dtype=np.float64).reshape(-1))
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        values.append(info["portfolio_value"])
        turnover.append(getattr(env.unwrapped, "last_turnover", np.nan))
        costs.append(getattr(env.unwrapped, "last_cost", np.nan))
    env.unwrapped.episode_turnover = np.array(turnover)
    env.unwrapped.episode_costs = np.array(costs)
    return np.array(values, dtype=np.float64), np.array(actions)


def run_episode(agent: Predictor, env: gym.Env) -> np.ndarray:
    return run_episode_with_actions(agent, env)[0]


def action_stats(actions: np.ndarray, env: gym.Env) -> dict[str, float]:
    """How varied the policy's behaviour was. A trained agent with one
    distinct action on every step has collapsed to a fixed rule (the 5k-step
    trading SAC did: max buy every day = buy-and-hold in disguise), and its
    metrics say nothing about learning. Clip share = actions at the edge of
    the action space, where the policy output is saturated."""
    low, high = env.action_space.low, env.action_space.high
    clipped = np.clip(actions, low, high)
    at_edge = np.isclose(clipped, low, atol=1e-3 * (high - low)) | np.isclose(clipped, high, atol=1e-3 * (high - low))
    distinct = len(np.unique(actions.round(4), axis=0))
    return {"distinct_actions": int(distinct), "clip_share": float(at_edge.mean())}


def backtest(agent: Predictor, env: gym.Env, label: str) -> dict:
    """Runs one episode and returns {label, equity_curve, **summary metrics,
    distinct_actions, clip_share}."""
    values, actions = run_episode_with_actions(agent, env)
    e = env.unwrapped
    trading = {
        "turnover": float(np.nanmean(e.episode_turnover)),  # mean daily sum|dw| (or traded / value)
        "cost_frac": float(np.nansum(e.episode_costs) / values[0]),  # total costs / initial capital
    }
    rf = getattr(e, "rf_daily", None)  # E03: per-day risk-free returns, attached by the pipeline
    stats = {**summarize(values, 0.0 if rf is None else rf), **action_stats(actions, env), **trading,
             "rf_mean_annual": float(np.mean(rf) * 252) if rf is not None else 0.0}
    log.info(
        "backtest[%s]: final_value=%.0f sharpe=%.3f max_drawdown=%.3f distinct_actions=%d clip_share=%.2f",
        label, stats["final_value"], stats["sharpe"], stats["max_drawdown"],
        stats["distinct_actions"], stats["clip_share"],
    )
    return {"label": label, "equity_curve": values, **stats}
