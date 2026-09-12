"""
Continuous portfolio-weight allocation environment.

PROVENANCE: adapted from FinRL `meta/env_portfolio_allocation/env_portfolio.py`.
Action is a raw score vector; softmax-normalized internally so the agent
doesn't have to learn the simplex constraint (weights >= 0, sum to 1)
itself.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from configs.settings import INITIAL_AMOUNT, REWARD_SCALING, TRANSACTION_COST_PCT


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


class PortfolioAllocationEnv(gym.Env):
    """One episode = one pass through `prices`/`features`. Action is a
    raw score vector over assets, softmax-normalized into portfolio
    weights each step; reward is the portfolio's return minus a turnover
    (reallocation) cost."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: np.ndarray,  # (T, N)
        features: np.ndarray,  # (T, N, F)
        tickers: list[str],
        initial_amount: float = INITIAL_AMOUNT,
        transaction_cost_pct: float = TRANSACTION_COST_PCT,
        reward_scaling: float = REWARD_SCALING,
    ):
        super().__init__()
        assert prices.shape[0] == features.shape[0]
        assert prices.shape[1] == features.shape[1] == len(tickers)

        self.prices = prices
        self.features = features
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.n_features = features.shape[2]
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.max_step = prices.shape[0] - 1

        obs_dim = self.n_assets + self.n_assets * self.n_features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-10, high=10, shape=(self.n_assets,), dtype=np.float32
        )

        self.day = 0
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = initial_amount
        self.asset_history: list[float] = []

    def _get_obs(self) -> np.ndarray:
        flat_features = self.features[self.day].reshape(-1)
        return np.concatenate((self.weights, flat_features)).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.day = 0
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = self.initial_amount
        self.asset_history = [self.portfolio_value]
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        new_weights = _softmax(np.asarray(action, dtype=np.float64))
        turnover = np.abs(new_weights - self.weights).sum()
        cost = turnover * self.transaction_cost_pct

        price_today = self.prices[self.day]
        price_next = self.prices[min(self.day + 1, self.max_step)]
        asset_returns = price_next / price_today - 1.0
        portfolio_return = float(np.dot(new_weights, asset_returns)) - cost

        prev_value = self.portfolio_value
        self.portfolio_value *= 1.0 + portfolio_return
        self.weights = new_weights
        self.day += 1

        terminated = self.day >= self.max_step
        reward = (self.portfolio_value - prev_value) / prev_value * self.reward_scaling
        self.asset_history.append(self.portfolio_value)

        return (
            self._get_obs(),
            float(reward),
            terminated,
            False,
            {"portfolio_value": self.portfolio_value},
        )
