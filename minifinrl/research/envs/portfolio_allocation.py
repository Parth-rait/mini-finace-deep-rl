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

from minifinrl.research.settings import INITIAL_AMOUNT, REWARD_SCALING, TRANSACTION_COST_PCT
from minifinrl.research.envs.scaling import OBS_SCALING, default_feature_names, scale_features


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
    # v2 (6 Oct 2026, EXPERIMENTS.md E02): turnover is charged against the
    # weights after price drift, not against yesterday's target. Recorded on
    # every model card; a v1-trained policy is refused.
    env_version = "portfolio-v2"

    def __init__(
        self,
        prices: np.ndarray,  # (T, N)
        features: np.ndarray,  # (T, N, F)
        tickers: list[str],
        initial_amount: float = INITIAL_AMOUNT,
        transaction_cost_pct: float = TRANSACTION_COST_PCT,
        reward_scaling: float = REWARD_SCALING,
        feature_names: list[str] | None = None,
    ):
        super().__init__()
        assert prices.shape[0] == features.shape[0]
        assert prices.shape[1] == features.shape[1] == len(tickers)

        self.prices = prices
        self.features = features
        self.feature_names = feature_names or default_feature_names(features.shape[2])
        self.obs_features = scale_features(features, prices, self.feature_names)
        self.obs_scaling = OBS_SCALING
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
        # weights are already in [0, 1]; features scaled per envs/scaling.py
        flat_features = self.obs_features[self.day].reshape(-1)
        return np.concatenate((self.weights, flat_features)).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.day = 0
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = self.initial_amount
        self.asset_history = [self.portfolio_value]
        self.last_turnover = 0.0
        self.last_cost = 0.0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # self.weights are the weights held *now*, i.e. after yesterday's
        # price moves. Rebalancing to new_weights trades their difference:
        #   turnover_t = sum_i |w_new_i - w_drift_i|,  cost_t = c * turnover_t
        new_weights = _softmax(np.asarray(action, dtype=np.float64))
        turnover = np.abs(new_weights - self.weights).sum()
        cost = turnover * self.transaction_cost_pct

        price_today = self.prices[self.day]
        price_next = self.prices[min(self.day + 1, self.max_step)]
        asset_returns = price_next / price_today - 1.0
        gross = float(np.dot(new_weights, asset_returns))
        portfolio_return = gross - cost

        prev_value = self.portfolio_value
        self.portfolio_value *= 1.0 + portfolio_return
        # drift: w_drift_i = w_i (1 + r_i) / (1 + w . r)
        self.weights = new_weights * (1.0 + asset_returns) / (1.0 + gross)
        self.last_turnover = turnover
        self.last_cost = cost * prev_value
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
