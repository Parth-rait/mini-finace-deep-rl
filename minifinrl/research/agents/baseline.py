"""
Non-RL baselines. Same predict() signature as DRLAgentWrapper so
eval.backtest can run them through the identical evaluation code path -
without a baseline, a "PPO beat SAC" result tells you nothing about
whether either agent is earning its complexity over doing nothing clever.
"""

from __future__ import annotations

import numpy as np


class BuyAndHoldAgent:
    """For StockTradingEnv: buy the max per-asset share amount on day 0,
    then hold (act zero) for the rest of the episode."""

    def __init__(self, n_assets: int):
        self.n_assets = n_assets
        self._bought = False

    def predict(self, obs, deterministic: bool = True):
        action = np.ones(self.n_assets) if not self._bought else np.zeros(self.n_assets)
        self._bought = True
        return action, None


class EqualWeightAgent:
    """For PortfolioAllocationEnv: constant equal weights every step (the
    softmax of an all-zero score vector is uniform)."""

    def __init__(self, n_assets: int):
        self.n_assets = n_assets

    def predict(self, obs, deterministic: bool = True):
        return np.zeros(self.n_assets), None


class FullyInvestedBuyAndHoldAgent:
    """For StockTradingEnv: equal-dollar buy-and-hold of the whole capital.

    BuyAndHoldAgent buys HMAX (100) shares per ticker once, which on the
    test window is 10.3% of capital, leaving ~90% in cash: a low-volatility
    benchmark the agents shouldn't be measured against (PLAN_SHIPPABLE.md
    F13). This one targets 1/n of capital per ticker (after costs and a
    small cash buffer), priced on the first day, and buys toward it at up to
    HMAX shares per ticker per step (the env's cap, so ~15 days on the test
    window), then holds. It reads absolute prices and holdings from the env
    (the observation only carries scaled ones), which is fine for a fixed
    rule that learns nothing.
    """

    def __init__(self, env, cash_buffer: float = 0.005):
        self.env = getattr(env, "unwrapped", env)
        self.cash_buffer = cash_buffer
        self._target: np.ndarray | None = None

    def predict(self, obs, deterministic: bool = True):
        e = self.env
        if self._target is None or e.day == 0:
            per_ticker = e.initial_amount * (1 - self.cash_buffer) / e.n_assets / (1 + e.transaction_cost_pct)
            self._target = per_ticker / e.prices[e.day]
        remaining = np.maximum(self._target - e.holdings, 0.0)
        return np.clip(remaining / e.hmax, 0.0, 1.0), None
