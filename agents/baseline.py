"""
Non-RL baselines. Same predict() signature as DRLAgentWrapper so
eval.backtest can run them through the identical evaluation code path —
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
