"""
Multi-asset discrete-share stock trading environment.

PROVENANCE: adapted from FinRL `meta/env_stock_trading/env_stocktrading.py`
and `finrl/applications/stock_trading/*.py`. Trimmed to what this project
needs: single stock universe, cash + per-asset share-count state, no
short selling, optional turbulence-based risk control. Share amounts are
treated as continuous (a Box action, not integer lots) - simpler for SB3
and immaterial at this position-sizing scale.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from minifinrl.core.configs.settings import (
    HMAX,
    INITIAL_AMOUNT,
    REWARD_SCALING,
    TRANSACTION_COST_PCT,
    USE_TURBULENCE,
)
from minifinrl.core.envs.scaling import OBS_SCALING, default_feature_names, scale_features


class StockTradingEnv(gym.Env):
    """One episode = one full pass through `prices`/`features` (a
    historical or synthetic path). Action is a per-ticker share-delta in
    [-hmax, hmax]; observation is [cash, prices, holdings, flat features],
    each scaled to roughly +-1 (see _get_obs and envs/scaling.py).
    """

    metadata = {"render_modes": []}
    env_version = "trading-v1"  # costs are charged on actual share trades, so already exact

    def __init__(
        self,
        prices: np.ndarray,  # (T, N)
        features: np.ndarray,  # (T, N, F) - last F column is turbulence if USE_TURBULENCE
        tickers: list[str],
        initial_amount: float = INITIAL_AMOUNT,
        hmax: int = HMAX,
        transaction_cost_pct: float = TRANSACTION_COST_PCT,
        reward_scaling: float = REWARD_SCALING,
        turbulence_threshold: float | None = None,
        feature_names: list[str] | None = None,
    ):
        super().__init__()
        assert prices.shape[0] == features.shape[0]
        assert prices.shape[1] == features.shape[1] == len(tickers)

        self.prices = prices
        self.features = features  # raw: the turbulence threshold compares raw values
        self.feature_names = feature_names or default_feature_names(features.shape[2])
        self.obs_features = scale_features(features, prices, self.feature_names)
        self.obs_scaling = OBS_SCALING
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.n_features = features.shape[2]
        self.initial_amount = initial_amount
        self.hmax = hmax
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.use_turbulence = USE_TURBULENCE
        self.turbulence_threshold = turbulence_threshold
        self.max_step = prices.shape[0] - 1

        obs_dim = 1 + self.n_assets + self.n_assets + self.n_assets * self.n_features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_assets,), dtype=np.float32
        )

        self.day = 0
        self.cash = initial_amount
        self.holdings = np.zeros(self.n_assets, dtype=np.float64)
        self.portfolio_value = initial_amount
        self.asset_history: list[float] = []

    def _turbulence(self) -> float:
        if not self.use_turbulence:
            return 0.0
        return float(self.features[self.day, 0, -1])

    def _compute_portfolio_value(self) -> float:
        return self.cash + float(np.dot(self.holdings, self.prices[self.day]))

    def _get_obs(self) -> np.ndarray:
        # Same layout as before (so obs_dim is unchanged), scaled:
        #   cash            -> share of initial capital (1.0 at reset)
        #   prices          -> relative to the episode's first day
        #   holdings        -> position value as share of initial capital
        #   features        -> envs/scaling.py rules
        price = self.prices[self.day]
        obs = np.concatenate((
            [self.cash / self.initial_amount],
            price / self.prices[0],
            self.holdings * price / self.initial_amount,
            self.obs_features[self.day].reshape(-1),
        ))
        return obs.astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.day = 0
        self.cash = self.initial_amount
        self.holdings = np.zeros(self.n_assets, dtype=np.float64)
        self.portfolio_value = self._compute_portfolio_value()
        self.asset_history = [self.portfolio_value]
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        prev_value = self.portfolio_value

        actions = np.clip(action, -1, 1) * self.hmax
        turbulence = self._turbulence()
        if self.turbulence_threshold is not None and turbulence > self.turbulence_threshold:
            # risk-off: liquidate everything, ignore the requested action
            actions = -self.holdings.copy()

        # sells first (frees up cash), then buys - both are share-value
        # truncated to what's actually available (no shorting, no margin)
        traded, paid = 0.0, 0.0
        order = np.argsort(actions)
        for i in order:
            price = self.prices[self.day, i]
            if price <= 0:
                continue
            shares = actions[i]
            if shares < 0:
                sell = min(-shares, self.holdings[i])
                proceeds = sell * price * (1 - self.transaction_cost_pct)
                traded += sell * price
                paid += sell * price * self.transaction_cost_pct
                self.cash += proceeds
                self.holdings[i] -= sell
            elif shares > 0:
                affordable = self.cash / (price * (1 + self.transaction_cost_pct))
                buy = min(shares, affordable)
                cost = buy * price * (1 + self.transaction_cost_pct)
                traded += buy * price
                paid += buy * price * self.transaction_cost_pct
                self.cash -= cost
                self.holdings[i] += buy

        # turnover = traded notional / portfolio value before trading (same
        # definition as the portfolio env's sum |dw|)
        self.last_turnover = traded / prev_value if prev_value > 0 else 0.0
        self.last_cost = paid
        self.day += 1
        terminated = self.day >= self.max_step
        self.portfolio_value = self._compute_portfolio_value()
        reward = (self.portfolio_value - prev_value) * self.reward_scaling
        self.asset_history.append(self.portfolio_value)

        return (
            self._get_obs(),
            float(reward),
            terminated,
            False,
            {"portfolio_value": self.portfolio_value},
        )
