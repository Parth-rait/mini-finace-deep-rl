"""E03/E04 maths against known answers: classical optimisers, the trailing-
window baseline, and the risk-free conversion in Sharpe."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from minifinrl.core.agents.classical import (
    REBALANCE_EVERY,
    RebalancingBaseline,
    min_variance_weights,
    risk_contributions,
    risk_parity_weights,
)
from minifinrl.core.configs.settings import INDICATORS
from minifinrl.core.envs.portfolio_allocation import PortfolioAllocationEnv
from minifinrl.core.eval.metrics import sharpe_ratio, summarize
from minifinrl.core.meta.features import to_array
from minifinrl.core.meta.panel import compute_features
from minifinrl.core.pipeline import risk_free_daily
from tests.conftest import random_walk_panel

# ---- optimisers ----------------------------------------------------------------------


def test_min_variance_two_asset_closed_form():
    s1, s2, s12 = 0.04, 0.09, 0.006
    w = min_variance_weights(np.array([[s1, s12], [s12, s2]]))
    w1 = (s2 - s12) / (s1 + s2 - 2 * s12)  # interior solution, = 0.7119
    assert w == pytest.approx([w1, 1 - w1], abs=1e-6)


def test_min_variance_long_only_corner():
    # unconstrained optimum would short asset 2 (w1 = 1.33); long-only -> [1, 0]
    w = min_variance_weights(np.array([[0.04, 0.05], [0.05, 0.09]]))
    assert w == pytest.approx([1.0, 0.0], abs=1e-6)


def test_risk_parity_uncorrelated_is_inverse_vol():
    sig = np.array([0.1, 0.2, 0.4])
    w = risk_parity_weights(np.diag(sig**2))
    assert w == pytest.approx((1 / sig) / (1 / sig).sum(), abs=1e-9)


def test_risk_parity_equal_contributions_and_scale_invariance():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 8))
    cov = (a @ a.T + 8 * np.eye(8)) * 1e-4  # daily-sized, correlated
    w = risk_parity_weights(cov)
    assert risk_contributions(w, cov) == pytest.approx(np.full(8, 1 / 8), abs=1e-9)
    assert risk_parity_weights(cov * 1e4) == pytest.approx(w, abs=1e-9)
    assert min_variance_weights(cov * 1e4) == pytest.approx(min_variance_weights(cov), abs=1e-6)


# ---- the rebalancing baseline in the env ------------------------------------------------


@pytest.fixture(scope="module")
def setup():
    panel = compute_features(random_walk_panel(n_days=700))
    dates = sorted(panel["date"].unique())
    window = panel[panel["date"] >= dates[400]]
    closes = panel.pivot(index="date", columns="tic", values="close")
    prior = closes[closes.index < dates[400]].iloc[-253:]
    prices, features, _d, tickers = to_array(window, INDICATORS)
    return prices, features, tickers, prior.to_numpy()


def _env(setup, prices=None):
    p, f, t, prior = setup
    env = PortfolioAllocationEnv(prices=p if prices is None else prices, features=f, tickers=t)
    env.prior_prices = prior
    return env


def test_holds_between_rebalances_and_pays_nothing(setup):
    env = _env(setup)
    agent = RebalancingBaseline(env, risk_parity_weights)
    obs, _ = env.reset()
    for day in range(3 * REBALANCE_EVERY):
        obs, *_ = env.step(agent.predict(obs)[0])
        if day % REBALANCE_EVERY != 0:
            assert env.last_turnover == pytest.approx(0.0, abs=1e-12)
    assert agent.rebalances == 3


def test_day_zero_uses_prior_history_not_one_over_n(setup):
    env = _env(setup)
    env.reset()
    w = RebalancingBaseline(env, min_variance_weights).target_weights()
    assert not np.allclose(w, 1 / len(w)) and w.sum() == pytest.approx(1.0)


def test_no_lookahead_in_weights(setup):
    """Weights decided on rebalance day 42 ignore every price after day 42."""
    prices = setup[0]
    changed = prices.copy()
    changed[43:] *= np.linspace(0.5, 2.0, prices.shape[1])
    weights = []
    for p in (prices, changed):
        env = _env(setup, p)
        env.reset()
        env.day = 2 * REBALANCE_EVERY
        weights.append(RebalancingBaseline(env, min_variance_weights).target_weights())
    assert weights[0] == pytest.approx(weights[1], abs=1e-12)


# ---- risk-free rate in Sharpe ------------------------------------------------------------


def test_risk_free_daily_conversion_and_carry_forward():
    yields = pd.Series([4.0, 5.0], index=["2024-01-02", "2024-01-04"])  # 01-03 missing (holiday)
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2027-06-01"]
    rf = risk_free_daily(dates, yields)
    d4, d5 = 1.04 ** (1 / 252) - 1, 1.05 ** (1 / 252) - 1
    assert rf == pytest.approx([d4, d4, d5, d5])  # length T-1; holiday and far future carry forward


def test_sharpe_uses_excess_returns():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0008, 0.01, 500)
    rf = np.full(500, 1.04 ** (1 / 252) - 1)
    ex = r - rf
    assert sharpe_ratio(r, rf) == pytest.approx(ex.mean() / ex.std(ddof=1) * np.sqrt(252))
    assert sharpe_ratio(r, rf) < sharpe_ratio(r, 0.0)
    with pytest.raises(ValueError, match="risk-free series"):
        sharpe_ratio(r, rf[:-1])
    values = 100 * np.cumprod(np.r_[1.0, 1 + r])
    assert summarize(values, rf)["sharpe"] == pytest.approx(sharpe_ratio(r, rf))


def test_negative_yields_are_valid():
    """DTB3 printed -0.05% on 26 Mar 2020: the daily return is slightly negative, not an error."""
    rf = risk_free_daily(["2020-03-26", "2020-03-27"], pd.Series([-0.05], index=["2020-03-26"]))
    assert rf[0] == pytest.approx((1 - 0.0005) ** (1 / 252) - 1) and rf[0] < 0
