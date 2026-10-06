"""Environment invariants and observation scaling (R0 in PLAN_SHIPPABLE.md)."""

from __future__ import annotations

import numpy as np
import pytest

from minifinrl.core.agents.baseline import BuyAndHoldAgent
from minifinrl.core.configs.settings import INDICATORS, TRANSACTION_COST_PCT
from minifinrl.core.envs.portfolio_allocation import PortfolioAllocationEnv
from minifinrl.core.envs.scaling import rule_for, scale_features
from minifinrl.core.envs.stock_trading import StockTradingEnv
from minifinrl.core.eval.backtest import backtest
from minifinrl.core.meta.features import to_array
from minifinrl.core.meta.panel import compute_features
from tests.conftest import random_walk_panel


@pytest.fixture(scope="module")
def arrays():
    panel = compute_features(random_walk_panel(n_days=400))
    panel = panel[panel["date"] >= sorted(panel["date"].unique())[260]]  # past turbulence warm-up
    prices, features, _dates, tickers = to_array(panel, INDICATORS)
    return prices, features, tickers


def _random_episode(env, seed=0, n=None):
    rng = np.random.default_rng(seed)
    obs, _ = env.reset()
    observations = [obs]
    done, steps = False, 0
    while not done and (n is None or steps < n):
        a = rng.uniform(env.action_space.low, env.action_space.high)
        obs, _, term, trunc, info = env.step(a)
        observations.append(obs)
        yield env, obs, info
        done, steps = term or trunc, steps + 1


# ---- scaling ------------------------------------------------------------------

def test_unknown_indicator_has_no_rule():
    with pytest.raises(ValueError, match="no scaling rule"):
        rule_for("boll_ub")


def test_known_values():
    c = np.array([[100.0]])
    assert rule_for("rsi_30")(np.array([[50.0]]), c)[0, 0] == 0.0
    assert rule_for("close_30_sma")(np.array([[110.0]]), c)[0, 0] == pytest.approx(1.0)
    assert rule_for("turbulence")(np.array([[0.0]]), c)[0, 0] == 0.0


def test_every_configured_indicator_has_a_rule():
    for name in INDICATORS + ["turbulence"]:
        rule_for(name)


def test_scale_rejects_name_count_mismatch(arrays):
    prices, features, _ = arrays
    with pytest.raises(ValueError, match="names"):
        scale_features(features, prices, INDICATORS[:2])


@pytest.mark.parametrize("env_cls", [StockTradingEnv, PortfolioAllocationEnv])
def test_observations_are_scaled(arrays, env_cls):
    prices, features, tickers = arrays
    env = env_cls(prices=prices, features=features, tickers=tickers)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    worst = np.abs(obs).max()
    for _env, o, _ in _random_episode(env):
        worst = max(worst, np.abs(o).max())
        assert np.isfinite(o).all()
    assert worst < 10, worst  # raw obs had cash = 1e6 and CCI up to 580


def test_trading_obs_dim_unchanged(arrays):
    prices, features, tickers = arrays
    env = StockTradingEnv(prices=prices, features=features, tickers=tickers)
    n, f = len(tickers), features.shape[2]
    assert env.observation_space.shape == (1 + n + n + n * f,)


# ---- trading invariants -----------------------------------------------------------

def test_trading_accounting_invariants(arrays):
    prices, features, tickers = arrays
    env = StockTradingEnv(prices=prices, features=features, tickers=tickers)
    for e, _obs, info in _random_episode(env, seed=3):
        assert e.cash >= -1e-6
        assert (e.holdings >= -1e-9).all()
        expected = e.cash + float(np.dot(e.holdings, e.prices[e.day]))
        assert info["portfolio_value"] == pytest.approx(expected, rel=1e-12)


def test_round_trip_pays_costs(arrays):
    prices, features, tickers = arrays
    env = StockTradingEnv(prices=prices, features=features, tickers=tickers)
    env.reset()
    n = len(tickers)
    day0 = env.prices[0].copy()
    env.step(np.ones(n))  # buy at day 0 prices
    bought = env.holdings.copy()
    env.day = 0  # sell back at the same prices
    env.step(-np.ones(n))
    gross = float(np.dot(bought, day0))
    lost = env.initial_amount - env.cash
    assert lost == pytest.approx(gross * 2 * TRANSACTION_COST_PCT, rel=1e-9)


def test_turbulence_threshold_uses_raw_values(arrays):
    prices, features, tickers = arrays
    env = StockTradingEnv(prices=prices, features=features, tickers=tickers, turbulence_threshold=1e-9)
    env.reset()
    env.day = int(np.argmax(features[:, 0, -1] > 0))
    env.holdings[:] = 5.0
    env.step(np.ones(len(tickers)))  # asked to buy, but risk-off must liquidate
    assert (env.holdings == 0).all()


# ---- portfolio invariants ----------------------------------------------------------

def test_portfolio_weights_and_value(arrays):
    prices, features, tickers = arrays
    env = PortfolioAllocationEnv(prices=prices, features=features, tickers=tickers)
    for e, _obs, info in _random_episode(env, seed=5, n=50):
        assert e.weights.sum() == pytest.approx(1.0) and (e.weights >= 0).all()
        assert info["portfolio_value"] > 0


# ---- degenerate-policy detection -------------------------------------------------------

class _Constant:
    def __init__(self, n):
        self.n = n

    def predict(self, obs, deterministic=True):
        return np.ones(self.n), None


def test_action_stats_flag_a_constant_policy(arrays):
    prices, features, tickers = arrays
    env = StockTradingEnv(prices=prices, features=features, tickers=tickers)
    r = backtest(_Constant(len(tickers)), env, "t")
    assert r["distinct_actions"] == 1 and r["clip_share"] == 1.0
    r2 = backtest(BuyAndHoldAgent(len(tickers)), StockTradingEnv(prices=prices, features=features, tickers=tickers), "t")
    assert r2["distinct_actions"] == 2


def test_fully_invested_baseline_invests_equal_dollars(arrays):
    from minifinrl.core.agents.baseline import FullyInvestedBuyAndHoldAgent
    from minifinrl.core.eval.backtest import run_episode_with_actions

    prices, features, tickers = arrays
    env = StockTradingEnv(prices=prices, features=features, tickers=tickers)
    values, actions = run_episode_with_actions(FullyInvestedBuyAndHoldAgent(env), env)
    # replay the build-up to find when it finished buying
    filled_day = int(np.nonzero(actions.sum(axis=1) > 1e-9)[0].max()) + 1
    env2 = StockTradingEnv(prices=prices, features=features, tickers=tickers)
    agent = FullyInvestedBuyAndHoldAgent(env2)
    obs, _ = env2.reset()
    for _ in range(filled_day):
        obs, *_ = env2.step(agent.predict(obs)[0])
    position = env2.holdings * env2.prices[0]  # the target: equal dollars at day-0 prices
    # share targets are fixed at day-0 prices, so drift during the build-up
    # leaves a little cash; the old 100-shares-once baseline left ~90%
    assert env2.cash / env2.initial_amount < 0.05
    assert position.std() / position.mean() < 0.01
    assert (actions[filled_day:] == 0).all()                 # then it only holds
    assert env2.cash >= 0


def test_portfolio_turnover_is_charged_against_drifted_weights(arrays):
    """E02: holding 1/N still trades every day to undo price drift.
    Hand-computed: w_drift = w(1+r)/(1+w.r), turnover = sum|1/N - w_drift|."""
    prices, features, tickers = arrays
    env = PortfolioAllocationEnv(prices=prices, features=features, tickers=tickers)
    env.reset()
    n = len(tickers)
    w = np.ones(n) / n
    value = env.initial_amount
    for t in range(5):
        r = prices[t + 1] / prices[t] - 1.0
        turnover = np.abs(w - env.weights).sum()
        expected = value * (1 + w @ r - TRANSACTION_COST_PCT * turnover)
        _, _, _, _, info = env.step(np.zeros(n))  # softmax(0) = 1/N
        assert info["portfolio_value"] == pytest.approx(expected, rel=1e-12)
        assert env.weights == pytest.approx(w * (1 + r) / (1 + w @ r))
        assert env.last_turnover == pytest.approx(turnover)
        if t > 0:
            assert turnover > 0  # v1 charged exactly 0 here
        value = expected
