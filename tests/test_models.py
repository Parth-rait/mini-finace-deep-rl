"""M1/M2: model cards are written, and a policy is refused unless its card
matches what the env feeds it. R2: fair win rates."""

from __future__ import annotations

import dataclasses
import logging

import pandas as pd
import pytest

from minifinrl.capabilities import CapabilityRegistry
from minifinrl.core.agents.registry import (
    ModelCard,
    ModelMismatch,
    ModelRegistry,
    ModelSpec,
    card_path,
    make_card,
    read_card,
    verify,
)
from minifinrl.core.agents.sb3_wrapper import DRLAgentWrapper
from minifinrl.core.configs.settings import INDICATORS, PPO_PARAMS
from minifinrl.core.envs.stock_trading import StockTradingEnv
from minifinrl.core.eval.report import rank_stability
from minifinrl.core.meta.features import to_array
from minifinrl.core.meta.panel import compute_features
from tests.conftest import random_walk_panel


@pytest.fixture(scope="module")
def env():
    panel = compute_features(random_walk_panel(n_days=320))
    prices, features, _d, tickers = to_array(panel[panel["date"] >= sorted(panel["date"].unique())[260]], INDICATORS)
    return StockTradingEnv(prices=prices, features=features, tickers=tickers)


@pytest.fixture(scope="module")
def saved(env, tmp_path_factory):
    path = tmp_path_factory.mktemp("models") / "trading_ppo_seed0.zip"
    agent = DRLAgentWrapper("ppo", env, {**PPO_PARAMS, "n_steps": 64, "batch_size": 32}, seed=0).train(64)
    spec = ModelSpec.from_env("trading", env)
    agent.save(path, card=make_card(algo="ppo", seed=0, timesteps=64, spec=spec, train_start="a",
                                    train_end="b", data_hash="h" * 64, train_seconds=agent.train_seconds))
    return path, spec


def test_card_written_and_round_trips(saved):
    path, spec = saved
    card = read_card(path)
    assert card_path(path).exists() and card.spec == spec and card.model_id == "trading_ppo_seed0"
    assert card.spec.obs_scaling == "scale-v1" and card.versions["stable-baselines3"]
    assert ModelCard.from_json(card.to_json()) == card


def test_matching_env_loads(saved, env):
    path, spec = saved
    agent = DRLAgentWrapper.load("ppo", path, env, expect=spec, data_hash="h" * 64)
    action, _ = agent.predict(env.reset()[0])
    assert action.shape == (len(env.tickers),)


@pytest.mark.parametrize("change,needle", [
    ({"tickers": ("BBB", "AAA", "CCC")}, "tickers"),                 # same set, other order
    ({"feature_names": tuple(INDICATORS[:-1]) + ("rsi_14", "turbulence")}, "feature_names"),
    ({"obs_scaling": "none"}, "obs_scaling"),                          # the pre-30-Sep models
    ({"app": "portfolio", "env_class": "PortfolioAllocationEnv"}, "app"),
])
def test_mismatch_refused(saved, change, needle):
    path, spec = saved
    with pytest.raises(ModelMismatch, match=needle):
        verify(path, dataclasses.replace(spec, **change), "ppo")


def test_wrong_algo_refused(saved):
    path, spec = saved
    with pytest.raises(ModelMismatch, match="algo"):
        verify(path, spec, "sac")


def test_no_card_refused(saved, tmp_path):
    path, spec = saved
    bare = tmp_path / "trading_ppo_seed0.zip"
    bare.write_bytes(path.read_bytes())
    with pytest.raises(ModelMismatch, match="no metadata card"):
        verify(bare, spec, "ppo")


def test_data_hash_warns_or_fails(saved, caplog):
    path, spec = saved
    with caplog.at_level(logging.WARNING, logger="mini_finrl"):
        verify(path, spec, "ppo", data_hash="x" * 64)
    with pytest.raises(ModelMismatch, match="trained on data"):
        verify(path, spec, "ppo", data_hash="x" * 64, strict_data=True)


def test_registry_flags_legacy(saved, tmp_path):
    path, _ = saved
    (tmp_path / "old_model.zip").write_bytes(path.read_bytes())
    entries = {e.model_id: e for e in ModelRegistry(tmp_path).list()}
    assert entries["old_model"].legacy


def test_backtests_skip_a_stale_model_and_say_why(rw_engine):
    reg = CapabilityRegistry.from_engine(rw_engine)
    reg.invoke("fetch_data", {}, interface="cli")
    reg.invoke("fit_hmm", {"n_paths": 1, "length": 30}, interface="cli")
    trained = reg.invoke("train", {"app": "trading", "model": "ppo", "seed": 0, "timesteps": 128}, interface="cli")
    card_path(trained.model_path).unlink()  # now it looks like a pre-M1 model
    out = reg.invoke("run_backtests", {"paths": "both"}, interface="cli")
    assert len(out.skipped) == 1 and "no metadata card" in out.skipped[0]
    assert out.rows == 4 * 2  # only the 4 baselines (historical + 1 path)
    models = reg.invoke("list_models", {}).models
    assert [m.legacy for m in models] == [True]
    assert reg.invoke("health", {}).legacy_models == ["trading_ppo_seed0"]


def test_win_rate_is_fair_across_seed_counts():
    """3 noisy PPO seeds vs 1 baseline. Best-of-rows would hand PPO every
    path (its best seed, 1.5, beats 1.0); seed-median (0.9) loses them all."""
    recs = [{"path": f"p{i}", "model": "baseline", "sharpe": 1.0} for i in range(10)]
    recs += [{"path": f"p{i}", "model": "ppo", "sharpe": s} for i in range(10) for s in (0.5, 0.9, 1.5)]
    t = rank_stability(recs, "path", "model").set_index("model")
    assert t.loc["baseline", "wins"] == 10 and t.loc["ppo", "wins"] == 0
    assert t.loc["ppo", "beats_baseline"] == 0 and pd.isna(t.loc["baseline", "beats_baseline"])
    assert t.loc["baseline", "ci_low"] == pytest.approx(0.7225, abs=1e-3)
