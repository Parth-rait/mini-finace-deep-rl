"""E07 selection study on constructed returns with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from minifinrl.core.walkforward import fold_table, make_folds, persistence, selection_study, strategy_returns


def test_folds():
    f = make_folds(2019, 2025, final_end="2026-06-30")
    assert [x.name for x in f] == ["2019", "2020", "2021", "2022", "2023", "2024", "2025-2026"]
    assert (f[0].train_start, f[0].train_end, f[0].test_start, f[0].test_end) == ("2014-01-01", "2018-12-31", "2019-01-01", "2019-12-31")
    assert f[-1].test_end == "2026-06-30" and f[-1].train_start == "2020-01-01"


def _sr(spec: dict[str, dict[str, float]], days=250, seed=0):
    """spec: fold -> strategy -> daily mean return (same noise for all, so
    the ranking within a fold is exactly the ranking of the means)."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, (fold, strats) in enumerate(spec.items()):
        dates = pd.bdate_range(f"{2019 + k}-01-02", periods=days).strftime("%Y-%m-%d")
        noise = rng.normal(0, 0.01, days)
        for s, mu in strats.items():
            rows += [{"fold": fold, "app": "portfolio", "strategy": s, "date": d, "ret": mu + e, "rf": 0.0}
                     for d, e in zip(dates, noise)]
    return pd.DataFrame(rows)


def test_follow_winner_picks_previous_folds_best_and_hindsight_this_folds():
    spec = {"f1": {"equal_weight": 0.003, "risk_parity": 0.001, "min_variance": 0.0},
            "f2": {"equal_weight": 0.0, "risk_parity": 0.003, "min_variance": 0.001},
            "f3": {"equal_weight": 0.001, "risk_parity": 0.0, "min_variance": 0.003}}
    t = selection_study(_sr(spec), "portfolio", ["f1", "f2", "f3"]).set_index("rule")
    assert t.loc["follow_winner", "picks"] == "equal_weight,risk_parity"  # f1's winner in f2, f2's in f3
    assert t.loc["hindsight_best", "picks"] == "risk_parity,min_variance"
    assert t.loc["hindsight_best", "sharpe"] >= t["sharpe"].drop("hindsight_best").max() - 1e-12
    # here the winner always flips, so following it is the worst rule
    assert t.loc["follow_winner", "sharpe"] < t.loc["mix_classical", "sharpe"]


def test_mix_is_the_average_of_daily_returns():
    spec = {"f1": {"equal_weight": 0.0, "risk_parity": 0.0, "min_variance": 0.0},
            "f2": {"equal_weight": 0.003, "risk_parity": 0.0, "min_variance": -0.003}}
    sr = _sr(spec)
    t = selection_study(sr, "portfolio", ["f1", "f2"]).set_index("rule")
    ew = sr[(sr.fold == "f2") & (sr.strategy == "equal_weight")]["ret"].to_numpy()
    days = len(ew)
    expected_cagr = np.prod(1 + (ew - 0.003)) ** (252 / days) - 1  # mean of +3bp, 0, -3bp offsets = 0 offset
    assert t.loc["mix_classical", "cagr"] == pytest.approx(expected_cagr, rel=1e-9)


def test_persistence_detects_stable_and_flipping_rankings():
    stable = {"f1": {"a": 0.003, "b": 0.002, "c": 0.001}, "f2": {"a": 0.003, "b": 0.002, "c": 0.001}}
    flip = {"f1": {"a": 0.003, "b": 0.002, "c": 0.001}, "f2": {"a": 0.001, "b": 0.002, "c": 0.003}}
    assert persistence(fold_table(_sr(stable)), "portfolio", ["f1", "f2"])[0]["spearman"] == pytest.approx(1.0)
    assert persistence(fold_table(_sr(flip)), "portfolio", ["f1", "f2"])[0]["spearman"] == pytest.approx(-1.0)


def test_strategy_returns_collapses_seeds_by_median():
    d = pd.DataFrame({"fold": "f", "app": "a", "model": "ppo", "seed": [0, 1, 2], "date": "2020-01-02",
                      "ret": [0.01, 0.02, 0.09], "rf": 0.0001})
    out = strategy_returns(d)
    assert len(out) == 1 and out.loc[0, "ret"] == 0.02 and out.loc[0, "strategy"] == "ppo"


def test_last_fold_only_absorbs_a_partial_final_year():
    assert make_folds(2020, 2021, final_end="2026-06-30")[-1].test_end == "2021-12-31"


def test_walk_forward_end_to_end(rw_engine):
    from minifinrl.capabilities import CapabilityRegistry

    reg = CapabilityRegistry.from_engine(rw_engine)
    reg.invoke("fetch_data", {}, interface="cli")
    out = reg.invoke("walk_forward", {"first_test_year": 2020, "last_test_year": 2021, "train_years": 2,
                                      "apps": ["portfolio"], "models": ["ppo"], "seeds": [0], "timesteps": 128},
                     interface="cli")
    assert out.folds == ["2020", "2021"]
    st = out.study["portfolio"]
    rules = {r["rule"] for r in st["selection"]}
    assert {"follow_winner", "hindsight_best", "mix_classical", "static:equal_weight", "static:ppo"} <= rules
    assert len(st["persistence"]) == 1
    # each fold's PPO model was trained only on that fold's window
    from minifinrl.core.agents.registry import read_card
    card = read_card(rw_engine.cfg.paths.model_dir / "2021" / "portfolio_ppo_seed0.zip")
    assert (card.train_start, card.train_end) == ("2019-01-01", "2020-12-31")
