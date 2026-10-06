"""No look-ahead, no cross-ticker leakage, and real history behind every
window's turbulence (fixes F1/F4 in PLAN_SHIPPABLE.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from minifinrl.core.configs.settings import INDICATORS, TURBULENCE_LOOKBACK
from minifinrl.core.meta.features import fill_indicators, train_test_split
from minifinrl.core.meta.panel import compute_features, features_with_history
from tests.conftest import random_walk_panel

FEATURES = INDICATORS + ["turbulence"]


@pytest.fixture(scope="module")
def raw():
    return random_walk_panel()


@pytest.fixture(scope="module")
def full(raw):
    return compute_features(raw)


@pytest.mark.parametrize("cut", [TURBULENCE_LOOKBACK + 10, 330, 419])
def test_no_lookahead(raw, full, cut):
    """Features on day t must be the same whether or not later days exist."""
    dates = sorted(raw["date"].unique())
    t = dates[cut]
    prefix = compute_features(raw[raw["date"] <= t])
    a = full[full["date"] == t].set_index("tic")[FEATURES]
    b = prefix[prefix["date"] == t].set_index("tic")[FEATURES]
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)


def test_fill_indicators_stays_within_ticker():
    df = pd.DataFrame({
        "date": ["d1", "d1", "d2", "d2", "d3", "d3"],
        "tic": ["A", "B"] * 3,
        "x": [np.nan, 5.0, 1.0, 6.0, np.nan, 7.0],
    })
    out = fill_indicators(df, ["x"])
    # d1 is A's warm-up: dropped for everyone (rectangular), never filled from B
    assert list(out["date"].unique()) == ["d2", "d3"]
    # d3 gap in A is forward-filled from A's own d2, not from B
    assert out.set_index(["date", "tic"]).loc[("d3", "A"), "x"] == 1.0


def test_test_window_turbulence_has_history(full):
    """Slicing after computing = no warm-up zeros inside the test window.
    (The old 04_backtest computed on the window alone: 29% zeros.)"""
    test_start = sorted(full["date"].unique())[TURBULENCE_LOOKBACK + 20]
    window = train_test_split(full, test_start, full["date"].max())
    assert (window.drop_duplicates("date")["turbulence"] == 0).sum() == 0


def test_window_alone_would_have_zeros(raw):
    """The bug the slice fixes, shown directly."""
    late = raw[raw["date"] >= sorted(raw["date"].unique())[100]]
    turb = compute_features(late).drop_duplicates("date")["turbulence"]
    assert (turb == 0).sum() >= TURBULENCE_LOOKBACK


def test_synthetic_path_warmed_up_on_history(raw):
    from minifinrl.core.meta.synthetic import to_tidy_panel

    hist_last = raw["date"].max()
    closes = raw.pivot(index="date", columns="tic", values="close")
    path = pd.DataFrame({t: closes[t].iloc[-1] * np.cumprod(1 + np.random.default_rng(1).normal(0, 0.01, 60))
                         for t in closes.columns})
    start = (pd.Timestamp(hist_last) + pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")
    out = features_with_history(to_tidy_panel(path, start_date=start), raw, warmup_days=TURBULENCE_LOOKBACK + 100)
    assert out["date"].min() == start and out["date"].nunique() == 60
    assert (out.drop_duplicates("date")["turbulence"] == 0).sum() == 0
    assert not out[FEATURES].isna().any().any()


def test_synthetic_path_overlapping_history_is_rejected(raw):
    from minifinrl.core.meta.synthetic import to_tidy_panel

    path = pd.DataFrame({t: [100.0] * 10 for t in ("AAA", "BBB", "CCC")})
    with pytest.raises(ValueError, match="not before the path start"):
        features_with_history(to_tidy_panel(path, start_date="2020-06-01"), raw, warmup_days=300)
