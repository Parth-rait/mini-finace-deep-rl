"""Luck test (L1): calibrated, no look-ahead, deterministic, and the
starting regime actually matters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from minifinrl.capabilities import CapabilityRegistry, CapabilityUnavailable
from minifinrl.core.luck.luck_test import LuckTestError, luck_test, percentile_of, verdict_for
from minifinrl.core.meta.synthetic import RegimeSyntheticGenerator


def regime_market(n_days=1500, block=150, seed=3):
    """Closes for 3 tickers driven by a market that alternates calm and crisis
    blocks (odd blocks are crisis), with the per-regime volatilities the HMM
    finds on the real 2014-2026 panel: market 0.7% vs 2.5% daily, VIX log
    changes 4% vs 16%. Day 599 ends a crisis block, day 449 a calm one."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_days).strftime("%Y-%m-%d")
    crisis = (np.arange(n_days) // block) % 2 == 1
    market = np.where(crisis, rng.normal(-0.001, 0.025, n_days), rng.normal(0.0005, 0.007, n_days))
    close = {}
    for k, t in enumerate(("AAA", "BBB", "CCC")):
        r = (0.8 + 0.2 * k) * market + rng.normal(0, 0.006, n_days)
        close[t] = 100 * np.cumprod(1 + r)
    vix_change = np.where(crisis, rng.normal(0, 0.16, n_days), rng.normal(0, 0.04, n_days))
    vix = pd.Series(15 * np.exp(np.cumsum(vix_change) * 0.2), index=dates)
    return pd.DataFrame(close, index=dates), vix, crisis


@pytest.fixture(scope="module")
def market():
    return regime_market()


def test_percentile_and_verdict():
    s = np.array([1.0, 2.0, 3.0, 4.0])
    assert percentile_of(2.5, s) == 50.0 and percentile_of(2.0, s) == 37.5
    assert percentile_of(0.0, s) == 0.0 and percentile_of(9.0, s) == 100.0
    assert [verdict_for(p) for p in (4.9, 5, 50, 95, 95.1)] == [
        "unusually_bad", "within_luck_range", "within_luck_range", "within_luck_range", "unusually_good"]


def test_calibration_uniform_percentiles(market):
    """If the world really were the fitted model, realized outcomes would land
    uniformly on the percentile scale. 200 'true' trades drawn from the model
    vs 500 reference paths each: KS test against uniform."""
    close, vix, _ = market
    gen = RegimeSyntheticGenerator().fit(close, vix)
    start = gen.regime_probs_last() @ gen.hmm.transmat_
    pcts = []
    for k in range(200):
        truth = np.prod(1 + gen.sample_ticker_returns("BBB", 10, 1, start, seed=10_000 + k)) - 1
        ref = np.prod(1 + gen.sample_ticker_returns("BBB", 10, 500, start, seed=k), axis=1) - 1
        pcts.append(percentile_of(truth, ref) / 100)
    assert stats.kstest(pcts, "uniform").pvalue > 0.01
    assert 0.03 <= np.mean(np.array(pcts) < 0.05) <= 0.12  # ~5% flagged unusually_bad, as advertised


def test_no_lookahead(market):
    """Prices after entry change only the realized outcome; prices after exit change nothing."""
    close, vix, _ = market
    entry, h = close.index[1200], 10
    base = luck_test(close, vix, "AAA", entry, "long", h, n_paths=400)

    after_entry = close.copy()
    after_entry.iloc[1201:] *= 1.5  # a jump right after entry
    r1 = luck_test(after_entry, vix, "AAA", entry, "long", h, n_paths=400)
    assert (r1.band_low, r1.band_high, r1.synthetic_median, r1.regime) == \
           (base.band_low, base.band_high, base.synthetic_median, base.regime)
    assert r1.realized_return != base.realized_return
    assert r1.fit_end == entry

    after_exit = close.copy()
    after_exit.iloc[1200 + h + 1:] *= 0.1
    assert luck_test(after_exit, vix, "AAA", entry, "long", h, n_paths=400) == base


def test_deterministic(market):
    close, vix, _ = market
    a = luck_test(close, vix, "CCC", close.index[900], "short", 5, n_paths=300, seed=4)
    b = luck_test(close, vix, "CCC", close.index[900], "short", 5, n_paths=300, seed=4)
    c = luck_test(close, vix, "CCC", close.index[900], "short", 5, n_paths=300, seed=5)
    assert a == b and a.synthetic_median != c.synthetic_median


def test_starting_regime_matters(market):
    """Entered at the end of a crisis block, the plausible band must be much
    wider than entered at the end of a calm block."""
    close, vix, crisis = market
    end_crisis = close.index[599]  # last day of block 3 (odd blocks are crisis)
    end_calm = close.index[449]  # last day of block 2 (calm)
    assert crisis[599] and not crisis[449]
    rc = luck_test(close, vix, "AAA", end_crisis, "long", 5, n_paths=1000, min_history=300)
    rq = luck_test(close, vix, "AAA", end_calm, "long", 5, n_paths=1000, min_history=300)
    assert rc.regime == "crisis" and rc.regime_probs["crisis"] > 0.9
    assert rq.regime == "calm"
    # measured 1.9x here; unconditioned sampling would give the same band for both
    assert (rc.band_high - rc.band_low) > 1.5 * (rq.band_high - rq.band_low)


def test_short_mirrors_long(market):
    close, vix, _ = market
    lo = luck_test(close, vix, "AAA", close.index[1000], "long", 10, n_paths=500)
    sh = luck_test(close, vix, "AAA", close.index[1000], "short", 10, n_paths=500)
    assert sh.realized_return == pytest.approx(-lo.realized_return)
    assert sh.band_high == pytest.approx(-lo.band_low) and sh.percentile == pytest.approx(100 - lo.percentile)


def test_errors_and_date_snapping(market):
    close, vix, _ = market
    with pytest.raises(LuckTestError, match="unknown ticker"):
        luck_test(close, vix, "ZZZ", close.index[900], "long", 5)
    with pytest.raises(LuckTestError, match="days of history"):
        luck_test(close, vix, "AAA", close.index[100], "long", 5)
    with pytest.raises(LuckTestError, match="not observable yet"):
        luck_test(close, vix, "AAA", close.index[-3], "long", 5)
    saturday = (pd.Timestamp(close.index[900]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if pd.Timestamp(saturday).weekday() >= 5:
        assert luck_test(close, vix, "AAA", saturday, "long", 5, n_paths=200).entry_date == close.index[900]


def test_capability_caches_fit_per_entry_day(rw_engine, monkeypatch):
    reg = CapabilityRegistry.from_engine(rw_engine)
    reg.invoke("fetch_data", {}, interface="cli")
    fits = []
    real_fit = RegimeSyntheticGenerator.fit
    monkeypatch.setattr(RegimeSyntheticGenerator, "fit", lambda self, p, v: fits.append(1) or real_fit(self, p, v))
    a = reg.invoke("luck_test", {"ticker": "AAA", "date": "2020-03-02", "horizon_days": 10})
    b = reg.invoke("luck_test", {"ticker": "BBB", "date": "2020-03-02", "direction": "short", "horizon_days": 5})
    assert len(fits) == 1  # same entry day -> one fit, any ticker/direction
    assert a.fit_end == a.entry_date == b.entry_date and a.verdict in ("unusually_bad", "within_luck_range", "unusually_good")
    with pytest.raises(CapabilityUnavailable, match="not observable"):
        reg.invoke("luck_test", {"ticker": "AAA", "date": "2026-06-26", "horizon_days": 20})
