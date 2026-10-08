"""Shared fixtures. Everything here is offline: providers are fakes that
generate deterministic bars and count how often they were called."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minifinrl.market.providers import PROVIDER_COLUMNS  # noqa: E402


class FakePriceProvider:
    """Business-day bars with a known close path. `adj_factor` plays the
    provider's back-adjustment: adj_close = close * adj_factor, so changing
    it between calls simulates a dividend paid since the last fetch."""

    name = "fake"

    def __init__(self, adj_factor: float = 0.9, empty: bool = False):
        self.adj_factor = adj_factor
        self.empty = empty
        self.calls: list[tuple[str, str, str]] = []

    def fetch(self, tic: str, start: str, end: str) -> pd.DataFrame:
        self.calls.append((tic, start, end))
        dates = pd.bdate_range(start, pd.Timestamp(end) - pd.Timedelta(days=1))
        if self.empty or len(dates) == 0:
            return pd.DataFrame(columns=PROVIDER_COLUMNS)
        # price depends only on the date, so overlapping fetches agree
        base = 100 + (sum(map(ord, tic)) % 50)
        ordinal = np.array([(d - pd.Timestamp("2000-01-03")).days for d in dates], dtype=float)
        close = base * (1 + 0.0002 * ordinal)
        return pd.DataFrame(
            {
                "date": dates.strftime("%Y-%m-%d"),
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "adj_close": close * self.adj_factor,
                "volume": 1_000_000,
            }
        )


class FakeSeriesProvider:
    name = "fakevix"

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def fetch(self, start: str, end: str) -> pd.Series:
        self.calls.append((start, end))
        dates = pd.bdate_range(start, pd.Timestamp(end) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        s = pd.Series(np.linspace(15, 25, len(dates)), index=dates, name="vix")
        s.index.name = "date"
        return s


def make_panel(tickers=("AAA", "BBB", "CCC"), start="2024-01-01", end="2024-07-01") -> pd.DataFrame:
    """A clean tidy panel that passes every validation rule."""
    from minifinrl.market.store import adjust

    p = FakePriceProvider()
    frames = []
    for t in tickers:
        f = adjust(p.fetch(t, start, end))
        f["tic"] = t
        frames.append(f)
    return pd.concat(frames, ignore_index=True).sort_values(["date", "tic"]).reset_index(drop=True)


@pytest.fixture
def fake_provider():
    return FakePriceProvider()


@pytest.fixture
def panel():
    return make_panel()


def random_walk_panel(n_days=420, tickers=("AAA", "BBB", "CCC"), start="2020-01-01", seed=0) -> pd.DataFrame:
    """Correlated random walks with real-looking OHLC, so the turbulence
    covariance is not degenerate."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days).strftime("%Y-%m-%d")
    market = rng.normal(0.0003, 0.01, n_days)
    rows = []
    for k, t in enumerate(tickers):
        r = 0.8 * market + rng.normal(0, 0.012, n_days)
        close = (100 + 20 * k) * np.cumprod(1 + r)
        wiggle = np.abs(rng.normal(0, 0.005, n_days))
        rows.append(pd.DataFrame({
            "date": dates, "tic": t, "open": close * (1 - wiggle / 2), "high": close * (1 + wiggle),
            "low": close * (1 - wiggle), "close": close, "volume": 1_000_000,
        }))
    return pd.concat(rows, ignore_index=True).sort_values(["date", "tic"]).reset_index(drop=True)


class RandomWalkProvider:
    """Realistic-enough bars for end-to-end runs: a correlated random walk
    on a fixed business-day calendar, so any [start, end) slice is the same
    numbers no matter how the store splits its fetches."""

    name = "rw"
    CAL = pd.bdate_range("2013-01-01", "2027-12-31")

    def __init__(self):
        rng = np.random.default_rng(7)
        market = rng.normal(0.0003, 0.01, len(self.CAL))
        self._close = {}
        for k, t in enumerate(("AAA", "BBB", "CCC")):
            r = 0.8 * market + rng.normal(0, 0.012, len(self.CAL))
            self._close[t] = (100 + 20 * k) * np.cumprod(1 + r)
        self._vix = 15 * np.exp(np.cumsum(rng.normal(0, 0.03, len(self.CAL))) * 0.3)
        # outside the research universe: a trade target and the market proxy
        self._close["OUT"] = 50 * np.cumprod(1 + 1.4 * market + rng.normal(0, 0.02, len(self.CAL)))
        self._close["SPY"] = 300 * np.cumprod(1 + market)
        meme = 1 + 1.5 * market + rng.normal(0, 0.03, len(self.CAL))
        meme[self.CAL.get_loc(pd.Timestamp("2021-01-27"))] = 4.0  # a real-looking +300% day
        self._close["MEME"] = 5 * np.cumprod(meme)
        self._close["META"] = 80 * np.cumprod(1 + 1.2 * market + rng.normal(0, 0.018, len(self.CAL)))
        self._close["BRK-B"] = 200 * np.cumprod(1 + 0.9 * market + rng.normal(0, 0.008, len(self.CAL)))
        # recent listings: NEW starts trading 2020-01-02, LATE on 2021-06-01
        self._listed = {"NEW": "2020-01-02", "LATE": "2021-06-01"}
        self._close["NEW"] = 20 * np.cumprod(1 + 1.3 * market + rng.normal(0, 0.025, len(self.CAL)))
        self._close["LATE"] = 30 * np.cumprod(1 + market + rng.normal(0, 0.02, len(self.CAL)))

    def fetch(self, tic, start, end):
        if tic not in self._close:  # like Yahoo: an unknown symbol comes back empty
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])
        m = (self.CAL >= max(start, self._listed.get(tic, start))) & (self.CAL < end)
        c = self._close[tic][m]
        return pd.DataFrame({
            "date": self.CAL[m].strftime("%Y-%m-%d"), "open": c * 0.998, "high": c * 1.01,
            "low": c * 0.99, "close": c, "adj_close": c, "volume": 1_000_000,
        })


class RandomWalkVix:
    name = "rwvix"

    def __init__(self):
        self._rw = RandomWalkProvider()

    def fetch(self, start, end):
        m = (self._rw.CAL >= start) & (self._rw.CAL < end)
        s = pd.Series(self._rw._vix[m], index=self._rw.CAL[m].strftime("%Y-%m-%d"), name="vix")
        s.index.name = "date"
        return s


class ConstantRates:
    """4% a year on every business day: a known risk-free rate for tests."""

    name = "constrates"

    def fetch(self, start, end):
        cal = RandomWalkProvider.CAL
        m = (cal >= start) & (cal < end)
        s = pd.Series(4.0, index=cal[m].strftime("%Y-%m-%d"), name="rate")
        s.index.name = "date"
        return s


SYMBOLS_NASDAQ = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc. - Common Stock|Q|N|N|40|N|N
META|Meta Platforms, Inc. - Class A Common Stock|Q|N|N|40|N|N
NEW|Newco Holdings Inc. - Common Stock|Q|N|N|100|N|N
LATE|Lateco Inc. - Common Stock|Q|N|N|100|N|N
TLT|iShares 20+ Year Treasury Bond ETF|G|N|N|100|Y|N
ZZZT|Test Issue Corp|Q|Y|N|100|N|N
File Creation Time: 1007202600:00|||||||
"""
SYMBOLS_OTHER = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
AAA|Alpha Industries Inc. Common Stock|N|AAA|N|100|N|AAA
BBB|Beta Corp Common Stock|N|BBB|N|100|N|BBB
CCC|Gamma Co Common Stock|N|CCC|N|100|N|CCC
OUT|Outside Systems Inc. Common Stock|N|OUT|N|100|N|OUT
MEME|Meme Entertainment Holdings Inc. Class A Common Stock|N|MEME|N|100|N|MEME
APLE|Apple Hospitality REIT, Inc. Common Shares|N|APLE|N|100|N|APLE
BRK.B|Berkshire Hathaway Inc. New Common Stock|N|BRK.B|N|40|N|BRK.B
SPY|State Street SPDR S&P 500 ETF Trust|P|SPY|Y|40|N|SPY
AGG|iShares Core U.S. Aggregate Bond ETF|P|AGG|Y|100|N|AGG
AGG$A|Preferred Thing|N|AGGpA|N|100|N|AGG-A
File Creation Time: 1007202600:00|||||||
"""


@pytest.fixture
def rw_engine(monkeypatch, tmp_path):
    """An Engine wired entirely to tmp_path and random-walk providers."""
    from minifinrl.research import pipeline
    from minifinrl.market import providers
    from minifinrl.market.panel import DataSpec
    from minifinrl.engine import Engine, EngineConfig
    from minifinrl.sentiment.rules import RulesBiasClassifier

    monkeypatch.setitem(providers.PRICE_PROVIDERS, "rw", RandomWalkProvider)
    monkeypatch.setitem(providers.VIX_PROVIDERS, "rwvix", RandomWalkVix)
    monkeypatch.setitem(providers.RATE_PROVIDERS, "constrates", ConstantRates)
    spec = DataSpec(tickers=("AAA", "BBB", "CCC"), price_provider="rw", vix_provider="rwvix", rate_provider="constrates",
                    store_root=str(tmp_path / "store"), manifest_dir=str(tmp_path / "manifests"), today="2027-01-01")
    paths = pipeline.Paths(model_dir=tmp_path / "models", synthetic_dir=tmp_path / "synthetic",
                           processed_dir=tmp_path / "processed", results_path=tmp_path / "results" / "backtest_results.csv")
    sym = tmp_path / "symbols"  # fresh files, so the directory never downloads in tests
    sym.mkdir()
    (sym / "nasdaqlisted.txt").write_text(SYMBOLS_NASDAQ)
    (sym / "otherlisted.txt").write_text(SYMBOLS_OTHER)
    cfg = EngineConfig(profile="test", spec=spec, paths=paths, manifest_dir=tmp_path / "manifests", today="2027-01-01",
                       experiment_dir=tmp_path / "experiments", experiments_md=tmp_path / "EXPERIMENTS.md", symbols_dir=sym,
                       database_url=f"sqlite:///{tmp_path / 'journal.db'}")
    return Engine(cfg, bias=RulesBiasClassifier())
