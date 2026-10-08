"""Incremental store: fetch only what's missing, never mix adjustment
anchors, never cache an empty answer, never store today's open bar."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from minifinrl.market.providers import ProviderError
from minifinrl.market.store import PriceStore, SeriesStore, adjust
from tests.conftest import FakePriceProvider, FakeSeriesProvider

TODAY = "2025-01-01"


def _store(provider, tmp_path, today=TODAY):
    return PriceStore(provider, tmp_path, today=today)


def test_first_fetch_then_hit(tmp_path, fake_provider):
    s = _store(fake_provider, tmp_path)
    a = s.get(["AAA"], "2024-01-01", "2024-03-01")
    b = s.get(["AAA"], "2024-01-01", "2024-03-01")
    assert len(fake_provider.calls) == 1
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == ["date", "tic", "open", "high", "low", "close", "volume"]


def test_sub_range_is_a_hit(tmp_path, fake_provider):
    s = _store(fake_provider, tmp_path)
    s.get(["AAA"], "2024-01-01", "2024-06-01")
    part = s.get(["AAA"], "2024-02-01", "2024-03-01")
    assert len(fake_provider.calls) == 1
    assert part["date"].min() >= "2024-02-01" and part["date"].max() < "2024-03-01"


def test_tail_extension_fetches_only_tail_plus_overlap(tmp_path, fake_provider):
    s = _store(fake_provider, tmp_path)
    s.get(["AAA"], "2024-01-01", "2024-03-01")
    s.get(["AAA"], "2024-01-01", "2024-04-01")
    assert len(fake_provider.calls) == 2
    _, tail_start, tail_end = fake_provider.calls[1]
    assert "2024-02-20" <= tail_start < "2024-03-01"  # a few stored days re-requested
    assert tail_end == "2024-04-01"
    cov = s.coverage("AAA")
    assert (cov["covered_start"], cov["covered_end"]) == ("2024-01-01", "2024-04-01")


def test_incremental_equals_single_fetch(tmp_path):
    inc = _store(FakePriceProvider(), tmp_path / "inc")
    inc.get(["AAA"], "2024-03-01", "2024-04-01")
    inc.get(["AAA"], "2024-01-01", "2024-04-01")  # head
    inc.get(["AAA"], "2024-01-01", "2024-06-01")  # tail
    one = _store(FakePriceProvider(), tmp_path / "one").get(["AAA"], "2024-01-01", "2024-06-01")
    pd.testing.assert_frame_equal(inc.get(["AAA"], "2024-01-01", "2024-06-01", refresh=False), one)


def test_dividend_since_last_fetch_triggers_full_refetch(tmp_path):
    p = FakePriceProvider(adj_factor=0.90)
    s = _store(p, tmp_path)
    s.get(["AAA"], "2024-01-01", "2024-03-01")
    p.adj_factor = 0.89  # provider re-anchored its adjusted history
    s.get(["AAA"], "2024-01-01", "2024-04-01")
    assert p.calls[-1] == ("AAA", "2024-01-01", "2024-04-01")  # whole range again
    raw = pd.read_csv(tmp_path / "fake" / "AAA.csv")
    assert np.allclose(raw["adj_close"] / raw["close"], 0.89)  # one anchor for every row


def test_empty_first_fetch_raises_and_is_not_cached(tmp_path):
    p = FakePriceProvider(empty=True)
    s = _store(p, tmp_path)
    with pytest.raises(ProviderError):
        s.get(["AAA"], "2024-01-01", "2024-03-01")
    assert s.coverage("AAA") is None
    p.empty = False
    assert not s.get(["AAA"], "2024-01-01", "2024-03-01").empty


def test_empty_tail_with_overlap_raises(tmp_path):
    p = FakePriceProvider()
    s = _store(p, tmp_path)
    s.get(["AAA"], "2024-01-01", "2024-03-01")
    p.empty = True
    with pytest.raises(ProviderError):
        s.get(["AAA"], "2024-01-01", "2024-04-01")
    assert s.coverage("AAA")["covered_end"] == "2024-03-01"  # unchanged


def test_end_clamped_to_today(tmp_path, fake_provider):
    s = _store(fake_provider, tmp_path, today="2024-02-15")
    out = s.get(["AAA"], "2024-01-01", "2024-12-31")
    assert out["date"].max() < "2024-02-15"
    assert s.coverage("AAA")["covered_end"] == "2024-02-15"


def test_read_only_never_fetches(tmp_path, fake_provider):
    s = _store(fake_provider, tmp_path)
    assert s.get(["AAA"], "2024-01-01", "2024-03-01", refresh=False).empty
    assert fake_provider.calls == []


def test_adjust_matches_meta_data_rule():
    raw = pd.DataFrame(
        {"date": ["2024-01-02"], "open": [10.0], "high": [12.0], "low": [9.0],
         "close": [11.0], "adj_close": [5.5], "volume": [100]}
    )
    out = adjust(raw)
    assert out.loc[0, "close"] == 5.5 and out.loc[0, "open"] == 5.0 and out.loc[0, "high"] == 6.0
    assert "adj_close" not in out.columns and out.loc[0, "volume"] == 100


def test_series_store_incremental(tmp_path):
    p = FakeSeriesProvider()
    s = SeriesStore(p, "VIX", tmp_path, today=TODAY)
    a = s.get("2024-01-01", "2024-03-01")
    s.get("2024-01-01", "2024-03-01")
    assert len(p.calls) == 1
    b = s.get("2024-01-01", "2024-04-01")
    assert p.calls[-1] == ("2024-03-01", "2024-04-01")
    assert len(b) > len(a) and b.index.is_monotonic_increasing and not b.index.duplicated().any()
    meta = json.loads((tmp_path / "fakevix" / "VIX.json").read_text())
    assert meta["covered_end"] == "2024-04-01"
