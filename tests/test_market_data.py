"""End to end through load_market_data with fake providers registered."""

from __future__ import annotations

import json

import pytest

import minifinrl.core.meta.market_data as md
from minifinrl.core.meta import providers
from minifinrl.core.meta.validate import DataValidationError
from tests.conftest import FakePriceProvider, FakeSeriesProvider


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    price = FakePriceProvider()
    vix = FakeSeriesProvider()
    monkeypatch.setitem(providers.PRICE_PROVIDERS, "fake", lambda: price)
    monkeypatch.setitem(providers.VIX_PROVIDERS, "fakevix", lambda: vix)
    monkeypatch.setattr(md, "MANIFEST_DIR", tmp_path / "manifests")
    return price, vix, tmp_path


def _load(tmp_path, **kw):
    args = dict(start="2024-01-01", end="2024-07-01", price_provider="fake", vix_provider="fakevix",
                store_root=tmp_path / "store", today="2025-01-01", mode="research")
    args.update(kw)
    return md.load_market_data(["AAA", "BBB"], **args)


def test_load_validates_and_writes_manifest(fakes):
    _, _, tmp = fakes
    out = _load(tmp)
    assert out.report.ok
    m = json.loads((tmp / "manifests" / "latest.json").read_text())
    assert m["data_hash"] == out.manifest["data_hash"] and len(m["data_hash"]) == 64
    assert m["providers"] == {"prices": "fake", "vix": "fakevix"}
    assert set(m["panel"]["per_ticker"]) == {"AAA", "BBB"}
    assert m["requested"]["end_exclusive"] == "2024-07-01"


def test_hash_is_stable_across_cache_hits(fakes):
    _, _, tmp = fakes
    assert _load(tmp).manifest["data_hash"] == _load(tmp, refresh=False).manifest["data_hash"]


def test_hash_changes_when_data_changes(fakes):
    price, _, tmp = fakes
    h1 = _load(tmp).manifest["data_hash"]
    price.adj_factor = 0.5
    h2 = _load(tmp, store_root=tmp / "store2").manifest["data_hash"]
    assert h1 != h2


def test_strict_raises_but_manifest_still_written(fakes):
    _, _, tmp = fakes
    with pytest.raises(DataValidationError):
        # live mode, a year after the data ends -> stale
        _load(tmp, mode="live", today="2025-07-01", end="2024-07-01")
    m = json.loads((tmp / "manifests" / "latest.json").read_text())
    assert not m["validation"]["ok"] and "business days old" in m["validation"]["errors"][0]


def test_resolve_end():
    assert md.resolve_end("2024-01-01") == "2024-01-01"
    assert md.resolve_end(None, "research") == md.TEST_END
    with pytest.raises(ValueError):
        md.resolve_end(None, "bogus")
