"""Tiingo / Alpaca / FRED-rate connectors against recorded responses (no
keys, no network), the cross-source check, and the experiment log."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from minifinrl.capabilities import CapabilityRegistry
from minifinrl.core.meta import providers
from minifinrl.core.meta.crosscheck import crosscheck
from minifinrl.core.meta.providers import AlpacaProvider, ProviderNotConfigured, TiingoProvider
from minifinrl.core.meta.store import PriceStore

TIINGO = [  # shape of api.tiingo.com/tiingo/daily/<t>/prices
    {"date": "2024-06-07T00:00:00.000Z", "open": 194.65, "high": 196.94, "low": 194.14, "close": 196.89,
     "volume": 53103912, "adjClose": 195.9, "adjHigh": 0, "adjLow": 0, "adjOpen": 0, "adjVolume": 0, "divCash": 0.0, "splitFactor": 1.0},
    {"date": "2024-06-10T00:00:00.000Z", "open": 196.9, "high": 197.3, "low": 192.15, "close": 193.12,
     "volume": 97262077, "adjClose": 192.15, "adjHigh": 0, "adjLow": 0, "adjOpen": 0, "adjVolume": 0, "divCash": 0.0, "splitFactor": 1.0},
]


def _bar(t, o, h, l, c, v):
    return {"t": t, "o": o, "h": h, "l": l, "c": c, "v": v, "n": 1, "vw": c}


def test_tiingo_parses_raw_and_adjusted(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "k")
    calls = []
    p = TiingoProvider(http_get=lambda url, headers: calls.append((url, headers)) or json.dumps(TIINGO))
    df = p.fetch("AAPL", "2024-06-07", "2024-06-11")
    url, headers = calls[0]
    assert "/daily/aapl/prices" in url and "endDate=2024-06-10" in url  # exclusive end -> inclusive API
    assert headers == {"Authorization": "Token k"}
    assert list(df["date"]) == ["2024-06-07", "2024-06-10"]
    assert df.loc[0, "close"] == 196.89 and df.loc[0, "adj_close"] == 195.9


def test_tiingo_missing_key_is_not_retried(monkeypatch):
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    calls = []
    with pytest.raises(ProviderNotConfigured, match="TIINGO_API_KEY"):
        TiingoProvider(http_get=lambda *a, **k: calls.append(1)).fetch("AAPL", "2024-06-07", "2024-06-11")
    assert calls == []


def test_alpaca_paginates_merges_adjustment_and_uses_new_york_dates(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY_ID", "id")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "secret")
    # daily bars are stamped at midnight New York = 04:00Z in summer
    raw_p1 = {"bars": {"AAPL": [_bar("2024-06-07T04:00:00Z", 194.65, 196.94, 194.14, 196.89, 1000)]}, "next_page_token": "tok"}
    raw_p2 = {"bars": {"AAPL": [_bar("2024-06-10T04:00:00Z", 196.9, 197.3, 192.15, 193.12, 2000)]}, "next_page_token": None}
    adj = {"bars": {"AAPL": [_bar("2024-06-07T04:00:00Z", 0, 0, 0, 195.9, 0), _bar("2024-06-10T04:00:00Z", 0, 0, 0, 192.15, 0)]},
           "next_page_token": None}
    seen = []

    def fake(url, headers):
        seen.append(url)
        assert headers == {"APCA-API-KEY-ID": "id", "APCA-API-SECRET-KEY": "secret"}
        if "adjustment=all" in url:
            return json.dumps(adj)
        return json.dumps(raw_p2 if "page_token=tok" in url else raw_p1)

    df = AlpacaProvider(http_get=fake).fetch("AAPL", "2024-06-07", "2024-06-11")
    assert sum("adjustment=raw" in u for u in seen) == 2  # followed the page token
    assert list(df["date"]) == ["2024-06-07", "2024-06-10"]
    assert list(df["close"]) == [196.89, 193.12] and list(df["adj_close"]) == [195.9, 192.15]


def test_alpaca_rejected_key_is_not_configured(monkeypatch):
    import urllib.error

    monkeypatch.setenv("ALPACA_API_KEY_ID", "id")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "bad")

    def urlopen(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(providers.urllib.request, "urlopen", urlopen)
    with pytest.raises(ProviderNotConfigured, match="403"):
        AlpacaProvider().fetch("AAPL", "2024-06-07", "2024-06-11")


def test_new_connectors_plug_into_the_store(monkeypatch, tmp_path):
    """Same contract as Yahoo: the store adjusts at read time."""
    monkeypatch.setenv("TIINGO_API_KEY", "k")
    store = PriceStore(TiingoProvider(http_get=lambda url, headers: json.dumps(TIINGO)), tmp_path, today="2025-01-01")
    out = store.get(["AAPL"], "2024-06-07", "2024-06-11")
    assert out.loc[0, "close"] == pytest.approx(195.9)  # adjusted close
    assert out.loc[0, "open"] == pytest.approx(194.65 * 195.9 / 196.89)


def test_crosscheck_flags_disagreement():
    dates = pd.bdate_range("2024-01-01", periods=200).strftime("%Y-%m-%d")
    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, 200))
    ref = pd.DataFrame({"date": dates, "tic": "AAA", "close": close})
    same_but_rescaled = ref.assign(close=close * 0.97)  # different adjustment anchor: still agrees
    noisy = ref.assign(close=close * (1 + rng.normal(0, 0.003, 200)))  # ~30 bps noise: disagrees
    gappy = ref.iloc[::2]  # half the days missing
    assert crosscheck(ref, same_but_rescaled, ["AAA"])[0]["agrees"]
    assert not crosscheck(ref, noisy, ["AAA"])[0]["agrees"]
    r = crosscheck(ref, gappy, ["AAA"])[0]
    assert not r["agrees"] and r["missing_in_cand"] == 100


def test_crosscheck_capability_same_source_agrees(rw_engine):
    out = CapabilityRegistry.from_engine(rw_engine).invoke(
        "crosscheck_prices", {"candidate": "rw", "reference": "rw", "start": "2020-01-01", "end": "2020-06-01"},
        interface="cli")
    assert out.agrees and all(t.p99_abs_diff == 0 for t in out.tickers)


def test_experiment_log_records_and_freezes(rw_engine):
    reg = CapabilityRegistry.from_engine(rw_engine)
    reg.invoke("fetch_data", {}, interface="cli")
    reg.invoke("fit_hmm", {"n_paths": 1, "length": 30}, interface="cli")
    reg.invoke("run_backtests", {"paths": "both"}, interface="cli")
    exp = {"title": "test run", "change": "c", "math": "x = 1", "hypothesis": "h", "conclusion": "k"}
    a = reg.invoke("record_experiment", {"id": "E01", **exp}, interface="cli")
    b = reg.invoke("record_experiment", {"id": "E02", "compare_to": "E01", **exp}, interface="cli")
    assert a.results_sha256 and b.results_sha256 == a.results_sha256
    md = rw_engine.cfg.experiments_md.read_text()
    assert "## E01" in md and "## E02" in md and "compared with E01" in md and "| equal_weight |" in md
    assert (rw_engine.cfg.experiment_dir / "E01" / "backtest_results.csv").exists()
    with pytest.raises(ValueError, match="already recorded"):
        reg.invoke("record_experiment", {"id": "E01", **exp}, interface="cli")
