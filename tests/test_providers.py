"""Provider parsing and retry behaviour, with the HTTP call injected."""

from __future__ import annotations

import json

import pytest

from minifinrl.core.meta.providers import FredSeriesProvider, ProviderError, with_retries

CSV = "observation_date,VIXCLS\n2026-06-24,18.63\n2026-06-25,.\n2026-06-26,18.41\n2026-06-29,17.65\n2026-06-30,16.45\n"


def test_fred_csv_parse_drops_missing_and_respects_exclusive_end(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    urls = []
    p = FredSeriesProvider(http_get=lambda u: urls.append(u) or CSV)
    s = p.fetch("2026-06-24", "2026-06-30")
    assert "fredgraph.csv" in urls[0] and "coed=2026-06-29" in urls[0]
    assert list(s.index) == ["2026-06-24", "2026-06-26", "2026-06-29"]  # '.' dropped, 06-30 excluded
    assert s.name == "vix" and s.dtype == float


def test_fred_json_api_used_when_key_set(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "k")
    payload = {"observations": [{"date": "2026-06-24", "value": "18.63"}, {"date": "2026-06-25", "value": "."}]}
    urls = []
    s = FredSeriesProvider(http_get=lambda u: urls.append(u) or json.dumps(payload)).fetch("2026-06-24", "2026-06-26")
    assert "api.stlouisfed.org" in urls[0]
    assert s.to_dict() == {"2026-06-24": 18.63}


def test_fred_api_error_payload_is_provider_error(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "k")
    body = json.dumps({"error_code": 400, "error_message": "Bad Request"})
    with pytest.raises(ProviderError, match="Bad Request"):
        FredSeriesProvider(http_get=lambda u: body).fetch("2026-06-24", "2026-06-26")


def test_retries_then_succeeds():
    attempts, sleeps = [], []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ProviderError("down")
        return "ok"

    assert with_retries(flaky, what="t", retries=3, sleep=sleeps.append) == "ok"
    assert sleeps == [1.0, 2.0]


def test_retries_give_up():
    def down():
        raise ProviderError("down")

    with pytest.raises(ProviderError):
        with_retries(down, what="t", retries=2, sleep=lambda _s: None)


def test_bugs_are_not_retried():
    calls = []

    def bug():
        calls.append(1)
        raise KeyError("x")

    with pytest.raises(KeyError):
        with_retries(bug, what="t", sleep=lambda _s: None)
    assert len(calls) == 1
