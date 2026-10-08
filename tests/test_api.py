"""The generated HTTP API: routes from the registry, status codes from
error_kind, batch jobs unreachable, place_order always refused."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from minifinrl.platform.capabilities import CapabilityRegistry
from minifinrl.interfaces.api import create_app
from minifinrl.system import System, SystemConfig


@pytest.fixture
def client(rw_engine):
    system = System(SystemConfig.from_profile("research"), rw_engine, CapabilityRegistry.from_engine(rw_engine))
    return TestClient(create_app(system)), system


def test_index_lists_api_capabilities_only(client):
    c, _ = client
    r = c.get("/api")
    assert r.status_code == 200
    paths = {x["path"] for x in r.json()["routes"]}
    assert {"/health", "/luck-test", "/rank-stability", "/classify-biases", "/place-order"} <= paths
    assert not paths & {"/train", "/fetch-data", "/run-backtests", "/walk-forward", "/record-experiment"}
    assert r.headers["X-Request-Id"] and "X-Duration-Ms" in r.headers


def test_batch_jobs_have_no_route(client):
    c, _ = client
    assert c.post("/train", json={"app": "trading", "model": "ppo", "seed": 0}).status_code == 404


def test_health_get_and_post(client):
    c, _ = client
    for r in (c.get("/health"), c.post("/health")):
        assert r.status_code == 200 and r.json()["profile"] == "test"


def test_place_order_is_always_403(client):
    c, _ = client
    r = c.post("/place-order", json={"ticker": "AAA", "side": "buy", "quantity": 1})
    assert r.status_code == 403 and r.json()["detail"]["error"] == "forbidden"


def test_bad_input_is_422(client):
    c, _ = client
    assert c.post("/rank-stability", json={"app": "crypto"}).status_code == 422
    assert c.post("/rank-stability", json={"app": "trading", "surprise": 1}).status_code == 422  # extra fields forbidden
    assert c.post("/luck-test", json={"ticker": "AAA", "date": "1 Feb 2023"}).status_code == 422


def test_missing_results_is_503_with_retry_after(client):
    c, _ = client
    r = c.post("/rank-stability", json={"app": "trading"})
    assert r.status_code == 503 and r.headers["Retry-After"] == "30"
    assert r.json()["detail"]["error"] == "unavailable" and r.json()["detail"]["request_id"]


def test_end_to_end_after_batch_jobs(client):
    c, system = client
    reg = system.capabilities
    reg.invoke("fetch_data", {}, interface="cli")
    reg.invoke("fit_hmm", {"n_paths": 2, "length": 40}, interface="cli")
    reg.invoke("run_backtests", {"paths": "both"}, interface="cli")

    rs = c.post("/rank-stability", json={"app": "portfolio"})
    assert rs.status_code == 200 and rs.json()["n_paths"] == 2
    lt = c.post("/luck-test", json={"ticker": "AAA", "date": "2021-03-01", "horizon_days": 5, "n_paths": 300})
    assert lt.status_code == 200 and lt.json()["verdict"] in ("unusually_bad", "within_luck_range", "unusually_good")
    late = c.post("/luck-test", json={"ticker": "AAA", "date": "2026-06-25", "horizon_days": 20})
    assert late.status_code == 503 and "not observable" in late.json()["detail"]["message"]
    bias = c.post("/classify-biases", json={"text": "The whole sub is buying, all in."})
    assert {s["label"] for s in bias.json()["signals"]} == {"herding", "overconfidence"}


def test_bugs_are_500_without_internals(client, monkeypatch):
    c, system = client

    def boom(req):
        raise ZeroDivisionError("secret internals")

    monkeypatch.setattr(system.engine, "health", boom)
    r = c.get("/health")
    assert r.status_code == 500 and "secret" not in r.text and r.json()["detail"]["request_id"]


def test_openapi_has_typed_schemas(client):
    c, _ = client
    spec = c.get("/openapi.json").json()
    body = spec["paths"]["/luck-test"]["post"]["requestBody"]["content"]["application/json"]["schema"]
    assert body["$ref"].endswith("/LuckTestIn")
    assert "LuckTestOut" in spec["components"]["schemas"]


def test_website_is_served(client):
    c, _ = client
    page = c.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"] and "Trade Review" in page.text
    for asset in ("/static/js/main.js", "/static/js/trades.js", "/static/style.css"):
        assert c.get(asset).status_code == 200
    assert "/review-trade" in {x["path"] for x in c.get("/api").json()["routes"]}


def test_env_selects_the_bias_backend(monkeypatch):
    monkeypatch.setenv("MINIFINRL_PROFILE", "ci")
    monkeypatch.setenv("MINIFINRL_BIAS", "aip")
    app = create_app()
    assert app.state.system.engine.bias.name == "aip-small-v1"
    assert app.state.system.engine.parser is not None and app.state.system.engine.explainer is not None
