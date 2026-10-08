"""The trade journal: IDs, logging, reviewing, closing and deleting trades,
and that one ID can never reach another's trades. SQLite in tmp_path;
random-walk prices; no LLM."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from minifinrl.journal import ids
from minifinrl.platform.capabilities import CapabilityError, CapabilityNotFound, CapabilityRegistry, error_kind


@pytest.fixture
def reg(rw_engine):
    return CapabilityRegistry.from_engine(rw_engine)


def new(reg) -> str:
    return reg.invoke("create_profile", {}).id


def log(reg, pid, **kw):
    return reg.invoke("log_trade", {"profile_id": pid, "ticker": "AAA", "buy_date": "2020-03-02", **kw})


# ---- IDs -----------------------------------------------------------------------------------

def test_ids_are_random_readable_and_checked():
    a, b = ids.new_profile_id(), ids.new_profile_id()
    assert a != b and ids.is_profile_id(a) and len(a) == len("TR-XXXX-XXXX-XXXX-XXXX")
    assert not set(a.replace("TR-", "").replace("-", "")) & set("01OILU")  # no look-alike characters
    assert ids.normalise(a.lower().replace("-", " ")) == a  # what people type still works
    assert not ids.is_profile_id("TR-1234") and not ids.is_profile_id("user123")


def test_create_and_open_a_journal(reg):
    pid = new(reg)
    out = reg.invoke("get_profile", {"profile_id": pid.lower()})
    assert out.id == pid and out.trades == 0


def test_unknown_or_malformed_id(reg):
    with pytest.raises(CapabilityNotFound, match="No journal has that ID"):
        reg.invoke("get_profile", {"profile_id": ids.new_profile_id()})
    with pytest.raises(ValidationError, match="not a journal ID"):
        reg.invoke("get_profile", {"profile_id": "hello"})


# ---- logging -------------------------------------------------------------------------------

def test_status_follows_the_dates(reg):
    pid = new(reg)
    assert log(reg, pid, sell_date="2020-03-16").status == "closed"
    assert log(reg, pid).status == "open"
    assert reg.invoke("log_trade", {"profile_id": pid, "ticker": "aaa", "planned": True}).status == "planned"
    out = reg.invoke("list_trades", {"profile_id": pid})
    assert (out.summary.count, out.summary.closed, out.summary.open, out.summary.planned) == (3, 1, 1, 1)
    assert {t.ticker for t in out.trades} == {"AAA"}


@pytest.mark.parametrize("bad, match", [
    ({"buy_date": None}, "needs its buy date"),
    ({"sell_date": "2020-03-01"}, "after the buy date"),
    ({"planned": True, "sell_date": "2020-03-16"}, "planned trade"),
    ({"reasoning": "x" * 2001}, "2000"),
    ({"price_paid": -5}, "greater than 0"),
])
def test_bad_trades_are_refused(reg, bad, match):
    with pytest.raises(ValidationError, match=match):
        log(reg, new(reg), **bad)


# ---- privacy -------------------------------------------------------------------------------

def test_one_id_never_reaches_another_journal(reg):
    mine, theirs = new(reg), new(reg)
    t = log(reg, theirs, reasoning="private reasons")
    assert reg.invoke("list_trades", {"profile_id": mine}).trades == []
    for name in ("delete_trade", "review_logged_trade"):
        with pytest.raises(CapabilityNotFound):
            reg.invoke(name, {"profile_id": mine, "trade_id": t.id})
    with pytest.raises(CapabilityNotFound):
        reg.invoke("close_trade", {"profile_id": mine, "trade_id": t.id, "sell_date": "2020-03-16"})
    assert reg.invoke("list_trades", {"profile_id": theirs}).trades[0].reasoning == "private reasons"


# ---- reviewing and closing ---------------------------------------------------------------------

def test_review_a_closed_trade_and_keep_the_result(reg):
    pid = new(reg)
    t = log(reg, pid, sell_date="2020-03-16", reasoning="everyone is buying and it can't lose")
    out = reg.invoke("review_logged_trade", {"profile_id": pid, "trade_id": t.id})
    assert out.review.status == "ok" and out.review.understood.exit_date == "2020-03-16"
    r = out.trade.result
    assert r["status"] == "ok" and r["verdict"] == out.review.outcome.verdict and r["biases"] == ["herding", "overconfidence"]
    assert reg.invoke("list_trades", {"profile_id": pid}).trades[0].result == r  # saved, not just returned


def test_open_trade_is_measured_to_the_latest_close(reg):
    pid = new(reg)
    t = log(reg, pid, buy_date="2026-11-02")
    out = reg.invoke("review_logged_trade", {"profile_id": pid, "trade_id": t.id})
    assert out.review.status == "ok" and out.review.understood.exit_date == "2026-12-31"


def test_close_an_open_trade(reg):
    pid = new(reg)
    t = log(reg, pid, price_paid=101.5)
    out = reg.invoke("close_trade", {"profile_id": pid, "trade_id": t.id, "sell_date": "2020-03-20", "price_sold": 99})
    assert out.trade.status == "closed" and out.trade.sell_date == "2020-03-20" and out.trade.price_sold == 99
    assert out.trade.price_paid == 101.5 and out.trade.result["days"] == 14
    with pytest.raises(CapabilityError, match="after the buy date"):
        reg.invoke("close_trade", {"profile_id": pid, "trade_id": t.id, "sell_date": "2020-02-01"})


def test_planned_trades_have_no_outcome_yet(reg):
    pid = new(reg)
    t = reg.invoke("log_trade", {"profile_id": pid, "ticker": "AAA", "planned": True})
    with pytest.raises(CapabilityError, match="no outcome yet") as e:
        reg.invoke("review_logged_trade", {"profile_id": pid, "trade_id": t.id})
    assert error_kind(e.value) == "bad_input"


def test_a_review_that_cant_run_is_saved_with_its_reason(reg):
    pid = new(reg)
    t = reg.invoke("log_trade", {"profile_id": pid, "ticker": "NOPE", "buy_date": "2020-03-02", "sell_date": "2020-03-16"})
    out = reg.invoke("review_logged_trade", {"profile_id": pid, "trade_id": t.id})
    assert out.trade.result["status"] == "rejected" and "No price history" in out.trade.result["message"]


# ---- deleting ------------------------------------------------------------------------------------

def test_delete_a_trade_and_a_whole_journal(reg):
    pid = new(reg)
    a, _ = log(reg, pid), log(reg, pid)
    assert reg.invoke("delete_trade", {"profile_id": pid, "trade_id": a.id}).deleted == 1
    assert reg.invoke("get_profile", {"profile_id": pid}).trades == 1
    assert reg.invoke("delete_profile", {"profile_id": pid}).deleted == 1
    with pytest.raises(CapabilityNotFound):
        reg.invoke("list_trades", {"profile_id": pid})


def test_journal_survives_a_restart(rw_engine):
    from minifinrl.engine import Engine

    pid = new(CapabilityRegistry.from_engine(rw_engine))
    log(CapabilityRegistry.from_engine(rw_engine), pid)
    again = CapabilityRegistry.from_engine(Engine(rw_engine.cfg))  # new engine, same database file
    assert again.invoke("get_profile", {"profile_id": pid}).trades == 1


# ---- interfaces ----------------------------------------------------------------------------------

def test_http_routes_and_status_codes(rw_engine):
    from minifinrl.interfaces.api import create_app
    from minifinrl.system import System, SystemConfig

    reg = CapabilityRegistry.from_engine(rw_engine)
    c = TestClient(create_app(System(SystemConfig.from_profile("research"), rw_engine, reg)))
    pid = c.post("/create-profile", json={}).json()["id"]
    t = c.post("/log-trade", json={"profile_id": pid, "ticker": "AAA", "buy_date": "2020-03-02"}).json()
    assert c.post("/list-trades", json={"profile_id": pid}).json()["summary"]["open"] == 1
    assert c.post("/get-profile", json={"profile_id": ids.new_profile_id()}).status_code == 404
    assert c.post("/get-profile", json={"profile_id": "nope"}).status_code == 422
    assert c.post("/delete-trade", json={"profile_id": pid, "trade_id": t["id"]}).json()["deleted"] == 1


def test_agents_get_no_write_tools(reg):
    """An AI agent may read and analyse, but never change someone's journal."""
    tools = set(reg.callables("agent"))
    assert "list_trades" in tools and not tools & {"create_profile", "log_trade", "delete_trade", "delete_profile"}
