"""The check before a trade: outcome range from today's regime, state and
bias reading of the reason, the nudges, and saving and starting a plan in
the journal. Random-walk prices; rules readers; no LLM."""

from __future__ import annotations

import pytest

from minifinrl.platform.capabilities import CapabilityError, CapabilityRegistry


@pytest.fixture
def reg(rw_engine):
    return CapabilityRegistry.from_engine(rw_engine)


HOT = "everyone on reddit is buying, it can't go down, I have to get in now before it's too late"


def plan(reg, **kw):
    return reg.invoke("check_plan", {"ticker": "AAA", "horizon_days": 10, **kw})


def test_range_from_todays_regime(reg):
    out = plan(reg)
    o = out.outlook
    assert out.status == "ok" and o.as_of == "2026-12-31"  # the latest close: nothing after today is used
    assert o.p05 < o.p25 < o.p50 < o.p75 < o.p95 and 0 < o.prob_loss < 1
    assert len(o.days) == len(o.fan_p05) == 10 and o.fan_p05[-1] < o.fan_p95[-1]
    short = plan(reg, direction="short").outlook
    assert short.p50 == pytest.approx(-o.p50, abs=0.02)  # the same paths, seen from the other side


def test_hot_reason_gets_a_pause(reg):
    out = plan(reg, reasoning=HOT)
    assert out.state.level == "hot" and out.nudges[0].level == "pause"
    texts = " ".join(n.text for n in out.nudges)
    assert "has to happen today" in texts and "Other people buying" in texts and "Would you be fine at" in texts


def test_calm_reason_and_no_reason(reg):
    calm = plan(reg, reasoning="margins improved two quarters in a row and the valuation looks fair to me")
    assert calm.state.level == "calm" and calm.nudges[0].text.startswith("Your reason reads calmly")
    bare = plan(reg)
    assert bare.state is None and bare.nudges[0].text.startswith("Add a sentence on why")


def test_never_advice(reg):
    for kw in ({}, {"reasoning": HOT}, {"direction": "short", "reasoning": "it will crash, I'm scared"}):
        for n in plan(reg, **kw).nudges:
            low = n.text.lower()
            assert not any(p in low for p in ("you should buy", "you should sell", "don't buy", "do not buy", "will go up", "will go down"))


@pytest.mark.parametrize("ticker, status", [("NOPE", "rejected"), ("TWTR", "rejected"), ("DOGE", "unsupported")])
def test_symbols_that_cant_be_checked(reg, ticker, status):
    out = plan(reg, ticker=ticker)
    assert out.status == status and out.messages


def test_recent_listing_needs_history(reg):
    assert plan(reg, ticker="NEW").status == "ok"  # listed in 2020, plenty by the data end


def last_close(reg):
    return plan(reg).outlook.last_price


def test_plan_numbers_and_odds(reg):
    px = last_close(reg)
    out = plan(reg, entry_price=px, stop_price=round(px * 0.95, 4), target_price=round(px * 1.10, 4), amount=2000, account_size=50000)
    r, o = out.risk, out.outlook
    assert out.status == "ok" and out.entry_price == px
    assert r.risk_pct == pytest.approx(0.05, abs=1e-3) and r.risk_money == pytest.approx(100, abs=0.5)
    assert r.reward_risk == pytest.approx(2.0, abs=0.01) and r.account_risk_pct == pytest.approx(0.002, abs=1e-4)
    assert 0 <= o.prob_target_first + o.prob_stop_first <= 1 and o.prob_stop >= o.prob_stop_first
    wide = plan(reg, stop_price=px * 0.6).outlook.prob_stop
    tight = plan(reg, stop_price=px * 0.995).outlook.prob_stop
    assert tight > wide  # a stop inside daily noise gets hit far more often


def test_entry_defaults_to_the_latest_close(reg):
    assert plan(reg, stop_price=1.0).entry_price == last_close(reg)


@pytest.mark.parametrize("kw, message", [
    ({"stop_price": 2.0}, "stop has to be below"),
    ({"stop_price": 0.5, "target_price": 0.9}, "target has to be above"),
])
def test_prices_the_wrong_way_round(reg, kw, message):
    px = last_close(reg)
    out = plan(reg, entry_price=px, **{k: v * px for k, v in kw.items()})
    assert out.status == "needs_input" and any(message in m for m in out.messages)
    short = plan(reg, direction="short", entry_price=px, stop_price=px * 0.9)
    assert short.status == "needs_input" and "above the entry" in short.messages[-1]


def test_plan_nudges(reg):
    px = last_close(reg)
    texts = lambda out: " ".join(n.text for n in out.nudges)  # noqa: E731
    assert "There's no stop yet" in texts(plan(reg))
    assert "ordinary day-to-day movement" in texts(plan(reg, stop_price=px * 0.997))
    assert "risk more than you aim to make" in texts(plan(reg, stop_price=px * 0.9, target_price=px * 1.02))
    assert "of your account on one idea" in texts(plan(reg, stop_price=px * 0.9, amount=20000, account_size=50000))


def save(reg, pid, **kw):
    px = last_close(reg)
    return reg.invoke("save_plan", {"profile_id": pid, "ticker": "AAA", "horizon_days": 10, "reasoning": HOT,
                                    "entry_price": px, "stop_price": px * 0.95, "target_price": px * 1.1, "amount": 1000, **kw})


def test_a_plan_needs_a_stop_and_a_reason(reg):
    pid = reg.invoke("create_profile", {}).id
    with pytest.raises(CapabilityError, match="needs a stop"):
        reg.invoke("save_plan", {"profile_id": pid, "ticker": "AAA", "reasoning": HOT})
    with pytest.raises(CapabilityError, match="needs a reason"):
        reg.invoke("save_plan", {"profile_id": pid, "ticker": "AAA", "stop_price": 1.0})


def test_save_a_plan_then_make_the_trade(reg):
    pid = reg.invoke("create_profile", {}).id
    saved = save(reg, pid, planned_date="2026-11-02")
    t = saved.trade
    assert t.status == "planned" and t.state["level"] == "hot"
    assert t.plan_stop == pytest.approx(saved.plan.stop_price) and t.plan_horizon == 10
    assert t.plan["risk"]["risk_money"] == pytest.approx(50, abs=0.5) and t.plan["nudges"][0]["level"] == "pause"
    started = reg.invoke("start_planned_trade", {"profile_id": pid, "trade_id": t.id, "buy_date": "2026-11-02",
                                                 "price_paid": t.plan_entry * 1.01})
    s = started.trade
    assert s.status == "open" and s.result["status"] == "ok" and s.plan == t.plan and s.state == t.state
    pc = s.result["plan_check"]
    assert pc["entry_gap"] == pytest.approx(0.01) and pc["verdict"] in ("on_plan", "stop", "target", "time")
    with pytest.raises(CapabilityError, match="Only a planned trade"):
        reg.invoke("start_planned_trade", {"profile_id": pid, "trade_id": t.id, "buy_date": "2026-11-02"})


def test_a_plan_that_cant_be_checked_is_not_saved(reg):
    pid = reg.invoke("create_profile", {}).id
    with pytest.raises(CapabilityError, match="No price history"):
        reg.invoke("save_plan", {"profile_id": pid, "ticker": "NOPE", "stop_price": 1.0, "reasoning": HOT})
    assert reg.invoke("list_trades", {"profile_id": pid}).trades == []


def test_account_size_is_remembered_and_used(reg):
    pid = reg.invoke("create_profile", {}).id
    assert reg.invoke("set_account_size", {"profile_id": pid, "account_size": 10000}).account_size == 10000
    t = save(reg, pid, amount=1000).trade  # no account size passed: the remembered one is used
    assert t.plan["risk"]["account_risk_pct"] == pytest.approx(0.005, abs=1e-4)
    assert reg.invoke("set_account_size", {"profile_id": pid}).account_size is None


def test_last_close_for_the_entry_box(reg):
    q = reg.invoke("last_close", {"ticker": "aaa"})
    assert q.ticker == "AAA" and q.as_of == "2026-12-31" and q.last_close == pytest.approx(last_close(reg))
    from minifinrl.platform.capabilities import CapabilityNotFound
    with pytest.raises(CapabilityNotFound):
        reg.invoke("last_close", {"ticker": "NOPE"})


def test_a_plan_comes_before_the_trade(reg):
    pid = reg.invoke("create_profile", {}).id
    t = save(reg, pid).trade
    with pytest.raises(CapabilityError, match="before you saved the plan"):
        reg.invoke("start_planned_trade", {"profile_id": pid, "trade_id": t.id, "buy_date": "2020-01-02"})


def test_price_levels(reg):
    out = reg.invoke("plan_levels", {"ticker": "AAA", "horizon_days": 21})
    assert out.last_close == pytest.approx(last_close(reg)) and out.atr > 0 and out.entries[0].key == "now"
    stops = {s.key: s for s in out.stops}
    # defined by how often normal luck reaches them within the time limit
    assert stops["tight"].prob == pytest.approx(0.5, abs=0.03) and stops["normal"].prob == pytest.approx(0.25, abs=0.03)
    assert stops["wide"].prob == pytest.approx(0.10, abs=0.03)
    assert all(s.price < out.entry for s in out.stops) and all(t.price > out.entry for t in out.targets)
    assert [abs(s.pct) for s in out.stops] == sorted(abs(s.pct) for s in out.stops)
    r2 = next(t for t in out.targets if t.key == "r2")
    assert r2.price - out.entry == pytest.approx(2 * (out.entry - stops["normal"].price), abs=0.03)  # prices are in cents
    dips = [e for e in out.entries if e.key.startswith("dip")]
    assert [d.prob for d in dips] == sorted((d.prob for d in dips), reverse=True)  # deeper dips fill less often


def test_levels_follow_your_stop_and_side(reg):
    px = last_close(reg)
    mine = reg.invoke("plan_levels", {"ticker": "AAA", "stop_price": px * 0.9})
    assert next(t for t in mine.targets if t.key == "r2").price == pytest.approx(px * 1.2, rel=1e-3)
    short = reg.invoke("plan_levels", {"ticker": "AAA", "direction": "short"})
    assert all(s.price > short.entry for s in short.stops) and all(t.price < short.entry for t in short.targets)
    week = reg.invoke("plan_levels", {"ticker": "AAA", "horizon_days": 5})
    normal = lambda o: next(s for s in o.stops if s.key == "normal")  # noqa: E731
    assert abs(normal(week).pct) < abs(normal(reg.invoke("plan_levels", {"ticker": "AAA", "horizon_days": 63})).pct)


def test_levels_for_an_unknown_symbol(reg):
    from minifinrl.platform.capabilities import CapabilityNotFound
    with pytest.raises(CapabilityNotFound):
        reg.invoke("plan_levels", {"ticker": "NOPE"})
