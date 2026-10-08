"""Did a trade follow its plan? Hand-made price paths with known triggers."""

from __future__ import annotations

import pandas as pd

from minifinrl.journal.adherence import check

DATES = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2025-03-03", periods=12)]


def path(*prices):
    return pd.Series(prices, index=DATES[:len(prices)])


PLAN = {"direction": "long", "buy_date": DATES[0], "plan_entry": 100, "plan_stop": 95, "plan_target": 110, "plan_horizon": 8}


def test_open_on_plan_then_stop_crossed():
    up = path(100, 101, 102, 101, 103)
    out = check(PLAN, up)
    assert out["verdict"] == "on_plan" and "4 trading days left" in out["message"]
    down = path(100, 98, 94, 96)
    out = check(PLAN, down)
    assert out["verdict"] == "stop" and DATES[2] in out["message"] and out["label"] == "Stop crossed"


def test_followed_broke_and_early():
    p = path(100, 98, 94, 93, 92, 91, 90)
    assert check({**PLAN, "sell_date": DATES[3]}, p)["verdict"] == "followed"  # sold 1 day after the stop
    broke = check({**PLAN, "sell_date": DATES[6]}, p)
    assert broke["verdict"] == "broke" and "held 4 more trading days" in broke["message"]
    assert check({**PLAN, "sell_date": DATES[1]}, p)["verdict"] == "early"  # sold before any trigger


def test_target_and_time_limit():
    p = path(100, 104, 111, 112)
    assert check({**PLAN, "sell_date": DATES[2]}, p)["verdict"] == "followed"
    flat = path(*([100] * 12))
    late = check({**PLAN, "sell_date": DATES[11]}, flat)
    assert late["verdict"] == "broke" and "time limit" in late["message"]


def test_short_plans_flip_the_triggers():
    short = {**PLAN, "direction": "short", "plan_stop": 105, "plan_target": 90}
    assert check(short, path(100, 103, 106))["verdict"] == "stop"
    assert check(short, path(100, 95, 89))["verdict"] == "target"


def test_entry_gap_and_selling_past_the_stop():
    out = check({**PLAN, "price_paid": 102, "sell_date": DATES[2], "price_sold": 90}, path(100, 98, 94))
    assert abs(out["entry_gap"] - 0.02) < 1e-9 and "past your stop" in out["message"]


def test_no_plan_no_check():
    assert check({"direction": "long", "buy_date": DATES[0]}, path(100, 101)) is None
