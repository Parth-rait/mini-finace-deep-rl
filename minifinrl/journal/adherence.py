"""
Did the trade follow its plan? Checked on daily closes, so a move that only
happened intraday is not seen.

While open, the latest state against the plan: on plan, stop crossed, target
reached, or time limit reached. Once closed, how the exit compares:

    followed   sold within 2 trading days of the first trigger (stop, target or time limit)
    broke      held more than 2 trading days past the first trigger
    early      sold before any trigger (not breaking the plan, but not following it either)
"""

from __future__ import annotations

import bisect

import pandas as pd

GRACE_DAYS = 2  # trading days allowed between a trigger and the sale

LABELS = {"followed": "Followed plan", "broke": "Broke plan", "early": "Sold early", "on_plan": "On plan",
          "stop": "Stop crossed", "target": "Target reached", "time": "Time limit reached"}
TRIGGER_WORDS = {"stop": "stop", "target": "target", "time": "time limit"}


def check(t: dict, closes: pd.Series) -> dict | None:
    """`t`: a trade row with plan_* fields. `closes`: the ticker's daily closes (date -> price)."""
    if not t.get("plan_stop") or not t.get("buy_date"):
        return None
    closes = closes.dropna()
    dates = list(closes.index)
    i = bisect.bisect_left(dates, t["buy_date"])
    if i >= len(dates):
        return None
    j = bisect.bisect_right(dates, t["sell_date"]) - 1 if t.get("sell_date") else len(dates) - 1
    long = t["direction"] == "long"
    stop, target, horizon = t["plan_stop"], t.get("plan_target"), t.get("plan_horizon")

    def first(cond) -> str | None:
        for k in range(i + 1, len(dates)):
            if cond(closes.iloc[k]):
                return dates[k]
        return None

    events = {
        "stop": first(lambda c: c <= stop if long else c >= stop),
        "target": first(lambda c: c >= target if long else c <= target) if target else None,
        "time": dates[i + horizon] if horizon and i + horizon < len(dates) else None,
    }
    out = {"events": events, "as_of": dates[j]}
    if t.get("price_paid") and t.get("plan_entry"):
        out["entry_gap"] = t["price_paid"] / t["plan_entry"] - 1

    triggered = sorted((d, k) for k, d in events.items() if d is not None and d <= dates[j])
    if not t.get("sell_date"):  # still open
        if not triggered:
            left = (horizon - (j - i)) if horizon else None
            out.update(verdict="on_plan", message="Between your stop and target" + (f", {left} trading days left of your time limit." if left and left > 0 else "."))
        else:
            d, k = triggered[0]
            out.update(verdict=k, message=f"Your plan said to sell: the {TRIGGER_WORDS[k]} came on {d}"
                       + (f" (close {closes[d]:,.2f})." if k != "time" else "."))
        out["label"] = LABELS[out["verdict"]]
        return out

    if not triggered:
        out.update(verdict="early", message="You sold before your stop, target or time limit was reached.")
    else:
        d, k = triggered[0]
        late = j - dates.index(d)
        if late <= GRACE_DAYS:
            out.update(verdict="followed", message=f"You sold within {late} trading day{'s' if late != 1 else ''} of your {TRIGGER_WORDS[k]} ({d}).")
        else:
            out.update(verdict="broke", message=f"Your {TRIGGER_WORDS[k]} came on {d}, but you held {late} more trading days.")
    if t.get("price_sold") and (t["price_sold"] < stop * 0.98 if long else t["price_sold"] > stop * 1.02):
        out["message"] += f" You sold at {t['price_sold']:,.2f}, past your stop of {stop:,.2f}."
    out["label"] = LABELS[out["verdict"]]
    return out
