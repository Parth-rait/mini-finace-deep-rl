"""
Price levels to plan with, so nobody has to invent an entry, stop or target
from nothing. Every level comes from how this stock has actually moved, and
carries how often normal luck reached it in the regime model's paths. They
describe the stock; they are not recommendations.

    typical daily move  average true range over 14 days:
                        true range = max(high - low, |high - previous close|, |low - previous close|)
    entry               the latest close, or a dip of 2, 5 or 10% (a rise, for a short),
                        with how often the price got there within the time limit (a fill)
    stop                the level normal luck reached in about half, a quarter and a tenth
                        of the paths within the time limit (tight, normal, wide), so it
                        adapts to the stock and the time limit; and just past the recent
                        20-day low (high, for a short)
    target              1.5, 2 and 3 times the stop's distance (yours, or the normal stop), the recent 20-day high
                        (low) and the 52-week high (low), when they're on the right side

Probabilities are on daily closes from the latest close, so a level touched
only intraday doesn't count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ATR_DAYS = 14
RECENT_DAYS = 20
YEAR_DAYS = 252


@dataclass(frozen=True)
class Level:
    key: str
    label: str
    price: float
    pct: float  # from the entry (signed, raw price change)
    prob: float | None  # share of paths that reached it within the time limit
    note: str = ""


def atr(bars: pd.DataFrame, days: int = ATR_DAYS) -> float:
    """`bars`: date-indexed high, low, close."""
    prev = bars["close"].shift(1)
    tr = pd.concat([bars["high"] - bars["low"], (bars["high"] - prev).abs(), (bars["low"] - prev).abs()], axis=1).max(axis=1)
    return float(tr.iloc[-days:].mean())


def _extreme(bars: pd.DataFrame, col: str, days: int, fn) -> tuple[float, str]:
    window = bars[col].iloc[-days:]
    day = window.idxmax() if fn is max else window.idxmin()
    return float(window[day]), str(day)


def build(bars: pd.DataFrame, paths: np.ndarray, direction: str, entry: float | None = None,
          stop: float | None = None) -> dict:
    """`paths`: (n, h) cumulative raw price returns from the latest close."""
    last = float(bars["close"].iloc[-1])
    entry = entry or last
    long = direction == "long"
    lo_path = last * (1 + paths.min(axis=1))
    hi_path = last * (1 + paths.max(axis=1))
    below = lambda price: float((lo_path <= price).mean())  # noqa: E731
    above = lambda price: float((hi_path >= price).mean())  # noqa: E731
    toward_loss = below if long else above
    toward_gain = above if long else below
    sign = 1 if long else -1
    a = atr(bars)

    def lvl(key, label, price, prob, note=""):
        return Level(key=key, label=label, price=round(price, 2), pct=price / entry - 1, prob=prob, note=note)

    entries = [lvl("now", "Now", last, None, "the latest close")]
    for d in (0.02, 0.05, 0.10):
        p = last * (1 - sign * d)
        entries.append(lvl(f"dip{int(d * 100)}", f"{'Dip' if long else 'Rise'} {int(d * 100)}%", p, toward_loss(p),
                           "how often the price got there: the chance the order fills"))

    # the price that a share q of paths reached on the losing side, moved to be measured from the entry
    worst = lo_path if long else hi_path
    stops = []
    for key, label, q, note in (("tight", "Tight", 0.5, "reached in about half the paths"),
                                ("normal", "Normal", 0.25, "reached in about 1 in 4 paths"),
                                ("wide", "Wide", 0.10, "reached in about 1 in 10 paths")):
        move = float(np.quantile(worst, q if long else 1 - q)) / last - 1
        price = entry * (1 + move)
        if (price < entry) if long else (price > entry):
            stops.append(lvl(key, label, price, toward_loss(price), note))
    swing, day = _extreme(bars, "low" if long else "high", RECENT_DAYS, min if long else max)
    swing_stop = swing * (1 - sign * 0.005)  # just past it, so touching the level itself doesn't count
    if (swing_stop < entry) if long else (swing_stop > entry):
        stops.append(lvl("swing", f"Past the recent {'low' if long else 'high'}", swing_stop, toward_loss(swing_stop),
                         f"{'low' if long else 'high'} of {swing:,.2f} on {day}"))

    stops.sort(key=lambda x: abs(x.pct))
    normal = next((x for x in stops if x.key == "normal"), stops[0] if stops else None)
    risk = abs(entry - stop) if stop and ((stop < entry) if long else (stop > entry)) else (abs(entry - normal.price) if normal else 2 * a)
    targets = [lvl(f"r{m}", f"{m:g}x risk", entry + sign * m * risk, toward_gain(entry + sign * m * risk),
                   f"{m:g} times the distance to {'your' if stop else 'a normal'} stop") for m in (1.5, 2, 3)]
    seen = set()
    for key, label, days in (("recent", "Recent", RECENT_DAYS), ("year", "52-week", YEAR_DAYS)):
        price, day = _extreme(bars, "high" if long else "low", min(days, len(bars)), max if long else min)
        if round(price, 2) in seen:
            targets[-1] = Level(**{**vars(targets[-1]), "label": f"Recent and 52-week {'high' if long else 'low'}"})
            continue
        if (price > entry * 1.005) if long else (price < entry * 0.995):
            targets.append(lvl(key, f"{label} {'high' if long else 'low'}", price, toward_gain(price), f"on {day}"))
            seen.add(round(price, 2))
    targets.sort(key=lambda x: abs(x.pct))
    return {"last_close": last, "as_of": str(bars.index[-1]), "entry": entry, "atr": a, "atr_pct": a / last,
            "entries": entries, "stops": stops, "targets": targets}
