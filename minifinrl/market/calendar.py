"""Holding windows counted on the exchange calendar.

A buy date moves forward to the first trading day on or after it; a sell
date moves back to the last trading day on or before it; still holding
means the latest close."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field


class WindowProblem(ValueError):
    def __init__(self, status: str, message: str, missing: list[str] | None = None):
        super().__init__(message)
        self.status, self.message, self.missing = status, message, missing or []


@dataclass
class Window:
    i: int  # entry row in the trading-day index
    j: int  # exit row
    notes: list[str] = field(default_factory=list)

    @property
    def days(self) -> int:
        return self.j - self.i


def trading_window(dates: list[str], today: str, date: str | None, sell_date: str | None = None,
                   still_holding: bool | None = None, horizon_days: int | None = None,
                   max_days: int | None = None) -> Window:
    """Entry and exit rows in `dates` (trading days, ascending). The buy date
    moves forward to the first trading day on or after it; the sell date moves
    back to the last trading day on or before it; still holding means the latest
    close. With only a sell date and a length, the buy date is counted back."""
    notes = []
    if not dates:
        raise WindowProblem("rejected", "No trading days are available to measure the trade against.")
    if sell_date and sell_date > today:
        raise WindowProblem("rejected", f"The sell date {sell_date} is in the future.", ["sell_date"])
    if not date:
        if not (sell_date and horizon_days):
            raise WindowProblem("needs_input", "Pick the buy date.", ["date"])
        j = bisect.bisect_right(dates, sell_date) - 1
        if j - horizon_days < 0:
            raise WindowProblem("rejected", "There is not enough price history before that sell date.")
        date = dates[j - horizon_days]
        notes.append(f"Buy date counted back {horizon_days} trading days from the sell date: {date}.")
    if date > today:
        raise WindowProblem("rejected", f"{date} is in the future, so there is no outcome to review yet.")
    i = bisect.bisect_left(dates, date)
    if i >= len(dates):
        raise WindowProblem("rejected", f"There is no trading day on or after {date} in the data yet (it ends {dates[-1]}).")
    if dates[i] != date:
        notes.append(f"{date} was not a trading day; the trade is dated from the next one, {dates[i]}.")
    if sell_date:
        if sell_date <= date:
            raise WindowProblem("needs_input", "The sell date has to be after the buy date.", ["sell_date"])
        j = bisect.bisect_right(dates, sell_date) - 1
        if sell_date > dates[-1]:
            notes.append(f"Prices are available up to {dates[-1]}, so the trade is measured to that close.")
        elif dates[j] != sell_date:
            notes.append(f"{sell_date} was not a trading day; the exit uses the close of {dates[j]}, the last trading day before it.")
        if j <= i:
            raise WindowProblem("needs_input", f"Bought and sold within the same trading day ({dates[i]}). The luck check "
                                "works on daily closes, so it needs at least one close after the buy.", ["sell_date"])
    elif still_holding:
        j = len(dates) - 1
        if j <= i:
            raise WindowProblem("rejected", f"Bought on {dates[i]}, the latest close in the data. There is nothing to measure yet.")
        notes.append(f"Still holding: measured up to the latest close, {dates[j]}.")
    elif horizon_days:
        j = i + horizon_days
        if j >= len(dates):
            raise WindowProblem("rejected", f"The outcome is not observable yet: {dates[i]} plus {horizon_days} trading days "
                                f"is past the latest close, {dates[-1]}. Pick 'still holding' to measure it so far.")
    else:
        raise WindowProblem("needs_input", "Pick the sell date, or say you're still holding.", ["sell_date"])
    if max_days and j - i > max_days:
        raise WindowProblem("needs_input", f"That is {j - i} trading days. The luck check covers holds of up to "
                            f"{max_days} trading days (about a year).", ["sell_date"])
    return Window(i, j, notes)
