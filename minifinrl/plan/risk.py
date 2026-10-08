"""
The arithmetic of a trade plan, done by code so it is never wrong:

    shares         = amount / entry
    risk (money)   = amount x |entry - stop| / entry        what the stop costs if hit
    reward (money) = amount x |target - entry| / entry      what the target makes if reached
    reward : risk  = |target - entry| / |entry - stop|
    account risk   = risk / account size

A long plan needs stop < entry < target; a short plan the reverse.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Risk:
    risk_pct: float  # of the position, if the stop is hit
    reward_pct: float | None
    reward_risk: float | None
    shares: float | None
    risk_money: float | None
    reward_money: float | None
    account_risk_pct: float | None


def problems(direction: str, entry: float, stop: float | None, target: float | None) -> list[str]:
    """What is inconsistent about the prices, in plain words."""
    out = []
    if stop is not None:
        if direction == "long" and stop >= entry:
            out.append("For a buy, the stop has to be below the entry price.")
        if direction == "short" and stop <= entry:
            out.append("For a short, the stop has to be above the entry price.")
    if target is not None:
        if direction == "long" and target <= entry:
            out.append("For a buy, the target has to be above the entry price.")
        if direction == "short" and target >= entry:
            out.append("For a short, the target has to be below the entry price.")
    return out


def compute(direction: str, entry: float, stop: float, target: float | None = None, amount: float | None = None,
            account: float | None = None) -> Risk:
    risk_pct = abs(entry - stop) / entry
    reward_pct = abs(target - entry) / entry if target is not None else None
    risk_money = amount * risk_pct if amount else None
    return Risk(
        risk_pct=risk_pct, reward_pct=reward_pct,
        reward_risk=reward_pct / risk_pct if reward_pct is not None and risk_pct > 0 else None,
        shares=amount / entry if amount else None, risk_money=risk_money,
        reward_money=amount * reward_pct if amount and reward_pct is not None else None,
        account_risk_pct=risk_money / account if risk_money is not None and account else None,
    )


def as_returns(direction: str, entry: float, stop: float | None, target: float | None) -> tuple[float | None, float | None]:
    """Stop and target as directional returns from entry (stop negative, target positive)."""
    sign = 1.0 if direction == "long" else -1.0
    return (sign * (stop / entry - 1) if stop is not None else None,
            sign * (target / entry - 1) if target is not None else None)
