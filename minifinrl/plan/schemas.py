"""Inputs and outputs of the plan capabilities."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from minifinrl.platform.schemas import ISO_DATE, Input
from minifinrl.review.schemas import BiasSpan
from minifinrl.review.settings import MAX_HORIZON
from minifinrl.sentiment.ports import StateReading


class PlanIn(Input):
    ticker: str = Field(min_length=1, max_length=12)
    direction: Literal["long", "short"] = "long"
    entry_price: float | None = Field(default=None, gt=0, description="the price you plan to enter at; the latest close if empty")
    stop_price: float | None = Field(default=None, gt=0, description="the price at which the idea is wrong and you get out")
    target_price: float | None = Field(default=None, gt=0, description="the price at which you take profit")
    amount: float | None = Field(default=None, gt=0, description="how much money you plan to put in")
    account_size: float | None = Field(default=None, gt=0, description="your trading account, to show risk as a share of it")
    horizon_days: int = Field(default=21, ge=1, le=MAX_HORIZON, description="the longest you'll wait, in trading days")
    reasoning: str = Field(default="", max_length=2000, description="why you want to make the trade, in your own words")
    planned_date: str | None = Field(default=None, pattern=ISO_DATE, description="when you plan to trade (for your records)")


class OutlookOut(BaseModel):
    as_of: str = Field(description="the latest close the model saw")
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float
    prob_loss: float = Field(description="share of simulated outcomes that lost money")
    regime: str
    regime_probs: dict[str, float]
    days: list[int]
    fan_p05: list[float]
    fan_p50: list[float]
    fan_p95: list[float]
    n_paths: int
    last_price: float
    prob_stop: float | None = None
    prob_target: float | None = None
    prob_target_first: float | None = None
    prob_stop_first: float | None = None


class RiskOut(BaseModel):
    risk_pct: float = Field(description="share of the position lost if the stop is hit")
    reward_pct: float | None
    reward_risk: float | None = Field(description="what the target makes for each unit the stop costs")
    shares: float | None
    risk_money: float | None
    reward_money: float | None
    account_risk_pct: float | None


class Nudge(BaseModel):
    level: Literal["info", "caution", "pause"]
    text: str


class PlanOut(BaseModel):
    status: Literal["ok", "needs_input", "unsupported", "rejected"]
    messages: list[str] = Field(default_factory=list)
    ticker: str
    name: str | None = None
    direction: str
    horizon_days: int
    planned_date: str | None = None
    entry_price: float | None = Field(default=None, description="the entry used: yours, or the latest close")
    stop_price: float | None = None
    target_price: float | None = None
    amount: float | None = None
    risk: RiskOut | None = None
    outlook: OutlookOut | None = None
    state: StateReading | None = None
    biases: list[BiasSpan] = Field(default_factory=list)
    bias_source: str | None = None
    nudges: list[Nudge] = Field(default_factory=list)
    disclaimer: str = ("A check of your own words and of how wide luck is over the hold you plan. "
                       "Not a forecast and not investment advice: it never says whether to trade.")


class LevelsIn(Input):
    ticker: str = Field(min_length=1, max_length=12)
    direction: Literal["long", "short"] = "long"
    horizon_days: int = Field(default=21, ge=1, le=MAX_HORIZON)
    entry_price: float | None = Field(default=None, gt=0, description="the entry to measure from; the latest close if empty")
    stop_price: float | None = Field(default=None, gt=0, description="your stop, so targets are multiples of your risk")


class LevelOut(BaseModel):
    key: str
    label: str
    price: float
    pct: float = Field(description="change from the entry")
    prob: float | None = Field(description="share of normal-luck paths that reached it within the time limit")
    note: str


class LevelsOut(BaseModel):
    ticker: str
    last_close: float
    as_of: str
    entry: float
    atr: float = Field(description="typical daily move: 14-day average true range")
    atr_pct: float
    horizon_days: int
    entries: list[LevelOut]
    stops: list[LevelOut]
    targets: list[LevelOut]
    note: str = ("Levels from how this stock has moved, with how often normal luck reached them on daily closes. "
                 "They describe the stock; they are not recommendations.")
