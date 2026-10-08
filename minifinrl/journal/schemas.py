"""Inputs and outputs of the journal capabilities."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from minifinrl.journal import ids
from minifinrl.plan.schemas import PlanIn, PlanOut
from minifinrl.platform.schemas import ISO_DATE, Input
from minifinrl.review.schemas import ReviewOut


class ProfileIn(Input):
    profile_id: str = Field(description="your journal ID, TR-XXXX-XXXX-XXXX-XXXX")

    @field_validator("profile_id", mode="before")
    @classmethod
    def _id(cls, v):
        v = ids.normalise(str(v))
        if not ids.is_profile_id(v):
            raise ValueError("not a journal ID (it looks like TR-XXXX-XXXX-XXXX-XXXX)")
        return v


class ProfileOut(BaseModel):
    id: str
    created_at: str
    trades: int = 0
    account_size: float | None = None


class AccountIn(ProfileIn):
    account_size: float | None = Field(default=None, gt=0, description="your trading account; empty to forget it")


class TradeIn(ProfileIn):
    """A trade to log. With a sell date it is closed, with `planned` it hasn't
    happened yet, otherwise it is open (still held)."""

    ticker: str = Field(min_length=1, max_length=12)
    name: str | None = Field(default=None, max_length=200, description="the listing's name, for display")
    direction: Literal["long", "short"] = "long"
    buy_date: str | None = Field(default=None, pattern=ISO_DATE)
    sell_date: str | None = Field(default=None, pattern=ISO_DATE)
    planned: bool = Field(default=False, description="not traded yet")
    price_paid: float | None = Field(default=None, gt=0)
    price_sold: float | None = Field(default=None, gt=0)
    quantity: float | None = Field(default=None, gt=0)
    reasoning: str | None = Field(default=None, max_length=2000, description="why, in your own words, at the time")

    @model_validator(mode="after")
    def _dates(self):
        if not self.planned and not self.buy_date:
            raise ValueError("a trade that has happened needs its buy date")
        if self.sell_date and not self.buy_date:
            raise ValueError("a sell date needs a buy date")
        if self.sell_date and self.buy_date and self.sell_date <= self.buy_date:
            raise ValueError("the sell date has to be after the buy date")
        if self.planned and self.sell_date:
            raise ValueError("a planned trade can't have a sell date yet")
        return self


class TradeRecord(BaseModel):
    id: str
    status: Literal["planned", "open", "closed"]
    ticker: str
    name: str | None
    direction: str
    buy_date: str | None
    sell_date: str | None
    price_paid: float | None
    price_sold: float | None
    quantity: float | None
    reasoning: str | None
    state: dict | None = Field(description="what was read from the reasoning when the trade was logged")
    result: dict | None = Field(description="the latest review: return, luck percentile, verdict, market, and how it compares with the plan")
    plan_entry: float | None = None
    plan_stop: float | None = None
    plan_target: float | None = None
    plan_amount: float | None = None
    plan_horizon: int | None = None
    plan: dict | None = Field(default=None, description="the plan check when the plan was saved: risk, odds, range, nudges")
    created_at: str
    updated_at: str


class JournalSummary(BaseModel):
    count: int
    planned: int
    open: int
    closed: int


class TradesOut(BaseModel):
    trades: list[TradeRecord]
    summary: JournalSummary


class TradeRef(ProfileIn):
    trade_id: str = Field(min_length=1, max_length=32)


class CloseTradeIn(TradeRef):
    sell_date: str = Field(pattern=ISO_DATE)
    price_sold: float | None = Field(default=None, gt=0)


class ReviewedTrade(BaseModel):
    trade: TradeRecord
    review: ReviewOut


class DeletedOut(BaseModel):
    deleted: int = Field(description="how many trades were deleted")


class PlanSaveIn(ProfileIn, PlanIn):
    """A plan to check and keep in the journal."""

    name: str | None = Field(default=None, max_length=200, description="the listing's name, for display")


class SavedPlan(BaseModel):
    trade: TradeRecord
    plan: PlanOut


class StartPlanIn(TradeRef):
    buy_date: str = Field(pattern=ISO_DATE, description="the day you made the trade")
    price_paid: float | None = Field(default=None, gt=0)


class Group(BaseModel):
    key: str
    label: str
    trades: int
    avg_return: float | None
    win_rate: float | None


class InsightsOut(BaseModel):
    closed: int
    avg_return: float | None
    win_rate: float | None
    by_plan: list[Group] = Field(description="closed trades by how they compared with their plan")
    by_state: list[Group] = Field(description="closed trades by the state their reasoning was in when logged")
    needs_attention: list[dict] = Field(description="open trades whose plan says it's time to act")
    headlines: list[str]
