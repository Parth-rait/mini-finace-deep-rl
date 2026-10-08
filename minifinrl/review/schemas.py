"""Inputs and outputs of the trade review capabilities."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from minifinrl.platform.schemas import ISO_DATE, Input
from minifinrl.regime.schemas import LuckTestOut
from minifinrl.review.settings import MAX_STATED_DAYS


class ReviewIn(Input):
    """The form fields are the trade; `text` is optional context (your reasoning,
    or the whole trade in words, which is parsed to fill fields left empty)."""

    text: str = Field(default="", max_length=2000, description="your reasoning, or the trade in your own words")
    ticker: str | None = Field(default=None, max_length=12, description="fills or overrides what was read from the text")
    date: str | None = Field(default=None, pattern=ISO_DATE, description="buy (or short) date")
    sell_date: str | None = Field(default=None, pattern=ISO_DATE, description="sell (or cover) date")
    still_holding: bool | None = Field(default=None, description="measure up to the latest close")
    direction: Literal["long", "short"] | None = None
    horizon_days: int | None = Field(default=None, ge=1, le=MAX_STATED_DAYS, description="trading days held, if no sell date")


class ParseIn(Input):
    text: str = Field(min_length=1, max_length=2000)


class ParseOut(BaseModel):
    ticker: str | None
    name: str | None = Field(description="the listing's name in the symbol directory")
    date: str | None
    sell_date: str | None
    still_holding: bool | None
    direction: str | None
    horizon_days: int | None
    reasoning: str | None
    notes: list[str] = Field(default_factory=list, description="anything that needs checking before the review")
    parser: str


class Understood(BaseModel):
    ticker: str | None
    company: str | None
    date: str | None
    direction: str | None
    horizon_days: int | None
    reasoning: str | None
    sell_date: str | None = None
    still_holding: bool | None = None
    name: str | None = Field(default=None, description="the listing's name in the symbol directory")
    entry_date: str | None = Field(default=None, description="first trading day on or after the date")
    exit_date: str | None = None
    parser: str = Field(description="which parser read the text")


class BiasSpan(BaseModel):
    label: str
    evidence: str
    confidence: float
    start: int | None = Field(description="character offsets of the evidence in the reasoning, for highlighting")
    end: int | None


class Surface(BaseModel):
    days: list[int]
    returns: list[float]
    density: list[list[float]] = Field(description="[day][return bucket] probability; each row sums to 1")
    p05: list[float]
    p50: list[float]
    p95: list[float]
    realized: list[float] = Field(description="your position's directional return after each day held")


class MarketContext(BaseModel):
    spy_return: float | None = Field(description="S&P 500 ETF over the same days")
    universe_return: float | None = Field(description="equal weight of the research stocks over the same days")


class ReviewOut(BaseModel):
    status: Literal["ok", "needs_input", "unsupported", "rejected"]
    messages: list[str] = Field(default_factory=list, description="what could not be done, and why")
    missing: list[str] = Field(default_factory=list, description="fields to fill in before the review can run")
    suggestions: list[str] = Field(default_factory=list)
    understood: Understood
    what_you_did: str | None = None
    outcome: LuckTestOut | None = None
    surface: Surface | None = None
    biases: list[BiasSpan] = Field(default_factory=list)
    bias_source: str | None = None
    market: MarketContext | None = None
    explanation: str | None = None
    explanation_source: Literal["llm", "template"] | None = None
    disclaimer: str = "Paper analysis of a past trade. Not investment advice, not a forecast, not a judgement of the person."
