"""Inputs and outputs of the regime capabilities."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from minifinrl.platform.schemas import Input


class LuckTestIn(Input):
    ticker: str
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="entry day (YYYY-MM-DD); snapped to the next trading day")
    direction: Literal["long", "short"] = "long"
    horizon_days: int = Field(default=20, ge=1, le=60, description="trading days held")
    n_paths: int = Field(default=1000, ge=200, le=5000)
    seed: int = Field(default=0, ge=0)


class LuckTestOut(BaseModel):
    ticker: str
    direction: str
    entry_date: str
    exit_date: str
    horizon_days: int
    realized_return: float
    synthetic_median: float
    band_low: float = Field(description="5th percentile of outcomes the regime model considers plausible")
    band_high: float = Field(description="95th percentile")
    prob_profit: float = Field(description="share of synthetic outcomes above 0; near 0.5 means the model has no view")
    percentile: float = Field(description="where the realized outcome falls among synthetic outcomes, 0-100")
    verdict: Literal["unusually_bad", "within_luck_range", "unusually_good"]
    regime: str = Field(description="most likely market regime on the entry day, from data up to that day only")
    regime_probs: dict[str, float]
    n_paths: int
    seed: int
    fit_start: str
    fit_end: str
    note: str = ("Locates the outcome within the range of luck under a regime model with no predictive "
                 "signal. Not a forecast, not a measure of skill, not investment advice.")
