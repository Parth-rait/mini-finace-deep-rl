"""Inputs and outputs of the market capabilities."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from minifinrl.platform.schemas import ISO_DATE, Input


class SnapshotIn(Input):
    tickers: list[str] | None = Field(default=None, description="default: the configured universe")
    days: int = Field(default=5, ge=1, le=60)


class Bar(BaseModel):
    date: str
    tic: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class Snapshot(BaseModel):
    as_of: str
    bars: list[Bar]


class SymbolIn(Input):
    q: str = Field(min_length=1, max_length=60)
    limit: int = Field(default=8, ge=1, le=25)


class SymbolHit(BaseModel):
    symbol: str
    name: str
    exchange: str
    etf: bool


class SymbolsOut(BaseModel):
    results: list[SymbolHit]
    note: str | None = None


class TradingDaysIn(Input):
    date: str = Field(pattern=ISO_DATE, description="buy date")
    sell_date: str | None = Field(default=None, pattern=ISO_DATE, description="sell date; empty means still holding")


class TradingDaysOut(BaseModel):
    entry_date: str | None = Field(description="first trading day on or after the buy date")
    exit_date: str | None = Field(description="last trading day on or before the sell date, or the latest close")
    days: int | None = Field(description="trading days between them, counted on the exchange calendar")
    notes: list[str] = Field(default_factory=list)


class FetchIn(Input):
    mode: Literal["research", "live"] | None = Field(default=None, description="default: the profile's mode")
    start: str | None = None
    end: str | None = Field(default=None, description="exclusive; default TEST_END (research) or today (live)")
    offline: bool = Field(default=False, description="validate what's on disk, no network")


class FetchOut(BaseModel):
    ok: bool
    data_hash: str
    rows: int
    requested: dict
    providers: dict
    errors: list[str]
    warnings: list[str]


class CrosscheckIn(Input):
    candidate: str = Field(description="price provider to test, e.g. tiingo or alpaca")
    reference: str = "yahoo"
    start: str = Field(default="2023-01-01", pattern=r"^\d{4}-\d{2}-\d{2}$")
    end: str = Field(default="2026-06-30", pattern=r"^\d{4}-\d{2}-\d{2}$", description="exclusive")


class TickerAgreement(BaseModel):
    ticker: str
    ref_days: int
    cand_days: int
    overlap_days: int
    missing_in_cand: int
    extra_in_cand: int
    max_abs_diff: float | None
    p99_abs_diff: float | None = Field(description="99th percentile of |daily return difference|")
    return_corr: float | None
    flagged_days: list[dict] = Field(default_factory=list, description="days where |r_ref - r_cand| > 50 bps, largest first")
    agrees: bool


class CrosscheckOut(BaseModel):
    reference: str
    candidate: str
    agrees: bool
    tickers: list[TickerAgreement]
    rule: str = "p99 |r_ref - r_cand| <= 10 bps on overlapping days, and < 1% of reference days missing"
