"""What the trade review needs from outside: reading a trade written in
words, and explaining a result. Implementations: review.parse_rules and
review.adapters.aip."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from minifinrl.review.settings import MAX_STATED_DAYS


class ParsedTrade(BaseModel):
    """What a parser understood from a free-text trade. Anything it could not
    read with confidence stays None; it never guesses."""

    ticker: str | None = Field(default=None, description="exchange symbol, e.g. TSLA, if stated or unambiguous")
    company: str | None = Field(default=None, description="company or asset name as written, e.g. 'tesla'")
    date: str | None = Field(default=None, description="entry date as YYYY-MM-DD, relative dates resolved against today")
    direction: str | None = Field(default=None, description="long (bought) or short (shorted, bought puts)")
    horizon_days: int | None = Field(default=None, ge=1, le=MAX_STATED_DAYS, description="trading days held: a week is 5, a month 21")
    sell_date: str | None = Field(default=None, description="exit date as YYYY-MM-DD, if stated")
    still_holding: bool | None = Field(default=None, description="true if they say they still hold it")
    reasoning: str | None = Field(default=None, description="the author's stated reason, copied from the text")
    unclear: list[str] = Field(default_factory=list, description="short notes on anything ambiguous")


@runtime_checkable
class TradeParser(Protocol):
    name: str

    def parse(self, text: str, today: str) -> ParsedTrade: ...


@runtime_checkable
class TradeExplainer(Protocol):
    """Writes a short plain-language explanation from computed results only."""

    name: str

    def explain(self, facts: dict) -> str: ...
