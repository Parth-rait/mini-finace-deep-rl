"""
Ports: what the Engine needs from outside the finance core, as interfaces.

The Engine is written against these Protocols only. Implementations live
in minifinrl/adapters/ (the only place aip is imported), and system.py
picks which one to plug in. So the finance logic never depends on an LLM,
and a test or the offline `ci` profile can plug in a rules-based one.

Framing rule, enforced by the schema: biases are detected in *text*, with
the quoted evidence, never attributed to a person.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class BudgetExhausted(RuntimeError):
    """An adapter's spending budget is used up. Adapters translate their
    library's own error (e.g. aip's BudgetExceeded) into this, so the
    engine layer never has to import that library."""


class ClassificationFailed(RuntimeError):
    """The classifier could not produce a valid answer for one text (e.g. an
    LLM's output failed validation after its repair attempts). Evaluation
    counts these instead of treating them as "no bias"."""


class BiasLabel(str, Enum):
    """Named biases from behavioural finance, each with a literature
    definition to label against."""

    FOMO = "fomo"
    REVENGE_TRADING = "revenge_trading"
    OVERCONFIDENCE = "overconfidence"
    LOSS_AVERSION = "loss_aversion"
    HERDING = "herding"
    ANCHORING = "anchoring"


class BiasSignal(BaseModel):
    label: BiasLabel
    evidence: str = Field(description="the exact span of the input text that signals the bias")
    confidence: float = Field(ge=0.0, le=1.0)


class BiasClassification(BaseModel):
    signals: list[BiasSignal]
    classifier: str = Field(description="which implementation produced this, e.g. 'rules-v1'")
    note: str = "Signals describe the text, not the person who wrote it. Not a psychological assessment."


@runtime_checkable
class BiasClassifier(Protocol):
    name: str

    def classify(self, text: str) -> BiasClassification: ...


# ---- trade review: reading a free-text trade, and explaining the result ----------------


class ParsedTrade(BaseModel):
    """What a parser understood from a free-text trade. Anything it could not
    read with confidence stays None; it never guesses."""

    ticker: str | None = Field(default=None, description="exchange symbol, e.g. TSLA, if stated or unambiguous")
    company: str | None = Field(default=None, description="company or asset name as written, e.g. 'tesla'")
    date: str | None = Field(default=None, description="entry date as YYYY-MM-DD, relative dates resolved against today")
    direction: str | None = Field(default=None, description="long (bought) or short (shorted, bought puts)")
    horizon_days: int | None = Field(default=None, ge=1, le=260, description="trading days held: a week is 5, a month 21")
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
