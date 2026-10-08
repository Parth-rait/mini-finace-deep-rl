"""What the sentiment feature needs from a classifier, as an interface.

Implementations: sentiment.rules (no model) and sentiment.adapters.aip
(an LLM through aip). system.py picks one.

Framing rule, enforced by the schema: biases are detected in *text*, with
the quoted evidence, never attributed to a person."""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


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
