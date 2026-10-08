"""What the sentiment feature needs from a classifier, as an interface.

Implementations: sentiment.rules (no model) and sentiment.adapters.aip
(an LLM through aip). system.py picks one.

Framing rule, enforced by the schema: biases are detected in *text*, with
the quoted evidence, never attributed to a person."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Protocol, runtime_checkable

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


# ---- trader state: mood and emotional signals ---------------------------------------------

class Mood(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class StateKind(str, Enum):
    """Signals of the state someone is in when they write about a trade.
    The emotions people trade on badly, and the habits of speech that go
    with acting in a hurry."""

    FEAR = "fear"  # scared, panicking, can't sleep
    GREED = "greed"  # easy money, getting rich, euphoria
    REGRET = "regret"  # should have, missed it, too late
    FRUSTRATION = "frustration"  # anger at the market or at a loss
    CERTAINTY = "certainty"  # can't lose, guaranteed, 100%
    URGENCY = "urgency"  # now, before it's too late, last chance
    HERD = "herd"  # everyone is buying, the whole sub


class StateSignal(BaseModel):
    kind: StateKind
    evidence: str = Field(min_length=1, max_length=200, description="the words in the text that show it, copied exactly")
    strength: float = Field(ge=0.0, le=1.0, description="how strongly the text shows it")


class StateReading(BaseModel):
    mood: Mood
    mood_strength: float = Field(ge=0.0, le=1.0, description="0 = no lean, 1 = as strong as it gets")
    signals: list[StateSignal] = Field(default_factory=list)
    heat: float = Field(default=0.0, ge=0.0, le=1.0, description="how emotionally charged, computed from the signals")
    level: Literal["calm", "warm", "hot"] = "calm"
    reader: str = Field(description="which implementation read the text")
    note: str = "Describes the text, not the person who wrote it. Not a psychological assessment."


@runtime_checkable
class StateReader(Protocol):
    """Reads mood and state signals from text. Implementations fill mood and
    signals; heat and level are computed afterwards by sentiment.state."""

    name: str

    def read(self, text: str) -> StateReading: ...
