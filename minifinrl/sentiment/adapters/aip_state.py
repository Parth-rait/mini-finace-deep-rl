"""
LLM state reader through aip: the author's mood (bullish, bearish, neutral)
and the state signals in their words (fear, greed, regret, frustration,
certainty, urgency, herd talk).

Same guards as the bias classifier: the text is passed as delimited
untrusted input, and every signal must quote words that are in the text, or
it is dropped. Heat is not asked of the model; sentiment.state computes it
from the signals.
"""

from __future__ import annotations

import logging
from typing import Callable

from pydantic import BaseModel, Field

from minifinrl.platform.errors import BudgetExhausted, ClassificationFailed
from minifinrl.sentiment.adapters.aip import _norm
from minifinrl.sentiment.ports import Mood, StateKind, StateReading, StateSignal

log = logging.getLogger("mini_finrl.adapters.aip_state")

SYSTEM = """You read a short text someone wrote about a trade or a market, and describe the writer's own state. You describe the text, not the person.

mood: the writer's own stance on the asset.
- bullish: expects it to rise, is buying, holding long, or cheering it on.
- bearish: expects it to fall, is selling, shorting, or warning others off.
- neutral: no stance, a question, news without an opinion, or mixed.
mood_strength: 0 (barely leans) to 1 (as strong as it gets).

signals, each only if the writer's own words show it:
- fear: being scared, nervous, panicking about the position or the market.
- greed: excitement about easy or large gains, getting rich, euphoria.
- regret: wishing they had acted differently, feeling they missed out.
- frustration: anger or exasperation at the market, a stock, or a loss.
- certainty: claiming the outcome is sure ("can't lose", "guaranteed", "100%").
- urgency: pressure to act now, before it's too late.
- herd: acting or arguing because others are doing it.

Rules:
- Only the writer's own state counts, not what they say about other people.
- Sarcasm or mockery is not the attitude it mocks.
- For each signal, quote the exact words from the text as evidence and give a strength from 0 to 1.
- Give each kind of signal at most once. Return no signals if none are shown."""


class _Signal(BaseModel):
    kind: StateKind
    evidence: str = Field(description="exact words copied from the text")
    strength: float = Field(ge=0.0, le=1.0)


class _Answer(BaseModel):
    mood: Mood
    mood_strength: float = Field(ge=0.0, le=1.0)
    signals: list[_Signal] = Field(default_factory=list)


class AipStateReader:
    def __init__(self, tier: str = "SMALL", *, structured: Callable | None = None,
                 delimit: Callable[[str, str], str] | None = None,
                 budget_error: type[BaseException] | None = None,
                 output_error: type[BaseException] | None = None):
        self.tier = tier
        self.name = f"aip-state-{tier.lower()}-v1"
        self._structured, self._delimit = structured, delimit
        self._budget_error, self._output_error = budget_error, output_error
        self.dropped_evidence = 0

    def _load(self) -> None:
        if self._structured is not None:
            return
        import aip
        from aip.cost import BudgetExceeded
        from aip.guards import delimit_untrusted
        from aip.llm import StructuredOutputError

        self._structured, self._delimit = aip.structured, delimit_untrusted
        self._budget_error, self._output_error = BudgetExceeded, StructuredOutputError

    def read(self, text: str) -> StateReading:
        self._load()
        prompt = self._delimit(text, "TEXT") if self._delimit else text
        try:
            answer = self._structured(prompt, schema=_Answer, system=SYSTEM, tier=self.tier)
        except Exception as exc:
            if self._budget_error and isinstance(exc, self._budget_error):
                raise BudgetExhausted(str(exc)) from exc
            if self._output_error and isinstance(exc, self._output_error):
                raise ClassificationFailed(str(exc)) from exc
            raise
        kept, seen = [], set()
        for s in answer.signals:
            if s.kind in seen:
                continue
            if _norm(s.evidence) and _norm(s.evidence) in _norm(text):
                kept.append(StateSignal(kind=s.kind, evidence=s.evidence, strength=s.strength))
                seen.add(s.kind)
            else:
                self.dropped_evidence += 1
                log.warning("dropped %s: evidence %r is not in the text", s.kind.value, s.evidence[:60])
        return StateReading(mood=answer.mood, mood_strength=answer.mood_strength, signals=kept, reader=self.name)
