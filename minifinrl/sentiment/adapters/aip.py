"""
LLM bias classifier, through the aip toolkit (aip.structured: JSON schema in
the prompt, Pydantic validation, repair loop, cache, cost and trace records).

The instructions are the labelling rules in corpus/LABELLING.md, so the model
and the people labelling follow the same written definitions. No labelled
text is used as an example in the prompt: the evaluation set stays unseen.

Two guards:
- the post is passed as delimited untrusted text (aip.guards.delimit_untrusted),
  so instructions inside a post are treated as content;
- every signal must quote words that appear in the post; a signal whose
  evidence isn't in the text is dropped (and counted), so the model can't
  justify a label with words the author never wrote.

aip is imported lazily, on first use, after system.py has pointed its cache
and trace directories at this repo.
"""

from __future__ import annotations

import logging
import re
from typing import Callable

from pydantic import BaseModel, Field

from minifinrl.sentiment.ports import BiasClassification, BiasLabel, BiasSignal
from minifinrl.platform.errors import BudgetExhausted, ClassificationFailed

log = logging.getLogger("mini_finrl.adapters.aip_bias")

SYSTEM = """You label short posts about trading for behavioural-bias signals. You label the text, not the person.

Labels:
- fomo: urgency to enter because others are already gaining.
- revenge_trading: trading to win back a recent loss, or trading driven by anger at a loss.
- overconfidence: certainty about the outcome with the risk ignored ("can't lose", "all in", "trust me").
- loss_aversion: refusing to sell at a loss, waiting to get back to break even.
- herding: doing something because the crowd is doing it.
- anchoring: judging the current price against a past reference (a past price, the purchase price, a previous high).

Rules:
- Only the author's own reasoning counts. A bias the author describes in, or attributes to, other people does not.
- Sarcasm or mockery of an attitude is not that attitude.
- Going against the crowd is not herding.
- A future price target is not anchoring.
- A post can have several labels or none. Return an empty list when no bias is signalled.

For each label you give, quote the exact words from the post that signal it as evidence, and give a confidence between 0 and 1."""


class _Signal(BaseModel):
    label: BiasLabel
    evidence: str = Field(description="exact words copied from the post")
    confidence: float = Field(ge=0.0, le=1.0)


class _Answer(BaseModel):
    signals: list[_Signal] = Field(default_factory=list)


def _norm(s: str) -> str:
    s = s.casefold().replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip(" \"'.…")


class AipBiasClassifier:
    def __init__(self, tier: str = "SMALL", *, structured: Callable | None = None,
                 delimit: Callable[[str, str], str] | None = None,
                 budget_error: type[BaseException] | None = None,
                 output_error: type[BaseException] | None = None):
        self.tier = tier
        self.name = f"aip-{tier.lower()}-v1"
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

    def classify(self, text: str) -> BiasClassification:
        self._load()
        prompt = self._delimit(text, "POST") if self._delimit else text
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
            if _norm(s.evidence) and _norm(s.evidence) in _norm(text) and s.label not in seen:
                kept.append(BiasSignal(label=s.label, evidence=s.evidence, confidence=s.confidence))
                seen.add(s.label)
            elif s.label not in seen:
                self.dropped_evidence += 1
                log.warning("dropped %s: evidence %r is not in the post", s.label.value, s.evidence[:60])
        return BiasClassification(signals=kept, classifier=self.name)
