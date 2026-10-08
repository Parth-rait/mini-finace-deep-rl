"""
Sentiment capabilities: reading trading text for the writer's state (mood,
fear, greed, urgency and the rest, with a heat score) and for behavioural-bias
signals, and scoring the readers against labelled text.
"""

from __future__ import annotations

from pathlib import Path

from minifinrl.platform.capabilities import CapabilityUnavailable, capability
from minifinrl.platform.log import get_logger
from minifinrl.platform.settings import ROOT
from minifinrl.sentiment import corpus, evaluation, mood_eval, state
from minifinrl.sentiment.ports import BiasClassification, BiasClassifier, StateReader, StateReading
from minifinrl.sentiment.schemas import BiasEvalIn, BiasEvalOut, MoodEvalIn, MoodEvalOut, TextIn
from minifinrl.sentiment.state_rules import RulesStateReader

log = get_logger(__name__)


class SentimentService:
    def __init__(self, cfg, bias: BiasClassifier | None = None, bias_fallback: BiasClassifier | None = None,
                 state_reader: StateReader | None = None, state_fallback: StateReader | None = None):
        self.cfg = cfg
        self.bias = bias
        self.bias_fallback = bias_fallback
        self.state_reader = state_reader
        self.state_fallback = state_fallback

    def read_state_with_fallback(self, text: str) -> StateReading | None:
        """The configured reader, then the fallback if it fails; scored for heat.
        None if neither is available."""
        for reader in (self.state_reader, self.state_fallback):
            if reader is None:
                continue
            try:
                reading = reader.read(text)
            except Exception as exc:
                log.warning("state reader %s failed: %s: %s", reader.name, type(exc).__name__, exc)
                continue
            if reader is not self.state_reader:
                reading = reading.model_copy(update={"reader": f"{reader.name} (fallback: the language model was unavailable)"})
            return state.scored(reading)
        return None

    def classify_with_fallback(self, text: str) -> tuple[BiasClassification | None, str | None]:
        """(result, source). The configured classifier, then the fallback if it
        fails (LLM down or out of budget); (None, None) if neither works."""
        for clf in (self.bias, self.bias_fallback):
            if clf is None:
                continue
            try:
                result = clf.classify(text)
            except Exception as exc:
                log.warning("bias classifier %s failed: %s: %s", clf.name, type(exc).__name__, exc)
                continue
            source = clf.name
            if source != (self.bias.name if self.bias else source):
                source += " (fallback: the language model was unavailable)"
            return result, source
        return None, None

    @capability("read_state", TextIn, StateReading, effect="llm", budget_ms=8000)
    def read_state(self, req: TextIn) -> StateReading:
        """The writer's mood and emotional state in a piece of trading text, each signal quoted, with a heat score. Describes the text, not the person."""
        reading = self.read_state_with_fallback(req.text)
        if reading is None:
            raise CapabilityUnavailable(f"no state reader configured in profile '{self.cfg.profile}'")
        return reading

    @capability("classify_biases", TextIn, BiasClassification, effect="llm", budget_ms=8000)
    def classify_biases(self, req: TextIn) -> BiasClassification:
        """Behavioural-bias signals in a piece of trading text, with the quoted evidence. Describes the text, not the person."""
        if self.bias is None:
            raise CapabilityUnavailable(f"no bias classifier configured in profile '{self.cfg.profile}'")
        return self.bias.classify(req.text)

    @capability("evaluate_biases", BiasEvalIn, BiasEvalOut, effect="batch")
    def evaluate_biases(self, req: BiasEvalIn) -> BiasEvalOut:
        """Score the configured bias classifier against hand-labelled texts: precision, recall, F1 and Cohen's kappa per bias."""
        if self.bias is None:
            raise CapabilityUnavailable(f"no bias classifier configured in profile '{self.cfg.profile}'")
        path = Path(req.labels_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise CapabilityUnavailable(f"no labels at {path}: see corpus/LABELLING.md")
        return BiasEvalOut(**evaluation.evaluate(evaluation.load_labels(path), self.bias))

    @capability("evaluate_mood", MoodEvalIn, MoodEvalOut, effect="batch")
    def evaluate_mood(self, req: MoodEvalIn) -> MoodEvalOut:
        """Score a state reader's mood against StockTwits posts tagged bullish or bearish by their authors (untagged posts as neutral)."""
        reader = RulesStateReader() if req.reader == "rules" else self.state_reader
        if reader is None:
            raise CapabilityUnavailable(f"no state reader configured in profile '{self.cfg.profile}'")
        rows = corpus.mood_sample(req.n_per_class, req.seed)
        return MoodEvalOut(**mood_eval.evaluate(rows, reader))
