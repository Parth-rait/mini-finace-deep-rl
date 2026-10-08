"""
Sentiment capabilities: reading behavioural-bias signals in trading text,
and scoring the classifier against hand labels.
"""

from __future__ import annotations

from pathlib import Path

from minifinrl.platform.capabilities import CapabilityUnavailable, capability
from minifinrl.platform.log import get_logger
from minifinrl.platform.settings import ROOT
from minifinrl.sentiment import evaluation
from minifinrl.sentiment.ports import BiasClassification, BiasClassifier
from minifinrl.sentiment.schemas import BiasEvalIn, BiasEvalOut, TextIn

log = get_logger(__name__)


class SentimentService:
    def __init__(self, cfg, bias: BiasClassifier | None = None, bias_fallback: BiasClassifier | None = None):
        self.cfg = cfg
        self.bias = bias
        self.bias_fallback = bias_fallback

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
