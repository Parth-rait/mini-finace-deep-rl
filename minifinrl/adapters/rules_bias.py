"""
Keyword/regex bias classifier: no LLM, no network, deterministic.

Two jobs: it is what the offline `ci` profile plugs in, so the whole system
runs without keys; and it is the baseline the LLM classifier has to beat
on the hand-labelled set (Phase F2 reports Cohen's kappa for both). The
patterns are deliberately simple; anything they miss is the LLM's job.
"""

from __future__ import annotations

import re

from minifinrl.ports import BiasClassification, BiasLabel, BiasSignal

_PATTERNS: dict[BiasLabel, list[str]] = {
    BiasLabel.FOMO: [
        r"\bfomo\b", r"missing out", r"(can'?t|don'?t want to) miss", r"before it'?s too late",
        r"get in (now|before)", r"to the moon", r"everyone'?s? (is )?(up|making|getting rich)",
    ],
    BiasLabel.REVENGE_TRADING: [
        r"win (it|that|this) back", r"make (it|that|this) back", r"get (it|my money) back", r"\brevenge\b",
        r"doubl(e|ing) down", r"\brecoup", r"after (losing|that loss|the loss)",
    ],
    BiasLabel.OVERCONFIDENCE: [
        r"can'?t (lose|go wrong)", r"\bguaranteed\b", r"\ball[- ]in\b", r"sure thing", r"easy money",
        r"100% (sure|certain)", r"no way (it|this) (drops|goes down|fails)",
    ],
    BiasLabel.LOSS_AVERSION: [
        r"(hold|wait)(ing)? (until|till) (it'?s )?(back|even|break ?even)", r"what i paid",
        r"back to (my )?(entry|cost|purchase price)", r"(won'?t|can'?t|not going to) sell at a loss", r"bag ?hold",
    ],
    BiasLabel.HERDING: [
        r"(the )?(whole|entire) (sub|forum|internet|chat|group)", r"everyone (is|'s) (buying|selling|in)",
        r"all my friends", r"\beverybody\b", r"following the crowd", r"(reddit|twitter|the sub) (is|says)",
    ],
    BiasLabel.ANCHORING: [
        r"(it )?(was|used to be) \$?\d[\d,.]*\b", r"52[- ]week high", r"down \d+% from",
        r"below (its|the) (high|peak|ath)", r"compared to (its|the) (high|peak)",
    ],
}
_COMPILED = {label: [re.compile(p, re.IGNORECASE) for p in pats] for label, pats in _PATTERNS.items()}
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")


def _sentence_around(text: str, start: int, end: int) -> str:
    for m in _SENTENCE.finditer(text):
        if m.start() <= start and end <= m.end():
            return m.group(0).strip()
    return text[start:end]


class RulesBiasClassifier:
    name = "rules-v1"

    def classify(self, text: str) -> BiasClassification:
        signals = []
        for label, patterns in _COMPILED.items():
            hits = [m for p in patterns if (m := p.search(text))]
            if hits:
                first = min(hits, key=lambda m: m.start())
                # more distinct cues -> more confident, capped: rules are a baseline
                signals.append(BiasSignal(
                    label=label, evidence=_sentence_around(text, first.start(), first.end()),
                    confidence=min(0.9, 0.5 + 0.15 * (len(hits) - 1)),
                ))
        return BiasClassification(signals=signals, classifier=self.name)
