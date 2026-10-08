"""
Rules state reader: no model, no network. Word and emoji lists for mood and
for each state signal. The fallback when the LLM reader is unavailable, and
the floor it is measured against (E09).

Mood: bullish cues minus bearish cues, with a negation just before a cue
("not bullish") flipping it. Signals: every match is evidence; strength
grows with the number of matches of that kind.
"""

from __future__ import annotations

import re

from minifinrl.sentiment.ports import Mood, StateKind, StateReading, StateSignal


def _rx(words: list[str]) -> re.Pattern:
    """Whole words for text, anywhere for emoji."""
    parts = [re.escape(w) if not w.isascii() else rf"(?<![\w']){re.escape(w)}(?![\w'])" for w in words]
    return re.compile("|".join(sorted(parts, key=len, reverse=True)), re.IGNORECASE)


BULLISH = _rx([
    "moon", "mooning", "to the moon", "bull", "bullish", "buy the dip", "btfd", "btd", "long", "calls", "rally",
    "breakout", "pump", "pumping", "ath", "new high", "undervalued", "lfg", "send it", "squeeze", "rip", "ripping",
    "green", "going up", "up big", "loading up", "accumulate", "accumulating", "hodl", "diamond hands", "cheap",
    "bargain", "buying more", "bounce", "recovery", "strong",
    "🚀", "🌙", "💎", "🙌", "📈", "🐂", "🔥", "💪", "🤑", "🟢", "⬆️",
])
BEARISH = _rx([
    "dump", "dumping", "crash", "crashing", "bear", "bearish", "short", "shorting", "puts", "rug", "rug pull",
    "rekt", "tank", "tanking", "bleed", "bleeding", "overvalued", "sell off", "selloff", "red", "going down",
    "drop", "dropping", "dead", "scam", "bubble", "collapse", "capitulation", "falling", "down bad", "bagholder",
    "bag holder", "weak", "sell", "selling", "exit", "get out",
    "📉", "🐻", "🩸", "💀", "🔻", "⬇️", "🔴", "😭",
])
NEGATION = re.compile(r"\b(not|no|isn'?t|aren'?t|wasn'?t|never|don'?t|doesn'?t|won'?t|hardly)\s+(\w+\s+)?$", re.IGNORECASE)

SIGNALS: dict[StateKind, re.Pattern] = {
    StateKind.FEAR: _rx(["scared", "afraid", "nervous", "worried", "worry", "panic", "panicking", "terrified", "anxious",
                         "can't sleep", "cant sleep", "freaking out", "scary", "fear", "😬", "😱", "😰", "😨"]),
    StateKind.GREED: _rx(["easy money", "free money", "get rich", "rich", "lambo", "millionaire", "10x", "100x", "1000x",
                          "print money", "printing", "gains", "retire", "life changing", "🤑", "💰", "💸", "🤤"]),
    StateKind.REGRET: _rx(["should have", "should've", "shouldn't have", "wish i had", "wish i'd", "missed", "regret",
                           "kicking myself", "too late", "if only", "could have"]),
    StateKind.FRUSTRATION: _rx(["wtf", "ffs", "bullshit", "sick of", "hate", "fml", "angry", "screw this", "rigged",
                                "manipulated", "ridiculous", "tired of", "🤬", "😡", "😤", "🖕"]),
    StateKind.CERTAINTY: _rx(["can't lose", "cant lose", "guaranteed", "100%", "definitely", "no way it", "sure thing",
                              "can't go down", "cant go down", "literally can't", "trust me", "no brainer", "easy",
                              "for sure", "obviously", "inevitable"]),
    StateKind.URGENCY: _rx(["right now", "now or never", "before it's too late", "before its too late", "last chance",
                            "hurry", "asap", "don't miss", "dont miss", "quick", "immediately", "tonight", "today",
                            "this week", "now", "fast", "⏰", "🚨"]),
    StateKind.HERD: _rx(["everyone", "everybody", "all my friends", "reddit", "wsb", "wallstreetbets", "twitter",
                         "the whole sub", "people are buying", "the crowd", "all in on it", "whole world", "everyones",
                         "everyone's", "all the apes", "apes"]),
}


def _strength(n: int) -> float:
    return min(1.0, 0.5 + 0.25 * (n - 1))


class RulesStateReader:
    name = "rules-state-v1"

    def read(self, text: str) -> StateReading:
        score = 0
        for pattern, sign in ((BULLISH, 1), (BEARISH, -1)):
            for m in pattern.finditer(text):
                negated = bool(NEGATION.search(text[max(0, m.start() - 25):m.start()]))
                score += -sign if negated else sign
        mood = Mood.BULLISH if score > 0 else Mood.BEARISH if score < 0 else Mood.NEUTRAL
        matches = [(kind, m.start(), m.end(), m.group(0)) for kind, pattern in SIGNALS.items() for m in pattern.finditer(text)]
        # a phrase inside a longer phrase of another kind belongs to the longer one:
        # "too late" in "before it's too late" is urgency, not regret
        matches = [a for a in matches if not any(b[0] != a[0] and b[1] <= a[1] and a[2] <= b[2] and (b[2] - b[1]) > (a[2] - a[1])
                                                 for b in matches)]
        signals = []
        for kind in SIGNALS:
            hits = [m[3] for m in matches if m[0] == kind]
            if hits:
                signals.append(StateSignal(kind=kind, evidence=hits[0], strength=_strength(len(hits))))
        return StateReading(mood=mood, mood_strength=min(1.0, abs(score) / 3), signals=signals, reader=self.name)
