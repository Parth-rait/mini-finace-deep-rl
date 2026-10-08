"""
Heat: how emotionally charged a piece of text is, from its state signals.

Computed by code, not by a model, so the same signals always give the same
score and the rule can be read in one line:

    heat = 1 - product over kinds of (1 - weight[kind] * strength[kind])

using each kind's strongest signal. Independent signals compound (two medium
signals are hotter than one), and no single kind can reach 1 on its own
unless its weight is 1. Mood is not part of heat: a strong opinion is not
the same thing as an emotional state.
"""

from __future__ import annotations

from minifinrl.sentiment.ports import StateKind, StateReading

WEIGHTS = {
    StateKind.GREED: 0.6, StateKind.FEAR: 0.6, StateKind.FRUSTRATION: 0.6,
    StateKind.URGENCY: 0.5, StateKind.CERTAINTY: 0.5, StateKind.HERD: 0.4, StateKind.REGRET: 0.3,
}
WARM, HOT = 0.25, 0.55  # level thresholds


def heat(reading: StateReading) -> float:
    strongest: dict[StateKind, float] = {}
    for s in reading.signals:
        strongest[s.kind] = max(strongest.get(s.kind, 0.0), s.strength)
    cool = 1.0
    for kind, strength in strongest.items():
        cool *= 1.0 - WEIGHTS[kind] * strength
    return round(1.0 - cool, 4)


def level(h: float) -> str:
    return "hot" if h >= HOT else ("warm" if h >= WARM else "calm")


def scored(reading: StateReading) -> StateReading:
    h = heat(reading)
    return reading.model_copy(update={"heat": h, "level": level(h)})
