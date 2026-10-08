"""Inputs and outputs of the sentiment capabilities."""

from __future__ import annotations


from typing import Literal

from pydantic import BaseModel, Field

from minifinrl.platform.schemas import Input


class TextIn(Input):
    text: str = Field(min_length=3, max_length=4000)


class BiasEvalIn(Input):
    labels_path: str = Field(default="corpus/bias_labels.jsonl", description="JSONL of hand-labelled texts (corpus/LABELLING.md)")


class LabelScore(BaseModel):
    label: str
    support: int = Field(description="texts with this label in the reference")
    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float | None
    f1: float | None
    kappa_model: float | None = Field(description="Cohen's kappa, classifier vs reference labels")
    kappa_humans: float | None = Field(default=None, description="Cohen's kappa, labeller A vs labeller B")


class BiasEvalOut(BaseModel):
    classifier: str
    texts: int
    failures: list[str] = Field(default_factory=list, description="ids where the classifier gave no valid answer (scored as no prediction)")
    double_labelled: int
    exact_match: float
    macro_f1: float | None
    macro_kappa_model: float | None
    macro_kappa_humans: float | None
    per_label: list[LabelScore]


class MoodEvalIn(Input):
    n_per_class: int = Field(default=100, ge=5, le=1000, description="posts per mood (bullish, bearish, neutral)")
    seed: int = 0
    reader: Literal["configured", "rules"] = Field(default="configured", description="the profile's reader, or the rules baseline")


class MoodScore(BaseModel):
    mood: str
    support: int
    precision: float | None
    recall: float | None
    f1: float | None


class MoodEvalOut(BaseModel):
    reader: str
    texts: int
    failures: list[str]
    accuracy: float
    macro_f1: float | None
    per_mood: list[MoodScore]
    confusion: dict[str, dict[str, int]] = Field(description="label -> predicted -> count")
    direction_accuracy: float | None = Field(description="posts labelled bullish/bearish given the right side (neutral counts wrong)")
    side_coverage: float | None = Field(description="of those, share where the reader took a side")
    side_precision: float | None = Field(description="of the answers that took a side, share on the right side")
