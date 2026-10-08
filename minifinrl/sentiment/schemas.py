"""Inputs and outputs of the sentiment capabilities."""

from __future__ import annotations


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
