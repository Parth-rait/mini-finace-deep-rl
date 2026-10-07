"""
Score a bias classifier against hand labels.

Input: a JSONL file, one text per line (see corpus/LABELLING.md):

    {"id": "t001", "text": "...", "labels": ["fomo", "herding"], "labels_b": ["fomo"], "source": "..."}

`labels` are the reference labels; `labels_b` (optional) a second person's
labels for the same text, used to measure how much people agree before
asking how well a model does. Each of the six biases is scored as its own
yes/no question over all texts:

    precision = TP / (TP + FP)     recall = TP / (TP + FN)     F1 = 2TP / (2TP + FP + FN)
    Cohen's kappa = (p_o - p_e) / (1 - p_e)
        p_o = share of texts where the two sides agree (both yes or both no)
        p_e = p_yes(a) p_yes(b) + p_no(a) p_no(b)   (agreement expected by chance)

Kappa is undefined (None) when p_e = 1, e.g. a label nobody ever used.
"""

from __future__ import annotations

import json
from pathlib import Path

from minifinrl.ports import BiasClassifier, BiasLabel, ClassificationFailed

LABELS = [b.value for b in BiasLabel]


def load_labels(path: str | Path) -> list[dict]:
    rows = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        r = json.loads(line)
        if not isinstance(r.get("labels"), list):
            raise ValueError(f"line {n}: add a \"labels\" list (use [] when no bias is signalled)")
        for key in ("labels", "labels_b"):
            unknown = set(r.get(key) or []) - set(LABELS)
            if unknown:
                raise ValueError(f"line {n}: unknown labels {sorted(unknown)} (allowed: {LABELS})")
        if not r.get("text"):
            raise ValueError(f"line {n}: empty text")
        rows.append(r)
    if not rows:
        raise ValueError(f"no labelled texts in {path}")
    return rows


def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n == 0 or n != len(b):
        raise ValueError("kappa needs two equally long, non-empty label lists")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def _prf(ref: list[bool], pred: list[bool]) -> dict:
    tp = sum(r and p for r, p in zip(ref, pred))
    fp = sum(p and not r for r, p in zip(ref, pred))
    fn = sum(r and not p for r, p in zip(ref, pred))
    precision = tp / (tp + fp) if tp + fp else None  # undefined when the label is never predicted
    recall = tp / (tp + fn) if tp + fn else None  # undefined when the label never occurs
    # F1 = 2TP / (2TP + FP + FN): 0 when the label occurs but is never found;
    # None only when it appears in neither the reference nor the predictions
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _predict(classifier: BiasClassifier, text: str) -> set[str] | None:
    try:
        return {s.label.value for s in classifier.classify(text).signals}
    except ClassificationFailed:
        return None


def evaluate(rows: list[dict], classifier: BiasClassifier) -> dict:
    raw = [_predict(classifier, r["text"]) for r in rows]
    failures = [r["id"] for r, p in zip(rows, raw) if p is None]
    # a failed text counts as "predicted nothing" in the scores, and is reported
    preds = [p if p is not None else set() for p in raw]
    refs = [set(r["labels"]) for r in rows]
    second = [set(r["labels_b"]) for r in rows if "labels_b" in r]
    per_label = []
    for label in LABELS:
        ref = [label in s for s in refs]
        pred = [label in s for s in preds]
        row = {"label": label, "support": sum(ref), **_prf(ref, pred), "kappa_model": cohen_kappa(ref, pred)}
        if second:
            both = [r for r in rows if "labels_b" in r]
            row["kappa_humans"] = cohen_kappa([label in r["labels"] for r in both], [label in r["labels_b"] for r in both])
        per_label.append(row)
    return {
        "classifier": classifier.name,
        "texts": len(rows),
        "failures": failures,
        "double_labelled": len(second),
        "exact_match": sum(p == r for p, r in zip(preds, refs)) / len(rows),
        "macro_f1": _mean([r["f1"] for r in per_label]),
        "macro_kappa_model": _mean([r["kappa_model"] for r in per_label]),
        "macro_kappa_humans": _mean([r.get("kappa_humans") for r in per_label]) if second else None,
        "per_label": per_label,
    }
