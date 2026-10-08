"""
Score a state reader's mood against labelled posts.

    accuracy          share of posts whose mood matches the label
    per mood          precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN)
    macro F1          mean F1 over bullish, bearish, neutral
    direction         over posts labelled bullish or bearish only: share given the right
                      direction (a neutral answer counts as wrong), and, of the answers
                      that took a side, the share on the right side
"""

from __future__ import annotations

from minifinrl.platform.errors import ClassificationFailed
from minifinrl.sentiment.ports import StateReader

MOODS = ["bullish", "bearish", "neutral"]


def evaluate(rows: list[dict], reader: StateReader) -> dict:
    preds, failures = [], []
    for r in rows:
        try:
            preds.append(reader.read(r["text"]).mood.value)
        except ClassificationFailed:
            preds.append("neutral")  # counted, and reported as a failure
            failures.append(r["id"])
    refs = [r["mood"] for r in rows]
    confusion = {a: {p: sum(1 for x, y in zip(refs, preds) if x == a and y == p) for p in MOODS} for a in MOODS}
    per = []
    for m in MOODS:
        tp = confusion[m][m]
        fp = sum(confusion[a][m] for a in MOODS if a != m)
        fn = sum(confusion[m][p] for p in MOODS if p != m)
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        per.append({"mood": m, "support": tp + fn, "precision": prec, "recall": rec, "f1": f1})
    sided = [(x, y) for x, y in zip(refs, preds) if x != "neutral"]
    took_side = [(x, y) for x, y in sided if y != "neutral"]
    f1s = [p["f1"] for p in per if p["f1"] is not None]
    return {
        "reader": reader.name, "texts": len(rows), "failures": failures,
        "accuracy": sum(x == y for x, y in zip(refs, preds)) / len(rows),
        "macro_f1": sum(f1s) / len(f1s) if f1s else None,
        "per_mood": per, "confusion": confusion,
        "direction_accuracy": sum(x == y for x, y in sided) / len(sided) if sided else None,
        "side_coverage": len(took_side) / len(sided) if sided else None,
        "side_precision": sum(x == y for x, y in took_side) / len(took_side) if took_side else None,
    }
