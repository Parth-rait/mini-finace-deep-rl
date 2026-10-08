"""Scoring the bias classifier against hand labels: kappa and F1 against
hand-computed values, file validation, and the capability."""

from __future__ import annotations

import json

import pytest

from minifinrl.sentiment.rules import RulesBiasClassifier
from minifinrl.sentiment.evaluation import _prf, cohen_kappa, evaluate, load_labels
from minifinrl.platform.capabilities import CapabilityRegistry, CapabilityUnavailable


def test_kappa_known_values():
    # p_o = 0.75; p_yes = 0.5 and 0.25 -> p_e = 0.5; kappa = (0.75 - 0.5) / 0.5
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == pytest.approx(-1.0)
    assert cohen_kappa([0, 0, 0], [0, 0, 0]) is None  # label never used: undefined


def test_f1_conventions():
    assert _prf([True, False], [False, False])["f1"] == 0.0  # occurs, never found
    assert _prf([False, False], [False, False])["f1"] is None  # nowhere to score
    r = _prf([True, True, False, False], [True, False, True, False])
    assert (r["precision"], r["recall"], r["f1"]) == (0.5, 0.5, 0.5)


def _write(tmp_path, rows):
    p = tmp_path / "labels.jsonl"
    p.write_text("// comment line\n" + "\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_evaluate_rules_classifier(tmp_path):
    rows = [
        {"id": "a", "text": "The whole sub is buying, all in.", "labels": ["herding", "overconfidence"], "labels_b": ["herding"]},
        {"id": "b", "text": "Lost big, doubling down to win it back.", "labels": ["revenge_trading"], "labels_b": ["revenge_trading"]},
        {"id": "c", "text": "Rebalanced to target weights.", "labels": [], "labels_b": []},
        {"id": "d", "text": "Everyone in the chat is up big, getting in now.", "labels": ["fomo"]},
    ]
    out = evaluate(load_labels(_write(tmp_path, rows)), RulesBiasClassifier())
    by = {r["label"]: r for r in out["per_label"]}
    assert out["texts"] == 4 and out["double_labelled"] == 3
    assert by["herding"]["tp"] == 1 and by["revenge_trading"]["f1"] == 1.0
    assert out["exact_match"] == pytest.approx(0.75)  # d: "getting in now" is not one of the rule phrases
    assert by["fomo"]["fn"] == 1 and by["fomo"]["f1"] == 0.0
    assert by["anchoring"]["f1"] is None
    assert by["overconfidence"]["kappa_humans"] is not None


def test_bad_files_fail_loudly(tmp_path):
    with pytest.raises(ValueError, match="unknown labels"):
        load_labels(_write(tmp_path, [{"id": "x", "text": "hi", "labels": ["greed"]}]))
    with pytest.raises(ValueError, match="no labelled texts"):
        load_labels(_write(tmp_path, []))


def test_template_file_is_valid():
    from minifinrl.platform.settings import ROOT

    rows = load_labels(ROOT / "corpus" / "bias_labels.template.jsonl")
    assert len(rows) == 3 and all("example" in r["source"] for r in rows)


def test_capability(rw_engine, tmp_path):
    reg = CapabilityRegistry.from_engine(rw_engine)
    with pytest.raises(CapabilityUnavailable, match="LABELLING"):
        reg.invoke("evaluate_biases", {"labels_path": str(tmp_path / "missing.jsonl")}, interface="cli")
    p = _write(tmp_path, [{"id": "a", "text": "can't lose, all in", "labels": ["overconfidence"]}])
    out = reg.invoke("evaluate_biases", {"labels_path": str(p)}, interface="cli")
    assert out.classifier == "rules-v1" and out.exact_match == 1.0
