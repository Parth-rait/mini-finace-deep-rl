"""The LLM bias classifier with the model call replaced by a fake: evidence
grounding, error translation, failure counting in the evaluation, wiring.
No network, no API key, and the real aip is never imported here."""

from __future__ import annotations

import os
import sys

import pytest

from minifinrl.adapters.aip_bias import SYSTEM, AipBiasClassifier, _Answer, _Signal
from minifinrl.bias_eval import evaluate
from minifinrl.ports import BiasLabel, BudgetExhausted, ClassificationFailed


class Budget(Exception):
    pass


class BadOutput(Exception):
    pass


def fake(answer: _Answer | Exception, calls: list | None = None):
    def structured(prompt, *, schema, system, tier):
        if calls is not None:
            calls.append({"prompt": prompt, "schema": schema, "system": system, "tier": tier})
        if isinstance(answer, Exception):
            raise answer
        return answer
    return structured


def clf(answer, calls=None):
    return AipBiasClassifier(structured=fake(answer, calls), delimit=lambda t, label: f"<{label}>{t}</{label}>",
                             budget_error=Budget, output_error=BadOutput)


def sig(label, evidence, c=0.8):
    return _Signal(label=BiasLabel(label), evidence=evidence, confidence=c)


TEXT = "Lost big yesterday, doubling down to win it back. Can't lose."


def test_prompt_uses_the_labelling_rules_and_delimits_the_post():
    calls = []
    clf(_Answer(signals=[]), calls).classify(TEXT)
    c = calls[0]
    assert c["prompt"] == f"<POST>{TEXT}</POST>" and c["schema"] is _Answer and c["tier"] == "SMALL"
    assert "Sarcasm" in c["system"] and "not herding" in c["system"] and "future price target" in c["system"]
    assert "st-00" not in SYSTEM  # no labelled text used as an example


def test_evidence_must_be_in_the_post():
    c = clf(_Answer(signals=[
        sig("revenge_trading", "doubling down to win it back"),
        sig("overconfidence", "  can't LOSE. "),          # case/space/punctuation differences are fine
        sig("fomo", "everyone is buying"),                  # not in the post: dropped
        sig("revenge_trading", "lost big yesterday"),       # duplicate label: kept once
    ]))
    out = c.classify(TEXT)
    assert [s.label.value for s in out.signals] == ["revenge_trading", "overconfidence"]
    assert c.dropped_evidence == 1 and out.classifier == "aip-small-v1"


def test_errors_are_translated():
    with pytest.raises(BudgetExhausted):
        clf(Budget("spent")).classify(TEXT)
    with pytest.raises(ClassificationFailed):
        clf(BadOutput("invalid json after repairs")).classify(TEXT)
    with pytest.raises(ZeroDivisionError):  # anything else is a bug and propagates
        clf(ZeroDivisionError()).classify(TEXT)


def test_evaluation_counts_failures_instead_of_hiding_them():
    class Flaky:
        name = "flaky"

        def classify(self, text):
            if "fail" in text:
                raise ClassificationFailed("no valid answer")
            return clf(_Answer(signals=[sig("overconfidence", "all in")])).classify(text)

    rows = [{"id": "a", "text": "all in, easy", "labels": ["overconfidence"]},
            {"id": "b", "text": "this one will fail", "labels": ["fomo"]}]
    out = evaluate(rows, Flaky())
    assert out["failures"] == ["b"]
    by = {r["label"]: r for r in out["per_label"]}
    assert by["overconfidence"]["tp"] == 1 and by["fomo"]["fn"] == 1  # failed text scored as no prediction


def test_system_wires_the_aip_backend_without_importing_aip(monkeypatch):
    from minifinrl.system import build_system

    monkeypatch.setenv("MINIFINRL_BIAS_TIER", "MAIN")
    already = "aip" in sys.modules
    s = build_system("ci", bias_backend="aip")
    assert s.engine.bias.name == "aip-main-v1"
    assert ("aip" in sys.modules) == already  # imported lazily, on the first classification


def test_dotenv_in_repo_root_is_loaded_without_overriding(monkeypatch, tmp_path):
    import minifinrl.system as system

    (tmp_path / ".env").write_text("MINIFINRL_TEST_KEY=from-file\nMINIFINRL_TEST_KEEP=from-file\n")
    monkeypatch.setattr(system, "ROOT", tmp_path)
    monkeypatch.delenv("MINIFINRL_TEST_KEY", raising=False)
    monkeypatch.setenv("MINIFINRL_TEST_KEEP", "from-shell")
    for k in ("AIP_CACHE_DIR", "AIP_TRACE_DIR", "AIP_OFFLINE"):
        monkeypatch.delenv(k, raising=False)
    system.configure_aip_env(system.SystemConfig.from_profile("research"))
    assert os.environ["MINIFINRL_TEST_KEY"] == "from-file"
    assert os.environ["MINIFINRL_TEST_KEEP"] == "from-shell"
    monkeypatch.delenv("MINIFINRL_TEST_KEY")
