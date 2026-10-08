"""Reading the writer's state: mood and signals (rules and LLM readers), the
heat score, fallbacks, and the mood evaluation harness. No network, no LLM:
the LLM reader gets a fake `structured`."""

from __future__ import annotations

import pytest

from minifinrl.platform.capabilities import CapabilityRegistry, CapabilityUnavailable
from minifinrl.platform.errors import BudgetExhausted, ClassificationFailed
from minifinrl.sentiment import corpus, mood_eval, state
from minifinrl.sentiment.adapters.aip_state import SYSTEM, AipStateReader, _Answer, _Signal
from minifinrl.sentiment.ports import Mood, StateKind, StateReading, StateSignal
from minifinrl.sentiment.service import SentimentService
from minifinrl.sentiment.state_rules import RulesStateReader

R = RulesStateReader()


def kinds(reading: StateReading) -> set[str]:
    return {s.kind.value for s in reading.signals}


# ---- rules reader -----------------------------------------------------------------------------

@pytest.mark.parametrize("text, mood, expected", [
    ("everyone on reddit is buying, it literally can't go down 🚀🚀", "bullish", {"herd", "certainty"}),
    ("down bad and scared, should have sold, this is rigged 📉", "bearish", {"fear", "regret", "frustration"}),
    ("easy money, buy now before it's too late 🤑", "bullish", {"greed", "urgency"}),
    ("read the annual report; margins are improving", "neutral", set()),
])
def test_rules_reader(text, mood, expected):
    r = R.read(text)
    assert r.mood.value == mood and kinds(r) == expected
    for s in r.signals:  # evidence is always the writer's own words
        assert s.evidence.lower() in text.lower()


def test_negation_flips_mood():
    assert R.read("I'm bullish").mood == Mood.BULLISH
    assert R.read("I'm not bullish at all").mood == Mood.BEARISH


def test_more_matches_mean_a_stronger_signal():
    one = R.read("everyone is in").signals[0].strength
    three = R.read("everyone on reddit and wsb is in").signals[0].strength
    assert one < three <= 1.0


# ---- heat -------------------------------------------------------------------------------------

def reading(*signals):
    return StateReading(mood="neutral", mood_strength=0, reader="t",
                        signals=[StateSignal(kind=k, evidence="x", strength=s) for k, s in signals])


def test_heat_formula():
    assert state.heat(reading()) == 0 and state.level(0) == "calm"
    assert state.heat(reading(("greed", 1.0))) == pytest.approx(0.6)
    # two kinds compound: 1 - (1 - 0.6)(1 - 0.5 * 0.5)
    assert state.heat(reading(("greed", 1.0), ("urgency", 0.5))) == pytest.approx(0.7)
    # the strongest signal of a kind counts, not the number of them
    assert state.heat(reading(("fear", 0.5), ("fear", 1.0))) == state.heat(reading(("fear", 1.0)))
    scored = state.scored(reading(("greed", 1.0), ("urgency", 0.5)))
    assert scored.heat == pytest.approx(0.7) and scored.level == "hot"


def test_every_signal_kind_has_a_weight():
    assert set(state.WEIGHTS) == set(StateKind)


def test_strong_mood_alone_is_not_heat():
    calm = StateReading(mood="bullish", mood_strength=1.0, reader="t")
    assert state.scored(calm).level == "calm"


# ---- LLM reader (fake model) ---------------------------------------------------------------------

def fake(answer=None, error=None):
    calls = []

    def structured(prompt, *, schema, system, tier):
        calls.append({"prompt": prompt, "system": system, "schema": schema})
        if error:
            raise error
        return answer
    return structured, calls


class Budget(Exception):
    pass


class BadOutput(Exception):
    pass


def llm(answer=None, error=None):
    structured, calls = fake(answer, error)
    return AipStateReader(structured=structured, delimit=lambda t, tag: f"<{tag}>{t}</{tag}>",
                          budget_error=Budget, output_error=BadOutput), calls


def test_llm_reader_keeps_only_quoted_evidence():
    text = "Everyone is buying and I can't lose this time"
    ans = _Answer(mood="bullish", mood_strength=0.8, signals=[
        _Signal(kind="herd", evidence="everyone is buying", strength=0.7),
        _Signal(kind="certainty", evidence="guaranteed win", strength=0.9),  # not in the text
        _Signal(kind="herd", evidence="Everyone", strength=0.2),  # second herd signal
    ])
    reader, calls = llm(ans)
    out = reader.read(text)
    assert out.mood == Mood.BULLISH and kinds(out) == {"herd"} and out.signals[0].strength == 0.7
    assert reader.dropped_evidence == 1
    assert calls[0]["prompt"] == f"<TEXT>{text}</TEXT>" and "Sarcasm" in calls[0]["system"]


def test_llm_errors_map_to_platform_errors():
    with pytest.raises(BudgetExhausted):
        llm(error=Budget("spent"))[0].read("hi there")
    with pytest.raises(ClassificationFailed):
        llm(error=BadOutput("bad json"))[0].read("hi there")


def test_prompt_names_every_signal_kind():
    for k in StateKind:
        assert f"- {k.value}:" in SYSTEM


# ---- service: fallback and capability ------------------------------------------------------------

class Broken:
    name = "broken-llm"

    def read(self, text):
        raise TimeoutError("down")


def test_fallback_when_the_model_is_down():
    svc = SentimentService(cfg=None, state_reader=Broken(), state_fallback=R)
    out = svc.read_state_with_fallback("everyone is buying, easy money")
    assert out.reader.startswith("rules-state-v1 (fallback") and out.level in ("warm", "hot")


def test_read_state_capability(rw_engine):
    reg = CapabilityRegistry.from_engine(rw_engine)
    out = reg.invoke("read_state", {"text": "scared and should have sold, everyone is panicking"})
    assert out.heat > 0 and {"fear", "regret", "herd"} <= kinds(out)
    assert "read_state" in reg.callables("agent")  # reading is fine for an agent


def test_no_reader_configured():
    class Cfg:
        profile = "test"
    with pytest.raises(CapabilityUnavailable):
        SentimentService(Cfg()).read_state(type("Req", (), {"text": "hello"})())


# ---- evaluation ----------------------------------------------------------------------------------

class Fixed:
    name = "fixed"

    def __init__(self, answers):
        self.answers = answers

    def read(self, text):
        a = self.answers[text]
        if a == "fail":
            raise ClassificationFailed("x")
        return StateReading(mood=a, mood_strength=1, reader=self.name)


def test_mood_metrics():
    rows = [{"id": str(i), "text": t, "mood": m} for i, (t, m) in enumerate(
        [("a", "bullish"), ("b", "bullish"), ("c", "bearish"), ("d", "bearish"), ("e", "neutral"), ("f", "neutral")])]
    out = mood_eval.evaluate(rows, Fixed({"a": "bullish", "b": "neutral", "c": "bearish", "d": "bullish", "e": "neutral", "f": "fail"}))
    assert out["accuracy"] == pytest.approx(4 / 6) and out["failures"] == ["5"]
    assert out["confusion"]["bearish"]["bullish"] == 1
    assert out["direction_accuracy"] == pytest.approx(2 / 4)  # a, c right; b neutral; d wrong side
    assert out["side_coverage"] == pytest.approx(3 / 4) and out["side_precision"] == pytest.approx(2 / 3)
    bull = next(p for p in out["per_mood"] if p["mood"] == "bullish")
    assert (bull["precision"], bull["recall"]) == (pytest.approx(1 / 2), pytest.approx(1 / 2))


def test_mood_sample_is_balanced_clean_and_seeded(tmp_path):
    lines = {"bullish": ["to the moon @someone 🚀🚀🚀 lets go", "x", "see https://spam.example now buy", "to the moon @other 🚀🚀🚀 lets go"]
             + [f"bull post number {i} with enough text" for i in range(10)],
             "bearish": [f"bear post number {i} with enough text" for i in range(10)],
             "neutral": [f"plain post number {i} with enough text" for i in range(10)]}
    for mood, name in corpus.TEST_FILES.items():
        (tmp_path / name).write_text("\n".join(lines[mood]))
    a = corpus.mood_sample(5, seed=1, cache=tmp_path)
    assert [r["mood"] for r in a].count("bullish") == 5 and len(a) == 15
    texts = [r["text"] for r in a]
    assert not any("http" in t or "@" in t or len(t) < 20 for t in texts)
    assert a == corpus.mood_sample(5, seed=1, cache=tmp_path)
    with pytest.raises(ValueError, match="usable"):
        corpus.mood_sample(50, cache=tmp_path)
