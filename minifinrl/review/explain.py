"""Facts for the explanation, and the checks that keep it honest: the
model writes {placeholders}, the code fills in computed values, and any
other number sends the explanation back to a plain template."""

from __future__ import annotations

import re


def evidence_span(reasoning: str, evidence: str) -> tuple[int, int] | None:
    """Character span of `evidence` in `reasoning`, ignoring case and runs of
    whitespace, for highlighting. None if it can't be located exactly."""
    words = [re.escape(w) for w in evidence.strip(" \"'.…").split()]
    if not words:
        return None
    m = re.search(r"\s+".join(words), reasoning, flags=re.IGNORECASE)
    return (m.start(), m.end()) if m else None


def pct(x: float, digits: int = 1) -> str:
    """+19.8%; small moves get two decimals so they don't print as -0.0%."""
    if abs(x) < 0.0005:
        digits = 2
    return f"{x * 100:+.{digits}f}%"


def ordinal(p: float) -> str:
    """Percentile as an ordinal; one decimal at the extremes (0.5th, 99.7th)."""
    if 1 <= p <= 99 or p == round(p):
        n = round(p)
        return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"
    return f"{p:.1f}th"


_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    return {m.group(0).replace(",", "").lstrip("+-") for m in _NUMBER.finditer(text)}


def unsupported_numbers(explanation: str, facts: dict) -> list[str]:
    """Numbers in the explanation that appear nowhere in the facts. Signs are
    ignored here (the facts carry them); any other figure means the text
    stated something that wasn't computed."""
    allowed = _numbers(" ".join(str(v) for v in _flatten(facts)))
    return sorted(n for n in _numbers(explanation) if n not in allowed)


def _flatten(x):
    """Every key and value: keys count too ("S&P 500", "95th percentile" are facts the text may repeat)."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from _flatten(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _flatten(v)
    else:
        yield x


# fact keys, written as plain English because the explainer copies their wording
F_WHAT = "what you did"
F_OUTCOME = "outcome"

F_RETURN = "your return"
F_LOW = "low end of the plausible range (5th percentile)"

F_HIGH = "high end of the plausible range (95th percentile)"
F_PCT = "where your return falls among simulated outcomes (percentile)"

F_VERDICT = "verdict"
F_REGIME = "market regime at entry"

F_BIASES = "bias signals in your reasoning"
F_BIAS_NOTE = "note on bias signals"

F_SPY = "S&P 500 (SPY) return over the same days"


def template_explanation(facts: dict) -> str:
    """The explanation built without a model. Always available."""
    parts = [facts[F_WHAT]]
    if F_OUTCOME in facts:
        o = facts[F_OUTCOME]
        parts.append(
            f"The position returned {o[F_RETURN]}, against a plausible range of {o[F_LOW]} to {o[F_HIGH]} "
            f"given the {o[F_REGIME]} market regime at entry, which puts it at the {o[F_PCT]} percentile: {o[F_VERDICT]}."
        )
    if facts.get(F_BIASES):
        parts.append("Bias signals in your reasoning: " + "; ".join(f"{b['bias']} (\"{b['your words']}\")" for b in facts[F_BIASES]) + ".")
    elif facts.get(F_BIAS_NOTE):
        parts.append(facts[F_BIAS_NOTE])
    if F_SPY in facts:
        parts.append(f"Over the same days the S&P 500 (SPY) returned {facts[F_SPY]}.")
    return " ".join(parts)


SLOT = re.compile(r"\{([a-z_]+)\}")


class SlotError(ValueError):
    pass


NAMES_WITH_DIGITS = ("S&P 500",)  # names, not statistics


def fill_slots(text: str, slots: dict[str, str], quotes: list[str], user_text: str = "") -> str:
    """Replace {name} placeholders with computed values. Any other digit in the
    model's text must be part of a name ("S&P 500"), a quotation of the user's
    words ("1000% pump"), or a number the user wrote themselves ("its 2021 high");
    so every figure the explanation asserts comes from `slots` or from the user."""
    unknown = sorted(set(SLOT.findall(text)) - set(slots))
    if unknown:
        raise SlotError(f"unknown placeholders {unknown}")
    bare = SLOT.sub("", text)
    for q in sorted([*quotes, *NAMES_WITH_DIGITS], key=len, reverse=True):
        if q:
            bare = re.sub(re.escape(q), "", bare, flags=re.IGNORECASE)
    own = {m.group(0) for m in re.finditer(r"\d[\d,.]*", user_text)}
    stray = [m.group(0).rstrip(".,") for m in re.finditer(r"\d[\d,.]*", bare) if m.group(0).rstrip(".,") not in own]
    if stray:
        raise SlotError(f"the text states numbers directly instead of using placeholders: {stray}")
    return SLOT.sub(lambda m: slots[m.group(1)], text)
