"""Errors any feature can raise, mapped to one outcome by every interface.

An exception class that sets `error_kind` is reported as that kind
(see platform.capabilities.error_kind), so a feature never needs the
interfaces to know about it."""

from __future__ import annotations


class BudgetExhausted(RuntimeError):
    """An adapter's spending budget is used up. Adapters translate their
    library's own error (e.g. aip's BudgetExceeded) into this, so the
    engine layer never has to import that library."""

    error_kind = "budget"


class ClassificationFailed(RuntimeError):
    """The classifier could not produce a valid answer for one text (e.g. an
    LLM's output failed validation after its repair attempts). Evaluation
    counts these instead of treating them as "no bias"."""
