"""
LLM adapters for the trade review, through aip.structured: one reads a
free-text trade into fields, one writes the plain-language explanation.

Both receive user text as delimited untrusted input. The explainer is given
facts already formatted as strings and told to copy numbers exactly; the
review step then checks every number in its text against those facts and
falls back to a template if any number doesn't match, so the explanation
can't state a figure that wasn't computed.

aip is imported lazily, after system.py has pointed its cache at this repo.
"""

from __future__ import annotations

import json
from typing import Callable

from pydantic import BaseModel, Field

from minifinrl.platform.errors import BudgetExhausted, ClassificationFailed
from minifinrl.review.ports import ParsedTrade

PARSE_SYSTEM = """You read one person's description of a stock trade and fill in fields. Today is {today}.

- ticker: the exchange symbol if written (TSLA, $TSLA) or if the company name is unambiguous (tesla -> TSLA). For crypto, put the coin as written in capitals (DOGE, BTC). Otherwise null.
- company: the company or asset name as the person wrote it, or null.
- date: the day they bought (or shorted), as YYYY-MM-DD. Resolve relative days ("last friday", "two weeks ago") against today. If the text gives only a month or a year, or a numeric date that reads two ways (03/04/2025), or a date that doesn't exist (30 February), put null and say why in unclear. Never pick a day yourself.
- sell_date: the day they sold (or covered), as YYYY-MM-DD, under the same rules. "sold on 3 March after a week" means sell_date is 3 March, not date. Null if not said.
- still_holding: true if they say they still hold the position, otherwise null.
- direction: "long" if they bought, bought calls, or only mention selling shares they owned; "short" if they shorted or bought puts; otherwise null.
- horizon_days: how long they held if they say it as a length ("a week", "10 days"): a week is 5 trading days, a month 21. Do not compute it from two dates. Null if not said.
- reasoning: their stated reason, copied word for word from the text, or null.
- unclear: short notes on anything you could not read with confidence.

Never invent a value. A missing field stays null."""

EXPLAIN_SYSTEM = """You explain the result of a trade review to the person who made the trade, in 3 or 4 short sentences, the way a calm, knowledgeable friend would say it.

Never write a digit. Wherever a number or date belongs, write one of the placeholders listed under "placeholders" in braces, exactly as named, for example {your_return} or {entry_date}; they are filled in with the real values afterwards. The facts show you what each value is, so you can tell whether it was good or bad.

Cover: what they did; whether the outcome sits inside or outside the range the market model considered plausible (and what that says about luck, not skill); the bias signals in their reasoning, quoting their words in double quotes if there are any; and how the S&P 500 did over the same days. Do not give advice, predict prices, or judge the person. Do not repeat the fact labels word for word."""


class _Explanation(BaseModel):
    explanation: str = Field(description="3 to 5 sentences")


class _AipCalls:
    def __init__(self, tier: str, structured: Callable | None, delimit: Callable | None,
                 budget_error: type[BaseException] | None, output_error: type[BaseException] | None):
        self.tier = tier
        self._structured, self._delimit = structured, delimit
        self._budget_error, self._output_error = budget_error, output_error

    def _load(self) -> None:
        if self._structured is not None:
            return
        import aip
        from aip.cost import BudgetExceeded
        from aip.guards import delimit_untrusted
        from aip.llm import StructuredOutputError

        self._structured, self._delimit = aip.structured, delimit_untrusted
        self._budget_error, self._output_error = BudgetExceeded, StructuredOutputError

    def call(self, prompt: str, schema, system: str):
        self._load()
        try:
            return self._structured(prompt, schema=schema, system=system, tier=self.tier)
        except Exception as exc:
            if self._budget_error and isinstance(exc, self._budget_error):
                raise BudgetExhausted(str(exc)) from exc
            if self._output_error and isinstance(exc, self._output_error):
                raise ClassificationFailed(str(exc)) from exc
            raise


class AipTradeParser(_AipCalls):
    def __init__(self, tier: str = "SMALL", *, structured=None, delimit=None, budget_error=None, output_error=None):
        super().__init__(tier, structured, delimit, budget_error, output_error)
        self.name = f"aip-parse-{tier.lower()}-v1"

    def parse(self, text: str, today: str) -> ParsedTrade:
        self._load()
        prompt = self._delimit(text, "TRADE_DESCRIPTION") if self._delimit else text
        return self.call(prompt, ParsedTrade, PARSE_SYSTEM.format(today=today))


class AipTradeExplainer(_AipCalls):
    def __init__(self, tier: str = "SMALL", *, structured=None, delimit=None, budget_error=None, output_error=None):
        super().__init__(tier, structured, delimit, budget_error, output_error)
        self.name = f"aip-explain-{tier.lower()}-v1"

    def explain(self, facts: dict) -> str:
        """Returns text with {placeholders}; the review fills and checks them."""
        self._load()
        body = json.dumps(facts, indent=2, ensure_ascii=False)
        prompt = self._delimit(body, "FACTS") if self._delimit else body
        return self.call(prompt, _Explanation, EXPLAIN_SYSTEM).explanation
