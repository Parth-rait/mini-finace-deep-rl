"""
Building blocks of the trade review that don't depend on which model runs:
combining parses, resolving a ticker, locating bias evidence in the text,
formatting facts, and checking that an explanation only uses computed numbers.

The review itself (Engine.review_trade) wires these to the data, the luck
test, the bias classifier and the explainer.
"""

from __future__ import annotations

import bisect
import difflib
import re
from dataclasses import dataclass, field
from datetime import date as _date

from minifinrl.core.configs.tickers import CROSS_ASSET_ETFS, CRYPTO, NAME_ALIASES, SMALL_TICKER
from minifinrl.core.meta.symbols import yahoo_symbol
from minifinrl.ports import ParsedTrade

KNOWN_SYMBOLS = sorted(set(SMALL_TICKER) | set(CROSS_ASSET_ETFS) | set(NAME_ALIASES.values()))
REQUIRED = ("ticker", "date", "direction")  # plus an exit: sell_date, still_holding or horizon_days
MAX_HORIZON = 252  # the luck check covers holds up to about a year of trading days
BOND_ETFS = ("TLT", "IEF", "SHY", "AGG", "BND")


def merge(primary: ParsedTrade | None, fallback: ParsedTrade, overrides: dict) -> ParsedTrade:
    """Fields typed into the form win, then the LLM parse, then the rules parse.
    If the merge puts the same day in both the buy and the sell slot (the LLM
    read "sold on 3 March" as the buy date, the rules parser as the sell date),
    the rules parser's reading of which verb the date belongs to decides."""
    fields = {}
    for name in ParsedTrade.model_fields:
        if name == "unclear":
            continue
        for source in (overrides, (primary.model_dump() if primary else {}), fallback.model_dump()):
            value = source.get(name)
            if value not in (None, ""):
                fields[name] = value
                break
    unclear = list(dict.fromkeys((primary.unclear if primary else []) + fallback.unclear))
    if fields.get("date") and fields.get("date") == fields.get("sell_date") and "date" not in overrides:
        if fallback.sell_date == fields["date"] and fallback.date != fields["date"]:
            fields["date"] = fallback.date
            if fields["date"] is None:
                del fields["date"]
    return ParsedTrade(**fields, unclear=unclear)


# ---- what the review can't cover -----------------------------------------------------------

_FOREIGN_SUFFIX = re.compile(r"\.(NS|BO|L|TO|V|HK|T|AX|DE|PA|SA|SS|SZ|KS)$", re.IGNORECASE)
_BOND = re.compile(r"\b(bonds?|treasur(y|ies)|t-?bills?|t-?notes?|gilts?|munis?|debentures?|corporate debt)\b", re.IGNORECASE)
_FOREIGN = re.compile(r"\b(nse|bse|nifty|sensex|ftse|nikkei|hang seng|tsx|asx|dax|euronext|rupees?|indian (stock|market|share)s?)\b|₹",
                      re.IGNORECASE)
_OPTIONS = re.compile(r"\b(calls?|puts?|options?|strike|expir\w*|leaps|0dte|contracts?)\b", re.IGNORECASE)

BOND_MESSAGE = ("Individual bonds and Treasuries don't trade on an exchange with a daily closing price here, so they "
                "can't be reviewed. A bond ETF can: TLT (20+ year Treasuries), IEF (7 to 10 year), SHY (1 to 3 year) "
                "or AGG (the broad bond market).")
FOREIGN_MESSAGE = ("Only US-listed stocks and ETFs are covered, because the luck check models the US market. If the "
                   "company also lists in the US (an ADR, such as INFY for Infosys, HDB for HDFC Bank, TM for Toyota), "
                   "pick that listing.")
CRYPTO_MESSAGE = ("Crypto isn't supported: it trades around the clock and doesn't follow the stock-market regime "
                  "the luck check models, so any answer would be misleading.")
OPTIONS_MESSAGE = ("You mentioned options. This reviews the underlying stock over the same days; an option's profit "
                   "or loss differs (leverage, strike, time decay), so read the numbers as the stock's, not the option's.")


def scope_problem(text: str, ticker: str | None) -> str | None:
    """A message if the trade is in something the review doesn't cover."""
    if ticker and _FOREIGN_SUFFIX.search(ticker.strip()):
        return FOREIGN_MESSAGE
    if ticker:
        return None
    if _BOND.search(text):
        return BOND_MESSAGE
    if _FOREIGN.search(text):
        return FOREIGN_MESSAGE
    return None


def mentions_options(text: str) -> bool:
    return bool(_OPTIONS.search(text))


def is_bond_query(q: str) -> bool:
    return bool(_BOND.search(q))


# ---- dates a parser must not guess -------------------------------------------------------------

_MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
_NUMERIC_DATE = re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{2,4})\b")
_DAY_MONTH = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTHS})[a-z]*\.?,?\s+(\d{{4}})\b", re.IGNORECASE)
_MONTH_DAY = re.compile(rf"\b({_MONTHS})[a-z]*\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.IGNORECASE)
_DAY_LEVEL = re.compile(
    rf"\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b\d{{1,2}}[/.]\d{{1,2}}[/.]\d{{2,4}}\b|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTHS})"
    rf"|\b(?:{_MONTHS})[a-z]*\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b(?!\d)|\b(?:today|yesterday|ago|tonight|this morning)\b"
    r"|\b(?:mon|tues|wednes|thurs|fri|satur|sun)day\b", re.IGNORECASE)
_MONTH_NUM = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def _valid(y: int, m: int, d: int) -> bool:
    try:
        _date(y if y > 99 else 2000 + y, m, d)
        return True
    except ValueError:
        return False


def date_problems(text: str) -> list[str]:
    """Dates in the text that can't be read without guessing: impossible ones
    (30 February), numeric ones that read two ways (03/04/2025), or none at
    day level (only "in 2023" or "in March"). Any of these means the parsed
    dates are dropped and the person picks them."""
    out = []
    for m in _NUMERIC_DATE.finditer(text):
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if a <= 12 and b <= 12 and a != b and _valid(y, a, b) and _valid(y, b, a):
            out.append(f"'{m.group(0)}' could be {_MONTH_LABEL[b - 1]} {a} or {_MONTH_LABEL[a - 1]} {b}; pick the date")
        elif not (_valid(y, a, b) or _valid(y, b, a)):
            out.append(f"'{m.group(0)}' is not a real date")
    for m in _DAY_MONTH.finditer(text):
        if not _valid(int(m.group(3)), _MONTH_NUM[m.group(2).lower()[:3]], int(m.group(1))):
            out.append(f"'{m.group(0)}' is not a real date")
    for m in _MONTH_DAY.finditer(text):
        if not _valid(int(m.group(3)), _MONTH_NUM[m.group(1).lower()[:3]], int(m.group(2))):
            out.append(f"'{m.group(0)}' is not a real date")
    if not out and re.search(rf"\b(19|20)\d{{2}}\b|\b(?:{_MONTHS})[a-z]*\b", text, re.IGNORECASE) and not _DAY_LEVEL.search(text):
        if re.search(r"\b(19|20)\d{2}\b", text) or re.search(rf"\b(in|during|early|mid|late|last)\s+(?:{_MONTHS})", text, re.IGNORECASE):
            out.append("the text gives a month or a year but not the day; pick the dates")
    return out


_MONTH_LABEL = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                "November", "December"]


# ---- the holding window, counted on the exchange calendar ---------------------------------------

class WindowProblem(ValueError):
    def __init__(self, status: str, message: str, missing: list[str] | None = None):
        super().__init__(message)
        self.status, self.message, self.missing = status, message, missing or []


@dataclass
class Window:
    i: int  # entry row in the trading-day index
    j: int  # exit row
    notes: list[str] = field(default_factory=list)

    @property
    def days(self) -> int:
        return self.j - self.i


def trading_window(dates: list[str], today: str, date: str | None, sell_date: str | None = None,
                   still_holding: bool | None = None, horizon_days: int | None = None,
                   max_days: int | None = MAX_HORIZON) -> Window:
    """Entry and exit rows in `dates` (trading days, ascending). The buy date
    moves forward to the first trading day on or after it; the sell date moves
    back to the last trading day on or before it; still holding means the latest
    close. With only a sell date and a length, the buy date is counted back."""
    notes = []
    if not dates:
        raise WindowProblem("rejected", "No trading days are available to measure the trade against.")
    if sell_date and sell_date > today:
        raise WindowProblem("rejected", f"The sell date {sell_date} is in the future.", ["sell_date"])
    if not date:
        if not (sell_date and horizon_days):
            raise WindowProblem("needs_input", "Pick the buy date.", ["date"])
        j = bisect.bisect_right(dates, sell_date) - 1
        if j - horizon_days < 0:
            raise WindowProblem("rejected", "There is not enough price history before that sell date.")
        date = dates[j - horizon_days]
        notes.append(f"Buy date counted back {horizon_days} trading days from the sell date: {date}.")
    if date > today:
        raise WindowProblem("rejected", f"{date} is in the future, so there is no outcome to review yet.")
    i = bisect.bisect_left(dates, date)
    if i >= len(dates):
        raise WindowProblem("rejected", f"There is no trading day on or after {date} in the data yet (it ends {dates[-1]}).")
    if dates[i] != date:
        notes.append(f"{date} was not a trading day; the trade is dated from the next one, {dates[i]}.")
    if sell_date:
        if sell_date <= date:
            raise WindowProblem("needs_input", "The sell date has to be after the buy date.", ["sell_date"])
        j = bisect.bisect_right(dates, sell_date) - 1
        if sell_date > dates[-1]:
            notes.append(f"Prices are available up to {dates[-1]}, so the trade is measured to that close.")
        elif dates[j] != sell_date:
            notes.append(f"{sell_date} was not a trading day; the exit uses the close of {dates[j]}, the last trading day before it.")
        if j <= i:
            raise WindowProblem("needs_input", f"Bought and sold within the same trading day ({dates[i]}). The luck check "
                                "works on daily closes, so it needs at least one close after the buy.", ["sell_date"])
    elif still_holding:
        j = len(dates) - 1
        if j <= i:
            raise WindowProblem("rejected", f"Bought on {dates[i]}, the latest close in the data. There is nothing to measure yet.")
        notes.append(f"Still holding: measured up to the latest close, {dates[j]}.")
    elif horizon_days:
        j = i + horizon_days
        if j >= len(dates):
            raise WindowProblem("rejected", f"The outcome is not observable yet: {dates[i]} plus {horizon_days} trading days "
                                f"is past the latest close, {dates[-1]}. Pick 'still holding' to measure it so far.")
    else:
        raise WindowProblem("needs_input", "Pick the sell date, or say you're still holding.", ["sell_date"])
    if max_days and j - i > max_days:
        raise WindowProblem("needs_input", f"That is {j - i} trading days. The luck check covers holds of up to "
                            f"{max_days} trading days (about a year).", ["sell_date"])
    return Window(i, j, notes)


def resolve_ticker(ticker: str | None, company: str | None) -> tuple[str | None, str | None, list[str]]:
    """(symbol, problem, suggestions). problem is 'crypto' or 'missing'.
    Symbols come back the way Yahoo writes them (BRK.B -> BRK-B)."""
    raw = (ticker or "").strip().lstrip("$").upper()
    name = (company or "").strip().lower()
    if raw in CRYPTO or name.upper() in CRYPTO or raw.replace("-USD", "") in CRYPTO:
        return None, "crypto", []
    if name in NAME_ALIASES and not raw:
        return NAME_ALIASES[name], None, []
    if raw.lower() in NAME_ALIASES:
        return NAME_ALIASES[raw.lower()], None, []
    if raw:
        return yahoo_symbol(raw), None, []
    return None, "missing", suggest(name) if name else []


def suggest(text: str, n: int = 4) -> list[str]:
    """Close matches among known names and symbols, as 'NAME (SYMBOL)' or 'SYMBOL'."""
    t = text.strip().lower()
    names = difflib.get_close_matches(t, list(NAME_ALIASES), n=n, cutoff=0.6)
    syms = difflib.get_close_matches(t.upper(), KNOWN_SYMBOLS, n=n, cutoff=0.6)
    out = [f"{nm} ({NAME_ALIASES[nm]})" for nm in names] + [s for s in syms if s not in {NAME_ALIASES[nm] for nm in names}]
    return out[:n]


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


# ---- explanation slots: the model writes {placeholders}, the code fills in numbers ----------

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
