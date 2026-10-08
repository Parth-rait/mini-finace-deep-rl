"""Reading what the person gave us, without guessing: combining the form
with what parsers read, what the review can't cover (crypto, bonds,
non-US listings, options), and dates that would need a guess."""

from __future__ import annotations

import re

from minifinrl.review.dates import date_problems  # noqa: F401  (part of reading the input)
from minifinrl.review.ports import ParsedTrade
from minifinrl.review.settings import MAX_HORIZON  # noqa: F401

REQUIRED = ("ticker", "date", "direction")  # plus an exit: sell_date, still_holding or horizon_days


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


