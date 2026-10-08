"""
Regex trade parser: no LLM, no network. The fallback when the LLM parser is
unavailable, and a floor to compare it against. It reads only what is stated
plainly: $TICKER or a known company name, an ISO or written date, bought /
shorted, "held N days|weeks|months". Relative dates ("last week") are left
for the LLM parser or the user.
"""

from __future__ import annotations

import re

import pandas as pd

from minifinrl.review.dates import MONTHS
from minifinrl.review.names import CRYPTO, KNOWN_SYMBOLS, NAME_ALIASES
from minifinrl.review.settings import MAX_STATED_DAYS
from minifinrl.review.ports import ParsedTrade

_UNIT = {"day": 1, "days": 1, "week": 5, "weeks": 5, "month": 21, "months": 21}
_WORDNUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "ten": 10}
_M = rf"(?:{MONTHS})[a-z]*\.?"
_DATES = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_M},?\s+\d{{4}}\b"
    rf"|\b{_M}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b"
    r"|\b\d{1,2}/\d{1,2}/\d{4}\b"
)
_BUY_VERB = re.compile(r"\b(bought|buy|bot|got in|went in|entered|opened|shorted|short|aped|added|picked up)\b")
_SELL_VERB = re.compile(r"\b(sold|sell|exited|closed|covered|dumped|got out|took profits?)\b")
_HOLDING = re.compile(r"\b(still (holding|hold|own|have|in)|haven'?t sold|not sold yet|holding (it )?(now|still))\b")


def _read_date(raw: str) -> str | None:
    """ISO, '3 march 2025', 'march 3, 2025', or d/m/y written so it can only be
    read one way (13/04/2025). Ambiguous and impossible dates return None;
    review.date_problems reports them."""
    raw = re.sub(r"(\d)(st|nd|rd|th)", r"\1", raw).replace("sept", "sep").replace(" of ", " ").replace(".", "")
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", raw)
    if m:
        a, b, y = map(int, m.groups())
        if a <= 12 and b <= 12 and a != b:
            return None
        day, month = (a, b) if a > 12 else (b, a)
        raw = f"{y}-{month:02d}-{day:02d}"
    try:
        return pd.Timestamp(raw).strftime("%Y-%m-%d")
    except ValueError:
        return None


class RulesTradeParser:
    name = "rules-parse-v1"

    def parse(self, text: str, today: str) -> ParsedTrade:
        low = text.lower()
        out = ParsedTrade()

        cash = re.findall(r"\$([A-Za-z][A-Za-z.\-]{0,5})\b", text)
        if cash:
            out.ticker = cash[0].upper()
        else:
            for name in sorted(NAME_ALIASES, key=len, reverse=True):
                if re.search(rf"\b{re.escape(name)}\b", low):
                    out.company, out.ticker = name, NAME_ALIASES[name]
                    break
            else:
                caps = [w for w in re.findall(r"\b[A-Z]{2,5}\b", text) if w in KNOWN_SYMBOLS or w in CRYPTO]
                if caps:
                    out.ticker = caps[0]
        for word in re.findall(r"\b[a-z]+\b", low):
            if word.upper() in CRYPTO and not out.ticker:
                out.ticker = word.upper()

        # each date belongs to the nearest buy or sell verb before it; with no
        # verb, the first date is the buy and a second one the sell
        for m in _DATES.finditer(low):
            day = _read_date(m.group(0))
            if day is None:
                out.unclear.append(f"could not read the date '{m.group(0)}'")
                continue
            before = low[max(0, m.start() - 60):m.start()]
            buys, sells = list(_BUY_VERB.finditer(before)), list(_SELL_VERB.finditer(before))
            last_buy, last_sell = (buys[-1].end() if buys else -1), (sells[-1].end() if sells else -1)
            if last_sell > last_buy and not out.sell_date:
                out.sell_date = day
            elif not out.date:
                out.date = day
            elif not out.sell_date:
                out.sell_date = day
        if out.date and out.sell_date and out.sell_date < out.date:
            out.date, out.sell_date = out.sell_date, out.date
        if _HOLDING.search(low):
            out.still_holding = True

        if re.search(r"\b(short(ed|ing)?|puts?|sold short)\b", low):
            out.direction = "short"
        elif re.search(r"\b(bought|buy(ing)?|long|calls?|went in|got in|added)\b", low):
            out.direction = "long"
        elif re.search(r"\b(sold|sell|selling)\b", low):
            out.direction = "long"  # selling shares you own closes a long position

        m = re.search(r"\b(?:held|holding|hold|kept|for|after|about|around|roughly|almost|nearly)\b[^.]{0,20}?\b"
                      r"(\d+|a|an|one|two|three|four|five|six|ten)\s+(days?|weeks?|months?)\b(?!\s+ago)", low)
        if m:
            n = int(m.group(1)) if m.group(1).isdigit() else _WORDNUM[m.group(1)]
            out.horizon_days = max(1, min(MAX_STATED_DAYS, n * _UNIT[m.group(2)]))

        m = re.search(r"\b(?:because|cause|cuz|coz|since|as)\b\s+(.+)", text, flags=re.IGNORECASE | re.DOTALL)
        out.reasoning = m.group(1).strip() if m else None
        return out
