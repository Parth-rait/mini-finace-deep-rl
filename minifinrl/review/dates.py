"""Dates a parser must not guess: impossible ones (30 February), numeric
ones that read two ways (03/04/2025), and text that gives no day at all
("in 2023"). Any of these means the parsed dates are dropped and the person
picks them."""

from __future__ import annotations

import re
from datetime import date as _date

MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"  # shared with parse_rules
_NUMERIC_DATE = re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{2,4})\b")

_DAY_MONTH = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({MONTHS})[a-z]*\.?,?\s+(\d{{4}})\b", re.IGNORECASE)
_MONTH_DAY = re.compile(rf"\b({MONTHS})[a-z]*\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.IGNORECASE)

_DAY_LEVEL = re.compile(
    rf"\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b\d{{1,2}}[/.]\d{{1,2}}[/.]\d{{2,4}}\b|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{MONTHS})"
    rf"|\b(?:{MONTHS})[a-z]*\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b(?!\d)|\b(?:today|yesterday|ago|tonight|this morning)\b"
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
    if not out and re.search(rf"\b(19|20)\d{{2}}\b|\b(?:{MONTHS})[a-z]*\b", text, re.IGNORECASE) and not _DAY_LEVEL.search(text):
        if re.search(r"\b(19|20)\d{2}\b", text) or re.search(rf"\b(in|during|early|mid|late|last)\s+(?:{MONTHS})", text, re.IGNORECASE):
            out.append("the text gives a month or a year but not the day; pick the dates")
    return out


_MONTH_LABEL = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                "November", "December"]
