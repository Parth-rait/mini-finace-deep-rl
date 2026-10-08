"""Limits of the trade review."""

MAX_HORIZON = 252  # the luck check covers holds up to about a year of trading days
MAX_STATED_DAYS = 260  # longest holding period a parser may report; the review then asks for a shorter one
