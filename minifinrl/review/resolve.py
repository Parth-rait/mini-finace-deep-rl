"""Turning what was typed into an exchange symbol."""

from __future__ import annotations

import difflib

from minifinrl.market.symbols import yahoo_symbol
from minifinrl.review.names import CRYPTO, KNOWN_SYMBOLS, NAME_ALIASES


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
