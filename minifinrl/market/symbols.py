"""
Directory of US-listed stocks and ETFs, for the symbol search.

Source: NASDAQ Trader's public symbol directory (nasdaqlisted.txt and
otherlisted.txt, updated daily; together they cover NASDAQ, NYSE, NYSE
American, NYSE Arca and Cboe listings). Downloaded at most once a day into
data/raw/symbols/. Test issues are dropped, and symbols are written the way
Yahoo Finance expects them (BRK.B -> BRK-B), because Yahoo supplies the prices.
"""

from __future__ import annotations

import csv
import io
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from minifinrl.platform.settings import DATA_RAW, HTTP_TIMEOUT_S
from minifinrl.market.providers import ProviderError

BASE = "https://www.nasdaqtrader.com/dynamic/SymDir/"
CACHE = Path(DATA_RAW) / "symbols"
EXCHANGES = {"Q": "NASDAQ", "N": "NYSE", "A": "NYSE American", "P": "NYSE Arca", "Z": "Cboe", "V": "IEX"}

# Tickers people still type that no longer trade under that symbol.
RENAMED = {"FB": "META", "ANTM": "ELV", "FISV": "FI", "RTN": "RTX", "SQ": "XYZ"}
DELISTED = {"TWTR": "Twitter was taken private in October 2022",
            "ATVI": "Activision Blizzard was acquired by Microsoft in October 2023"}


@dataclass(frozen=True)
class Listing:
    symbol: str  # Yahoo style
    name: str
    exchange: str
    etf: bool


def yahoo_symbol(raw: str) -> str:
    """BRK.B / BRK/B -> BRK-B; uppercase; strip a leading $."""
    return raw.strip().lstrip("$").upper().replace(".", "-").replace("/", "-")


_CLEAN = re.compile(r"\s*-\s*(common stock|ordinary shares|class [a-z] (common stock|ordinary shares)|american depositary shares.*|"
                    r"common shares|units?|warrants?.*|rights?.*)$", re.IGNORECASE)


def _clean_name(name: str) -> str:
    name = _CLEAN.sub("", name.strip())
    name = re.sub(r"\s+(New\s+)?(Class [A-Z]\s+)?(Common Stock|Common Shares|Ordinary Shares)$", "", name, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", name).strip()


def parse(nasdaq_text: str, other_text: str) -> list[Listing]:
    out: dict[str, Listing] = {}
    for row in csv.DictReader(io.StringIO(nasdaq_text), delimiter="|"):
        if not row.get("Symbol") or row["Symbol"].startswith("File Creation") or row.get("Test Issue") == "Y":
            continue
        sym = yahoo_symbol(row["Symbol"])
        out.setdefault(sym, Listing(sym, _clean_name(row["Security Name"]), "NASDAQ", row.get("ETF") == "Y"))
    for row in csv.DictReader(io.StringIO(other_text), delimiter="|"):
        if not row.get("ACT Symbol") or row["ACT Symbol"].startswith("File Creation") or row.get("Test Issue") == "Y":
            continue
        if "$" in row["ACT Symbol"]:  # preferred shares: not what people mean by "the stock"
            continue
        sym = yahoo_symbol(row["ACT Symbol"])
        out.setdefault(sym, Listing(sym, _clean_name(row["Security Name"]), EXCHANGES.get(row.get("Exchange", ""), "US"),
                                    row.get("ETF") == "Y"))
    return sorted(out.values(), key=lambda l: l.symbol)


def _download(name: str) -> str:
    req = urllib.request.Request(BASE + name, headers={"User-Agent": "mini-finrl/0.2"})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as r:
            return r.read().decode("utf-8", errors="replace")
    except OSError as exc:
        raise ProviderError(f"symbol directory unavailable: {exc}") from exc


def load(cache: Path = CACHE, *, refresh: bool = True) -> list[Listing]:
    """Today's directory from the cache, downloading if it is older than a day.
    With refresh=False (or if the download fails) an older cache is used."""
    cache.mkdir(parents=True, exist_ok=True)
    files = [cache / "nasdaqlisted.txt", cache / "otherlisted.txt"]
    stale = not all(f.exists() for f in files) or any(
        (datetime.now(timezone.utc) - datetime.fromtimestamp(f.stat().st_mtime, timezone.utc)).days >= 1 for f in files)
    if stale and refresh:
        try:
            for f in files:
                f.write_text(_download(f.name))
        except ProviderError:
            if not all(f.exists() for f in files):
                raise
    if not all(f.exists() for f in files):
        raise ProviderError("no symbol directory on disk")
    return parse(files[0].read_text(), files[1].read_text())


def search(listings: list[Listing], query: str, limit: int = 8) -> list[Listing]:
    """Rank 0: the exact symbol, or (for a word typed in lowercase) a listing
    whose name starts with that word, so "ford" finds Ford Motor (F) as well
    as the FORD symbol. Then symbol prefixes for symbol-like queries, then
    names whose words start with every query word, then any name containing
    the query. Stocks before ETFs and shorter names first within a rank
    (the plain company before its funds and spin-offs)."""
    q = query.strip()
    if not q:
        return []
    qs, words = yahoo_symbol(q), [w for w in re.split(r"\W+", q.lower()) if w]
    symbol_like = " " not in q and len(qs) <= 6
    scored = []
    for l in listings:
        name = l.name.lower()
        name_words = [w for w in re.split(r"\W+", name) if w]
        if l.symbol == qs or (q.islower() and len(words) == 1 and name_words[:1] == words):
            rank = 0
        elif symbol_like and l.symbol.startswith(qs):
            rank = 1
        elif words and all(any(nw.startswith(w) for nw in name_words) for w in words):
            rank = 2 if name_words and name_words[0].startswith(words[0]) else 3
        elif q.lower() in name:
            rank = 4
        else:
            continue
        scored.append((rank, l.etf, len(l.name), l.symbol, l))
    scored.sort(key=lambda t: t[:4])
    return [t[-1] for t in scored[:limit]]


def lookup(listings: list[Listing], symbol: str) -> Listing | None:
    sym = yahoo_symbol(symbol)
    return next((l for l in listings if l.symbol == sym), None)
