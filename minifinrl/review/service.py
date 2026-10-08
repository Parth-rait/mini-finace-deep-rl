"""
Trade review capabilities: review a past trade (luck, bias signals, market
context, explanation), read a trade written in words into form fields,
find a symbol, and count trading days between two dates.
"""

from __future__ import annotations

import re


from minifinrl.market import calendar, symbols
from minifinrl.market.providers import ProviderError
from minifinrl.market.schemas import SymbolHit, SymbolIn, SymbolsOut, TradingDaysIn, TradingDaysOut
from minifinrl.platform.capabilities import CapabilityUnavailable, capability
from minifinrl.platform.errors import BudgetExhausted, ClassificationFailed
from minifinrl.platform.log import get_logger
from minifinrl.review import explain, intake, resolve, trade
from minifinrl.review.ports import ParsedTrade, TradeExplainer, TradeParser
from minifinrl.review.schemas import (
    BiasSpan,
    ParseIn,
    ParseOut,
    ReviewIn,
    ReviewOut,
)

log = get_logger(__name__)


class ReviewService:
    def __init__(self, cfg, market, regime, sentiment, *, parser: TradeParser | None = None,
                 parser_fallback: TradeParser | None = None, explainer: TradeExplainer | None = None):
        self.cfg = cfg
        self.market, self.regime, self.sentiment = market, regime, sentiment
        self.parser = parser
        self.parser_fallback = parser_fallback
        self.explainer = explainer

    def parse(self, text: str, today: str, overrides: dict) -> tuple[ParsedTrade, str, list[str]]:
        notes, primary, name = [], None, None
        if self.parser is not None and text.strip():
            try:
                primary, name = self.parser.parse(text, today), self.parser.name
            except (ClassificationFailed, BudgetExhausted, ProviderError) as exc:
                notes.append(f"The language model could not read the trade ({type(exc).__name__}); used the simple parser instead.")
            except Exception as exc:  # network errors from the model provider surface as many types
                log.warning("LLM parser failed: %s: %s", type(exc).__name__, exc)
                notes.append("The language model was unavailable; used the simple parser instead.")
        fallback = (self.parser_fallback.parse(text, today) if self.parser_fallback else ParsedTrade())
        name = name or (self.parser_fallback.name if self.parser_fallback else "form")
        parsed = intake.merge(primary, fallback, overrides)
        # dates that can't be read without guessing are dropped, whatever the parsers made of them
        problems = intake.date_problems(text)
        dropped = False
        for f in ("date", "sell_date"):
            if problems and f not in overrides and getattr(parsed, f):
                setattr(parsed, f, None)
                dropped = True
        if problems and (dropped or "date" not in overrides):
            notes += [f"Dates: {p}." for p in problems]
            parsed.unclear = [u for u in parsed.unclear if "date" not in u.lower()]  # said once, above
        return parsed, name, notes

    def check_symbol(self, symbol: str, company: str | None, from_form: bool) -> tuple[str, str | None, list[str], str | None]:
        """(symbol, listing name, notes, reason it can't be reviewed). Follows
        renames, refuses delisted symbols, and corrects a parser that matched
        only part of the name ("apple hospitality" read as AAPL)."""
        notes = []
        if symbol in symbols.DELISTED:
            return symbol, None, notes, (f"{symbol} no longer trades: {symbols.DELISTED[symbol]}. Its old prices aren't "
                                         "available from the data source, so the trade can't be checked.")
        if symbol in symbols.RENAMED:
            new = symbols.RENAMED[symbol]
            notes.append(f"{symbol} now trades as {new}, with the same price history; reviewed {new}.")
            symbol = new
        listings = self.market.listings()
        if not listings:
            return symbol, None, notes, None
        hit = symbols.lookup(listings, symbol)
        words = [w for w in re.split(r"\W+", (company or "").lower()) if w and w not in ("inc", "the", "shares", "stock")]
        if words and hit and not from_form:
            name_words = re.split(r"\W+", hit.name.lower())
            matched = [w for w in words if any(nw.startswith(w) for nw in name_words)]
            if matched and len(matched) < len(words):
                top = symbols.search(listings, company, 1)
                if top and top[0].symbol != symbol and all(
                        any(nw.startswith(w) for nw in re.split(r"\W+", top[0].name.lower())) for w in words):
                    notes.append(f"Read '{company}' as {top[0].symbol} ({top[0].name}), not {symbol} ({hit.name}).")
                    symbol, hit = top[0].symbol, top[0]
        return symbol, (hit.name if hit else None), notes, None

    def suggest(self, text: str) -> list[str]:
        listings = self.market.listings()
        if listings and text:
            return [f"{l.name} ({l.symbol})" for l in symbols.search(listings, text, 4)]
        return resolve.suggest(text)

    @capability("search_symbols", SymbolIn, SymbolsOut, effect="read", budget_ms=1500)
    def search_symbols(self, req: SymbolIn) -> SymbolsOut:
        """Find a US-listed stock or ETF by symbol or company name."""
        listings = self.market.listings()
        if listings is None:
            raise CapabilityUnavailable("the symbol directory is unavailable right now")
        q, note, first = req.q.strip(), None, []
        sym = symbols.yahoo_symbol(q)
        if intake.is_bond_query(q):
            note, first = intake.BOND_MESSAGE, [symbols.lookup(listings, s) for s in intake.BOND_ETFS]
        elif intake.scope_problem(q, q if "." in q else None) or intake.scope_problem(q, None):
            note = intake.FOREIGN_MESSAGE
        elif sym in symbols.RENAMED:
            note, first = f"{sym} now trades as {symbols.RENAMED[sym]}.", [symbols.lookup(listings, symbols.RENAMED[sym])]
        elif sym in symbols.DELISTED:
            note = f"{sym} no longer trades: {symbols.DELISTED[sym]}."
        hits = [h for h in first if h] + symbols.search(listings, q, req.limit)
        hits = list(dict.fromkeys(hits))[: req.limit]
        if not hits and note is None:
            note = f"No US-listed stock or ETF matches '{q}'."
        return SymbolsOut(results=[SymbolHit(**vars(h)) for h in hits], note=note)

    @capability("trading_days", TradingDaysIn, TradingDaysOut, effect="read", budget_ms=3000)
    def trading_days(self, req: TradingDaysIn) -> TradingDaysOut:
        """Trading days between a buy date and a sell date (or the latest close), counted on the exchange calendar."""
        today = self.market.today()
        _, dates = self.market.market_closes(today)
        try:
            w = calendar.trading_window(dates, today, req.date, req.sell_date, still_holding=not req.sell_date, max_days=None)
        except calendar.WindowProblem as exc:
            return TradingDaysOut(entry_date=None, exit_date=None, days=None, notes=[exc.message])
        if w.days > intake.MAX_HORIZON:
            w.notes.append(f"The luck check covers holds of up to {intake.MAX_HORIZON} trading days (about a year).")
        return TradingDaysOut(entry_date=dates[w.i], exit_date=dates[w.j], days=w.days, notes=w.notes)

    @capability("parse_trade", ParseIn, ParseOut, effect="llm", budget_ms=10000)
    def parse_trade(self, req: ParseIn) -> ParseOut:
        """Read a trade written in words into form fields, for the person to check before the review."""
        parsed, parser_name, notes = self.parse(req.text, self.market.today(), {})
        notes += [f"Unclear: {u}" for u in parsed.unclear]
        symbol, problem, _ = resolve.resolve_ticker(parsed.ticker, parsed.company)
        name = None
        if problem == "crypto":
            notes.append(intake.CRYPTO_MESSAGE)
        elif msg := intake.scope_problem(f"{req.text} {parsed.company or ''}", parsed.ticker if symbol else None):
            notes.append(msg)
        elif symbol:
            symbol, name, more, refusal = self.check_symbol(symbol, parsed.company, from_form=False)
            notes += more + ([refusal] if refusal else [])
        if intake.mentions_options(req.text):
            notes.append(intake.OPTIONS_MESSAGE)
        return ParseOut(ticker=symbol, name=name, date=parsed.date, sell_date=parsed.sell_date,
                        still_holding=parsed.still_holding, direction=parsed.direction, horizon_days=parsed.horizon_days,
                        reasoning=parsed.reasoning, notes=notes, parser=parser_name)

    def classify(self, reasoning: str) -> tuple[list[BiasSpan], str | None, str | None]:
        if len(reasoning.split()) < 3:
            return [], None, "No reasoning was given, so there was nothing to check for bias signals."
        result, source = self.sentiment.classify_with_fallback(reasoning)
        if result is None:
            return [], None, "No bias classifier was available."
        spans = []
        for sig in result.signals:
            span = explain.evidence_span(reasoning, sig.evidence)
            spans.append(BiasSpan(label=sig.label.value, evidence=sig.evidence, confidence=sig.confidence,
                                  start=span[0] if span else None, end=span[1] if span else None))
        note = None if spans else "No bias signals were found in your reasoning."
        return spans, source, note

    @capability("review_trade", ReviewIn, ReviewOut, effect="llm", budget_ms=20000)
    def review_trade(self, req: ReviewIn) -> ReviewOut:
        """Check whether a trade's outcome was luck, point out bias signals in the reasoning, and put it in market context."""
        return trade.run(self, req)
