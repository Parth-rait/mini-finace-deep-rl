"""
The trade review, as named steps:

    understand -> resolve the symbol -> check nothing is missing -> load prices
    -> luck check -> market context -> explanation

Each step either adds to the review or stops it with a clear status
(needs_input, unsupported, rejected) and a message, so every input ends in an
answer, never a crash or a guess. `run` wires the steps; the service
(review/service.py) supplies the parsers, data and classifiers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from minifinrl.market import calendar, symbols
from minifinrl.market import validate as checks
from minifinrl.market.providers import ProviderError
from minifinrl.platform.log import get_logger
from minifinrl.regime.luck import LuckResult, LuckTestError, OutcomeSurface, luck_test, outcome_surface
from minifinrl.regime.model import MIN_TICKER_RETURNS
from minifinrl.regime.schemas import LuckTestOut
from minifinrl.review import explain, intake, resolve
from minifinrl.review.ports import ParsedTrade
from minifinrl.review.schemas import BiasSpan, MarketContext, ReviewIn, ReviewOut, Surface, Understood

log = get_logger(__name__)

N_PATHS, SEED = 1000, 0
VERDICT_WORDS = {"unusually_good": "unusually good, better than the model's normal range of luck",
                 "unusually_bad": "unusually bad, worse than the model's normal range of luck",
                 "within_luck_range": "within the normal range of luck"}


class Stop(Exception):
    """End the review early with a status and a message."""

    def __init__(self, status: str, message: str, **extra):
        super().__init__(message)
        self.status, self.message, self.extra = status, message, extra


@dataclass
class Review:
    """What the steps build up."""

    req: ReviewIn
    today: str
    parsed: ParsedTrade
    und: Understood
    overrides: dict
    messages: list[str] = field(default_factory=list)
    symbol: str | None = None
    universe: list[str] = field(default_factory=list)
    closes: pd.DataFrame | None = None
    dates: list[str] = field(default_factory=list)
    window: calendar.Window | None = None


# ---- steps ----------------------------------------------------------------------------------

def understand(svc, req: ReviewIn) -> Review:
    """The form fields are the trade; text fills what the form leaves empty.
    When the form holds the whole trade, the text is only the reasoning."""
    today = svc.market.today()
    overrides = {k: v for k, v in req.model_dump().items() if k != "text" and v is not None}
    form_complete = set(overrides) >= {"ticker", "date", "direction"} and bool(
        overrides.keys() & {"sell_date", "still_holding", "horizon_days"})
    if form_complete:
        parsed, parser_name, messages = ParsedTrade(**overrides, reasoning=req.text.strip() or None), "form", []
    else:
        parsed, parser_name, messages = svc.parse(req.text, today, overrides)
    und = Understood(**{k: getattr(parsed, k) for k in ("ticker", "company", "date", "direction", "horizon_days",
                                                        "reasoning", "sell_date", "still_holding")},
                     parser=parser_name)
    messages += [f"Unclear: {u}" for u in parsed.unclear]
    return Review(req=req, today=today, parsed=parsed, und=und, overrides=overrides, messages=messages)


def resolve_symbol(svc, rv: Review) -> list[str]:
    """Crypto, bonds and non-US listings stop here; renamed and delisted
    symbols are followed or explained. Returns suggestions for a missing symbol."""
    symbol, problem, suggestions = resolve.resolve_ticker(rv.parsed.ticker, rv.parsed.company)
    if problem == "crypto":
        raise Stop("unsupported", intake.CRYPTO_MESSAGE)
    if msg := intake.scope_problem(f"{rv.req.text} {rv.parsed.company or ''}", rv.parsed.ticker if symbol else None):
        raise Stop("unsupported", msg)
    if symbol:
        symbol, rv.und.name, more, refusal = svc.check_symbol(symbol, rv.parsed.company, from_form="ticker" in rv.overrides)
        rv.messages += more
        if refusal:
            rv.und.ticker = symbol
            raise Stop("rejected", refusal)
    rv.und.ticker = rv.symbol = symbol
    return suggestions


def require_details(svc, rv: Review, suggestions: list[str]) -> None:
    und = rv.und
    missing = [f for f in intake.REQUIRED if getattr(und, f) in (None, "")]
    if "date" in missing and und.sell_date and und.horizon_days:
        missing.remove("date")  # counted back from the sell date
    if not (und.sell_date or und.still_holding or und.horizon_days):
        missing.append("sell_date")
    if missing:
        company = rv.parsed.company
        if not rv.symbol and company:
            suggestions = svc.suggest(company)
            listings = svc.market.listings()
            words = [w for w in re.split(r"\W+", company.lower()) if w]
            if listings and words and not any(all(w in h.name.lower() for w in words)
                                              for h in symbols.search(listings, company, 8)):
                rv.messages.append(f"No US-listed stock or ETF is named '{company}'. " + intake.FOREIGN_MESSAGE)
        raise Stop("needs_input", "Some details of the trade are missing. Fill them in to run the review.",
                   missing=missing, suggestions=suggestions)
    if und.direction not in ("long", "short"):
        raise Stop("needs_input", f"Direction '{und.direction}' should be long or short.", missing=["direction"])
    if intake.mentions_options(rv.req.text):
        rv.messages.append(intake.OPTIONS_MESSAGE)


def load_prices(svc, rv: Review) -> pd.Series:
    """The market reference, the holding window on its calendar, and the
    target's own history, checked. Returns the target's closes."""
    und, symbol, today = rv.und, rv.symbol, rv.today
    rv.universe = list(svc.cfg.spec.tickers)
    rv.closes, rv.dates = svc.market.market_closes(today)  # the market reference must load, or nothing can be judged
    try:
        rv.window = w = calendar.trading_window(rv.dates, today, und.date, und.sell_date, und.still_holding,
                                                und.horizon_days, max_days=intake.MAX_HORIZON)
    except calendar.WindowProblem as exc:
        raise Stop(exc.status, exc.message, missing=exc.missing) from exc
    rv.messages += w.notes
    entry, exit_ = rv.dates[w.i], rv.dates[w.j]
    und.date = und.date or entry
    und.horizon_days = w.days
    if symbol not in rv.universe:
        # an unknown symbol and a provider hiccup both come back empty from
        # Yahoo, so a failure here is reported as "no history", not an outage
        try:
            extra = svc.market.closes([symbol], today)
        except ProviderError:
            extra = pd.DataFrame()
        if symbol not in extra.columns or extra[symbol].dropna().empty:
            raise Stop("rejected", f"No price history found for '{symbol}'. Check the symbol, or try again "
                       "later if the data source is busy.", suggestions=svc.suggest(rv.parsed.company or symbol))
        rv.closes = rv.closes.join(extra[[symbol]], how="outer")
    own = rv.closes[symbol].dropna()
    if own.index[0] > entry:
        raise Stop("rejected", f"{symbol}'s price history starts on {own.index[0]}, after the buy date {entry}. "
                   "Check the date, or whether the symbol belonged to a different company then.")
    if exit_ not in own.index:
        raise Stop("rejected", f"There is no {symbol} price on {exit_} (its data ends {own.index[-1]}).")
    n_hist = int((own.index <= entry).sum()) - 1
    if n_hist < MIN_TICKER_RETURNS:
        raise Stop("rejected", f"{symbol} had only {n_hist} trading days of history before {entry}. The luck check "
                   f"needs at least {MIN_TICKER_RETURNS} to learn how the stock moves.")
    if symbol != "SPY":
        try:  # context only: without it the review still runs
            spy_px = svc.market.closes(["SPY"], today)
            rv.closes = rv.closes.join(spy_px[["SPY"]], how="left") if "SPY" in spy_px.columns else rv.closes
        except ProviderError:
            rv.messages.append("The S&P 500 comparison is unavailable right now.")
    check_quality(rv, own)
    return own


def check_quality(rv: Review, own: pd.Series) -> None:
    symbol = rv.symbol
    target = own.reset_index().rename(columns={symbol: "close"}).assign(tic=symbol)
    target[["open", "high", "low"]] = target[["close"]].values.repeat(3, axis=1)
    target["volume"] = 0
    problems = checks.check_prices(target) + checks.check_gaps(target)
    if problems:
        raise Stop("rejected", f"The price data for {symbol} failed quality checks: {'; '.join(problems)}")
    # Daily moves above 50% are almost always bad data in the large-cap research
    # universe, but real for meme and small-cap stocks (AMC rose ~301% on
    # 2021-01-27). For a reviewed trade they are a warning, not a rejection.
    moves = own.pct_change()
    big = moves.abs()[moves.abs() > 0.5]
    if len(big):
        worst = big.idxmax()
        rv.messages.append(
            f"{symbol} has {len(big)} daily move{'s' if len(big) > 1 else ''} above 50% in its history (largest "
            f"{explain.pct(float(moves[worst]), 0)} on {worst}). The luck check assumes "
            "ordinary daily moves, so read its range with care for a stock like this."
        )


def check_luck(svc, rv: Review) -> tuple[LuckResult, OutcomeSurface]:
    symbol, universe = rv.symbol, rv.universe
    market_cols = universe + [symbol] if symbol not in universe else universe
    close = rv.closes.loc[rv.dates, market_cols]  # a recent listing is NaN before its first day; the fit allows that
    vix = svc.market.vix(rv.today)
    fit = svc.regime.fitter(universe)
    entry, days = rv.dates[rv.window.i], rv.window.days
    try:
        r = luck_test(close, vix, symbol, entry, rv.und.direction, days, n_paths=N_PATHS, seed=SEED, fit=fit)
        surf = outcome_surface(close, vix, symbol, entry, rv.und.direction, days, n_paths=N_PATHS, seed=SEED, fit=fit)
    except LuckTestError as exc:
        raise Stop("rejected", f"The luck check can't run for this trade: {exc}.") from exc
    rv.und.entry_date, rv.und.exit_date = r.entry_date, r.exit_date
    return r, surf


def what_you_did(rv: Review, r: LuckResult) -> str:
    und, symbol, days = rv.und, rv.symbol, rv.window.days
    verb = "bought" if und.direction == "long" else "shorted"
    if und.sell_date:
        return (f"You {verb} {symbol} on {r.entry_date} and {'sold' if und.direction == 'long' else 'covered'} "
                f"on {r.exit_date}, {days} trading days later.")
    if und.still_holding:
        return (f"You {verb} {symbol} on {r.entry_date} and still hold it; measured to the latest close on "
                f"{r.exit_date}, {days} trading days later.")
    return f"You {verb} {symbol} on {r.entry_date} and held it for {days} trading days, closing on {r.exit_date}."


def market_context(rv: Review, r: LuckResult) -> tuple[float | None, float]:
    """(S&P 500 return, equal-weight research-stocks return) over the same days."""
    spy = None
    if "SPY" in rv.closes.columns and rv.symbol != "SPY":
        s_ = rv.closes["SPY"].dropna()
        if r.entry_date in s_.index and r.exit_date in s_.index:
            spy = float(s_[r.exit_date] / s_[r.entry_date] - 1)
    uni = rv.closes.loc[rv.dates, rv.universe]
    return spy, float((uni.loc[r.exit_date] / uni.loc[r.entry_date] - 1).mean())


def explanation(svc, rv: Review, r: LuckResult, what: str, spans: list[BiasSpan], bias_note: str | None,
                spy: float | None, uni_ret: float) -> tuple[str, str]:
    """(text, source). The template always works; the model's version is used
    only if every number in it was computed (it writes placeholders, the
    code fills them in)."""
    # plain-English keys: the model copies wording from the facts, so no
    # code-style names may appear here (an earlier version leaked "spy_return")
    facts = {explain.F_WHAT: what,
             explain.F_OUTCOME: {explain.F_RETURN: explain.pct(r.realized_return), explain.F_LOW: explain.pct(r.band_low),
                                 explain.F_HIGH: explain.pct(r.band_high), explain.F_PCT: explain.ordinal(r.percentile),
                                 explain.F_VERDICT: VERDICT_WORDS[r.verdict], explain.F_REGIME: r.regime},
             explain.F_BIASES: [{"bias": b.label.replace("_", " "), "your words": b.evidence} for b in spans]}
    if bias_note:
        facts[explain.F_BIAS_NOTE] = bias_note
    if spy is not None:
        facts[explain.F_SPY] = explain.pct(spy)
    template = explain.template_explanation(facts)
    if svc.explainer is None:
        return template, "template"
    slots = {"ticker": rv.symbol, "entry_date": r.entry_date, "exit_date": r.exit_date,
             "days": str(rv.window.days), "your_return": explain.pct(r.realized_return),
             "low": explain.pct(r.band_low), "high": explain.pct(r.band_high),
             "percentile": explain.ordinal(r.percentile), "research_stocks_return": explain.pct(uni_ret)}
    if spy is not None:
        slots["spy_return"] = explain.pct(spy)
    text = ""
    try:
        text = svc.explainer.explain({**facts, "placeholders": dict(slots)})
        filled = explain.fill_slots(text, slots, [b.evidence for b in spans], rv.und.reasoning or rv.req.text)
        bad = explain.unsupported_numbers(filled, {**facts, "s": slots})  # second, independent check
        if not bad:
            return filled, "llm"
        log.warning("explanation rejected: numbers %s are not computed values", bad)
    except explain.SlotError as exc:
        log.warning("explanation rejected: %s | text: %.300s", exc, text)
    except Exception as exc:  # model unavailable: the template explanation stands
        log.warning("explainer failed: %s: %s", type(exc).__name__, exc)
    return template, "template"


# ---- the review ------------------------------------------------------------------------------

def run(svc, req: ReviewIn) -> ReviewOut:
    rv = understand(svc, req)
    try:
        suggestions = resolve_symbol(svc, rv)
        require_details(svc, rv, suggestions)
        load_prices(svc, rv)
        r, surf = check_luck(svc, rv)
    except Stop as stop:
        return ReviewOut(status=stop.status, messages=rv.messages + [stop.message], understood=rv.und, **stop.extra)
    what = what_you_did(rv, r)
    spans, bias_source, bias_note = svc.classify(rv.und.reasoning or "")
    reasoning = (rv.und.reasoning or "").strip()
    state = svc.sentiment.read_state_with_fallback(reasoning) if len(reasoning.split()) >= 3 else None
    spy, uni_ret = market_context(rv, r)
    text, source = explanation(svc, rv, r, what, spans, bias_note, spy, uni_ret)
    return ReviewOut(
        status="ok", messages=rv.messages, understood=rv.und, what_you_did=what, outcome=LuckTestOut(**vars(r)),
        surface=Surface(**vars(surf)), biases=spans, bias_source=bias_source, state=state,
        market=MarketContext(spy_return=spy, universe_return=uni_ret), explanation=text, explanation_source=source,
    )
