"""
Plan capability: the check before a trade. Uses the market data, the regime
model, the state reader and the review's symbol checks, all passed in by the
Engine.
"""

from __future__ import annotations

import pandas as pd

from minifinrl.market.providers import ProviderError
from minifinrl.plan import nudges, risk
from minifinrl.plan.schemas import OutlookOut, PlanIn, PlanOut, RiskOut
from minifinrl.platform.capabilities import capability
from minifinrl.platform.log import get_logger
from minifinrl.regime.luck import LuckTestError
from minifinrl.regime.model import MIN_TICKER_RETURNS
from minifinrl.regime.outlook import outcome_range
from minifinrl.review import intake, resolve

log = get_logger(__name__)


class PlanService:
    def __init__(self, cfg, market, regime, sentiment, review):
        self.cfg = cfg
        self.market, self.regime, self.sentiment, self.review = market, regime, sentiment, review

    def _prices(self, symbol: str) -> tuple[pd.DataFrame | None, list[str], str | None]:
        """(closes with the target joined, market trading days, reason it can't be used)."""
        today = self.market.today()
        universe = list(self.cfg.spec.tickers)
        closes, dates = self.market.market_closes(today)
        if symbol not in universe:
            try:
                extra = self.market.closes([symbol], today)
            except ProviderError:
                extra = pd.DataFrame()
            if symbol not in extra.columns or extra[symbol].dropna().empty:
                return None, dates, f"No price history found for '{symbol}'. Check the symbol."
            closes = closes.join(extra[[symbol]], how="outer")
        own = closes[symbol].dropna()
        if len(own) - 1 < MIN_TICKER_RETURNS:
            return None, dates, (f"{symbol} has only {len(own) - 1} trading days of history; the check needs at least "
                                 f"{MIN_TICKER_RETURNS} to learn how it moves.")
        return closes, dates, None

    @capability("check_plan", PlanIn, PlanOut, effect="llm", budget_ms=20000)
    def check_plan(self, req: PlanIn) -> PlanOut:
        """Before a trade: the risk of your plan in money, how often normal luck reaches your stop and target, the state your reasoning is written in, and what to watch. Never says buy or don't buy."""
        symbol, problem, _ = resolve.resolve_ticker(req.ticker, None)
        base = {"ticker": symbol or req.ticker.upper(), "direction": req.direction, "horizon_days": req.horizon_days,
                "planned_date": req.planned_date, "stop_price": req.stop_price, "target_price": req.target_price,
                "amount": req.amount}
        if problem == "crypto":
            return PlanOut(status="unsupported", messages=[intake.CRYPTO_MESSAGE], **base)
        if msg := intake.scope_problem(req.reasoning, req.ticker):
            return PlanOut(status="unsupported", messages=[msg], **base)
        symbol, name, messages, refusal = self.review.check_symbol(symbol, None, from_form=True)
        base["ticker"] = symbol
        if refusal:
            return PlanOut(status="rejected", messages=messages + [refusal], name=name, **base)
        if intake.mentions_options(req.reasoning):
            messages.append(intake.OPTIONS_MESSAGE)

        closes, dates, why_not = self._prices(symbol)
        if closes is None:
            return PlanOut(status="rejected", messages=messages + [why_not], name=name, **base)
        last = float(closes[symbol].dropna().iloc[-1])
        entry = req.entry_price or last
        bad = risk.problems(req.direction, entry, req.stop_price, req.target_price)
        if bad:
            return PlanOut(status="needs_input", messages=messages + bad, name=name, entry_price=entry, **base)
        if req.entry_price and abs(req.entry_price / last - 1) > 0.05:
            messages.append(f"Your entry is {req.entry_price / last - 1:+.0%} from the latest close ({last:,.2f}); the odds "
                            "below are simulated from the latest close.")

        universe = list(self.cfg.spec.tickers)
        cols = universe + [symbol] if symbol not in universe else universe
        stop_ret, target_ret = risk.as_returns(req.direction, last, req.stop_price, req.target_price)
        try:
            o = outcome_range(closes.loc[dates, cols], self.market.vix(self.market.today()), symbol, req.direction,
                              req.horizon_days, fit=self.regime.fitter(universe), stop_return=stop_ret, target_return=target_ret)
        except LuckTestError as exc:
            return PlanOut(status="rejected", messages=messages + [f"The check can't run: {exc}."], name=name, **base)
        r = risk.compute(req.direction, entry, req.stop_price, req.target_price, req.amount, req.account_size) if req.stop_price else None

        reasoning = req.reasoning.strip()
        state = self.sentiment.read_state_with_fallback(reasoning) if len(reasoning.split()) >= 3 else None
        spans, bias_source, _ = self.review.classify(reasoning)
        extra = nudges.plan_nudges(r, o, has_stop=req.stop_price is not None)
        return PlanOut(
            status="ok", messages=messages, name=name, entry_price=entry, **base,
            risk=RiskOut(**vars(r)) if r else None,
            outlook=OutlookOut(**{k: v for k, v in vars(o).items() if k in OutlookOut.model_fields}),
            state=state, biases=spans, bias_source=bias_source,
            nudges=nudges.build(state, spans, o, symbol, has_reasoning=bool(reasoning), extra=extra),
        )
