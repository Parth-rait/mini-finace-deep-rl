"""
Journal capabilities: make a journal ID, log trades under it, review them
(the trade review, saved with the trade), close them, and delete them.

Every call that touches trades takes the profile ID, and the store scopes
every query by it, so an ID only ever reaches its own trades.
"""

from __future__ import annotations

from minifinrl.journal import adherence, ids
from minifinrl.journal.schemas import (
    AccountIn,
    CloseTradeIn,
    Group,
    InsightsOut,
    PlanSaveIn,
    SavedPlan,
    StartPlanIn,
    DeletedOut,
    JournalSummary,
    ProfileIn,
    ProfileOut,
    ReviewedTrade,
    TradeIn,
    TradeRecord,
    TradeRef,
    TradesOut,
)
from minifinrl.journal.settings import DATABASE_URL, MAX_TRADES_PER_PROFILE
from minifinrl.journal.store import JournalStore
from minifinrl.platform.capabilities import CapabilityError, CapabilityNotFound, capability
from minifinrl.platform.log import get_logger
from minifinrl.plan.schemas import PlanIn
from minifinrl.platform.schemas import Empty
from minifinrl.review.schemas import ReviewIn, ReviewOut

log = get_logger(__name__)


class BadRequest(CapabilityError):
    kind = "bad_input"


def summarise(review: ReviewOut, as_of: str) -> dict:
    """What is kept with the trade from a review: the headline numbers, or
    why it couldn't run."""
    out = {"status": review.status, "as_of": as_of}
    if review.status != "ok":
        out["message"] = review.messages[-1] if review.messages else None
        return out
    o = review.outcome
    out.update(entry_date=o.entry_date, exit_date=o.exit_date, days=o.horizon_days, your_return=o.realized_return,
               low=o.band_low, high=o.band_high, percentile=o.percentile, verdict=o.verdict, regime=o.regime,
               spy_return=review.market.spy_return if review.market else None,
               biases=sorted({b.label for b in review.biases}))
    return out


def plan_summary(plan) -> dict:
    """What is kept from a plan check: the risk, the odds, the range signed up for, and the nudges given."""
    o = plan.outlook
    out = {"as_of": o.as_of if o else None, "horizon_days": plan.horizon_days, "last_price": o.last_price if o else None,
           "nudges": [n.model_dump() for n in plan.nudges], "biases": sorted({b.label for b in plan.biases}),
           "risk": plan.risk.model_dump() if plan.risk else None}
    if o:
        out.update(p05=o.p05, p50=o.p50, p95=o.p95, prob_loss=o.prob_loss, regime=o.regime, prob_stop=o.prob_stop,
                   prob_target=o.prob_target, prob_target_first=o.prob_target_first)
    return out


def state_of(review: ReviewOut) -> dict | None:
    """The state read from the reasoning, kept with the trade as it was when logged."""
    return review.state.model_dump(mode="json") if review.state else None


class JournalService:
    def __init__(self, cfg, review, plan=None):
        self.cfg = cfg
        self.review = review  # the review service, for reviewing logged trades
        self.plan = plan  # the plan service, for checking planned trades
        self.store = JournalStore(getattr(cfg, "database_url", None) or DATABASE_URL)

    def _profile(self, profile_id: str) -> dict:
        p = self.store.profile(profile_id)
        if p is None:
            raise CapabilityNotFound("No journal has that ID. Check it, or start a new journal.")
        return p

    def _trade(self, profile_id: str, trade_id: str) -> dict:
        self._profile(profile_id)
        t = self.store.trade(profile_id, trade_id)
        if t is None:
            raise CapabilityNotFound("That trade isn't in this journal.")
        return t

    # ---- profiles ---------------------------------------------------------------------------

    @capability("create_profile", Empty, ProfileOut, effect="write", budget_ms=1000)
    def create_profile(self, req: Empty) -> ProfileOut:
        """Start a new journal and get its ID. Keep the ID private: it is the only key to the journal."""
        for _ in range(5):  # a collision is astronomically unlikely, but cheap to rule out
            pid = ids.new_profile_id()
            if self.store.profile(pid) is None:
                return ProfileOut(**self.store.create_profile(pid), trades=0)
        raise CapabilityError("could not allocate a journal ID")

    @capability("get_profile", ProfileIn, ProfileOut, effect="read", budget_ms=500)
    def get_profile(self, req: ProfileIn) -> ProfileOut:
        """Open a journal by its ID."""
        return ProfileOut(**self._profile(req.profile_id), trades=self.store.count_trades(req.profile_id))

    @capability("set_account_size", AccountIn, ProfileOut, effect="write", budget_ms=1000)
    def set_account_size(self, req: AccountIn) -> ProfileOut:
        """Remember your trading account size, so plans show risk as a share of it. Empty forgets it."""
        self._profile(req.profile_id)
        p = self.store.set_account_size(req.profile_id, req.account_size)
        return ProfileOut(**p, trades=self.store.count_trades(req.profile_id))

    @capability("delete_profile", ProfileIn, DeletedOut, effect="write", budget_ms=2000)
    def delete_profile(self, req: ProfileIn) -> DeletedOut:
        """Delete a journal and every trade in it. This can't be undone."""
        self._profile(req.profile_id)
        return DeletedOut(deleted=self.store.delete_profile(req.profile_id))

    # ---- trades -----------------------------------------------------------------------------

    @capability("log_trade", TradeIn, TradeRecord, effect="write", budget_ms=1000)
    def log_trade(self, req: TradeIn) -> TradeRecord:
        """Save a trade to your journal: planned, open (still held) or closed (with a sell date)."""
        self._profile(req.profile_id)
        if self.store.count_trades(req.profile_id) >= MAX_TRADES_PER_PROFILE:
            raise BadRequest(f"a journal holds up to {MAX_TRADES_PER_PROFILE} trades")
        status = "planned" if req.planned else ("closed" if req.sell_date else "open")
        row = self.store.add_trade({
            "id": ids.new_trade_id(), "profile_id": req.profile_id, "status": status,
            "ticker": req.ticker.strip().upper(), "name": req.name, "direction": req.direction,
            "buy_date": req.buy_date, "sell_date": req.sell_date, "price_paid": req.price_paid,
            "price_sold": req.price_sold, "quantity": req.quantity,
            "reasoning": (req.reasoning or "").strip() or None, "state": None, "result": None,
        })
        return TradeRecord(**row)

    @capability("list_trades", ProfileIn, TradesOut, effect="read", budget_ms=1000)
    def list_trades(self, req: ProfileIn) -> TradesOut:
        """Every trade in your journal, newest first."""
        self._profile(req.profile_id)
        rows = [TradeRecord(**r) for r in self.store.trades(req.profile_id)]
        count = {s: sum(r.status == s for r in rows) for s in ("planned", "open", "closed")}
        return TradesOut(trades=rows, summary=JournalSummary(count=len(rows), **count))

    @capability("review_logged_trade", TradeRef, ReviewedTrade, effect="llm", budget_ms=20000)
    def review_logged_trade(self, req: TradeRef) -> ReviewedTrade:
        """Run the trade review on a logged trade (an open one is measured to the latest close) and save the result with it."""
        t = self._trade(req.profile_id, req.trade_id)
        if t["status"] == "planned":
            raise BadRequest("A planned trade has no outcome yet. Log it as open once you've made it.")
        return self._review_and_save(t)

    @capability("close_trade", CloseTradeIn, ReviewedTrade, effect="llm", budget_ms=20000)
    def close_trade(self, req: CloseTradeIn) -> ReviewedTrade:
        """Record that you sold (or covered) an open trade, then review it and save the result."""
        t = self._trade(req.profile_id, req.trade_id)
        if t["status"] == "planned" or not t["buy_date"]:
            raise BadRequest("Add the buy date first: a planned trade can't be closed.")
        if req.sell_date <= t["buy_date"]:
            raise BadRequest("The sell date has to be after the buy date.")
        t = self.store.update_trade(req.profile_id, req.trade_id, status="closed", sell_date=req.sell_date,
                                    price_sold=req.price_sold if req.price_sold is not None else t["price_sold"])
        return self._review_and_save(t)

    @capability("save_plan", PlanSaveIn, SavedPlan, effect="llm", budget_ms=20000)
    def save_plan(self, req: PlanSaveIn) -> SavedPlan:
        """Check a planned trade and keep it in your journal, with the state your reason was written in and the range you signed up for."""
        profile = self._profile(req.profile_id)
        if self.store.count_trades(req.profile_id) >= MAX_TRADES_PER_PROFILE:
            raise BadRequest(f"a journal holds up to {MAX_TRADES_PER_PROFILE} trades")
        if req.stop_price is None:
            raise BadRequest("A plan needs a stop: the price at which you'd accept you were wrong.")
        if len(req.reasoning.split()) < 3:
            raise BadRequest("A plan needs a reason, in a sentence or two.")
        fields = req.model_dump(include=set(PlanIn.model_fields))
        fields["account_size"] = req.account_size or profile.get("account_size")
        plan = self.plan.check_plan(PlanIn(**fields))
        if plan.status != "ok":
            raise BadRequest(plan.messages[-1] if plan.messages else "This plan can't be checked.")
        row = self.store.add_trade({
            "id": ids.new_trade_id(), "profile_id": req.profile_id, "status": "planned", "ticker": plan.ticker,
            "name": req.name or plan.name, "direction": req.direction, "buy_date": req.planned_date, "sell_date": None,
            "price_paid": None, "price_sold": None, "quantity": None, "reasoning": req.reasoning.strip() or None,
            "state": plan.state.model_dump(mode="json") if plan.state else None, "result": None,
            "plan_entry": plan.entry_price, "plan_stop": plan.stop_price, "plan_target": plan.target_price,
            "plan_amount": plan.amount, "plan_horizon": plan.horizon_days, "plan": plan_summary(plan),
        })
        return SavedPlan(trade=TradeRecord(**row), plan=plan)

    @capability("start_planned_trade", StartPlanIn, ReviewedTrade, effect="llm", budget_ms=20000)
    def start_planned_trade(self, req: StartPlanIn) -> ReviewedTrade:
        """You made a planned trade: record the buy date, and review it so far. The plan's state and range are kept for comparison."""
        t = self._trade(req.profile_id, req.trade_id)
        if t["status"] != "planned":
            raise BadRequest("Only a planned trade can be started.")
        planned_on = t["created_at"][:10]
        if req.buy_date < planned_on:
            raise BadRequest(f"A plan comes before the trade: the buy date can't be before you saved the plan ({planned_on}). "
                             "For a trade made earlier, use Review instead.")
        t = self.store.update_trade(req.profile_id, req.trade_id, status="open", buy_date=req.buy_date,
                                    price_paid=req.price_paid if req.price_paid is not None else t["price_paid"])
        return self._review_and_save(t)

    @capability("journal_insights", ProfileIn, InsightsOut, effect="read", budget_ms=1000)
    def journal_insights(self, req: ProfileIn) -> InsightsOut:
        """What your closed trades say about you: results when you followed your plan or not, and by the state your reasoning was in."""
        self._profile(req.profile_id)
        rows = self.store.trades(req.profile_id)
        closed = [t for t in rows if t["status"] == "closed" and (t.get("result") or {}).get("status") == "ok"]

        def group(key, label, ts):
            rets = [t["result"]["your_return"] for t in ts]
            return Group(key=key, label=label, trades=len(ts), avg_return=sum(rets) / len(rets) if rets else None,
                         win_rate=sum(r > 0 for r in rets) / len(rets) if rets else None)

        def verdict(t):
            pc = t["result"].get("plan_check")
            return pc["verdict"] if pc else "no_plan"
        by_plan = [group(k, lbl, [t for t in closed if verdict(t) == k]) for k, lbl in
                   (("followed", "Followed the plan"), ("broke", "Broke the plan"), ("early", "Sold early"), ("no_plan", "No plan"))]
        lvl = lambda t: (t.get("state") or {}).get("level", "none")  # noqa: E731
        by_state = [group(k, lbl, [t for t in closed if lvl(t) == k]) for k, lbl in
                    (("calm", "Calm"), ("warm", "Warm"), ("hot", "Hot"), ("none", "No reason given"))]
        attention = [{"trade_id": t["id"], "ticker": t["ticker"], "label": t["result"]["plan_check"]["label"],
                      "message": t["result"]["plan_check"]["message"]}
                     for t in rows if t["status"] == "open" and ((t.get("result") or {}).get("plan_check") or {}).get("verdict") in ("stop", "target", "time")]
        headlines = []
        f, b = by_plan[0], by_plan[1]
        if f.trades >= 2 and b.trades >= 2:
            headlines.append(f"When you followed your plan: {f.trades} trades, average {f.avg_return:+.1%}. "
                             f"When you broke it: {b.trades} trades, average {b.avg_return:+.1%}.")
        calm, hot = by_state[0], by_state[2]
        if calm.trades >= 2 and hot.trades >= 2:
            headlines.append(f"Trades planned calmly: average {calm.avg_return:+.1%}. Trades planned hot: average {hot.avg_return:+.1%}.")
        if not headlines:
            headlines.append("Not enough closed trades yet to compare. Once a few planned trades are closed, this shows how "
                             "following your plan, and the state you planned in, relate to your results.")
        all_rets = [t["result"]["your_return"] for t in closed]
        return InsightsOut(closed=len(closed), avg_return=sum(all_rets) / len(all_rets) if all_rets else None,
                           win_rate=sum(r > 0 for r in all_rets) / len(all_rets) if all_rets else None,
                           by_plan=by_plan, by_state=by_state, needs_attention=attention, headlines=headlines)

    @capability("delete_trade", TradeRef, DeletedOut, effect="write", budget_ms=1000)
    def delete_trade(self, req: TradeRef) -> DeletedOut:
        """Delete one trade from your journal. This can't be undone."""
        self._trade(req.profile_id, req.trade_id)
        return DeletedOut(deleted=self.store.delete_trade(req.profile_id, req.trade_id))

    def _review_and_save(self, t: dict) -> ReviewedTrade:
        review = self.review.review_trade(ReviewIn(
            text=t["reasoning"] or "", ticker=t["ticker"], date=t["buy_date"], direction=t["direction"],
            **({"sell_date": t["sell_date"]} if t["sell_date"] else {"still_holding": True}),
        ))
        fields = {"result": summarise(review, self.review.market.today())}
        if t.get("plan_stop") and review.status == "ok":
            try:
                px = self.review.market.closes([t["ticker"]], self.review.market.today())
                if t["ticker"] in px.columns:
                    fields["result"]["plan_check"] = adherence.check(t, px[t["ticker"]])
            except Exception as exc:  # the review stands without the plan comparison
                log.warning("plan comparison failed for %s: %s", t["ticker"], exc)
        if t.get("state") is None and review.state is not None:  # first reading wins: the state at the time
            fields["state"] = state_of(review)
        saved = self.store.update_trade(t["profile_id"], t["id"], **fields)
        return ReviewedTrade(trade=TradeRecord(**saved), review=review)
