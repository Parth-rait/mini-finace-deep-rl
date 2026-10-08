"""
Journal capabilities: make a journal ID, log trades under it, review them
(the trade review, saved with the trade), close them, and delete them.

Every call that touches trades takes the profile ID, and the store scopes
every query by it, so an ID only ever reaches its own trades.
"""

from __future__ import annotations

from minifinrl.journal import ids
from minifinrl.journal.schemas import (
    CloseTradeIn,
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
from minifinrl.platform.schemas import Empty
from minifinrl.review.schemas import ReviewIn, ReviewOut


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


class JournalService:
    def __init__(self, cfg, review):
        self.cfg = cfg
        self.review = review  # the review service, for reviewing logged trades
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
        saved = self.store.update_trade(t["profile_id"], t["id"], result=summarise(review, self.review.market.today()))
        return ReviewedTrade(trade=TradeRecord(**saved), review=review)
