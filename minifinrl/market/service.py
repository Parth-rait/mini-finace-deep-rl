"""
Market capabilities, and the shared price access every other feature uses
(today's market date, the price store, closes, VIX, the symbol directory).
"""

from __future__ import annotations

import pandas as pd

from minifinrl.market import symbols
from minifinrl.market.crosscheck import crosscheck
from minifinrl.market.dataset import load_market_data
from minifinrl.market.providers import ProviderError, get_price_provider, get_vix_provider
from minifinrl.market.schemas import (
    Bar,
    CrosscheckIn,
    CrosscheckOut,
    FetchIn,
    FetchOut,
    Snapshot,
    SnapshotIn,
    TickerAgreement,
)
from minifinrl.market.settings import TRAIN_START
from minifinrl.market.store import PriceStore, SeriesStore, market_today
from minifinrl.platform.capabilities import CapabilityUnavailable, capability
from minifinrl.platform.log import get_logger

log = get_logger(__name__)


class MarketService:
    def __init__(self, cfg):
        self.cfg = cfg
        self._symbols: tuple[str, list[symbols.Listing]] | None = None

    # ---- shared access (not capabilities) ---------------------------------------------

    def today(self) -> str:
        return self.cfg.today or market_today()

    def price_store(self) -> PriceStore:
        return PriceStore(get_price_provider(self.cfg.spec.price_provider), self.cfg.spec.store_root, today=self.cfg.today)

    def closes(self, tickers: list[str], end: str) -> pd.DataFrame:
        spec = self.cfg.spec
        panel = self.price_store().get(tickers, spec.start, end, refresh=spec.refresh)
        if panel.empty:
            return pd.DataFrame()
        return panel.pivot(index="date", columns="tic", values="close")

    def market_closes(self, today: str) -> tuple[pd.DataFrame, list[str]]:
        """The research universe's closes and its trading days (days every
        universe ticker traded): the calendar every holding window is counted on."""
        universe = list(self.cfg.spec.tickers)
        try:
            closes = self.closes(universe, today)
        except ProviderError as exc:
            raise CapabilityUnavailable(f"price data is unavailable right now: {exc}") from exc
        if closes.empty:
            raise CapabilityUnavailable("price data is unavailable right now")
        return closes, list(closes.index[closes[universe].notna().all(axis=1)])

    def vix(self, today: str) -> pd.Series:
        spec = self.cfg.spec
        return SeriesStore(get_vix_provider(spec.vix_provider), key="VIX", root=spec.store_root,
                           today=self.cfg.today).get(spec.start, today, refresh=spec.refresh)

    def listings(self) -> list[symbols.Listing] | None:
        """The US symbol directory, reloaded once a day. None if it can't be had:
        callers then run without the directory checks."""
        today = self.today()
        if self._symbols is None or self._symbols[0] != today:
            try:
                self._symbols = (today, symbols.load(self.cfg.symbols_dir, refresh=self.cfg.spec.refresh))
            except ProviderError as exc:
                log.warning("symbol directory unavailable: %s", exc)
                return None
        return self._symbols[1]

    # ---- read ---------------------------------------------------------------------------

    @capability("market_snapshot", SnapshotIn, Snapshot, effect="read", budget_ms=1000)
    def market_snapshot(self, req: SnapshotIn) -> Snapshot:
        """The last N daily bars per ticker from the local store. Never calls the provider."""
        tickers = req.tickers or list(self.cfg.spec.tickers)
        unknown = sorted(set(tickers) - set(self.cfg.spec.tickers))
        if unknown:
            raise CapabilityUnavailable(f"tickers {unknown} are not in the configured universe")
        panel = self.price_store().get(tickers, "1900-01-01", "9999-12-31", refresh=False)
        if panel.empty:
            raise CapabilityUnavailable("no market data on disk: run fetch_data first")
        keep = sorted(panel["date"].unique())[-req.days:]
        panel = panel[panel["date"].isin(keep)]
        return Snapshot(as_of=keep[-1], bars=[Bar(**r) for r in panel.to_dict("records")])

    # ---- batch (CLI only) -------------------------------------------------------------

    @capability("fetch_data", FetchIn, FetchOut, effect="batch")
    def fetch_data(self, req: FetchIn) -> FetchOut:
        """Fetch through the incremental store, validate, and write a provenance manifest."""
        spec = self.cfg.spec
        md = load_market_data(
            list(spec.tickers), req.start or TRAIN_START, req.end, mode=req.mode or self.cfg.mode,
            price_provider=spec.price_provider, vix_provider=spec.vix_provider,
            refresh=spec.refresh and not req.offline, strict=False, store_root=spec.store_root,
            today=self.cfg.today, manifest_dir=self.cfg.manifest_dir,
        )
        m = md.manifest
        return FetchOut(
            ok=md.report.ok, data_hash=m["data_hash"], rows=m["panel"]["rows"], requested=m["requested"],
            providers=m["providers"], errors=md.report.errors, warnings=md.report.warnings,
        )

    @capability("crosscheck_prices", CrosscheckIn, CrosscheckOut, effect="batch")
    def crosscheck_prices(self, req: CrosscheckIn) -> CrosscheckOut:
        """Do two price providers agree on daily returns over the same history? Run before trusting a new source."""
        spec = self.cfg.spec
        panels = []
        for name in (req.reference, req.candidate):
            store = PriceStore(get_price_provider(name), spec.store_root, today=self.cfg.today)
            panels.append(store.get(list(spec.tickers), req.start, req.end, refresh=spec.refresh))
        rows = crosscheck(panels[0], panels[1], list(spec.tickers))
        return CrosscheckOut(reference=req.reference, candidate=req.candidate,
                             agrees=all(r["agrees"] for r in rows), tickers=[TickerAgreement(**r) for r in rows])
