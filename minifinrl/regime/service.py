"""
Regime capabilities: the luck test. Also the one cache of fitted regime
models, shared by the luck_test capability and the trade review (fitting
takes about a second, and many questions share an entry day).
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import pandas as pd

from minifinrl.market.panel import build_panel
from minifinrl.platform.capabilities import CapabilityUnavailable, capability
from minifinrl.regime.luck import LuckTestError, luck_test
from minifinrl.regime.model import RegimeSyntheticGenerator
from minifinrl.regime.schemas import LuckTestIn, LuckTestOut


class RegimeService:
    CACHE_SIZE = 32  # fitted regime models (~1 s to fit each)

    def __init__(self, cfg):
        self.cfg = cfg
        self._fits: OrderedDict[str, RegimeSyntheticGenerator] = OrderedDict()
        self._lock = threading.Lock()

    def fitter(self, market_columns: list[str] | None = None):
        """A fit(prices, vix) function for luck_test/outcome_surface that reuses
        models fitted on the same data. The fit only ever sees data up to the
        entry day (the last row), so the key is that day plus the columns."""
        def fit(prices: pd.DataFrame, vix: pd.Series) -> RegimeSyntheticGenerator:
            key = f"{prices.index[-1]}|{','.join(prices.columns)}|{','.join(market_columns or [])}"
            with self._lock:
                if key in self._fits:
                    self._fits.move_to_end(key)
                    return self._fits[key]
            gen = RegimeSyntheticGenerator().fit(prices, vix, market_columns=market_columns)
            with self._lock:
                self._fits[key] = gen
                while len(self._fits) > self.CACHE_SIZE:
                    self._fits.popitem(last=False)
            return gen
        return fit

    @capability("luck_test", LuckTestIn, LuckTestOut, effect="compute", budget_ms=3000)
    def luck_test(self, req: LuckTestIn) -> LuckTestOut:
        """Was one trade's outcome inside the normal range of luck, given the market regime on its entry day?"""
        fp = build_panel(self.cfg.spec)
        close = fp.panel.pivot(index="date", columns="tic", values="close")
        try:
            r = luck_test(close, fp.vix, req.ticker, req.date, req.direction, req.horizon_days,
                          n_paths=req.n_paths, seed=req.seed, fit=self.fitter())
        except LuckTestError as exc:
            raise CapabilityUnavailable(str(exc)) from exc
        return LuckTestOut(**vars(r))
