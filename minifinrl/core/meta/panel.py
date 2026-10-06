"""
The one feature panel every script trains, backtests and fits the HMM on.

PROVENANCE: original to this project. Before this, 03_train.py and
04_backtest.py each downloaded their own date window and computed features
on it. The test window then had no history before 2023-01-01, so
turbulence (252-day lookback) was 0 on 254 of its 874 days, which is an input
pattern the agents never saw in training. Here features are computed once
on the full 2014 -> TEST_END history, and train/test are slices of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from minifinrl.core.configs.logging_config import get_logger
from minifinrl.core.configs.settings import (
    DATA_STORE,
    INDICATORS,
    PRICE_PROVIDER,
    TEST_END,
    TICKERS,
    TRAIN_START,
    TURBULENCE_LOOKBACK,
    USE_TURBULENCE,
    VIX_PROVIDER,
)
from minifinrl.core.meta.data import RAW_COLUMNS
from minifinrl.core.meta.features import add_indicators, add_turbulence, clean_panel
from minifinrl.core.meta.market_data import load_market_data

log = get_logger(__name__)


@dataclass(frozen=True)
class FeaturePanel:
    panel: pd.DataFrame  # tidy, with INDICATORS (+ turbulence) columns
    vix: pd.Series
    data_hash: str  # of the raw panel, from the provenance manifest


def compute_features(raw: pd.DataFrame) -> pd.DataFrame:
    panel = clean_panel(raw)
    panel = add_indicators(panel, INDICATORS)
    if USE_TURBULENCE:
        panel = add_turbulence(panel, TURBULENCE_LOOKBACK)
    return panel


@dataclass(frozen=True)
class DataSpec:
    """Everything that decides which raw data a panel is built from.
    Hashable, so it keys the per-process memo."""

    start: str = TRAIN_START
    end: str = TEST_END
    tickers: tuple[str, ...] = tuple(TICKERS)
    price_provider: str = PRICE_PROVIDER
    vix_provider: str = VIX_PROVIDER
    store_root: str = str(DATA_STORE)
    manifest_dir: str | None = None
    rate_provider: str = "fred"
    refresh: bool = True
    today: str | None = None


@lru_cache(maxsize=4)
def _build(spec: DataSpec) -> FeaturePanel:
    md = load_market_data(
        list(spec.tickers), spec.start, spec.end, mode="research", price_provider=spec.price_provider,
        vix_provider=spec.vix_provider, refresh=spec.refresh, store_root=spec.store_root,
        today=spec.today, manifest_dir=spec.manifest_dir,
    )
    panel = compute_features(md.panel)
    log.info("feature panel %s..%s: %s, data_hash %s", spec.start, spec.end, panel.shape, md.manifest["data_hash"][:12])
    return FeaturePanel(panel=panel, vix=md.vix, data_hash=md.manifest["data_hash"])


def build_panel(spec: DataSpec | None = None) -> FeaturePanel:
    """Validated raw data -> features on the full history. Memoized per
    process (a backtest run builds one env per model x path); callers get
    copies, so mutating one can't corrupt the next."""
    fp = _build(spec or DataSpec())
    return FeaturePanel(panel=fp.panel.copy(), vix=fp.vix.copy(), data_hash=fp.data_hash)


def features_with_history(path_raw: pd.DataFrame, history_raw: pd.DataFrame, warmup_days: int) -> pd.DataFrame:
    """Features for a synthetic path that continues from the real history.

    A synthetic path starts at the last real price, so its indicators and
    turbulence should see the real days before it, the same way the test
    window does. Computed on the path alone, turbulence would be 0 for its
    first TURBULENCE_LOOKBACK days, which is half of a 500-day path. The real
    tail is prepended, features computed, then only path dates are kept.
    """
    tail_dates = sorted(history_raw["date"].unique())[-warmup_days:]
    if tail_dates and tail_dates[-1] >= path_raw["date"].min():
        raise ValueError(f"history ends {tail_dates[-1]}, not before the path start {path_raw['date'].min()}")
    tail = history_raw.loc[history_raw["date"].isin(tail_dates), RAW_COLUMNS]
    combined = pd.concat([tail, path_raw[RAW_COLUMNS]], ignore_index=True).sort_values(["date", "tic"])
    feats = compute_features(combined)
    out = feats[feats["date"] >= path_raw["date"].min()].reset_index(drop=True)
    if out["date"].nunique() != path_raw["date"].nunique():
        raise ValueError("path dates lost while computing features")
    return out
