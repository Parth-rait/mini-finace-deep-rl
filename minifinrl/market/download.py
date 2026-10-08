"""
OHLCV + VIX download.

PROVENANCE: adapted from FinRL `meta/preprocessor/yahoodownloader.py`. We only
keep the Yahoo path (no Alpaca/WRDS) since that's all this project's
needs to do, and we cache to disk so the network is hit once.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from minifinrl.platform.log import get_logger
from minifinrl.platform.settings import DATA_RAW
from minifinrl.market.settings import VIX_TICKER

log = get_logger(__name__)

RAW_COLUMNS = ["date", "tic", "open", "high", "low", "close", "volume"]


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Recent yfinance returns MultiIndex columns (ticker, field) even for a
    single-ticker request. Always drop to the field level explicitly rather
    than assuming the old flat schema."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns.name = None
    return df


def _download_one(tic: str, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(tic, start=start, end=end, auto_adjust=False, progress=False)
    if raw.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)

    raw = _flatten_columns(raw).reset_index()
    raw = raw.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    # Use the split/dividend-adjusted close as the price series, scaling the
    # rest of OHLC by the same ratio, so corporate actions don't show up as
    # fake overnight price jumps.
    if "adj_close" in raw.columns:
        ratio = raw["adj_close"] / raw["close"]
        for col in ("open", "high", "low", "close"):
            raw[col] = raw[col] * ratio
        raw = raw.drop(columns=["adj_close"])

    raw["tic"] = tic
    raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
    return raw[RAW_COLUMNS]


def download(
    tickers: list[str], start: str, end: str, *, use_cache: bool = True
) -> pd.DataFrame:
    """Download OHLCV for `tickers` in [start, end) and return a tidy panel:
    columns [date, tic, open, high, low, close, volume], sorted by (date, tic).

    Caches the combined result to data/raw/ keyed by (tickers, start, end) so
    repeated runs never touch the network.
    """
    cache_name = f"ohlcv_{start}_{end}_{'-'.join(sorted(tickers))}.csv"
    cache_path = Path(DATA_RAW) / cache_name
    if use_cache and cache_path.exists():
        log.info("cache hit: %s", cache_path.name)
        return pd.read_csv(cache_path)

    log.info("downloading %d tickers from %s to %s (cache miss: %s)", len(tickers), start, end, cache_path.name)
    frames, failures = [], []
    for tic in tickers:
        df = _download_one(tic, start, end)
        if df.empty:
            failures.append(tic)
        else:
            frames.append(df)

    if not frames:
        log.error("no data downloaded for any ticker in %s", tickers)
        raise ValueError(f"no data downloaded for any ticker in {tickers}")
    if failures:
        log.warning("no data returned for: %s", failures)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["date", "tic"]).reset_index(drop=True)

    Path(DATA_RAW).mkdir(parents=True, exist_ok=True)
    panel.to_csv(cache_path, index=False)
    log.info("downloaded %d rows for %d tickers, cached to %s", len(panel), len(frames), cache_path)
    return panel


def download_vix(start: str, end: str, *, use_cache: bool = True) -> pd.Series:
    """Download the VIX close level, indexed by date string. Cached
    separately from the ticker panel since it's a single series reused
    across every run regardless of which tickers are in play."""
    cache_path = Path(DATA_RAW) / f"vix_{start}_{end}.csv"
    if use_cache and cache_path.exists():
        log.info("cache hit: %s", cache_path.name)
        return pd.read_csv(cache_path, index_col="date")["vix"]

    log.info("downloading %s from %s to %s", VIX_TICKER, start, end)
    raw = yf.download(VIX_TICKER, start=start, end=end, auto_adjust=False, progress=False)
    if raw.empty:
        log.error("no VIX data returned for %s to %s", start, end)
        raise ValueError(f"no VIX data returned for {start} to {end}")

    raw = _flatten_columns(raw).reset_index()
    raw = raw.rename(columns={"Date": "date", "Close": "vix"})
    raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
    out = raw.set_index("date")["vix"]

    Path(DATA_RAW).mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, header=True)
    log.info("downloaded %d VIX rows, cached to %s", len(out), cache_path)
    return out
