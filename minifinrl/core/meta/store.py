"""
Per-ticker incremental price store.

PROVENANCE: original to this project. `meta/data.py` caches one CSV per
exact (tickers, start, end), so asking for one more day re-downloads every
ticker from 2014. Here each ticker has its own file plus a small JSON note
of the date range already fetched, and only the missing head/tail is
requested.

The trap this module exists to avoid: providers return back-adjusted
prices, and back-adjusted history is rescaled every time a dividend or
split happens. Rows fetched in March and rows fetched in June are on
different scales if AAPL paid a dividend in between, and just appending
them puts a fake price jump in the series. So:

1. the store keeps UNADJUSTED OHLC plus the provider's `adj_close`,
2. every incremental fetch re-requests STORE_OVERLAP_DAYS already-stored
   rows, and if the adj_close/close ratio moved on that overlap, the whole
   ticker is re-downloaded so every row shares one adjustment anchor,
3. the split/dividend adjustment is applied at read time, exactly as
   `meta/data.py` does it.

The end date is clamped to today (exclusive) in New York time, so an
in-progress intraday bar is never stored as a daily close.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from minifinrl.core.configs.logging_config import get_logger
from minifinrl.core.configs.settings import ADJ_RATIO_TOLERANCE, DATA_STORE, STORE_OVERLAP_DAYS
from minifinrl.core.meta.data import RAW_COLUMNS
from minifinrl.core.meta.providers import PROVIDER_COLUMNS, PriceProvider, ProviderError, SeriesProvider

log = get_logger(__name__)

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


def market_today() -> str:
    """Today's date in New York: the first date whose bar may still be open."""
    return pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file + rename, so a reader (the service) never sees
    a half-written file while a refresh runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def adjust(raw: pd.DataFrame) -> pd.DataFrame:
    """Unadjusted bars + adj_close -> adjusted OHLC. Same rule as
    meta/data.py: scale open/high/low/close by adj_close/close so corporate
    actions don't show up as overnight jumps. Volume is left as reported."""
    out = raw.copy()
    ratio = out["adj_close"] / out["close"]
    for col in ("open", "high", "low", "close"):
        out[col] = out[col] * ratio
    return out.drop(columns=["adj_close"])


class PriceStore:
    """One directory per provider: `<TIC>.csv` (unadjusted rows) and
    `<TIC>.json` ({covered_start, covered_end, fetched_at}). `covered_*` is
    the range that was *asked for*, not the range that had data, so a
    pre-IPO start date is not re-requested on every call."""

    def __init__(self, provider: PriceProvider, root: Path | str = DATA_STORE, *, today: str | None = None):
        self.provider = provider
        self.root = Path(root) / provider.name
        self._today = today  # tests pin this; None = real market date

    # ---- paths / metadata -------------------------------------------------

    def _csv(self, tic: str) -> Path:
        return self.root / f"{tic}.csv"

    def _meta(self, tic: str) -> Path:
        return self.root / f"{tic}.json"

    def coverage(self, tic: str) -> dict | None:
        p = self._meta(tic)
        return json.loads(p.read_text()) if p.exists() else None

    def last_bar(self, tic: str) -> str | None:
        """Newest stored date for `tic`, from the coverage note (or the CSV
        for stores written before the note recorded it). Disk only."""
        cov = self.coverage(tic)
        if cov is None:
            return None
        if cov.get("last_bar"):
            return cov["last_bar"]
        raw = self._read_raw(tic)
        return raw["date"].max() if len(raw) else None

    def _read_raw(self, tic: str) -> pd.DataFrame:
        p = self._csv(tic)
        if not p.exists():
            return pd.DataFrame(columns=PROVIDER_COLUMNS)
        return pd.read_csv(p, dtype={"date": str})

    def _save(self, tic: str, rows: pd.DataFrame, start: str, end: str) -> None:
        rows = rows.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
        _atomic_write_text(self._csv(tic), rows[PROVIDER_COLUMNS].to_csv(index=False))
        meta = {
            "provider": self.provider.name,
            "covered_start": start,
            "covered_end": end,
            "rows": int(len(rows)),
            "last_bar": rows["date"].max() if len(rows) else None,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        _atomic_write_text(self._meta(tic), json.dumps(meta, indent=2))

    # ---- update -------------------------------------------------------------

    def _end_cap(self, end: str) -> str:
        return min(end, self._today or market_today())

    @staticmethod
    def _ratios_moved(old: pd.DataFrame, new: pd.DataFrame) -> bool:
        both = old.merge(new, on="date", suffixes=("_old", "_new"))
        if both.empty:
            return False
        r_old = both["adj_close_old"] / both["close_old"]
        r_new = both["adj_close_new"] / both["close_new"]
        return bool((np.abs(r_new / r_old - 1.0) > ADJ_RATIO_TOLERANCE).any())

    def update(self, tic: str, start: str, end: str) -> None:
        """Make sure [start, min(end, today)) is covered on disk."""
        end = self._end_cap(end)
        if start >= end:
            return
        with _lock_for(str(self._csv(tic))):
            cov = self.coverage(tic)
            if cov is None:
                log.info("store[%s] %s: first fetch %s..%s", self.provider.name, tic, start, end)
                rows = self.provider.fetch(tic, start, end)
                if rows.empty:
                    # yfinance answers a rate limit with an empty frame, not
                    # an error. Caching that would mark the range "covered"
                    # forever, so an empty first fetch is never saved.
                    raise ProviderError(f"{self.provider.name} returned no rows for {tic} {start}..{end}")
                self._save(tic, rows, start, end)
                return

            lo, hi = min(start, cov["covered_start"]), max(end, cov["covered_end"])
            if lo == cov["covered_start"] and hi == cov["covered_end"]:
                log.debug("store[%s] %s: hit %s..%s", self.provider.name, tic, start, end)
                return

            old = self._read_raw(tic)
            pieces = [old]
            if lo < cov["covered_start"]:
                # overlap: the first STORE_OVERLAP_DAYS stored rows
                overlap_end = old["date"].iloc[min(STORE_OVERLAP_DAYS, len(old)) - 1] if len(old) else cov["covered_start"]
                head_end = (pd.Timestamp(max(overlap_end, cov["covered_start"])) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                pieces.append(self.provider.fetch(tic, lo, head_end))
            if hi > cov["covered_end"]:
                overlap_start = old["date"].iloc[-min(STORE_OVERLAP_DAYS, len(old))] if len(old) else cov["covered_end"]
                tail = self.provider.fetch(tic, min(overlap_start, cov["covered_end"]), hi)
                if tail.empty and len(old):
                    # the overlap rows exist on disk, so an empty answer is
                    # a silent provider failure, not "no new bars"
                    raise ProviderError(f"{self.provider.name} returned no rows for {tic} overlap from {overlap_start}")
                pieces.append(tail)

            if any(self._ratios_moved(old, p) for p in pieces[1:]):
                log.warning(
                    "store[%s] %s: adjustment ratio changed since last fetch (dividend/split) - refetching %s..%s",
                    self.provider.name, tic, lo, hi,
                )
                self._save(tic, self.provider.fetch(tic, lo, hi), lo, hi)
                return

            merged = pd.concat([p for p in pieces if not p.empty], ignore_index=True)
            log.info("store[%s] %s: extended coverage to %s..%s", self.provider.name, tic, lo, hi)
            self._save(tic, merged, lo, hi)

    # ---- read ---------------------------------------------------------------

    def read(self, tic: str, start: str, end: str) -> pd.DataFrame:
        """Adjusted bars for [start, end), columns meta.data.RAW_COLUMNS
        (the tidy schema meta.features expects). Reads disk only."""
        raw = self._read_raw(tic)
        raw = raw[(raw["date"] >= start) & (raw["date"] < end)]
        if raw.empty:
            return pd.DataFrame(columns=RAW_COLUMNS)
        out = adjust(raw)
        out["tic"] = tic
        return out[RAW_COLUMNS].reset_index(drop=True)

    def get(self, tickers: list[str], start: str, end: str, *, refresh: bool = True) -> pd.DataFrame:
        """Tidy panel for `tickers` in [start, end), sorted by (date, tic).
        `refresh=False` never touches the network (the service's read path)."""
        frames = []
        for tic in tickers:
            if refresh:
                self.update(tic, start, end)
            frames.append(self.read(tic, start, end))
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame(columns=RAW_COLUMNS)
        return pd.concat(frames, ignore_index=True).sort_values(["date", "tic"]).reset_index(drop=True)


class SeriesStore:
    """Same incremental idea for a single unadjusted series (VIX). No
    ratio check needed: an index level is never back-adjusted."""

    def __init__(self, provider: SeriesProvider, key: str, root: Path | str = DATA_STORE, *, today: str | None = None,
                 column: str = "vix"):
        self.provider = provider
        self.column = column
        self.path = Path(root) / provider.name / f"{key}.csv"
        self.meta_path = self.path.with_suffix(".json")
        self._today = today

    def get(self, start: str, end: str, *, refresh: bool = True) -> pd.Series:
        capped_end = min(end, self._today or market_today())
        if refresh and start < capped_end:
            with _lock_for(str(self.path)):
                self._update(start, capped_end)
        if not self.path.exists():
            return pd.Series(dtype=float, name=self.column)
        s = pd.read_csv(self.path, dtype={"date": str}, index_col="date")[self.column]
        return s[(s.index >= start) & (s.index < end)]

    def _update(self, start: str, end: str) -> None:
        cov = json.loads(self.meta_path.read_text()) if self.meta_path.exists() else None
        if cov is None:
            s, lo, hi = self.provider.fetch(start, end), start, end
        else:
            lo, hi = min(start, cov["covered_start"]), max(end, cov["covered_end"])
            if lo == cov["covered_start"] and hi == cov["covered_end"]:
                return
            old = pd.read_csv(self.path, dtype={"date": str}, index_col="date")[self.column]
            parts = [old]
            if lo < cov["covered_start"]:
                parts.append(self.provider.fetch(lo, cov["covered_start"]))
            if hi > cov["covered_end"]:
                parts.append(self.provider.fetch(cov["covered_end"], hi))
            s = pd.concat([p for p in parts if len(p)])
            s = s[~s.index.duplicated(keep="last")]
        s = s.sort_index().rename(self.column)
        s.index.name = "date"
        _atomic_write_text(self.path, s.to_csv(header=True))
        meta = {
            "provider": self.provider.name,
            "covered_start": lo,
            "covered_end": hi,
            "rows": int(len(s)),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        _atomic_write_text(self.meta_path, json.dumps(meta, indent=2))
