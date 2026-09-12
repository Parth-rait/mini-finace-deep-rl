"""Fetch raw OHLCV for the configured universe plus VIX, for the full
train+test date range, and cache both to data/raw/."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.logging_config import get_logger
from configs.settings import TEST_END, TICKERS, TRAIN_START
from meta.data import download, download_vix

log = get_logger(__name__)


def main() -> None:
    panel = download(TICKERS, TRAIN_START, TEST_END)
    vix = download_vix(TRAIN_START, TEST_END)
    log.info("downloaded %d rows for %d tickers", len(panel), panel["tic"].nunique())
    log.info("downloaded %d VIX rows", len(vix))


if __name__ == "__main__":
    main()
