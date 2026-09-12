"""Compute features on the full historical panel, cache it, then fit the
regime HMM on it and sample synthetic paths to data/synthetic/."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.settings import (
    DATA_PROCESSED,
    DATA_SYNTHETIC,
    HMM_N_PATHS,
    HMM_PATH_LENGTH,
    INDICATORS,
    TEST_END,
    TICKERS,
    TRAIN_START,
    TURBULENCE_LOOKBACK,
)
from configs.logging_config import get_logger
from meta.data import download, download_vix
from meta.features import add_indicators, add_turbulence, clean_panel
from meta.synthetic import RegimeSyntheticGenerator, to_tidy_panel

log = get_logger(__name__)


def main() -> None:
    raw = download(TICKERS, TRAIN_START, TEST_END)
    vix = download_vix(TRAIN_START, TEST_END)

    panel = clean_panel(raw)
    panel = add_indicators(panel, INDICATORS)
    panel = add_turbulence(panel, TURBULENCE_LOOKBACK)

    Path(DATA_PROCESSED).mkdir(parents=True, exist_ok=True)
    panel.to_csv(Path(DATA_PROCESSED) / "panel.csv", index=False)
    log.info("processed panel: %s, cached to %s", panel.shape, Path(DATA_PROCESSED) / "panel.csv")

    prices = panel.pivot(index="date", columns="tic", values="close")
    generator = RegimeSyntheticGenerator().fit(prices, vix)

    Path(DATA_SYNTHETIC).mkdir(parents=True, exist_ok=True)
    for i, synthetic_prices in enumerate(generator.sample_many(HMM_N_PATHS, HMM_PATH_LENGTH)):
        synthetic_panel = to_tidy_panel(synthetic_prices, start_date=TEST_END)
        synthetic_panel = add_indicators(synthetic_panel, INDICATORS)
        synthetic_panel = add_turbulence(synthetic_panel, TURBULENCE_LOOKBACK)
        synthetic_panel.to_csv(Path(DATA_SYNTHETIC) / f"path_{i:03d}.csv", index=False)

    log.info("sampled %d synthetic paths of length %d to %s", HMM_N_PATHS, HMM_PATH_LENGTH, DATA_SYNTHETIC)


if __name__ == "__main__":
    main()
