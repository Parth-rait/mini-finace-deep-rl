"""Where research inputs and outputs live. Kept apart from pipeline.py so
code that only reads results never loads the training stack (torch)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from minifinrl.platform.settings import DATA_PROCESSED, DATA_SYNTHETIC, LOG_DIR, MODEL_DIR, RESULTS, ROOT

RESULTS_FILE = "backtest_results.csv"
EXPERIMENT_DIR = RESULTS / "experiments"
EXPERIMENTS_MD = ROOT / "EXPERIMENTS.md"


@dataclass(frozen=True)
class Paths:
    model_dir: Path = MODEL_DIR
    synthetic_dir: Path = DATA_SYNTHETIC
    processed_dir: Path = DATA_PROCESSED
    results_path: Path = LOG_DIR / RESULTS_FILE
