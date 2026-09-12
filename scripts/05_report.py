"""Turn raw backtest rows into the rank-stability summary this project
exists to produce: does the model that wins on real history keep winning
across synthetic paths, or was it luck?"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from configs.logging_config import get_logger
from configs.settings import LOG_DIR
from eval.report import aggregate_seeds, rank_stability

log = get_logger(__name__)


def main() -> None:
    results_path = Path(LOG_DIR) / "backtest_results.csv"
    if not results_path.exists():
        log.error("no backtest results at %s — run scripts/04_backtest.py first", results_path)
        raise SystemExit(1)

    log.info("reporting on %s", results_path)
    df = pd.read_csv(results_path)

    for app in df["app"].unique():
        sub = df[df["app"] == app]
        hist = sub[sub["path"] == "historical"]
        synth = sub[sub["path"] != "historical"]

        print(f"\n=== {app} ===")
        if not hist.empty:
            print("historical sharpe by model (median over seeds):")
            print(aggregate_seeds(hist.to_dict("records"), ["model"], "sharpe").to_string(index=False))

        if not synth.empty:
            print("\nsynthetic sharpe by model (median over seeds & paths):")
            print(aggregate_seeds(synth.to_dict("records"), ["model"], "sharpe").to_string(index=False))

            print("\nsynthetic path win rate by model:")
            print(
                rank_stability(
                    synth.to_dict("records"), path_col="path", model_col="model", metric="sharpe"
                ).to_string(index=False)
            )


if __name__ == "__main__":
    main()
