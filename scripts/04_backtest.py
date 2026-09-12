"""Evaluate every saved (app, model, seed) combination, plus a fixed
non-RL baseline per app (buy-and-hold for trading, equal-weight for
portfolio allocation), against the historical test window and/or the
synthetic paths. Writes one row of metrics per (app, model, seed, path)
to results/logs/backtest_results.csv.

Usage:
    python scripts/04_backtest.py --paths historical
    python scripts/04_backtest.py --paths synthetic
    python scripts/04_backtest.py --paths both   # default
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from agents.baseline import BuyAndHoldAgent, EqualWeightAgent
from agents.sb3_wrapper import DRLAgentWrapper
from configs.logging_config import get_logger
from configs.settings import (
    APPS,
    DATA_SYNTHETIC,
    INDICATORS,
    LOG_DIR,
    MODEL_DIR,
    MODELS,
    SEEDS,
    TEST_END,
    TEST_START,
    TICKERS,
    TURBULENCE_LOOKBACK,
)
from envs.portfolio_allocation import PortfolioAllocationEnv
from envs.stock_trading import StockTradingEnv
from eval.backtest import backtest
from meta.data import download
from meta.features import add_indicators, add_turbulence, clean_panel, to_array, train_test_split

log = get_logger(__name__)

ENVS = {"trading": StockTradingEnv, "portfolio": PortfolioAllocationEnv}
# non-RL baselines, one per app, so the report shows whether PPO/SAC are
# actually earning their complexity over doing nothing clever. Each entry
# is a factory (not an instance) since BuyAndHoldAgent carries per-episode
# state and needs a fresh instance for every env it runs against.
BASELINES = {
    "trading": lambda n_assets: BuyAndHoldAgent(n_assets=n_assets),
    "portfolio": lambda n_assets: EqualWeightAgent(n_assets=n_assets),
}


def historical_env(app: str):
    raw = download(TICKERS, TEST_START, TEST_END)
    panel = clean_panel(raw)
    panel = add_indicators(panel, INDICATORS)
    panel = add_turbulence(panel, TURBULENCE_LOOKBACK)
    panel = train_test_split(panel, TEST_START, TEST_END)
    prices, features, _dates, tickers = to_array(panel, INDICATORS)
    return ENVS[app](prices=prices, features=features, tickers=tickers)


def synthetic_env(app: str, path_csv: Path):
    panel = pd.read_csv(path_csv)
    prices, features, _dates, tickers = to_array(panel, INDICATORS)
    return ENVS[app](prices=prices, features=features, tickers=tickers)


def _run_one(app: str, model: str, seed: int | None, agent, env, path_label: str) -> dict:
    result = backtest(agent, env, label=path_label)
    result.pop("equity_curve")
    return {"app": app, "model": model, "seed": seed, "path": path_label, **result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", choices=["historical", "synthetic", "both"], default="both")
    args = parser.parse_args()

    records = []
    for app in APPS:
        n_assets = len(TICKERS)

        # baseline first: no seed, no saved model, just a fixed strategy —
        # this is what tells you whether PPO/SAC are worth their complexity.
        if args.paths in ("historical", "both"):
            baseline = BASELINES[app](n_assets)
            records.append(_run_one(app, "baseline", None, baseline, historical_env(app), "historical"))
        if args.paths in ("synthetic", "both"):
            for path_csv in sorted(Path(DATA_SYNTHETIC).glob("path_*.csv")):
                baseline = BASELINES[app](n_assets)  # fresh instance per env: carries per-episode state
                records.append(_run_one(app, "baseline", None, baseline, synthetic_env(app, path_csv), path_csv.stem))

        for model in MODELS:
            for seed in SEEDS:
                model_path = MODEL_DIR / f"{app}_{model}_seed{seed}.zip"
                if not model_path.exists():
                    log.warning("skip missing %s", model_path)
                    continue

                if args.paths in ("historical", "both"):
                    env = historical_env(app)
                    agent = DRLAgentWrapper.load(model, model_path, env=env)
                    records.append(_run_one(app, model, seed, agent, env, "historical"))

                if args.paths in ("synthetic", "both"):
                    for path_csv in sorted(Path(DATA_SYNTHETIC).glob("path_*.csv")):
                        env = synthetic_env(app, path_csv)
                        agent = DRLAgentWrapper.load(model, model_path, env=env)
                        records.append(_run_one(app, model, seed, agent, env, path_csv.stem))

    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    out_path = Path(LOG_DIR) / "backtest_results.csv"
    pd.DataFrame(records).to_csv(out_path, index=False)
    log.info("wrote %d backtest rows to %s", len(records), out_path)


if __name__ == "__main__":
    main()
