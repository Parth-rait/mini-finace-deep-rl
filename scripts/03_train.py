"""Train one (app, model, seed) combination on the historical training
window and save the resulting policy to results/models/.

Usage:
    python scripts/03_train.py --app trading --model ppo --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.sb3_wrapper import DRLAgentWrapper
from configs.logging_config import get_logger
from configs.settings import (
    INDICATORS,
    MODEL_DIR,
    PPO_PARAMS,
    SAC_PARAMS,
    TICKERS,
    TOTAL_TIMESTEPS,
    TRAIN_END,
    TRAIN_START,
    TURBULENCE_LOOKBACK,
)
from envs.portfolio_allocation import PortfolioAllocationEnv
from envs.stock_trading import StockTradingEnv
from meta.data import download
from meta.features import add_indicators, add_turbulence, clean_panel, to_array, train_test_split

log = get_logger(__name__)

PARAMS = {"ppo": PPO_PARAMS, "sac": SAC_PARAMS}
ENVS = {"trading": StockTradingEnv, "portfolio": PortfolioAllocationEnv}


def build_env(app: str, start: str, end: str):
    raw = download(TICKERS, start, end)
    panel = clean_panel(raw)
    panel = add_indicators(panel, INDICATORS)
    panel = add_turbulence(panel, TURBULENCE_LOOKBACK)
    panel = train_test_split(panel, start, end)
    prices, features, _dates, tickers = to_array(panel, INDICATORS)
    return ENVS[app](prices=prices, features=features, tickers=tickers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", choices=list(ENVS), required=True)
    parser.add_argument("--model", choices=list(PARAMS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    args = parser.parse_args()
    log.info("run: app=%s model=%s seed=%d timesteps=%d", args.app, args.model, args.seed, args.timesteps)

    env = build_env(args.app, TRAIN_START, TRAIN_END)
    agent = DRLAgentWrapper(args.model, env, PARAMS[args.model], seed=args.seed)
    agent.train(total_timesteps=args.timesteps)

    out_path = MODEL_DIR / f"{args.app}_{args.model}_seed{args.seed}.zip"
    agent.save(out_path)


if __name__ == "__main__":
    main()
