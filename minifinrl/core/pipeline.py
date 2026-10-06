"""
The research pipeline as plain functions: fetch, fit the regime model and
sample paths, train, backtest, report.

PROVENANCE: moved out of scripts/01-05 unchanged in behaviour, so the
Engine (and through it the CLI, the API and the gate) runs exactly the code
the scripts used to. Every path is a parameter with the settings value as
its default, so a test or another profile can point it anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from minifinrl.core.agents.baseline import EqualWeightAgent, FullyInvestedBuyAndHoldAgent
from minifinrl.core.agents.classical import LOOKBACK, min_variance_agent, risk_parity_agent
from minifinrl.core.agents.registry import ModelMismatch, ModelSpec, make_card
from minifinrl.core.agents.sb3_wrapper import DRLAgentWrapper
from minifinrl.core.configs.logging_config import get_logger
from minifinrl.core.configs.settings import (
    APPS,
    DATA_PROCESSED,
    DATA_SYNTHETIC,
    HMM_N_PATHS,
    HMM_PATH_LENGTH,
    INDICATORS,
    LOG_DIR,
    MODEL_DIR,
    MODELS,
    PPO_PARAMS,
    SAC_PARAMS,
    SEEDS,
    TD3_ACTION_NOISE,
    TD3_PARAMS,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    TURBULENCE_LOOKBACK,
)
from minifinrl.core.envs.portfolio_allocation import PortfolioAllocationEnv
from minifinrl.core.envs.stock_trading import StockTradingEnv
from minifinrl.core.eval.backtest import backtest
from minifinrl.core.eval.report import aggregate_seeds, rank_stability
from minifinrl.core.meta.features import to_array, train_test_split
from minifinrl.core.meta.providers import ProviderError
from minifinrl.core.meta.panel import DataSpec, build_panel, features_with_history
from minifinrl.core.meta.providers import get_rate_provider
from minifinrl.core.meta.store import SeriesStore
from minifinrl.core.meta.synthetic import RegimeSyntheticGenerator, to_tidy_panel

log = get_logger(__name__)

ENVS = {"trading": StockTradingEnv, "portfolio": PortfolioAllocationEnv}
PARAMS = {"ppo": PPO_PARAMS, "sac": SAC_PARAMS, "td3": TD3_PARAMS}
# non-RL baselines, one per app, so the report shows whether PPO/SAC are
# actually earning their complexity over doing nothing clever. Factories
# taking the env, since the buy-and-hold carries per-episode state and needs
# absolute prices. Trading uses the fully invested buy-and-hold (F13); the
# old 100-shares-once BuyAndHoldAgent stays in agents/baseline.py for reference.
BASELINES: dict[str, dict[str, Callable]] = {
    "trading": {"buy_hold": lambda env: FullyInvestedBuyAndHoldAgent(env)},
    "portfolio": {
        "equal_weight": lambda env: EqualWeightAgent(n_assets=env.n_assets),
        "min_variance": min_variance_agent,  # E04
        "risk_parity": risk_parity_agent,  # E04
    },
}
# the passive benchmark each app's agents are judged against ("beats ref")
REFERENCE_BASELINE = {"trading": "buy_hold", "portfolio": "equal_weight"}
BASELINE_NAMES = frozenset(n for d in BASELINES.values() for n in d) | {"baseline"}  # "baseline": pre-E04 rows
RESULTS_FILE = "backtest_results.csv"


@dataclass(frozen=True)
class Paths:
    model_dir: Path = MODEL_DIR
    synthetic_dir: Path = DATA_SYNTHETIC
    processed_dir: Path = DATA_PROCESSED
    results_path: Path = LOG_DIR / RESULTS_FILE


def model_path(paths: Paths, app: str, model: str, seed: int) -> Path:
    return Path(paths.model_dir) / f"{app}_{model}_seed{seed}.zip"


# ---- environments ------------------------------------------------------------------


def env_from_panel(app: str, panel: pd.DataFrame, prior: pd.DataFrame | None = None):
    """`prior`: closes (date x tic) strictly before the window, for
    baselines that estimate on a trailing window. Agents never see it."""
    prices, features, dates, tickers = to_array(panel, INDICATORS)
    env = ENVS[app](prices=prices, features=features, tickers=tickers)
    env.dates = list(dates)
    if prior is not None:
        if len(prior) and prior.index[-1] >= dates[0]:
            raise ValueError(f"prior history ends {prior.index[-1]}, not before the window start {dates[0]}")
        env.prior_prices = prior.reindex(columns=tickers).to_numpy()
    return env


def _closes(spec: DataSpec) -> pd.DataFrame:
    return build_panel(spec).panel.pivot(index="date", columns="tic", values="close")


def historical_env(app: str, spec: DataSpec, start: str, end: str):
    # features come from the full-history panel, then get sliced, so the
    # 252-day turbulence lookback always has real history behind it
    closes = _closes(spec)
    prior = closes[closes.index < start].iloc[-LOOKBACK - 1:]
    return env_from_panel(app, train_test_split(build_panel(spec).panel, start, end), prior)


def synthetic_env(app: str, path_csv: Path, spec: DataSpec | None = None):
    # a synthetic path continues from the last real close, so its "before"
    # is the real history up to the path start
    panel = pd.read_csv(path_csv)
    prior = None
    if spec is not None:
        closes = _closes(spec)
        prior = closes[closes.index < panel["date"].min()].iloc[-LOOKBACK - 1:]
    return env_from_panel(app, panel, prior)


def synthetic_paths(paths: Paths) -> list[Path]:
    return sorted(Path(paths.synthetic_dir).glob("path_*.csv"))


# ---- 02: regime model + synthetic paths --------------------------------------------


def fit_and_sample(
    spec: DataSpec, paths: Paths, *, n_paths: int = HMM_N_PATHS, length: int = HMM_PATH_LENGTH
) -> list[Path]:
    fp = build_panel(spec)
    panel, vix = fp.panel, fp.vix

    Path(paths.processed_dir).mkdir(parents=True, exist_ok=True)
    panel.to_csv(Path(paths.processed_dir) / "panel.csv", index=False)
    log.info("processed panel: %s, cached to %s", panel.shape, Path(paths.processed_dir) / "panel.csv")

    prices = panel.pivot(index="date", columns="tic", values="close")
    generator = RegimeSyntheticGenerator().fit(prices, vix)

    out_dir = Path(paths.synthetic_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, synthetic_prices in enumerate(generator.sample_many(n_paths, length)):
        synthetic_panel = to_tidy_panel(synthetic_prices, start_date=spec.end)
        # the path continues from the last real price, so its indicators and
        # turbulence are warmed up on the real history before it
        synthetic_panel = features_with_history(synthetic_panel, panel, warmup_days=TURBULENCE_LOOKBACK + 100)
        path = out_dir / f"path_{i:03d}.csv"
        synthetic_panel.to_csv(path, index=False)
        written.append(path)
    log.info("sampled %d synthetic paths of length %d to %s", n_paths, length, out_dir)
    return written


# ---- 03: train ---------------------------------------------------------------------


def train(app: str, model: str, seed: int, timesteps: int, spec: DataSpec, paths: Paths, *,
          train_start: str = TRAIN_START, train_end: str = TRAIN_END) -> Path:
    log.info("run: app=%s model=%s seed=%d timesteps=%d window=%s..%s", app, model, seed, timesteps, train_start, train_end)
    env = historical_env(app, spec, train_start, train_end)
    model_spec = ModelSpec.from_env(app, env)
    agent = DRLAgentWrapper(model, env, PARAMS[model], seed=seed)
    agent.train(total_timesteps=timesteps)
    card = make_card(
        algo=model, seed=seed, timesteps=timesteps, spec=model_spec, train_start=train_start,
        train_end=train_end, data_hash=build_panel(spec).data_hash, train_seconds=agent.train_seconds,
        params={**PARAMS[model], **({"action_noise_sigma_frac": TD3_ACTION_NOISE} if model == "td3" else {})},
    )
    out = model_path(paths, app, model, seed)
    agent.save(out, card=card)
    return out


# ---- 04: backtest ------------------------------------------------------------------


def risk_free_series(spec: DataSpec) -> pd.Series:
    """3-month T-bill yield, % per year, from the store (FRED DTB3, E03)."""
    store = SeriesStore(get_rate_provider(spec.rate_provider), key="DTB3", root=spec.store_root,
                        today=spec.today, column="rate")
    y = store.get(spec.start, "9999-12-31", refresh=spec.refresh)
    if y.empty:
        raise ProviderError("no risk-free rate data (FRED DTB3) on disk or from the provider")
    # Not >= 0: 3-month bill yields printed slightly negative in Sep-Oct 2015
    # (-0.02% low) and on 25-26 Mar 2020 (-0.05%). Real, and the conversion
    # (1 + y/100)^(1/252) - 1 handles them; only absurd values are rejected.
    if not np.isfinite(y.values).all() or (y < -1).any() or (y > 25).any():
        raise ProviderError("risk-free series has values outside [-1, 25]%")
    return y


def risk_free_daily(dates: list[str], yields: pd.Series) -> np.ndarray:
    """Per-step risk-free returns for an episode over `dates` (len T): step t
    holds from dates[t] to dates[t+1] and earns r_f,t = (1 + y_t/100)^(1/252) - 1,
    with y_t the last published yield on or before dates[t] (holidays and
    future synthetic dates carry the last known value forward). Length T-1."""
    y = yields.reindex(sorted(set(yields.index) | set(dates))).ffill().bfill().reindex(dates[:-1])
    return (1.0 + y.to_numpy() / 100.0) ** (1.0 / 252) - 1.0


def _row(app: str, model: str, seed: int | None, agent, env, path_label: str, returns: list | None = None) -> dict:
    result = backtest(agent, env, label=path_label)
    curve = result.pop("equity_curve")
    if returns is not None and path_label == "historical":
        # daily returns + risk-free for the walk-forward selection study (E07)
        rf = getattr(env.unwrapped, "rf_daily", None)
        r = curve[1:] / curve[:-1] - 1.0
        returns.append(pd.DataFrame({"app": app, "model": model, "seed": seed, "date": env.unwrapped.dates[:-1],
                                     "ret": r, "rf": rf if rf is not None else 0.0}))
    return {"app": app, "model": model, "seed": seed, "path": path_label, **result}


def run_backtests(
    which: str,
    spec: DataSpec,
    paths: Paths,
    *,
    apps: list[str] = APPS,
    models: list[str] = MODELS,
    seeds: list[int] = SEEDS,
    baselines: list[str] | None = None,
    risk_free: str = "dtb3",
    test_start: str = TEST_START,
    test_end: str = TEST_END,
    keep_returns: bool = False,
) -> pd.DataFrame:
    """Every saved (app, model, seed) plus the baselines, on the historical
    test window and/or every synthetic path. `which`: historical | synthetic
    | both. `baselines`: names to run (default: all for the app).
    `risk_free`: dtb3 (E03) | zero (the pre-E03 convention). Writes
    paths.results_path."""
    do_hist, do_synth = which in ("historical", "both"), which in ("synthetic", "both")
    data_hash = build_panel(spec).data_hash
    yields = risk_free_series(spec) if risk_free == "dtb3" else None
    records, skipped, returns = [], [], ([] if keep_returns else None)

    def envs(app: str):
        """(label, env) for every evaluation path, rf attached."""
        out = []
        if do_hist:
            out.append(("historical", historical_env(app, spec, test_start, test_end)))
        if do_synth:
            out += [(p.stem, synthetic_env(app, p, spec)) for p in synthetic_paths(paths)]
        for _label, env in out:
            if yields is not None:
                env.rf_daily = risk_free_daily(env.dates, yields)
        return out

    def load(model: str, mp: Path, env, app: str):
        # M2: refuse a policy whose card doesn't match what this env feeds it
        return DRLAgentWrapper.load(model, mp, env=env, expect=ModelSpec.from_env(app, env), data_hash=data_hash)

    for app in apps:
        wanted = BASELINES[app] if baselines is None else {k: v for k, v in BASELINES[app].items() if k in baselines}
        for name, make in wanted.items():
            for label, env in envs(app):
                records.append(_row(app, name, None, make(env), env, label, returns))

        for model in models:
            for seed in seeds:
                mp = model_path(paths, app, model, seed)
                if not mp.exists():
                    log.warning("skip missing %s", mp)
                    continue
                try:
                    for label, env in envs(app):
                        records.append(_row(app, model, seed, load(model, mp, env, app), env, label, returns))
                except ModelMismatch as exc:
                    # one stale model must not sink the whole run, but it is
                    # reported, not silently dropped
                    log.error("skip %s: %s", mp.name, exc)
                    records = [r for r in records if not (r["app"] == app and r["model"] == model and r["seed"] == seed)]
                    skipped.append(f"{mp.stem}: {exc}")

    out = Path(paths.results_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(out, index=False)
    log.info("wrote %d backtest rows to %s (%d models skipped, risk_free=%s)", len(records), out, len(skipped), risk_free)
    df.attrs["skipped"] = skipped
    df.attrs["returns"] = pd.concat(returns, ignore_index=True) if returns else None
    return df


# ---- 05: report ----------------------------------------------------------------------


def load_results(paths: Paths) -> pd.DataFrame:
    p = Path(paths.results_path)
    if not p.exists():
        raise FileNotFoundError(f"no backtest results at {p}: run the backtests first")
    return pd.read_csv(p)


def collapsed_policies(df: pd.DataFrame) -> pd.DataFrame:
    """Trained policies with one distinct action on every step (a fixed
    rule, not learning). Empty when the column predates the check."""
    if "distinct_actions" not in df.columns:
        return df.iloc[0:0][["app", "model", "seed", "path"]]
    c = df[(~df["model"].isin(BASELINE_NAMES)) & (df["distinct_actions"] <= 1)]
    return c[["app", "model", "seed", "path"]].drop_duplicates(["app", "model", "seed"])


def report_tables(df: pd.DataFrame, metric: str = "sharpe") -> dict[str, dict[str, pd.DataFrame]]:
    """Per app: historical median/IQR by model (over seeds), synthetic
    median/IQR by model (over seeds and paths), synthetic win rate by model."""
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for app in df["app"].unique():
        sub = df[df["app"] == app]
        hist, synth = sub[sub["path"] == "historical"], sub[sub["path"] != "historical"]
        tables: dict[str, pd.DataFrame] = {}
        if not hist.empty:
            tables["historical"] = aggregate_seeds(hist.to_dict("records"), ["model"], metric)
        if not synth.empty:
            tables["synthetic"] = aggregate_seeds(synth.to_dict("records"), ["model"], metric)
            ref = REFERENCE_BASELINE[app] if (synth["model"] == REFERENCE_BASELINE[app]).any() else "baseline"
            tables["win_rate"] = rank_stability(synth.to_dict("records"), path_col="path", model_col="model",
                                                metric=metric, baseline=ref)
        out[app] = tables
    return out
