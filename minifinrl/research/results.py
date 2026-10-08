"""Reading recorded research results: backtest rows, report tables and the
experiment log. Pandas only, so the read path never loads torch."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from minifinrl.research.evaluation.report import aggregate_seeds, rank_stability
from minifinrl.research.paths import EXPERIMENT_DIR, Paths

# the passive benchmark each app's agents are judged against ("beats ref")
REFERENCE_BASELINE = {"trading": "buy_hold", "portfolio": "equal_weight"}
# every non-RL strategy name in the results; pipeline.py checks this matches its BASELINES
BASELINE_NAMES = frozenset({"buy_hold", "equal_weight", "min_variance", "risk_parity"}) | {"baseline"}  # "baseline": pre-E04 rows


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


def load_log(directory: Path = EXPERIMENT_DIR) -> list[dict]:
    p = Path(directory) / "log.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []
