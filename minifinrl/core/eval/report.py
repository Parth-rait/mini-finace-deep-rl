"""
Aggregate backtest results across seeds and synthetic paths - this is
basically the whole point of the project: does whatever wins on the real
historical path keep winning across the plausible alternate ones, or was
it just luck of the draw?
"""

from __future__ import annotations

import math

import pandas as pd


def aggregate_seeds(
    records: list[dict], group_cols: list[str], metric: str = "sharpe"
) -> pd.DataFrame:
    """`records`: one dict per backtest run (as produced by
    eval.backtest.backtest(), minus the equity_curve array, plus whatever
    identifying keys the caller added - model/app/seed/path). Returns
    median + IQR of `metric` grouped by `group_cols`."""
    df = pd.DataFrame(records)
    grouped = df.groupby(group_cols)[metric]
    summary = grouped.median().rename(f"{metric}_median").to_frame()
    summary[f"{metric}_q25"] = grouped.quantile(0.25)
    summary[f"{metric}_q75"] = grouped.quantile(0.75)
    return summary.reset_index()


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a win rate. Honest at small n (20
    synthetic paths), where the normal approximation goes below 0."""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _per_path_model(df: pd.DataFrame, path_col: str, model_col: str, metric: str) -> pd.DataFrame:
    """One value per (path, model): the median over seeds. Without this, a
    model with 3 seeds gets 3 chances per path to post the best number and
    a seedless baseline gets 1, so win rates favour whoever has more seeds."""
    return df.groupby([path_col, model_col])[metric].median().unstack(model_col)


def rank_stability(
    records: list[dict], path_col: str, model_col: str, metric: str = "sharpe", baseline: str = "baseline"
) -> pd.DataFrame:
    """For each path, which model ranked first on `metric` (median over its
    seeds). Returns per model: wins, n_paths, win_rate with a 95% Wilson
    interval, and on how many paths it beat the baseline (the question that
    matters more than who came first)."""
    wide = _per_path_model(pd.DataFrame(records), path_col, model_col, metric)
    n_paths = len(wide)
    winners = wide.idxmax(axis=1).value_counts()
    rows = []
    for model in wide.columns:
        wins = int(winners.get(model, 0))
        lo, hi = wilson(wins, n_paths)
        beats = int((wide[model] > wide[baseline]).sum()) if baseline in wide.columns and model != baseline else None
        rows.append({model_col: model, "wins": wins, "n_paths": n_paths, "win_rate": wins / n_paths,
                     "ci_low": lo, "ci_high": hi, "beats_baseline": beats})
    out = pd.DataFrame(rows).sort_values(["win_rate", model_col], ascending=[False, True]).reset_index(drop=True)
    out["beats_baseline"] = out["beats_baseline"].astype("Int64")
    return out
