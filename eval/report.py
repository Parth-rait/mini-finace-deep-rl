"""
Aggregate backtest results across seeds and synthetic paths — this is
basically the whole point of the project: does whatever wins on the real
historical path keep winning across the plausible alternate ones, or was
it just luck of the draw?
"""

from __future__ import annotations

import pandas as pd


def aggregate_seeds(
    records: list[dict], group_cols: list[str], metric: str = "sharpe"
) -> pd.DataFrame:
    """`records`: one dict per backtest run (as produced by
    eval.backtest.backtest(), minus the equity_curve array, plus whatever
    identifying keys the caller added — model/app/seed/path). Returns
    median + IQR of `metric` grouped by `group_cols`."""
    df = pd.DataFrame(records)
    grouped = df.groupby(group_cols)[metric]
    summary = grouped.median().rename(f"{metric}_median").to_frame()
    summary[f"{metric}_q25"] = grouped.quantile(0.25)
    summary[f"{metric}_q75"] = grouped.quantile(0.75)
    return summary.reset_index()


def rank_stability(
    records: list[dict], path_col: str, model_col: str, metric: str = "sharpe"
) -> pd.DataFrame:
    """For each distinct path, find which model ranked first on `metric`;
    return each model's win rate (fraction of paths it won)."""
    df = pd.DataFrame(records)
    best_idx = df.groupby(path_col)[metric].idxmax()
    winners = df.loc[best_idx, [path_col, model_col]]
    win_counts = winners[model_col].value_counts()
    n_paths = df[path_col].nunique()
    return (
        (win_counts / n_paths)
        .rename("win_rate")
        .rename_axis(model_col)
        .reset_index()
    )
