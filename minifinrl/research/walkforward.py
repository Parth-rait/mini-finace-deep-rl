"""
Walk-forward evaluation and strategy-selection study (EXPERIMENTS.md E07).

PROVENANCE: original to this project. FinRL's ensemble strategy re-picks
the algorithm with the best recent validation Sharpe; this module tests
whether that kind of selection works at all, out of sample, against
simply sticking with one rule.

Folds: for each test year Y, train on the TRAIN_YEARS calendar years
before Y, test on Y (the last fold runs to TEST_END). Features are
computed once on the full history from trailing data only, so slicing a
fold never leaks the future (tests/test_features.py).

Selection rules, each evaluated on the chained out-of-sample daily returns
from the second fold on (the first has no previous fold to select from):

    static:<s>       always strategy s
    follow_winner    in fold k, the strategy with the best Sharpe in fold k-1
    mix_classical    1/3 each of equal_weight, risk_parity, min_variance
                     (daily returns averaged: a daily-rebalanced mix of the
                     three; the cost of rebalancing between them is ignored)
    hindsight_best   per fold, the strategy that turned out best: not
                     achievable, it is the yardstick for regret

Persistence: Spearman rank correlation of strategies' Sharpe between
consecutive folds. Near zero = last fold's ranking says nothing about the next.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from minifinrl.research.evaluation.metrics import max_drawdown, sharpe_ratio

TRAIN_YEARS = 5
CLASSICAL = ("equal_weight", "risk_parity", "min_variance")


@dataclass(frozen=True)
class Fold:
    name: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str  # inclusive, like train_test_split


def make_folds(first_test_year: int, last_test_year: int, *, train_years: int = TRAIN_YEARS,
               final_end: str | None = None) -> list[Fold]:
    folds = []
    for y in range(first_test_year, last_test_year + 1):
        # the last fold absorbs a partial final year (e.g. 2025 + 2026-H1), but
        # never stretches across more than that
        stretch = y == last_test_year and final_end and 0 <= int(final_end[:4]) - y <= 1
        end = final_end if stretch else f"{y}-12-31"
        folds.append(Fold(name=str(y) if end.startswith(str(y)) else f"{y}-{end[:4]}",
                          train_start=f"{y - train_years}-01-01", train_end=f"{y - 1}-12-31",
                          test_start=f"{y}-01-01", test_end=end))
    return folds


def strategy_returns(daily: pd.DataFrame) -> pd.DataFrame:
    """daily: rows (fold, app, model, seed, date, ret, rf). Returns
    (fold, app, strategy, date, ret, rf) with seeds collapsed by the daily
    median (a trained algorithm counts once, like the seed-median elsewhere)."""
    g = daily.groupby(["fold", "app", "model", "date"], as_index=False).agg(ret=("ret", "median"), rf=("rf", "first"))
    return g.rename(columns={"model": "strategy"})


def _sharpe(r: pd.Series, rf: pd.Series) -> float:
    return sharpe_ratio(r.to_numpy(), rf.to_numpy())


def fold_table(sr: pd.DataFrame) -> pd.DataFrame:
    """Sharpe per (app, fold, strategy)."""
    rows = [{"app": a, "fold": f, "strategy": s, "sharpe": _sharpe(g["ret"], g["rf"])}
            for (a, f, s), g in sr.groupby(["app", "fold", "strategy"])]
    return pd.DataFrame(rows)


def persistence(ft: pd.DataFrame, app: str, folds: list[str]) -> list[dict]:
    """Spearman rho of strategy Sharpe ranks between consecutive folds."""
    out = []
    piv = ft[ft["app"] == app].pivot(index="strategy", columns="fold", values="sharpe")
    for a, b in zip(folds[:-1], folds[1:]):
        rho = stats.spearmanr(piv[a], piv[b]).statistic if len(piv) > 2 else np.nan
        out.append({"from": a, "to": b, "spearman": float(rho)})
    return out


def selection_study(sr: pd.DataFrame, app: str, folds: list[str]) -> pd.DataFrame:
    """Chained out-of-sample performance of each selection rule for `app`,
    from folds[1] on. Columns: rule, sharpe, cagr, max_drawdown, picks."""
    sub = sr[sr["app"] == app]
    ft = fold_table(sub)
    best = ft.loc[ft.groupby("fold")["sharpe"].idxmax()].set_index("fold")["strategy"]
    strategies = sorted(sub["strategy"].unique())

    def series(fold: str, strategy: str) -> pd.DataFrame:
        return sub[(sub["fold"] == fold) & (sub["strategy"] == strategy)][["date", "ret", "rf"]]

    rules: dict[str, list[tuple[str, str]]] = {f"static:{s}": [(f, s) for f in folds[1:]] for s in strategies}
    rules["follow_winner"] = [(f, best[prev]) for prev, f in zip(folds[:-1], folds[1:])]
    rules["hindsight_best"] = [(f, best[f]) for f in folds[1:]]

    rows = []
    for rule, picks in rules.items():
        chained = pd.concat([series(f, s) for f, s in picks]).sort_values("date")
        rows.append(_summary(rule, chained, [s for _f, s in picks]))
    if all(c in strategies for c in CLASSICAL):
        parts = [sub[(sub["strategy"] == c) & (sub["fold"].isin(folds[1:]))].set_index("date")[["ret", "rf"]] for c in CLASSICAL]
        mix = pd.DataFrame({"ret": sum(p["ret"] for p in parts) / len(parts), "rf": parts[0]["rf"]}).reset_index()
        rows.append(_summary("mix_classical", mix.sort_values("date"), ["1/3 each"] * (len(folds) - 1)))
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


def _summary(rule: str, chained: pd.DataFrame, picks: list[str]) -> dict:
    r, rf = chained["ret"].to_numpy(), chained["rf"].to_numpy()
    values = np.r_[1.0, np.cumprod(1.0 + r)]
    years = len(r) / 252
    return {"rule": rule, "sharpe": sharpe_ratio(r, rf), "cagr": float(values[-1] ** (1 / years) - 1) if years > 0 else 0.0,
            "max_drawdown": max_drawdown(values), "days": int(len(r)), "picks": ",".join(picks)}


# ---- driver ----------------------------------------------------------------------------


def run_walk_forward(spec, base, folds: list[Fold], *, apps: list[str], models: list[str], seeds: list[int],
                     timesteps: int, risk_free: str = "dtb3", log=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train each (app, model, seed) on every fold's training window, backtest
    it and all baselines on that fold's test window. Models go to
    <base.model_dir>/<fold>/. Returns (metrics rows with a `fold` column,
    daily returns with `fold`), and writes both next to base.results_path."""
    from pathlib import Path

    from minifinrl.research import pipeline

    rows, daily = [], []
    for fold in folds:
        paths = pipeline.Paths(model_dir=Path(base.model_dir) / fold.name, synthetic_dir=base.synthetic_dir,
                               processed_dir=base.processed_dir,
                               results_path=Path(base.results_path).parent / "folds" / f"{fold.name}.csv")
        for app in apps:
            for model in models:
                for seed in seeds:
                    if log:
                        log.info("walk-forward %s: train %s %s seed %d on %s..%s", fold.name, app, model, seed,
                                 fold.train_start, fold.train_end)
                    pipeline.train(app, model, seed, timesteps, spec, paths,
                                   train_start=fold.train_start, train_end=fold.train_end)
        df = pipeline.run_backtests("historical", spec, paths, apps=apps, models=models, seeds=seeds,
                                    risk_free=risk_free, test_start=fold.test_start, test_end=fold.test_end,
                                    keep_returns=True)
        if df.attrs["skipped"]:
            raise RuntimeError(f"fold {fold.name}: models refused: {df.attrs['skipped']}")
        fold_returns = df.attrs["returns"]
        df.attrs = {}  # pd.concat compares attrs, and DataFrames in attrs can't be compared
        rows.append(df.assign(fold=fold.name))
        daily.append(fold_returns.assign(fold=fold.name))
    metrics, returns = pd.concat(rows, ignore_index=True), pd.concat(daily, ignore_index=True)
    out = Path(base.results_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out, index=False)
    returns.to_csv(out.with_name("daily_returns.csv"), index=False)
    return metrics, returns


def study(returns: pd.DataFrame, apps: list[str], folds: list[str]) -> dict:
    """Selection table + persistence per app, JSON-able."""
    sr = strategy_returns(returns)
    ft = fold_table(sr)
    return {
        app: {
            "selection": selection_study(sr, app, folds).to_dict("records"),
            "persistence": persistence(ft, app, folds),
            "fold_sharpe": ft[ft["app"] == app].pivot(index="strategy", columns="fold", values="sharpe")
                            .reindex(columns=folds).round(3).reset_index().to_dict("records"),
        }
        for app in apps
    }
