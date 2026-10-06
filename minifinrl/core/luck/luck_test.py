"""
Luck test: was a single trade's outcome within the normal range of luck,
given the market regime on the day it was entered?

PROVENANCE: original to this project; the single-trade version of the
repo's research question (does a result survive resampling of the path?).

What it does NOT answer: "was this trade skilful". The regime model has no
predictive signal (expected drift is about beta x market drift), so the
share of synthetic paths in which a trade wins sits near 50% for almost any
trade and would say nothing (PLAN_SHIPPABLE.md L1). What it CAN answer is
where the realized outcome falls in the distribution of outcomes the model
considers plausible from that starting regime:

    < 5th percentile   unusually_bad    worse than bad luck explains
    5th..95th          within_luck_range
    > 95th percentile  unusually_good   better than good luck explains

Look-ahead rules (each tested):
- the regime model is fit on closes up to and including the entry day only
  (the decision is made at that close, so it is known);
- the starting regime is the posterior on the entry day from that fit,
  pushed one step through the transition matrix (the first simulated day
  is the day after entry);
- prices after entry are used for exactly one thing: the realized outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import pandas as pd

from minifinrl.core.configs.logging_config import get_logger
from minifinrl.core.meta.synthetic import RegimeSyntheticGenerator

log = get_logger(__name__)

Direction = Literal["long", "short"]
MIN_HISTORY_DAYS = 500  # ~2 years: enough for 3 regimes to each be visited


class LuckTestError(ValueError):
    """The question can't be answered with the data available (unknown
    ticker, too little history, outcome not observable yet)."""


@dataclass(frozen=True)
class LuckResult:
    ticker: str
    direction: str
    entry_date: str
    exit_date: str
    horizon_days: int
    realized_return: float
    synthetic_median: float
    band_low: float  # 5th percentile of synthetic outcomes
    band_high: float  # 95th percentile
    prob_profit: float  # share of synthetic outcomes > 0 (expect ~0.5: no signal)
    percentile: float  # where the realized outcome falls, 0..100
    verdict: str
    regime: str
    regime_probs: dict[str, float] = field(default_factory=dict)
    n_paths: int = 0
    seed: int = 0
    fit_start: str = ""
    fit_end: str = ""


def percentile_of(value: float, sample: np.ndarray) -> float:
    """Mid-rank percentile: ties count half, so a value equal to every
    sample sits at 50, not 0 or 100."""
    below = float((sample < value).sum())
    equal = float((sample == value).sum())
    return 100.0 * (below + 0.5 * equal) / len(sample)


def verdict_for(pct: float) -> str:
    if pct < 5:
        return "unusually_bad"
    if pct > 95:
        return "unusually_good"
    return "within_luck_range"


def luck_test(
    close: pd.DataFrame,
    vix: pd.Series,
    ticker: str,
    date: str,
    direction: Direction,
    horizon_days: int,
    *,
    n_paths: int = 1000,
    seed: int = 0,
    min_history: int = MIN_HISTORY_DAYS,
    fit: Callable[[pd.DataFrame, pd.Series], RegimeSyntheticGenerator] | None = None,
) -> LuckResult:
    """`close`: (date x ticker) adjusted closes, date strings ascending.
    `date`: entry, snapped forward to the first trading day on or after it.
    `fit(prices, vix)` lets the caller cache fitted generators per entry day."""
    if ticker not in close.columns:
        raise LuckTestError(f"unknown ticker '{ticker}' (have {list(close.columns)})")
    if direction not in ("long", "short"):
        raise LuckTestError(f"direction must be long or short, got '{direction}'")
    dates = list(close.index)
    i = int(np.searchsorted(dates, date))
    if i >= len(dates):
        raise LuckTestError(f"no trading day on or after {date} (data ends {dates[-1]})")
    if i + 1 < min_history:
        raise LuckTestError(f"entry {dates[i]} has {i + 1} days of history, need {min_history}")
    if i + horizon_days >= len(dates):
        raise LuckTestError(
            f"outcome not observable yet: entry {dates[i]} + {horizon_days} trading days is past the data end {dates[-1]}"
        )

    entry, exit_ = dates[i], dates[i + horizon_days]
    sign = 1.0 if direction == "long" else -1.0
    realized = sign * (close[ticker].iloc[i + horizon_days] / close[ticker].iloc[i] - 1.0)

    history = close.iloc[: i + 1]  # up to and including the entry close, nothing after
    gen = (fit or (lambda p, v: RegimeSyntheticGenerator().fit(p, v)))(history, vix)
    posterior = gen.regime_probs_last()
    start = posterior @ gen.hmm.transmat_  # regime distribution of the first simulated day

    daily = gen.sample_ticker_returns(ticker, horizon_days, n_paths, start, seed)
    outcomes = sign * (np.prod(1.0 + daily, axis=1) - 1.0)
    pct = percentile_of(realized, outcomes)
    names = gen.regime_names()

    result = LuckResult(
        ticker=ticker, direction=direction, entry_date=entry, exit_date=exit_, horizon_days=horizon_days,
        realized_return=float(realized), synthetic_median=float(np.median(outcomes)),
        band_low=float(np.quantile(outcomes, 0.05)), band_high=float(np.quantile(outcomes, 0.95)),
        prob_profit=float((outcomes > 0).mean()), percentile=pct, verdict=verdict_for(pct),
        regime=names[int(np.argmax(posterior))],
        regime_probs={names[k]: float(posterior[k]) for k in range(len(names))},
        n_paths=n_paths, seed=seed, fit_start=gen.fit_start, fit_end=gen.fit_end,
    )
    log.info(
        "luck_test %s %s %s+%dd: realized %.3f, band [%.3f, %.3f], pct %.1f -> %s (regime %s)",
        direction, ticker, entry, horizon_days, realized, result.band_low, result.band_high, pct,
        result.verdict, result.regime,
    )
    return result
