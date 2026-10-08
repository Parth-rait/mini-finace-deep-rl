"""
The range of outcomes a planned trade is signing up for, given today's
market regime. NOT a forecast: the regime model has no predictive edge (the
median outcome sits near zero for almost any stock), so this says how wide
luck alone is over the planned hold, which is what someone should be ready
for before they trade.

Same model and sampling as the luck test, with the fit ending at the latest
close instead of at a past entry day.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from minifinrl.regime.luck import MIN_HISTORY_DAYS, LuckTestError
from minifinrl.regime.model import RegimeSyntheticGenerator


@dataclass(frozen=True)
class Outlook:
    ticker: str
    direction: str
    horizon_days: int
    as_of: str  # the latest close the fit saw
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float
    prob_loss: float  # share of simulated outcomes below zero
    regime: str
    regime_probs: dict[str, float]
    days: list[int]  # the fan: 5th, 50th and 95th percentile after each day of the hold
    fan_p05: list[float]
    fan_p50: list[float]
    fan_p95: list[float]
    n_paths: int
    last_price: float = 0.0  # the ticker's latest close
    # with a stop and/or target (directional returns from entry): how often the paths reach them
    prob_stop: float | None = None  # touched the stop at some close within the hold
    prob_target: float | None = None  # touched the target at some close within the hold
    prob_target_first: float | None = None  # reached the target before the stop
    prob_stop_first: float | None = None  # reached the stop before the target


def outcome_range(
    close: pd.DataFrame, vix: pd.Series, ticker: str, direction: str, horizon_days: int, *,
    n_paths: int = 1000, seed: int = 0, min_history: int = MIN_HISTORY_DAYS,
    fit: Callable[[pd.DataFrame, pd.Series], RegimeSyntheticGenerator] | None = None,
    stop_return: float | None = None, target_return: float | None = None,
) -> Outlook:
    """`close`: (date x ticker) closes up to the latest day. The fit uses all of
    it; paths start the day after the last close."""
    if ticker not in close.columns:
        raise LuckTestError(f"unknown ticker '{ticker}'")
    if len(close) < min_history:
        raise LuckTestError(f"{len(close)} days of market history, need {min_history}")
    gen = (fit or (lambda p, v: RegimeSyntheticGenerator().fit(p, v)))(close, vix)
    posterior = gen.regime_probs_last()
    start = posterior @ gen.hmm.transmat_
    sign = 1.0 if direction == "long" else -1.0
    daily = gen.sample_ticker_returns(ticker, horizon_days, n_paths, start, seed)
    paths = sign * (np.cumprod(1.0 + daily, axis=1) - 1.0)
    final = paths[:, -1]
    q = np.quantile(final, [0.05, 0.25, 0.5, 0.75, 0.95])
    fan = np.quantile(paths, [0.05, 0.5, 0.95], axis=0)
    names = gen.regime_names()
    hits = _hits(paths, stop_return, target_return)
    return Outlook(
        ticker=ticker, direction=direction, horizon_days=horizon_days, as_of=str(close.index[-1]),
        p05=float(q[0]), p25=float(q[1]), p50=float(q[2]), p75=float(q[3]), p95=float(q[4]),
        prob_loss=float((final < 0).mean()), regime=names[int(np.argmax(posterior))],
        regime_probs={names[k]: float(posterior[k]) for k in range(len(names))},
        days=list(range(1, horizon_days + 1)), fan_p05=fan[0].tolist(), fan_p50=fan[1].tolist(), fan_p95=fan[2].tolist(),
        n_paths=n_paths, last_price=float(close[ticker].dropna().iloc[-1]), **hits,
    )


def _hits(paths: np.ndarray, stop: float | None, target: float | None) -> dict:
    """Share of paths touching the stop / target at a daily close, and which came first."""
    out: dict = {}
    n, h = paths.shape
    first_stop = np.full(n, h)
    first_target = np.full(n, h)
    if stop is not None:
        hit = paths <= stop
        out["prob_stop"] = float(hit.any(axis=1).mean())
        first_stop = np.where(hit.any(axis=1), hit.argmax(axis=1), h)
    if target is not None:
        hit = paths >= target
        out["prob_target"] = float(hit.any(axis=1).mean())
        first_target = np.where(hit.any(axis=1), hit.argmax(axis=1), h)
    if stop is not None and target is not None:
        out["prob_target_first"] = float((first_target < first_stop).mean())
        out["prob_stop_first"] = float((first_stop < first_target).mean())
    return out
