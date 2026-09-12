"""
Performance metrics computed on a portfolio value series.

PROVENANCE: standard formulas, the same ones FinRL surfaces via pyfolio;
reimplemented directly since this project only needs a handful of
numbers, not a full tearsheet.
"""

from __future__ import annotations

import numpy as np

from configs.settings import RISK_FREE_RATE, TRADING_DAYS


def returns_from_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[1:] / values[:-1] - 1.0


def cagr(values: np.ndarray) -> float:
    total_return = values[-1] / values[0]
    n_years = (len(values) - 1) / TRADING_DAYS
    if n_years <= 0:
        return 0.0
    return float(total_return ** (1 / n_years) - 1)


def annualized_volatility(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: np.ndarray, risk_free: float = RISK_FREE_RATE) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / TRADING_DAYS
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: np.ndarray, risk_free: float = RISK_FREE_RATE) -> float:
    excess = returns - risk_free / TRADING_DAYS
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return 0.0
    return float(np.mean(excess) / downside_std * np.sqrt(TRADING_DAYS))


def max_drawdown(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    running_max = np.maximum.accumulate(values)
    drawdown = (values - running_max) / running_max
    return float(drawdown.min())


def summarize(values: np.ndarray) -> dict[str, float]:
    """One row of headline stats for a single equity curve."""
    returns = returns_from_values(values)
    return {
        "final_value": float(values[-1]),
        "cagr": cagr(values),
        "volatility": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(values),
    }
