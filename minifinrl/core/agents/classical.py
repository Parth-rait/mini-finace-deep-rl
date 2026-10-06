"""
Classical portfolio baselines for PortfolioAllocationEnv: minimum variance
and risk parity, re-estimated on a trailing window and rebalanced monthly.

PROVENANCE: original to this project (EXPERIMENTS.md E04). FinRL's notebooks
use PyPortfolioOpt for mean-variance; this implements the two optimisers
from their definitions instead, and deliberately leaves out max-Sharpe:
it needs expected returns, whose estimation error swamps the signal
(Michaud 1989; DeMiguel et al. 2009).

    covariance   Sigma = Ledoit-Wolf shrinkage of the trailing daily returns
                 (Ledoit & Wolf 2004, sklearn.covariance.LedoitWolf)
    min variance argmin_w  w' Sigma w   s.t.  sum w = 1,  0 <= w <= 1
    risk parity  w_i (Sigma w)_i equal for all i, via Spinu (2013):
                 x* = argmin_{x>0} 1/2 x' Sigma x - (1/N) sum log x_i,
                 w = x* / sum x*   (strictly convex: unique solution)

No look-ahead: weights decided on day t use closes up to and including t
(the env trades at t's close and earns t -> t+1). At the start of a window
the trailing data comes from `env.prior_prices` (real history before the
window), never from later days.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

REBALANCE_EVERY = 21  # trading days (~monthly)
LOOKBACK = 252  # trading days of returns used for Sigma
MIN_RETURNS = 60  # fewer than this -> 1/N


def shrunk_covariance(returns: np.ndarray) -> np.ndarray:
    return LedoitWolf().fit(returns).covariance_


def _normalised(cov: np.ndarray) -> np.ndarray:
    """Both optimisers' solutions are invariant to scaling Sigma by a
    constant; daily covariances are ~1e-4, so rescale to mean variance 1
    for numerical conditioning."""
    return cov / np.mean(np.diag(cov))


def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    c = _normalised(cov)
    res = minimize(
        lambda w: w @ c @ w, np.full(n, 1.0 / n), jac=lambda w: 2 * c @ w, method="SLSQP",
        bounds=[(0.0, 1.0)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}],
        options={"ftol": 1e-14, "maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(f"min-variance optimisation failed: {res.message}")
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()


def risk_parity_weights(cov: np.ndarray, *, tol: float = 1e-12, max_iter: int = 100) -> np.ndarray:
    """Damped Newton on Spinu's strictly convex objective
        f(x) = 1/2 x' C x - b' log x,   b = 1/N
        grad = C x - b / x,   Hessian = C + diag(b / x^2)  (positive definite)
    with a backtracking step that keeps x > 0. At the optimum
    x_i (C x)_i = b_i, i.e. equal risk contributions. The answer is checked."""
    n = cov.shape[0]
    c = _normalised(cov)
    b = np.full(n, 1.0 / n)
    x = 1.0 / np.sqrt(np.diag(c))
    x *= np.sqrt(1.0 / (x @ c @ x))  # start on the right scale
    f = lambda v: 0.5 * v @ c @ v - b @ np.log(v)  # noqa: E731
    for _ in range(max_iter):
        g = c @ x - b / x
        if np.max(np.abs(g)) < tol:
            break
        step = np.linalg.solve(c + np.diag(b / x**2), g)
        t = 1.0
        while np.any(x - t * step <= 0) or f(x - t * step) > f(x) - 0.25 * t * (g @ step):
            t *= 0.5
            if t < 1e-12:
                raise RuntimeError("risk-parity Newton step failed to make progress")
        x = x - t * step
    w = x / x.sum()
    rc = risk_contributions(w, cov)
    if np.max(np.abs(rc - 1.0 / n)) > 1e-8:
        raise RuntimeError(f"risk parity not reached: contributions {rc}")
    return w


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """RC_i = w_i (Sigma w)_i / (w' Sigma w); sums to 1."""
    m = cov @ w
    return w * m / (w @ m)


class RebalancingBaseline:
    """Holds (no trades, so no cost) between rebalances; on rebalance days
    moves to weight_fn(Sigma estimated on the trailing LOOKBACK returns)."""

    def __init__(self, env, weight_fn: Callable[[np.ndarray], np.ndarray],
                 every: int = REBALANCE_EVERY, lookback: int = LOOKBACK):
        self.env = getattr(env, "unwrapped", env)
        self.weight_fn = weight_fn
        self.every = every
        self.lookback = lookback
        self.rebalances = 0

    def _trailing_prices(self) -> np.ndarray:
        e = self.env
        upto = e.prices[: e.day + 1]
        prior = getattr(e, "prior_prices", None)
        if prior is not None and len(prior):
            upto = np.vstack([prior, upto])
        return upto[-(self.lookback + 1):]

    def target_weights(self) -> np.ndarray:
        prices = self._trailing_prices()
        rets = prices[1:] / prices[:-1] - 1.0
        if len(rets) < MIN_RETURNS:
            return np.full(self.env.n_assets, 1.0 / self.env.n_assets)
        return self.weight_fn(shrunk_covariance(rets))

    def predict(self, obs, deterministic: bool = True):
        e = self.env
        w = self.target_weights() if e.day % self.every == 0 else e.weights
        if e.day % self.every == 0:
            self.rebalances += 1
        # the env applies softmax(scores); softmax(log w) = w exactly
        return np.log(np.clip(w, 1e-300, None)), None


def min_variance_agent(env) -> RebalancingBaseline:
    return RebalancingBaseline(env, min_variance_weights)


def risk_parity_agent(env) -> RebalancingBaseline:
    return RebalancingBaseline(env, risk_parity_weights)
