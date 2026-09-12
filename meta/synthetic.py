"""
Regime-conditional synthetic path generation.

PROVENANCE: original to this project — FinRL trains/evaluates on a single
realized historical path; this module exists to answer this project's
research question by generating alternative, statistically-plausible
market histories from a regime-switching model fit to the real one.

Approach
--------
1. Fit a Gaussian HMM to a 2D observation series: the daily equal-weight
   portfolio log-return and the daily VIX log-change. This is the "market
   factor" — the thing that drives regime transitions (calm / normal /
   crisis).
2. Decode the historical hidden-state path (Viterbi, via `.predict`) and,
   per ticker, bucket historical idiosyncratic residuals
   (return - beta * market_return) by the state they occurred in.
3. To generate a synthetic path: sample a state sequence + market-factor
   path from the fitted HMM, then for each ticker reconstruct a return as
   `beta * market_return + bootstrapped_residual`, where the residual is
   drawn from that ticker's historical residual pool for the sampled
   state. This keeps cross-sectional structure (betas) and
   regime-appropriate volatility clustering while letting the specific
   sequence of daily moves differ from history.

This is deliberately a single-factor regime model, not a full multivariate
HMM over all tickers (which would need far more data to fit without
overfitting the covariance). It's good enough to stress-test rank
stability across plausible alternate histories — not a calibrated risk
model. See README limitations: Gaussian HMM understates tail risk, and
synthetic OHLC below collapses to close-only (see `to_tidy_panel`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from configs.logging_config import get_logger
from configs.settings import HMM_N_STATES, HMM_SEED

log = get_logger(__name__)


@dataclass
class _TickerResidualModel:
    beta: float
    residuals_by_state: dict[int, np.ndarray]
    last_price: float


class RegimeSyntheticGenerator:
    """Fits a 2D regime model on (market return, VIX change) and samples
    synthetic per-ticker price panels conditioned on it."""

    def __init__(self, n_states: int = HMM_N_STATES, seed: int = HMM_SEED):
        self.n_states = n_states
        self.seed = seed
        # "full" covariance is numerically fragile here: a regime with few
        # assigned days (a short-lived crisis, say) can produce a singular
        # 2x2 covariance and crash the Cholesky step inside hmmlearn.
        # "diag" gives up modeling the return/VIX covariance within a
        # regime but is stable on the data sizes this project uses.
        self.hmm = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            min_covar=1e-3,
            random_state=seed,
            n_iter=200,
        )
        self._rng = np.random.default_rng(seed)
        self._ticker_models: dict[str, _TickerResidualModel] = {}
        self._fitted = False

    def fit(self, prices: pd.DataFrame, vix: pd.Series) -> "RegimeSyntheticGenerator":
        """`prices`: (T, N) close price DataFrame indexed by date, columns
        = tickers. `vix`: VIX level Series indexed by the same dates
        (or a superset — it's reindexed and forward-filled)."""
        returns = prices.pct_change().dropna()
        vix = vix.reindex(returns.index).ffill()
        vix_change = np.log(vix / vix.shift(1)).fillna(0.0)

        market_return = returns.mean(axis=1)
        obs = np.column_stack([market_return.values, vix_change.values])

        log.info("fitting %d-state HMM on %d observations", self.n_states, len(obs))
        self.hmm.fit(obs)
        if not self.hmm.monitor_.converged:
            log.warning(
                "HMM did not converge in %d iterations (final log-likelihood delta present in monitor_.history)",
                self.hmm.n_iter,
            )
        states = self.hmm.predict(obs)
        state_counts = {int(s): int((states == s).sum()) for s in range(self.n_states)}
        log.info("HMM state occupancy: %s", state_counts)
        thin_states = [s for s, n in state_counts.items() if n < 5]
        if thin_states:
            log.warning(
                "states %s have <5 assigned days — their residual bootstrap pool will be small/noisy",
                thin_states,
            )

        market_var = float(np.var(market_return.values))
        for tic in prices.columns:
            r = returns[tic].values
            beta = float(np.cov(r, market_return.values)[0, 1] / market_var) if market_var > 0 else 0.0
            resid = r - beta * market_return.values
            by_state = {
                s: resid[states == s] for s in range(self.n_states) if (states == s).any()
            }
            self._ticker_models[tic] = _TickerResidualModel(
                beta=beta, residuals_by_state=by_state, last_price=float(prices[tic].iloc[-1])
            )

        self._fitted = True
        log.info("fit complete for %d tickers", len(self._ticker_models))
        return self

    def sample_path(self, length: int, seed: int | None = None) -> pd.DataFrame:
        """Returns a (length, N) DataFrame of synthetic close prices, one
        column per fitted ticker, starting from each ticker's last
        historical price."""
        if not self._fitted:
            raise RuntimeError("call fit() before sample_path()")

        draw_seed = seed if seed is not None else int(self._rng.integers(1 << 32))
        rng = np.random.default_rng(draw_seed)
        market_obs, states = self.hmm.sample(length, random_state=draw_seed)
        market_return = market_obs[:, 0]

        prices = {}
        for tic, model in self._ticker_models.items():
            resid = np.empty(length)
            for t, s in enumerate(states):
                pool = model.residuals_by_state.get(int(s))
                if pool is None or len(pool) == 0:
                    pool = np.concatenate(list(model.residuals_by_state.values()))
                resid[t] = rng.choice(pool)

            ticker_return = model.beta * market_return + resid
            price_path = model.last_price * np.cumprod(1.0 + ticker_return)
            prices[tic] = price_path

        return pd.DataFrame(prices)

    def sample_many(self, n_paths: int, length: int) -> list[pd.DataFrame]:
        log.info("sampling %d synthetic paths of length %d", n_paths, length)
        paths = [self.sample_path(length, seed=self.seed + i) for i in range(n_paths)]
        log.info("sampled %d synthetic paths", len(paths))
        return paths


def to_tidy_panel(price_df: pd.DataFrame, start_date: str) -> pd.DataFrame:
    """Reshape a synthetic (T, N) close-price DataFrame into the tidy
    [date, tic, open, high, low, close, volume] panel the rest of the
    pipeline (meta.features) expects.

    Synthetic data only models close-to-close returns, so open/high/low
    are set equal to close and volume to 0 — fine for computing technical
    indicators, not a realistic intraday model. Documented in
    README limitations.
    """
    dates = pd.bdate_range(start=start_date, periods=len(price_df)).strftime("%Y-%m-%d")
    rows = []
    for tic in price_df.columns:
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "tic": tic,
                    "open": price_df[tic].values,
                    "high": price_df[tic].values,
                    "low": price_df[tic].values,
                    "close": price_df[tic].values,
                    "volume": 0,
                }
            )
        )
    panel = pd.concat(rows, ignore_index=True)
    return panel.sort_values(["date", "tic"]).reset_index(drop=True)
