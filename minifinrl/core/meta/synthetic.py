"""
Regime-conditional synthetic path generation.

PROVENANCE: original to this project - FinRL trains/evaluates on a single
realized historical path; this module exists to answer this project's
research question by generating alternative, statistically-plausible
market histories from a regime-switching model fit to the real one.

Roughly: fit a Gaussian HMM on a 2D series (daily equal-weight portfolio
log-return + daily VIX log-change) - that's the "market factor" driving
regime transitions (calm / normal / crisis). Decode the historical
hidden-state path with Viterbi, and per ticker bucket the historical
idiosyncratic residuals (return - beta * market_return) by which state
they happened in. To sample a synthetic path: draw a state sequence +
market path from the HMM, then per ticker rebuild a return as
`beta * market_return + bootstrapped_residual`, pulling the residual from
that ticker's pool for whichever state got sampled. Keeps the
cross-sectional structure (betas) and regime-appropriate vol clustering
while letting the actual day-to-day path differ from history.

This is deliberately a single-factor regime model, not a full multivariate
HMM over all tickers (which would need far more data to fit without
overfitting the covariance). It's good enough to stress-test rank
stability across plausible alternate histories - not a calibrated risk
model. See README limitations: Gaussian HMM understates tail risk, and
synthetic OHLC below collapses to close-only (see `to_tidy_panel`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from minifinrl.core.configs.logging_config import get_logger
from minifinrl.core.configs.settings import HMM_N_STATES, HMM_SEED

log = get_logger(__name__)

MIN_TICKER_RETURNS = 60  # trading days a ticker needs in the fit window to get its own model


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

    def fit(self, prices: pd.DataFrame, vix: pd.Series, *, market_columns: list[str] | None = None) -> "RegimeSyntheticGenerator":
        """`prices`: (T, N) close price DataFrame indexed by date, columns
        = tickers. `vix`: VIX level Series indexed by the same dates
        (or a superset - it's reindexed and forward-filled).

        `market_columns`: the tickers that define the market factor (their
        equal-weight return). Default: all columns. Passing the research
        universe lets an outside ticker (a user's trade) get its own beta
        and residuals without being mixed into the market it is measured
        against."""
        all_returns = prices.pct_change(fill_method=None).iloc[1:]
        market_cols = list(market_columns) if market_columns else list(prices.columns)
        # the market factor and the regimes come from days when every market
        # ticker traded; a ticker outside the market with a shorter history
        # (a recent IPO) doesn't truncate them
        returns = all_returns.loc[all_returns[market_cols].notna().all(axis=1)]
        vix = vix.reindex(returns.index).ffill()
        vix_change = np.log(vix / vix.shift(1)).fillna(0.0)

        market_return = returns[market_cols].mean(axis=1)
        obs = np.column_stack([market_return.values, vix_change.values])

        log.info("fitting %d-state HMM on %d observations", self.n_states, len(obs))
        self.hmm.fit(obs)
        self._obs = obs
        self.fit_start, self.fit_end = str(returns.index[0]), str(returns.index[-1])
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
                "states %s have <5 assigned days - their residual bootstrap pool will be small/noisy",
                thin_states,
            )

        market_var = float(np.var(market_return.values))
        for tic in prices.columns:
            ok = returns[tic].notna().values  # all True for market tickers
            if ok.sum() < MIN_TICKER_RETURNS:
                log.warning("%s: only %d returns in the fit window; no model fitted for it", tic, int(ok.sum()))
                continue
            r, m, st = returns[tic].values[ok], market_return.values[ok], states[ok]
            m_var = float(np.var(m)) if not ok.all() else market_var
            beta = float(np.cov(r, m)[0, 1] / m_var) if m_var > 0 else 0.0
            resid = r - beta * m
            by_state = {
                s: resid[st == s] for s in range(self.n_states) if (st == s).any()
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

    # ---- regime-conditional sampling (used by core/luck) -------------------------

    def _market_std(self) -> np.ndarray:
        return np.sqrt(np.asarray(self.hmm._covars_)[:, 0])  # diag: column 0 = market return

    def regime_names(self) -> list[str]:
        """State index -> name, by market-return volatility (calm < normal < crisis
        for 3 states). hmmlearn's state numbering is arbitrary."""
        order = np.argsort(self._market_std())
        labels = ["calm", "normal", "crisis"] if self.n_states == 3 else [f"regime_{k}" for k in range(self.n_states)]
        names = [""] * self.n_states
        for rank, state in enumerate(order):
            names[state] = labels[rank]
        return names

    def regime_probs_last(self) -> np.ndarray:
        """Posterior over regimes on the last fitted day (forward-backward on
        the fit window only, so nothing after it is used)."""
        if not self._fitted:
            raise RuntimeError("call fit() first")
        return self.hmm.predict_proba(self._obs)[-1]

    def sample_ticker_returns(
        self, tic: str, horizon: int, n_paths: int, start_probs: np.ndarray, seed: int
    ) -> np.ndarray:
        """(n_paths, horizon) daily returns for one ticker, with the regime chain
        started from `start_probs` (the distribution of the FIRST sampled day's
        regime). Same model as sample_path: beta x market return drawn from the
        state's Gaussian, plus a residual bootstrapped from that state's pool,
        vectorised over paths."""
        if not self._fitted:
            raise RuntimeError("call fit() first")
        rng = np.random.default_rng(seed)
        trans = self.hmm.transmat_
        states = np.empty((n_paths, horizon), dtype=int)
        states[:, 0] = rng.choice(self.n_states, size=n_paths, p=start_probs / start_probs.sum())
        cum = np.cumsum(trans, axis=1)
        for t in range(1, horizon):
            u = rng.random(n_paths)[:, None]
            states[:, t] = np.minimum((u > cum[states[:, t - 1]]).sum(axis=1), self.n_states - 1)

        market = rng.normal(self.hmm.means_[states, 0], self._market_std()[states])
        model = self._ticker_models[tic]
        resid = np.empty_like(market)
        everything = np.concatenate(list(model.residuals_by_state.values()))
        for s in range(self.n_states):
            mask = states == s
            if mask.any():
                pool = model.residuals_by_state.get(s)
                pool = pool if pool is not None and len(pool) else everything
                resid[mask] = rng.choice(pool, size=int(mask.sum()))
        return model.beta * market + resid

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
    are set equal to close and volume to 0 - fine for computing technical
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
