"""
Feature engineering.

PROVENANCE: adapted from FinRL `meta/preprocessor/preprocessors.py`.

One deliberate departure from FinRL's `FeatureEngineer.add_technical_indicator`:
FinRL runs `Sdf.retype` on the WHOLE multi-ticker frame and then slices per
ticker, but stockstats' rolling windows don't know about ticker boundaries -
rows from the end of one ticker's history bleed into the start of the next
one's rolling stats. Here we `Sdf.retype` each ticker's slice independently,
which avoids that leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf

from minifinrl.platform.log import get_logger

log = get_logger(__name__)


def clean_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only dates where every ticker has a row. The environments assume
    a rectangular (date x tic) panel; nothing downstream re-checks this."""
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)
    n_tickers = df["tic"].nunique()
    counts = df.groupby("date")["tic"].nunique()
    complete_dates = counts[counts == n_tickers].index
    dropped = counts.index.difference(complete_dates)
    if len(dropped) > 0:
        log.warning(
            "clean_panel: dropping %d/%d dates with missing tickers (e.g. %s)",
            len(dropped), len(counts), list(dropped[:5]),
        )
    return df[df["date"].isin(complete_dates)].reset_index(drop=True)


def add_indicators(df: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    """Attach stockstats technicals, computed independently per ticker."""
    df = df.sort_values(["tic", "date"]).reset_index(drop=True)
    out = []
    for _tic, g in df.groupby("tic", sort=False):
        sdf = Sdf.retype(g.copy())
        for ind in indicators:
            _ = sdf[ind]  # stockstats computes + attaches the column lazily
        # Sdf.retype moves "date" into the index; bring it back as a column.
        out.append(pd.DataFrame(sdf).reset_index())

    merged = pd.concat(out, ignore_index=True)
    merged = merged.sort_values(["date", "tic"]).reset_index(drop=True)

    n_bad = int(merged[indicators].isin([np.inf, -np.inf]).sum().sum())
    if n_bad:
        log.warning("add_indicators: replacing %d inf/-inf values before fill", n_bad)
    merged[indicators] = merged[indicators].replace([np.inf, -np.inf], np.nan)
    merged = fill_indicators(merged, indicators)
    log.info("add_indicators: computed %s for %d tickers, %d rows", indicators, df["tic"].nunique(), len(merged))
    return merged


def fill_indicators(df: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    """Forward-fill gaps within each ticker, then drop the leading dates
    that are still NaN (indicator warm-up).

    This replaces FinRL's `groupby("tic").ffill().bfill()`, where the
    `.bfill()` ran on the whole frame: a ticker's warm-up NaN got the next
    row's value, i.e. another ticker's, or a later date's (look-ahead).
    Dropping whole dates keeps the panel rectangular. With the current
    INDICATORS stockstats emits no NaN, so this changes nothing today; it
    only matters once an indicator with a warm-up period is added.
    """
    df = df.copy()
    df[indicators] = df.groupby("tic")[indicators].ffill()
    still_nan = df.loc[df[indicators].isna().any(axis=1), "date"].unique()
    if len(still_nan):
        log.warning(
            "fill_indicators: dropping %d leading warm-up date(s) with NaN indicators (%s..%s)",
            len(still_nan), min(still_nan), max(still_nan),
        )
        df = df[~df["date"].isin(still_nan)].reset_index(drop=True)
    return df


def add_turbulence(df: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """Mahalanobis distance of today's cross-sectional return vector from its
    trailing `lookback`-day covariance - a market-wide "this looks unusual
    given recent history" signal, same value broadcast to every ticker on a
    given date.

    PROVENANCE: lifted close to verbatim from FinRL
    `FeatureEngineer.calculate_turbulence` - the pinv-on-a-near-singular-
    covariance trick and the "ignore the first couple of nonzero values as
    warm-up noise" rule are fiddly enough to get right that there's no value
    in re-deriving them from scratch.
    """
    prices = df.pivot(index="date", columns="tic", values="close")
    returns = prices.pct_change()
    dates = returns.index.to_list()

    turbulence = [0.0] * lookback
    count = 0
    for i in range(lookback, len(dates)):
        current = returns.iloc[[i]]
        hist = returns.iloc[i - lookback : i]
        hist = hist.iloc[hist.isna().sum().min() :].dropna(axis=1)

        if hist.shape[1] == 0:
            turbulence.append(0.0)
            continue

        cov = hist.cov()
        demeaned = current[hist.columns] - hist.mean()
        score = demeaned.values @ np.linalg.pinv(cov.values) @ demeaned.values.T
        value = float(score[0, 0]) if score[0, 0] > 0 else 0.0

        count += 1
        turbulence.append(value if count > 2 else 0.0)

    turb_df = pd.DataFrame({"date": dates, "turbulence": turbulence})
    log.info(
        "add_turbulence: lookback=%d, %d/%d days scored (rest are warm-up), max=%.2f",
        lookback, count, len(dates), max(turbulence) if turbulence else 0.0,
    )
    return df.merge(turb_df, on="date", how="left")


def train_test_split(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Date-sliced frame, re-indexed so index == trading-day ordinal (what
    the environments step through)."""
    mask = (df["date"] >= start) & (df["date"] <= end)
    out = df.loc[mask].sort_values(["date", "tic"]).reset_index(drop=True)
    out.index = out["date"].factorize()[0]
    return out


def to_array(
    df: pd.DataFrame, indicators: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Collapse the tidy panel into the tensors the environments consume.

    Returns:
        prices: (T, N) close price array
        features: (T, N, F) indicator array, turbulence appended as the last
            feature (broadcast across assets) if present in `df`
        dates: list[str] of length T
        tickers: list[str] of length N, columns order of the above arrays
    """
    prices_df = df.pivot(index="date", columns="tic", values="close")
    dates = prices_df.index.to_list()
    tickers = prices_df.columns.to_list()

    feature_cols = list(indicators)
    blocks = [
        df.pivot(index="date", columns="tic", values=c).reindex(columns=tickers)
        for c in feature_cols
    ]

    if "turbulence" in df.columns:
        turb = df.drop_duplicates("date").set_index("date")["turbulence"]
        turb = turb.reindex(prices_df.index)
        blocks.append(pd.DataFrame({t: turb for t in tickers}, index=prices_df.index))

    features = np.stack([b.values for b in blocks], axis=-1)  # (T, N, F)
    return prices_df.values, features, dates, tickers
