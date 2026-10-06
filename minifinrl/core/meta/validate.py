"""
Data validation: check a raw panel before anything trains or serves on it.

PROVENANCE: original to this project. `meta/features.clean_panel` quietly
drops incomplete dates and nothing downstream re-checks the data, so a
provider hiccup (a missing ticker, a stale day, an unadjusted split) turns
into a model trained on holes. Every rule here fails loudly instead.

Each rule returns problems as strings. `errors` block (`raise_for_errors`),
`warnings` are logged and kept in the provenance manifest. Rules take
plain arguments so a test can break exactly one thing and assert that
exactly one rule fires.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from minifinrl.core.configs.logging_config import get_logger
from minifinrl.core.configs.settings import (
    COVERAGE_TOLERANCE_BDAYS,
    MAX_ABS_DAILY_RETURN,
    MAX_DROPPED_DATE_FRAC,
    MAX_GAP_BDAYS,
    MAX_STALENESS_BDAYS,
)
from minifinrl.core.meta.data import RAW_COLUMNS

log = get_logger(__name__)

PRICE_COLS = ["open", "high", "low", "close"]


class DataValidationError(ValueError):
    pass


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        for w in self.warnings:
            log.warning("validation: %s", w)
        if self.errors:
            for e in self.errors:
                log.error("validation: %s", e)
            raise DataValidationError(f"{len(self.errors)} data validation error(s): " + "; ".join(self.errors))

    def to_dict(self) -> dict:
        return {"ok": self.ok, "errors": list(self.errors), "warnings": list(self.warnings)}


def _bday_gap(a: str, b: str) -> int:
    """Business days from a to b (1 for consecutive trading days)."""
    return int(np.busday_count(np.datetime64(a, "D"), np.datetime64(b, "D")))


# ---- rules -------------------------------------------------------------------


def check_schema(df: pd.DataFrame) -> list[str]:
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        return [f"missing columns {missing}"]
    if df.empty:
        return ["panel is empty"]
    return []


def check_tickers_present(df: pd.DataFrame, tickers: list[str]) -> list[str]:
    absent = sorted(set(tickers) - set(df["tic"].unique()))
    return [f"no rows for tickers {absent}"] if absent else []


def check_duplicates(df: pd.DataFrame) -> list[str]:
    n = int(df.duplicated(["date", "tic"]).sum())
    return [f"{n} duplicate (date, tic) rows"] if n else []


def check_prices(df: pd.DataFrame) -> list[str]:
    problems = []
    values = df[PRICE_COLS]
    n_nan = int(values.isna().sum().sum() + (~np.isfinite(values.fillna(0))).sum().sum())
    if n_nan:
        problems.append(f"{n_nan} NaN/inf price values")
    n_nonpos = int((values <= 0).sum().sum())
    if n_nonpos:
        bad = df.loc[(values <= 0).any(axis=1), ["date", "tic"]].head(3).values.tolist()
        problems.append(f"{n_nonpos} zero/negative prices (e.g. {bad})")
    # 1e-6 relative slack: adjustment multiplies all four by the same float
    hi_bad = df["high"] < df[["open", "close"]].max(axis=1) * (1 - 1e-6)
    lo_bad = df["low"] > df[["open", "close"]].min(axis=1) * (1 + 1e-6)
    n_ohlc = int((hi_bad | lo_bad).sum())
    if n_ohlc:
        bad = df.loc[hi_bad | lo_bad, ["date", "tic"]].head(3).values.tolist()
        problems.append(f"{n_ohlc} rows with high < max(open, close) or low > min(open, close) (e.g. {bad})")
    return problems


def check_rectangular(df: pd.DataFrame, max_dropped_frac: float = MAX_DROPPED_DATE_FRAC) -> tuple[list[str], list[str]]:
    """The environments need every ticker on every date; clean_panel drops
    the rest. A few dropped dates is a warning, more than the threshold
    means a ticker's history is broken and is an error."""
    n_tic = df["tic"].nunique()
    counts = df.groupby("date")["tic"].nunique()
    incomplete = counts[counts < n_tic]
    if incomplete.empty:
        return [], []
    frac = len(incomplete) / len(counts)
    per_tic = {
        t: int(len(counts) - g["date"].nunique()) for t, g in df.groupby("tic") if g["date"].nunique() < len(counts)
    }
    msg = f"{len(incomplete)}/{len(counts)} dates ({frac:.1%}) missing some ticker; missing days per ticker: {per_tic}"
    return ([msg], []) if frac > max_dropped_frac else ([], [msg])


def check_gaps(df: pd.DataFrame, max_gap: int = MAX_GAP_BDAYS) -> list[str]:
    dates = sorted(df["date"].unique())
    gaps = [(a, b, _bday_gap(a, b)) for a, b in zip(dates[:-1], dates[1:])]
    big = [(a, b, g) for a, b, g in gaps if g > max_gap]
    if not big:
        return []
    return [f"{len(big)} calendar gap(s) longer than {max_gap} business days (e.g. {big[:3]})"]


def check_returns(df: pd.DataFrame, max_abs: float = MAX_ABS_DAILY_RETURN) -> list[str]:
    """A >50% close-to-close move in a large-cap is nearly always an
    unadjusted split or a bad print, not a real move."""
    # duplicates are reported by check_duplicates; pivot would crash on them
    closes = df.drop_duplicates(["date", "tic"]).pivot(index="date", columns="tic", values="close").sort_index()
    moves = closes.pct_change(fill_method=None).abs()
    # pandas 3 .stack() keeps NaN, so filter explicitly
    hits = moves.stack()
    hits = hits[hits > max_abs]
    if hits.empty:
        return []
    examples = [(d, t, round(float(v), 3)) for (d, t), v in hits.head(3).items()]
    return [f"{len(hits)} daily moves above {max_abs:.0%} (e.g. {examples}) - unadjusted split or bad data?"]


def check_coverage(df: pd.DataFrame, start: str, end: str, tol: int = COVERAGE_TOLERANCE_BDAYS) -> list[str]:
    """First/last bar should sit within `tol` business days of the requested
    [start, end). Per ticker, since one late-listed ticker shortens the
    whole rectangular panel."""
    problems = []
    last_expected = (pd.Timestamp(end) - pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")
    for tic, g in df.groupby("tic"):
        first, last = g["date"].min(), g["date"].max()
        if _bday_gap(start, first) > tol:
            problems.append(f"{tic} starts {first}, {_bday_gap(start, first)} business days after requested {start}")
        if _bday_gap(last, last_expected) > tol:
            problems.append(f"{tic} ends {last}, {_bday_gap(last, last_expected)} business days before requested end {end}")
    return problems


def check_freshness(df: pd.DataFrame, today: str, max_stale: int = MAX_STALENESS_BDAYS) -> list[str]:
    """Live mode only: the newest bar must be recent. Business days, not
    exchange days, so a holiday week can use up some of the slack."""
    last = df["date"].max()
    stale = _bday_gap(last, today)
    return [f"newest bar {last} is {stale} business days old (max {max_stale})"] if stale > max_stale else []


# ---- entry point -------------------------------------------------------------


def validate_panel(
    df: pd.DataFrame,
    tickers: list[str],
    start: str,
    end: str,
    *,
    live_today: str | None = None,
) -> ValidationReport:
    """Run every rule. `end` is exclusive (same as download()). Pass
    `live_today` to enable the freshness check."""
    report = ValidationReport()
    report.errors += check_schema(df)
    if report.errors:
        return report

    report.errors += check_tickers_present(df, tickers)
    report.errors += check_duplicates(df)
    report.errors += check_prices(df)
    rect_err, rect_warn = check_rectangular(df)
    report.errors += rect_err
    report.warnings += rect_warn
    report.errors += check_gaps(df)
    report.errors += check_returns(df)
    report.errors += check_coverage(df, start, end)
    if live_today is not None:
        report.errors += check_freshness(df, live_today)
    log.info("validate_panel: %d rows, %d errors, %d warnings", len(df), len(report.errors), len(report.warnings))
    return report


def validate_series(s: pd.Series, name: str, start: str, end: str) -> ValidationReport:
    """Checks for a single daily series (VIX): present, finite, positive,
    no long gaps, covers the range."""
    report = ValidationReport()
    if s.empty:
        report.errors.append(f"{name}: empty")
        return report
    if not np.isfinite(s.values).all():
        report.errors.append(f"{name}: NaN/inf values")
    if (s <= 0).any():
        report.errors.append(f"{name}: non-positive values")
    as_panel = pd.DataFrame({"date": s.index, "tic": name})
    report.errors += check_gaps(as_panel)
    report.errors += check_coverage(as_panel, start, end)
    return report
