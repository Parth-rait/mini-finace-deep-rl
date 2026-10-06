"""
Cross-source price check: do two providers agree on the same history?

PROVENANCE: original to this project. Adjusted price *levels* legitimately
differ between providers (each back-adjusts to its own anchor), so the
comparison is on adjusted daily returns, which don't depend on the anchor:

    r_t = adj_close_t / adj_close_{t-1} - 1,   d_t = |r_t(ref) - r_t(cand)|

on dates both sources have. A source "agrees" when p99(d) <= 10 bps and
fewer than 1% of the reference's dates are missing from it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AGREE_P99 = 1e-3  # 10 bps on daily returns
MAX_MISSING_FRAC = 0.01


def crosscheck(ref: pd.DataFrame, cand: pd.DataFrame, tickers: list[str]) -> list[dict]:
    """`ref`, `cand`: tidy adjusted panels (date, tic, close, ...)."""
    out = []
    for tic in tickers:
        a = ref[ref["tic"] == tic].set_index("date")["close"].sort_index()
        b = cand[cand["tic"] == tic].set_index("date")["close"].sort_index()
        common = a.index.intersection(b.index)
        row = {"ticker": tic, "ref_days": int(len(a)), "cand_days": int(len(b)), "overlap_days": int(len(common)),
               "missing_in_cand": int(len(a.index.difference(b.index))),
               "extra_in_cand": int(len(b.index.difference(a.index)))}
        if len(common) >= 3:
            ra, rb = a[common].pct_change().dropna(), b[common].pct_change().dropna()
            d = (ra - rb).abs()
            row.update(max_abs_diff=float(d.max()), p99_abs_diff=float(np.quantile(d, 0.99)),
                       return_corr=float(np.corrcoef(ra, rb)[0, 1]))
        else:
            row.update(max_abs_diff=None, p99_abs_diff=None, return_corr=None)
        row["agrees"] = bool(
            row["p99_abs_diff"] is not None and row["p99_abs_diff"] <= AGREE_P99
            and row["missing_in_cand"] <= MAX_MISSING_FRAC * max(row["ref_days"], 1)
        )
        out.append(row)
    return out
