"""Each rule catches a deliberately broken panel, and a clean panel passes
all of them. One break per test, so a failure names exactly one rule."""

from __future__ import annotations

import pytest

from minifinrl.market.validate import DataValidationError, validate_panel, validate_series
from tests.conftest import make_panel

TICKERS = ["AAA", "BBB", "CCC"]
START, END = "2024-01-01", "2024-07-01"


def _run(df, **kw):
    return validate_panel(df, TICKERS, START, END, **kw)


def test_clean_panel_passes(panel):
    report = _run(panel)
    assert report.ok, report.errors
    assert report.warnings == []
    report.raise_for_errors()  # does not raise


def test_missing_column():
    r = _run(make_panel().drop(columns=["volume"]))
    assert r.errors == ["missing columns ['volume']"]


def test_missing_ticker(panel):
    r = _run(panel[panel["tic"] != "CCC"])
    assert any("no rows for tickers ['CCC']" in e for e in r.errors)


def test_duplicates(panel):
    import pandas as pd

    r = _run(pd.concat([panel, panel.iloc[:2]]))
    assert any("2 duplicate" in e for e in r.errors)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan")])
def test_bad_price(panel, value):
    panel.loc[10, "close"] = value
    r = _run(panel)
    assert not r.ok


def test_high_below_close(panel):
    panel.loc[5, "high"] = panel.loc[5, "close"] * 0.5
    r = _run(panel)
    assert len(r.errors) == 1 and "high < max(open, close)" in r.errors[0]


def test_few_missing_dates_is_warning(panel):
    # one date missing for one ticker: 1/130 dates < 1% -> warning only
    first_date = panel["date"].unique()[50]
    df = panel[~((panel["date"] == first_date) & (panel["tic"] == "BBB"))]
    r = _run(df)
    assert r.ok and len(r.warnings) == 1 and "'BBB': 1" in r.warnings[0]


def test_many_missing_dates_is_error(panel):
    dates = panel["date"].unique()[40:50]
    r = _run(panel[~(panel["date"].isin(dates) & (panel["tic"] == "BBB"))])
    assert any("missing some ticker" in e for e in r.errors)


def test_long_gap(panel):
    dates = panel["date"].unique()[60:70]  # 10 business days gone for everyone
    r = _run(panel[~panel["date"].isin(dates)])
    assert len(r.errors) == 1 and "calendar gap" in r.errors[0]


def test_unadjusted_split(panel):
    # a 4:1 split that wasn't adjusted: price quarters from one day on
    idx = panel.index[(panel["tic"] == "AAA") & (panel["date"] >= panel["date"].unique()[80])]
    panel.loc[idx, ["open", "high", "low", "close"]] /= 4
    r = _run(panel)
    assert len(r.errors) == 1 and "daily moves above 50%" in r.errors[0]


def test_late_start(panel):
    early = panel["date"].unique()[:20]
    r = _run(panel[~(panel["date"].isin(early) & (panel["tic"] == "CCC"))])
    assert any("CCC starts" in e for e in r.errors)


def test_early_end(panel):
    late = panel["date"].unique()[-20:]
    r = _run(panel[~panel["date"].isin(late)])
    assert any("ends" in e and "before requested end" in e for e in r.errors)


def test_freshness_only_in_live_mode(panel):
    assert _run(panel).ok
    r = _run(panel, live_today="2024-08-01")
    assert len(r.errors) == 1 and "business days old" in r.errors[0]
    assert _run(panel, live_today="2024-07-02").ok


def test_raise_for_errors(panel):
    panel.loc[0, "close"] = -1
    with pytest.raises(DataValidationError, match="data validation error"):
        _run(panel).raise_for_errors()


def test_series_rules():
    import pandas as pd

    dates = pd.bdate_range(START, "2024-06-28").strftime("%Y-%m-%d")
    s = pd.Series(20.0, index=dates)
    assert validate_series(s, "VIX", START, END).ok
    assert not validate_series(s.iloc[:50], "VIX", START, END).ok  # ends early
    s2 = s.copy()
    s2.iloc[3] = -1
    assert not validate_series(s2, "VIX", START, END).ok
