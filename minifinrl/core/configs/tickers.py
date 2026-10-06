"""Ticker universes. Kept separate from settings.py so adding a bigger
universe later doesn't require touching the tunables file."""

SMALL_TICKER = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "JPM",
    "JNJ",
    "XOM",
]

# E06: cross-asset universe. All 12 trade since before 2014-01-02 (checked
# 6 Oct 2026), so a 2014 start has no gaps and, being ETFs, no survivorship
# bias. XLC (2018) and XLRE (2015) are excluded for that reason.
CROSS_ASSET_ETFS = [
    "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",  # 9 original SPDR sectors
    "TLT", "IEF",  # long and intermediate US Treasuries
    "GLD",  # gold
]

UNIVERSES = {"mega8": SMALL_TICKER, "cross_asset12": CROSS_ASSET_ETFS}
