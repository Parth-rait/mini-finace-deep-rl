"""Market data: universe, date windows, indicators, providers, cache and validation."""

from minifinrl.market.universe import SMALL_TICKER
from minifinrl.platform.settings import DATA_RAW

# ---- universe and dates -------------------------------------------------
TICKERS = SMALL_TICKER
VIX_TICKER = "^VIX"

TRAIN_START = "2014-01-01"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2026-06-30"

# ---- features ------------------------------------------------------------
# stockstats names; keep this list short, every extra feature widens the
# observation space and slows training.
INDICATORS = [
    "macd",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]

USE_TURBULENCE = True
TURBULENCE_LOOKBACK = 252

# ---- providers, cache, validation ---------------------------------------
# "research" pins the end date to TEST_END so results are reproducible;
# "live" moves it to today and turns on the freshness check.
DATA_MODE = "research"  # "research" | "live"
PRICE_PROVIDER = "yahoo"  # see market.providers.PRICE_PROVIDERS
VIX_PROVIDER = "yahoo"  # "yahoo" (^VIX, what all results use) | "fred" (VIXCLS, unused for now)
FRED_VIX_SERIES = "VIXCLS"
# FRED_API_KEY is read from the environment, never stored here. Without it
# the public fredgraph CSV endpoint is used (same series, no key).

DATA_STORE = DATA_RAW / "store"  # per-ticker incremental cache
# Re-fetch this many stored days on every incremental update: if the
# provider's adjustment ratio moved on the overlap (a dividend or split
# since the last fetch), the whole ticker is re-downloaded.
STORE_OVERLAP_DAYS = 5
ADJ_RATIO_TOLERANCE = 1e-4

# validation thresholds
MAX_GAP_BDAYS = 5  # longest tolerated run of missing business days
MAX_DROPPED_DATE_FRAC = 0.01  # share of dates clean_panel may drop
MAX_ABS_DAILY_RETURN = 0.5  # bigger moves are almost always bad data
COVERAGE_TOLERANCE_BDAYS = 5  # first/last bar vs requested start/end
MAX_STALENESS_BDAYS = 3  # live mode only
