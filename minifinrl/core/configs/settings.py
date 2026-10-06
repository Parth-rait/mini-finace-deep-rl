"""Central configuration. Everything tunable lives here, nothing elsewhere."""

import os
from pathlib import Path

from minifinrl.core.configs.tickers import SMALL_TICKER

# ---- paths -------------------------------------------------------------
# Where data/ and results/ live. In a checkout: the repo root (the folder
# holding pyproject.toml, 3 levels above this file). Installed from git,
# this file sits in site-packages, so set MINIFINRL_HOME; without it the
# current directory is used rather than writing into site-packages.
_CHECKOUT_ROOT = Path(__file__).resolve().parents[3]
if os.environ.get("MINIFINRL_HOME"):
    ROOT = Path(os.environ["MINIFINRL_HOME"]).resolve()
elif (_CHECKOUT_ROOT / "pyproject.toml").exists() or (_CHECKOUT_ROOT / ".git").exists():
    ROOT = _CHECKOUT_ROOT
else:
    ROOT = Path.cwd()
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SYNTHETIC = ROOT / "data" / "synthetic"
RESULTS = ROOT / "results"
MODEL_DIR = RESULTS / "models"
FIGURE_DIR = RESULTS / "figures"
LOG_DIR = RESULTS / "logs"

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

# ---- environment ----------------------------------------------------------
INITIAL_AMOUNT = 1_000_000
HMAX = 100  # max shares traded per asset per step
TRANSACTION_COST_PCT = 0.001  # 10 bps each way
REWARD_SCALING = 1e-4  # keeps reward magnitudes sane for SB3

# ---- training ---------------------------------------------------------
MODELS = ["ppo", "sac", "td3"]
APPS = ["trading", "portfolio"]
SEEDS = [0, 1, 2]  # >=3. Single-seed DRL results are noise.
TOTAL_TIMESTEPS = 50_000  # smoke test: 5_000

PPO_PARAMS = {
    "n_steps": 2048,
    "batch_size": 128,
    "learning_rate": 2.5e-4,
    "ent_coef": 0.005,
}
SAC_PARAMS = {
    "batch_size": 256,
    "buffer_size": 100_000,
    "learning_rate": 3e-4,
    "learning_starts": 1000,
}
# E05: TD3 matched to SAC on every shared hyperparameter, so the comparison
# isolates the objective: SAC maximises reward + alpha * entropy, TD3 reward
# only (deterministic policy, explores with fixed Gaussian action noise).
# policy_delay=2, target_policy_noise=0.2, noise clip 0.5: SB3 defaults.
TD3_PARAMS = dict(SAC_PARAMS)
TD3_ACTION_NOISE = 0.1  # exploration sigma, as a fraction of the action half-range

# ---- synthetic data (HMM) ----------------------------------------------
HMM_N_STATES = 3  # e.g. calm / normal / crisis
HMM_N_PATHS = 20  # synthetic paths to sample
HMM_PATH_LENGTH = 500  # trading days per path
HMM_SEED = 42

# ---- evaluation ------------------------------------------------------------
TRADING_DAYS = 252
RISK_FREE_RATE = 0.0

# ---- data layer (meta/providers.py, meta/store.py, meta/validate.py) ------
# "research" pins the end date to TEST_END so results are reproducible;
# "live" moves it to today and turns on the freshness check.
DATA_MODE = "research"  # "research" | "live"
PRICE_PROVIDER = "yahoo"  # see meta.providers.PRICE_PROVIDERS
VIX_PROVIDER = "yahoo"  # "yahoo" (^VIX, what all results use) | "fred" (VIXCLS, unused for now)
FRED_VIX_SERIES = "VIXCLS"
# FRED_API_KEY is read from the environment, never stored here. Without it
# the public fredgraph CSV endpoint is used (same series, no key).
HTTP_TIMEOUT_S = 20
HTTP_RETRIES = 3

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
