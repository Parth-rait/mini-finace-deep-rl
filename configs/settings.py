"""Central configuration. Everything tunable lives here, nothing elsewhere."""

from pathlib import Path

from configs.tickers import SMALL_TICKER

# ---- paths -------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
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
MODELS = ["ppo", "sac"]
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

# ---- synthetic data (HMM) ----------------------------------------------
HMM_N_STATES = 3  # e.g. calm / normal / crisis
HMM_N_PATHS = 20  # synthetic paths to sample
HMM_PATH_LENGTH = 500  # trading days per path
HMM_SEED = 42

# ---- evaluation ------------------------------------------------------------
TRADING_DAYS = 252
RISK_FREE_RATE = 0.0
