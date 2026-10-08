"""Deep-RL research: environments, training and evaluation."""

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

# ---- evaluation ------------------------------------------------------------
TRADING_DAYS = 252
RISK_FREE_RATE = 0.0
