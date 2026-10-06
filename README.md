# mini-FinRL

A reduced financial RL system: one agent wrapper, two algorithms (PPO, SAC),
two applications (stock trading, portfolio allocation), evaluated on
historical data and on synthetic paths sampled from a fitted regime model.

## Research question

Do the performance rankings of DRL trading agents survive resampling of the
market path, or are they artifacts of the single realized history they were
trained and tested on?

## Layers

```
minifinrl/core/            the finance core (no LLM, no web)
  configs/  tickers + all tunables (paths, dates, hyperparameters)
  meta/     data (providers, incremental store, validation, provenance),
            features, synthetic generation
  envs/     gymnasium environments (one per application) + observation scaling
  agents/   SB3 wrapper + non-RL baselines
  eval/     metrics, backtest runner, multi-seed aggregation
minifinrl/engine.py        the one facade: every capability is a typed method
minifinrl/capabilities.py  registry that generates the CLI (and next the API, agent tools)
minifinrl/ports.py         what the engine needs from outside (bias classifier)
minifinrl/adapters/        port implementations; the only code allowed to import aip
minifinrl/system.py        composition root: build_system(profile) wires everything
minifinrl/interfaces/      CLI (API and agent next)
scripts/    thin shells over the CLI, kept for muscle memory
tests/      offline test suite (pytest), incl. tests/test_architecture.py
```

Each layer only imports from layers above it in this list (`envs` imports
`meta`, never the reverse). That's what keeps a change to, say, the data
source from rippling into the RL code.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # editable install, pinned by requirements.lock
pytest -q

python scripts/01_download_data.py
python scripts/02_fit_hmm.py
python scripts/03_train.py --app trading --model ppo --seed 0
python scripts/03_train.py --app trading --model sac --seed 0
# repeat 03_train.py across --app {trading,portfolio}, --model {ppo,sac},
# --seed {0,1,2} (see minifinrl/core/configs/settings.py APPS/MODELS/SEEDS) for the full grid
python scripts/04_backtest.py --paths both
python scripts/05_report.py
```

For a fast smoke test end-to-end, pass `--timesteps 5000` to `03_train.py`
instead of the full `TOTAL_TIMESTEPS`.

The scripts are thin shells over one generated CLI; everything the system
can do is a capability of `minifinrl.engine.Engine`, assembled in
`minifinrl/system.py`:

```bash
python -m minifinrl list                          # every capability + effect class
python -m minifinrl health                        # data freshness, models, results
python -m minifinrl rank-stability --app trading  # win rates with 95% intervals
python -m minifinrl train --app trading --model sac --seed 1
python -m minifinrl health --profile ci           # offline profile, no network
```

## Logging

Every module logs through `minifinrl/core/configs/logging_config.py` (`get_logger(__name__)`)
rather than `print()`. Output goes to both the console (INFO+) and a
rotating file at `results/logs/mini_finrl.log` (DEBUG+, 5 x 5MB, so a long
sweep doesn't grow the file unbounded). Cache hits/misses, HMM convergence
and state occupancy, per-run train/save/load events, and each backtest's
headline stats all land there. Run `tail -f results/logs/mini_finrl.log`
while a sweep runs, or grep it after the fact instead of relying on
terminal scrollback. The final report tables from `05_report.py` are
printed directly (they're the deliverable, not a log event).

## Extending

- **New algorithm**: add one entry to `_ALGOS` in `minifinrl/core/agents/sb3_wrapper.py`.
- **New application**: add a `gym.Env` in `minifinrl/core/envs/`, register it in the
  `ENVS` dict in `scripts/03_train.py` and `scripts/04_backtest.py`, add its
  name to `APPS` in `minifinrl/core/configs/settings.py`.
- **New ticker universe**: add a list to `minifinrl/core/configs/tickers.py`, point
  `TICKERS` at it in `minifinrl/core/configs/settings.py`.
- **New indicator**: add its stockstats name to `INDICATORS` in
  `minifinrl/core/configs/settings.py`; `minifinrl/core/meta/features.py` picks it up automatically.
  It also needs a scaling rule in `minifinrl/core/envs/scaling.py` (an unknown name raises).

## Results

All numbers come from the experiment log ([EXPERIMENTS.md](EXPERIMENTS.md)), where every
change is recorded with its definition, hypothesis and frozen raw results. Sharpe is
**in excess of the 3-month T-bill** (FRED DTB3, E03). Portfolio costs are charged on
drifted weights (E02). 50k training steps, 3 seeds, observations scaled, every model
verified against its metadata card.

### Single test window (2023-01-01 to 2026-06-30) and 20 synthetic paths (E05)

| app | strategy | hist Sharpe | hist CAGR | synth Sharpe | synth wins | beats ref | turnover/day | costs |
|---|---|---|---|---|---|---|---|---|
| trading | buy_hold | 1.25 | 30.8% | 1.10 | 8/20 | - | 0.2% | 0.1% |
| trading | td3 | 1.19 | 26.7% | 0.88 | 4/20 | 5/20 | 3.8% | 3.0% |
| trading | ppo | 1.02 | 23.2% | 0.95 | 5/20 | 7/20 | 0.5% | 0.2% |
| trading | sac | 0.72 | 18.1% | 0.94 | 3/20 | 3/20 | 5.9% | 4.6% |
| portfolio | equal_weight | 1.50 | 31.8% | 1.09 | 10/20 | - | 0.9% | 1.6% |
| portfolio | risk_parity | 1.49 | 26.4% | 1.08 | 3/20 | 8/20 | 0.3% | 0.4% |
| portfolio | ppo | 1.34 | 29.5% | 0.97 | 1/20 | 1/20 | 6.5% | 10.4% |
| portfolio | td3 | 1.15 | 26.9% | 0.79 | 0/20 | 0/20 | 22.2% | 31.5% |
| portfolio | min_variance | 0.87 | 15.4% | 0.96 | 6/20 | 6/20 | 0.6% | 0.6% |
| portfolio | sac | -0.35 | -5.9% | 0.09 | 0/20 | 0/20 | 77.3% | 68.9% |

Reference baseline ("beats ref"): fully invested buy-and-hold (trading), daily 1/N (portfolio).
Costs = total transaction costs as a share of starting capital.

### Walk-forward: 7 test years, each trained on the 5 years before (E07, portfolio)

| strategy | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025-2026 |
|---|---|---|---|---|---|---|---|
| equal_weight | 2.01 | 0.81 | 2.18 | -0.84 | 2.29 | 1.70 | 0.95 |
| min_variance | 1.52 | 0.56 | 1.22 | -0.13 | -0.28 | 0.74 | 1.65 |
| ppo | 1.85 | 0.75 | 1.95 | -0.88 | 2.13 | 1.70 | 0.91 |
| risk_parity | 1.95 | 0.76 | 2.18 | -0.55 | 1.73 | 1.47 | 1.37 |

Selection rules on the chained out-of-sample returns, 2020 to mid-2026:

| rule | Sharpe | CAGR | max drawdown |
|---|---|---|---|
| hindsight_best | 1.19 | 27.0% | -31.4% |
| static:risk_parity | 0.88 | 20.0% | -31.1% |
| static:equal_weight | 0.85 | 21.2% | -31.4% |
| mix_classical | 0.81 | 17.8% | -31.2% |
| static:ppo | 0.77 | 19.1% | -31.4% |
| follow_winner | 0.69 | 16.1% | -31.4% |
| static:min_variance | 0.58 | 12.1% | -31.1% |

### Cross-asset universe (12 ETFs: sectors, Treasuries, gold) (E06)

| app | strategy | hist Sharpe | synth Sharpe | synth wins | beats ref |
|---|---|---|---|---|---|
| trading | PPO | **0.94** | 0.60 | 5/20 | 9/20 |
| trading | buy-and-hold | 0.88 | 0.66 | 3/20 | - |
| trading | SAC | 0.35 | **0.76** | **9/20** | **14/20** |
| portfolio | equal weight | **0.93** | **0.68** | **14/20** | - |
| portfolio | risk parity | 0.83 | 0.63 | 3/20 | 6/20 |
| portfolio | PPO | 0.85 | 0.62 | 0/20 | 0/20 |

Risk-based allocations had lower risk but lower Sharpe here: they overweight Treasuries,
and over 2023-26 TLT (−0.3%/yr) and IEF (3.0%/yr) returned less than T-bills (~4.5%).

### What it says about the research question

- **No DRL agent beats the passive or classical baselines.** On the single window, on
  synthetic paths, and in every one of the 7 walk-forward years (in trading, buy-and-hold
  beats PPO in all 7).
- **Rankings do not survive resampling or time.** TD3 is the best trading agent on history
  (1.19) and the worst on synthetic paths (0.88). Year-to-year strategy rank correlation swings
  between +1.0 and −0.8.
- **Which baseline wins depends on the year**: 1/N in bull years, minimum variance in 2022
  and 2025-26. But **following last year's winner (0.69) is worse than always using risk parity
  (0.88)**. Knowing the regime in advance would be worth ~0.3 Sharpe (hindsight 1.19), and
  nothing tested captures it.
- **Where the agents lose:** they don't beat the baselines even in-sample, and their trading
  costs are about the size of their gap (portfolio PPO pays ~10% of capital, SAC 69%). E05 shows
  most of SAC's churn comes from its entropy bonus.

## Demo Video

[5-minute walkthrough](https://www.youtube.com/watch?v=yNb6URCGGPw): what was built, what was found, and what I'd do differently.

## Provenance

See [CITATION.md](CITATION.md).

## Limitations

- Daily bars only; no slippage or market-impact model; no short selling.
- The Gaussian HMM in `minifinrl/core/meta/synthetic.py` is a single-factor (market + VIX)
  regime model, not a full multivariate fit across tickers. It understates
  tail risk and cross-asset correlation breakdown during real crises.
- Synthetic OHLC collapses open/high/low to the synthetic close and volume
  to 0 (`minifinrl/core/meta/synthetic.to_tidy_panel`). That's fine for computing technical
  indicators, not a realistic intraday model.
- Cash in the environments earns nothing, while Sharpe is measured over the T-bill rate.
  This is conservative for any strategy that holds cash (the trading agents hold 3-24%).
- The Tiingo and Alpaca connectors are tested against recorded responses only; live use
  needs API keys and has not been verified. Alpaca's free feed is IEX-only, so its closes
  are not the consolidated close (run `crosscheck_prices` before relying on it).
- All research numbers use Yahoo data. Two downloads of the same range differ by about
  1e-6 in adjusted closes, so a data hash identifies a stored snapshot, not the market.
- Results are backtests, not evidence of live tradability, and nothing here is investment advice.
