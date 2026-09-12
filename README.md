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
configs/  tickers + all tunables (paths, dates, hyperparameters)
meta/     data acquisition, features, synthetic generation
envs/     gymnasium environments (one per application)
agents/   SB3 wrapper + non-RL baselines
eval/     metrics, backtest runner, multi-seed aggregation
scripts/  CLI entry points that wire the above into a pipeline
```

Each layer only imports from layers above it in this list (`envs` imports
`meta`, never the reverse) — that's what keeps a change to, say, the data
source from rippling into the RL code.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/01_download_data.py
python scripts/02_fit_hmm.py
python scripts/03_train.py --app trading --model ppo --seed 0
python scripts/03_train.py --app trading --model sac --seed 0
# repeat 03_train.py across --app {trading,portfolio}, --model {ppo,sac},
# --seed {0,1,2} (see configs/settings.py APPS/MODELS/SEEDS) for the full grid
python scripts/04_backtest.py --paths both
python scripts/05_report.py
```

For a fast smoke test end-to-end, pass `--timesteps 5000` to `03_train.py`
instead of the full `TOTAL_TIMESTEPS`.

## Logging

Every module logs through `configs/logging_config.py` (`get_logger(__name__)`)
rather than `print()`. Output goes to both the console (INFO+) and a
rotating file at `results/logs/mini_finrl.log` (DEBUG+, 5 x 5MB, so a long
sweep doesn't grow the file unbounded). Cache hits/misses, HMM convergence
and state occupancy, per-run train/save/load events, and each backtest's
headline stats all land there — `tail -f results/logs/mini_finrl.log`
while a sweep runs, or grep it after the fact instead of relying on
terminal scrollback. The final report tables from `05_report.py` are
printed directly (they're the deliverable, not a log event).

## Extending

- **New algorithm**: add one entry to `_ALGOS` in `agents/sb3_wrapper.py`.
- **New application**: add a `gym.Env` in `envs/`, register it in the
  `ENVS` dict in `scripts/03_train.py` and `scripts/04_backtest.py`, add its
  name to `APPS` in `configs/settings.py`.
- **New ticker universe**: add a list to `configs/tickers.py`, point
  `TICKERS` at it in `configs/settings.py`.
- **New indicator**: add its stockstats name to `INDICATORS` in
  `configs/settings.py` — `meta/features.py` picks it up automatically.

## Results

TODO — metrics table (median and IQR across seeds), equity curves,
rank-stability across synthetic paths. Run `scripts/05_report.py` after a
full training + backtest sweep to generate the numbers.

## Provenance

See [CITATION.md](CITATION.md).

## Limitations

- Daily bars only; no slippage or market-impact model; no short selling.
- The Gaussian HMM in `meta/synthetic.py` is a single-factor (market + VIX)
  regime model, not a full multivariate fit across tickers — it understates
  tail risk and cross-asset correlation breakdown during real crises.
- Synthetic OHLC collapses open/high/low to the synthetic close and volume
  to 0 (`meta/synthetic.to_tidy_panel`) — fine for computing technical
  indicators, not a realistic intraday model.
- Results are backtests, not evidence of live tradability.
