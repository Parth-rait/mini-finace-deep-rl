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
minifinrl/
  market/      prices, validated incremental cache, symbol directory, trading calendar
  regime/      market regime model (HMM) and the luck test
  research/    deep-RL envs, agents, backtests, walk-forward, experiment log (the only torch user)
  sentiment/   bias signals in trading text, and their evaluation
  review/      the trade review behind the website
  orders/      declared and always refused (paper analysis only)
  platform/    settings, logging, errors, the capability registry
  engine.py    builds one service per feature
  system.py    composition root: build_system(profile) picks rules or LLM adapters
  interfaces/  CLI, HTTP API, website
scripts/    thin shells over the CLI, kept for muscle memory
tests/      offline test suite (pytest), incl. tests/test_architecture.py
```

Every feature has the same shape (a service with its capabilities, its schemas,
its settings) and features depend on each other in one direction only. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the rules and for how to add a capability or a feature.

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
# --seed {0,1,2} (see minifinrl/research/settings.py APPS/MODELS/SEEDS) for the full grid
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

## Website: Trade Review

Pick a stock from a search over every US-listed stock and ETF (the NASDAQ Trader
symbol directory, refreshed daily), pick the buy date and the sell date (or tick
"still holding"), and optionally say why you made the trade. The site:

1. counts the trading days between the dates on the exchange calendar (a holiday buy
   moves to the next session, a weekend sell uses the Friday close);
2. checks where the outcome falls among 1,000 paths simulated from the market regime on
   the entry day, shown as a 3D surface with your trade's path drawn across it;
3. marks the bias signals in your reasoning, quoting your words;
4. compares the trade with the S&P 500 over the same days, and explains the result in
   plain words (the model writes placeholders, the code fills in the numbers).

You can also describe the trade in words; the LLM fills the form and you check it.
Dates that read two ways (03/04/2025), dates that don't exist, and days the text never
gave ("in 2023") are left for you to pick rather than guessed. Renamed tickers (FB) are
followed, delisted ones (TWTR) and listings that started after the buy date are
explained, recent listings work once they have 60 days of history, and options are
reviewed as the underlying with a note. Crypto, individual bonds and non-US listings
are declined with a pointer to what does work (bond ETFs, US-listed ADRs).

```bash
MINIFINRL_BIAS=aip python -m minifinrl.interfaces.api     # http://127.0.0.1:8000
PORT=8077 MINIFINRL_BIAS=aip python -m minifinrl.interfaces.api   # if 8000 is taken
```

With `GEMINI_API_KEY` in `.env` the LLM reads the trade, finds bias signals and writes
the explanation (`MINIFINRL_BIAS=aip`); with `MINIFINRL_BIAS=rules`, or no key, the
rules versions run.
Paper analysis only: nothing here places orders or gives advice.

## HTTP API

The same capabilities are served over HTTP (paper analysis only; nothing places orders):

```bash
uvicorn minifinrl.interfaces.api:create_app --factory --port 8000
curl -s localhost:8000/                       # routes
curl -s -X POST localhost:8000/luck-test -H 'content-type: application/json' \
     -d '{"ticker": "META", "date": "2023-02-01", "horizon_days": 5}'
```

Routes are generated from the capability registry. Training, fetching and
backtesting have no route; they run from the command line only.
`MINIFINRL_PROFILE`, `MINIFINRL_UNIVERSE` and `MINIFINRL_WORKSPACE` choose the
setup. Interactive docs are at `/docs`.

## Bias classifier evaluation

`corpus/LABELLING.md` explains how to label trading texts for six behavioural
biases. With the labels in `corpus/bias_labels.jsonl`:

```bash
python -m minifinrl evaluate-biases
```

reports precision, recall, F1 and Cohen's kappa per bias, and agreement
between two human labellers when a second set of labels is present.

## Logging

Every module logs through `minifinrl/platform/log.py` (`get_logger(__name__)`)
rather than `print()`. Output goes to both the console (INFO+) and a
rotating file at `results/logs/mini_finrl.log` (DEBUG+, 5 x 5MB, so a long
sweep doesn't grow the file unbounded). Cache hits/misses, HMM convergence
and state occupancy, per-run train/save/load events, and each backtest's
headline stats all land there. Run `tail -f results/logs/mini_finrl.log`
while a sweep runs, or grep it after the fact instead of relying on
terminal scrollback. The final report tables from `05_report.py` are
printed directly (they're the deliverable, not a log event).

## Extending

- **New algorithm**: add one entry to `_ALGOS` in `minifinrl/research/agents/sb3_wrapper.py`.
- **New application**: add a `gym.Env` in `minifinrl/research/envs/`, register it in the
  `ENVS` dict in `minifinrl/research/pipeline.py`, add its name to `APPS` in
  `minifinrl/research/settings.py`.
- **New ticker universe**: add a list to `minifinrl/market/universe.py`, point
  `TICKERS` at it in `minifinrl/market/settings.py`.
- **New indicator**: add its stockstats name to `INDICATORS` in
  `minifinrl/market/settings.py`; `minifinrl/market/indicators.py` picks it up automatically.
  It also needs a scaling rule in `minifinrl/research/envs/scaling.py` (an unknown name raises).

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
- The Gaussian HMM in `minifinrl/regime/model.py` is a single-factor (market + VIX)
  regime model, not a full multivariate fit across tickers. It understates
  tail risk and cross-asset correlation breakdown during real crises.
- Synthetic OHLC collapses open/high/low to the synthetic close and volume
  to 0 (`minifinrl/regime/model.to_tidy_panel`). That's fine for computing technical
  indicators, not a realistic intraday model.
- Cash in the environments earns nothing, while Sharpe is measured over the T-bill rate.
  This is conservative for any strategy that holds cash (the trading agents hold 3-24%).
- Tiingo was checked live against Yahoo on all 20 tickers, 2014 to mid-2026: identical
  trading calendars, daily returns within 1.2 basis points at the 99th percentile, one
  bad print found (Tiingo's XOM row for 2014-07-28 repeats the prices of 2014-07-30).
  Yahoo stays the research source and Tiingo is the cross-check. Tiingo's free plan is
  for internal use only, so its prices are not committed or served. The Alpaca connector
  is tested against recorded responses only, and its free feed is IEX-only.
- All research numbers use Yahoo data. Two downloads of the same range differ by about
  1e-6 in adjusted closes, so a data hash identifies a stored snapshot, not the market.
- Results are backtests, not evidence of live tradability, and nothing here is investment advice.
