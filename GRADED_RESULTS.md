# Results from the graded submission

This branch is the graded commit (571029a, 12 Sep 2026) plus the results that
run produced. They were excluded by .gitignore at submission and are added here
so they sit next to the version that was assessed. The code is unchanged.

Files added (all from the run on 12 Sep 2026, copied from a local backup):

| file | last modified | sha256 (first 16) |
|---|---|---|
| results/logs/backtest_results.csv | 2026-09-12 18:54 | a4448461d83a47f1 |
| results/models/trading_ppo_seed0.zip | 2026-09-12 18:53 | 520196fd8ce7eaea |
| results/models/trading_sac_seed0.zip | 2026-09-12 18:54 | ff764e3343b740cd |
| results/models/portfolio_ppo_seed0.zip | 2026-09-12 18:54 | 711cd06a7c93542d |
| results/models/portfolio_sac_seed0.zip | 2026-09-12 18:54 | 91e3d5af43417b4d |
| results/logs/mini_finrl.log | 2026-09-11 to 2026-09-12 | log lines from those two days only |
| results/logs/report.txt | | output of scripts/05_report.py on the CSV above |

The log was cut at the end of 12 Sep because the same file later collected
entries from post-grading work. The CSV was written at 18:54:45 that day
(see the log) and report.txt matches the report printed at 18:54:53.

## Numbers

One seed per model, 5,000 training steps, test window 2023-01-01 to
2026-06-30, 20 synthetic paths. Sharpe with a 0% risk-free rate.

| | baseline | PPO | SAC |
|---|---|---|---|
| Trading, historical Sharpe | 1.21 | 1.17 | 1.23 |
| Trading, synthetic median Sharpe | 1.09 | 1.18 | 1.20 |
| Trading, synthetic paths won (of 20) | 2 | 8 | 10 |
| Portfolio, historical Sharpe | 1.79 | 1.77 | 1.05 |
| Portfolio, synthetic median Sharpe | 1.29 | 1.24 | 0.87 |
| Portfolio, synthetic paths won (of 20) | 15 | 2 | 3 |

To print the tables again: `python scripts/05_report.py`.

These are the submitted numbers, not the current ones. Later work found
problems in this version (the trading baseline held about 90% cash, the
trading SAC policy repeated a single action, observations were unscaled) and
the corrected study reaches a different conclusion. That work is on the
shippable-core branch, with each change recorded in EXPERIMENTS.md.
