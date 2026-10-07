# Experiment log

Every change that can move a result is recorded here: what changed, its exact
definition, the hypothesis, and the numbers it produced. Written by the
`record_experiment` capability (`python -m minifinrl record-experiment ...`);
the machine-readable twin is `results/experiments/log.jsonl`, and each entry's
raw backtest rows are frozen in `results/experiments/<id>/`.

Sharpe = annualised mean / std of daily returns (excess over the stated
risk-free rate). "Synthetic" = 20 regime-model paths of 500 days. Wins and
"beats ref" use each model's median over seeds per path; the reference
baseline is fully invested buy-and-hold (trading) and daily 1/N (portfolio).

## E01: Reference: fair baseline, PPO/SAC, 50k steps

*2026-10-06T06:06:48+00:00, commit `571029a-dirty`, data `f555b4c0286c`, results sha256 `ec353f8c0d13`*

**Change.** State after F13 (fully invested buy-and-hold) and R2 (seed-median win rates). Starting point every later experiment is compared with.

**Definition.**

```
trading baseline: target shares_i = V0(1-0.005) / (N (1+c) p_i,0), bought at <= HMAX/day, then held
portfolio baseline: w_t = 1/N every day
costs: trading c * |shares traded| * price; portfolio c * sum_i |w_new,i - w_prev,i|   (c = 10 bps)
Sharpe = sqrt(252) * mean(r_t - 0) / std(r_t), risk-free = 0
wins: per path, model value = median over seeds; winner = argmax
```

**Hypothesis.** None (reference). Expect: no agent beats the passive baseline.

**Files.** `minifinrl/core/agents/baseline.py`, `minifinrl/core/eval/report.py`


**portfolio**

| model | seeds | hist Sharpe | hist CAGR | synth Sharpe | wins | beats ref | turnover/day | costs |
|---|---|---|---|---|---|---|---|---|
| equal_weight | 0 | 1.79 | 32.1% | 1.29 | 20/20 | - | - | - |
| ppo | 3 | 1.61 | 29.5% | 1.16 | 0/20 | 0/20 | - | - |
| sac | 3 | 0.07 | -1.3% | 0.23 | 0/20 | 0/20 | - | - |

**trading**

| model | seeds | hist Sharpe | hist CAGR | synth Sharpe | wins | beats ref | turnover/day | costs |
|---|---|---|---|---|---|---|---|---|
| buy_hold | 0 | 1.48 | 30.8% | 1.26 | 11/20 | - | - | - |
| ppo | 3 | 1.27 | 23.2% | 1.09 | 6/20 | 7/20 | - | - |
| sac | 3 | 0.94 | 18.1% | 1.11 | 3/20 | 3/20 | - | - |

**Conclusion.** No agent beats the passive baseline in either app. Trading: buy-and-hold 1.48 historical, PPO beats it on 7/20 paths, SAC 3/20. Portfolio: 1/N wins 20/20. Caveats found since: portfolio costs ignored price drift (fixed in E02) and Sharpe assumed a 0% risk-free rate (E03).


---

## E02: Portfolio costs charged against drifted weights

*2026-10-06T06:22:21+00:00, commit `571029a-dirty`, data `f555b4c0286c`, results sha256 `f2c81928cbc1`, compared with E01*

**Change.** PortfolioAllocationEnv now carries weights forward through each day's price moves and charges turnover against those drifted weights (env_version portfolio-v2). Portfolio PPO and SAC retrained (3 seeds each, 50k steps); v1 models are refused by their cards. Trading untouched: it is the control.

**Definition.**

```
v1:  turnover_t = sum_i |w_new,i - w_target,i(t-1)|           (ignores drift: holding 1/N costs 0)
v2:  w_drift,i = w_i (1 + r_i) / (1 + w . r)                  (weights after day t moves)
     turnover_t = sum_i |w_new,i - w_drift,i(t-1)|,  cost_t = c * turnover_t,  c = 10 bps
     V_{t+1} = V_t (1 + w_new . r_t - cost_t)
```

**Hypothesis.** v1 flattered steady strategies, the 1/N reference above all. Charging drift should lower 1/N's Sharpe slightly and narrow, but not close, its gap to the agents. Trading results must not move.

**Files.** `minifinrl/core/envs/portfolio_allocation.py`, `minifinrl/core/agents/registry.py`, `tests/test_envs.py`


**portfolio** (Δ vs E01)

| model | seeds | hist Sharpe | hist CAGR | synth Sharpe | wins | beats ref | turnover/day | costs |
|---|---|---|---|---|---|---|---|---|
| equal_weight | 0 | 1.77 (-0.02) | 31.8% | 1.28 (-0.01) | 20/20 | - | 0.9% | 1.6% |
| ppo | 3 | 1.60 | 29.5% | 1.16 | 0/20 | 0/20 | 6.5% | 10.4% |
| sac | 3 | -0.15 (-0.22) | -5.9% | 0.24 (+0.01) | 0/20 | 0/20 | 77.3% | 68.9% |

**trading** (Δ vs E01)

| model | seeds | hist Sharpe | hist CAGR | synth Sharpe | wins | beats ref | turnover/day | costs |
|---|---|---|---|---|---|---|---|---|
| buy_hold | 0 | 1.48 | 30.8% | 1.26 | 11/20 | - | 0.2% | 0.1% |
| ppo | 3 | 1.27 | 23.2% | 1.09 | 6/20 | 7/20 | 0.5% | 0.2% |
| sac | 3 | 0.94 | 18.1% | 1.11 | 3/20 | 3/20 | 5.9% | 4.6% |

**Conclusion.** Confirmed but small. 1/N now turns over 0.9%/day and pays 1.6% of capital over the test window: historical Sharpe 1.790 -> 1.775, still wins 20/20 paths. Retrained PPO is unchanged (1.605 -> 1.604, ~10% of capital in costs). SAC retrained on v2 is worse on history (0.07 -> -0.15) and still churns 77%/day. Trading identical to the last digit (control holds). The accounting flaw did not change any conclusion.


---

## E03: Sharpe and Sortino in excess of the 3-month T-bill

*2026-10-06T06:23:26+00:00, commit `571029a-dirty`, data `f555b4c0286c`, results sha256 `127cef5aec8d`, compared with E02*

**Change.** New FRED connector for DTB3 (3-month Treasury bill, secondary market, keyless). Sharpe and Sortino now use returns in excess of the daily T-bill return instead of 0. Metric-only: same models, same paths, same trades as E02.

**Definition.**

```
r_f,t = (1 + y_t / 100)^(1/252) - 1     y_t = DTB3 on day t (% p.a.); holidays carry the last value forward;
                                         synthetic paths (future dates) hold the last observed y constant
Sharpe  = sqrt(252) * mean(r_t - r_f,t) / std(r_t - r_f,t)
Sortino = sqrt(252) * mean(r_t - r_f,t) / std(min(r_t - r_f,t, 0) | < 0)
validation: y in [-1, 25]% (DTB3 printed -0.05% on 2020-03-26; negatives are real)
```

**Hypothesis.** Bills paid 4.55% on average over the test window, so a 0% risk-free rate overstated every Sharpe by roughly 0.2-0.3. The shift should be close to uniform (similar volatilities), so rankings should barely move.

**Files.** `minifinrl/core/meta/providers.py`, `minifinrl/core/meta/store.py`, `minifinrl/core/eval/metrics.py`, `minifinrl/core/pipeline.py`, `tests/test_classical.py`


**portfolio** (Δ vs E02)

| model | seeds | hist Sharpe | hist CAGR | synth Sharpe | wins | beats ref | turnover/day | costs |
|---|---|---|---|---|---|---|---|---|
| equal_weight | 0 | 1.50 (-0.27) | 31.8% | 1.09 (-0.19) | 19/20 | - | 0.9% | 1.6% |
| ppo | 3 | 1.34 (-0.26) | 29.5% | 0.97 (-0.19) | 1/20 | 1/20 | 6.5% | 10.4% |
| sac | 3 | -0.35 (-0.20) | -5.9% | 0.09 (-0.15) | 0/20 | 0/20 | 77.3% | 68.9% |

**trading** (Δ vs E02)

| model | seeds | hist Sharpe | hist CAGR | synth Sharpe | wins | beats ref | turnover/day | costs |
|---|---|---|---|---|---|---|---|---|
| buy_hold | 0 | 1.25 (-0.23) | 30.8% | 1.10 (-0.17) | 12/20 | - | 0.2% | 0.1% |
| ppo | 3 | 1.02 (-0.25) | 23.2% | 0.95 (-0.15) | 5/20 | 7/20 | 0.5% | 0.2% |
| sac | 3 | 0.72 (-0.23) | 18.1% | 0.94 (-0.17) | 3/20 | 3/20 | 5.9% | 4.6% |

**Conclusion.** Confirmed. Every historical Sharpe falls by 0.20-0.27 (applied rate averaged 4.45% on the test window, 3.9% on synthetic paths); synthetic medians fall by 0.15-0.19. Rankings are unchanged: trading buy-and-hold 1.25 > PPO 1.02 > SAC 0.72; portfolio 1/N 1.50 > PPO 1.34. Portfolio PPO now beats 1/N on 1/20 paths (was 0), within noise. All earlier Sharpe figures (E01, E02, README) used rf = 0 and are about 0.25 too high in absolute terms.


---

## E04: Classical baselines: minimum variance and risk parity

*2026-10-06T06:24:25+00:00, commit `571029a-dirty`, data `f555b4c0286c`, results sha256 `c89a9f12d8c2`, compared with E03*

**Change.** Two portfolio baselines implemented from their definitions (not PyPortfolioOpt, not FinRL): long-only minimum variance and equal-risk-contribution risk parity, both on a Ledoit-Wolf covariance of the trailing 252 daily returns, rebalanced every 21 trading days, holding (zero turnover) in between. Real history before each window supplies the first estimate. Max-Sharpe deliberately excluded (expected-return estimation error). Same agents, paths and rf as E03.

**Definition.**

```
Sigma_t   = LedoitWolf(r_{t-252..t})                    (closes up to and including day t; no look-ahead)
min-var   w = argmin w' Sigma w  s.t. 1'w = 1, 0 <= w <= 1          (SLSQP on Sigma / mean(diag Sigma))\nrisk par. x* = argmin_{x>0} 1/2 x' Sigma x - (1/N) sum log x_i, w = x*/1'x*\n          damped Newton: grad = Sigma x - 1/(N x), Hessian = Sigma + diag(1/(N x^2)); checked |RC_i - 1/N| < 1e-8\n          RC_i = w_i (Sigma w)_i / (w' Sigma w)\nrebalance on day t if t mod 21 = 0, else keep the drifted weights (cost 0 under portfolio-v2)
```

**Hypothesis.** Risk-based allocations trade return for lower volatility: lower drawdowns, Sharpe near 1/N. If either beats 1/N often, '1/N wins 20/20' (E01-E03) was an artefact of comparing agents against a single baseline.

**Files.** `minifinrl/core/agents/classical.py`, `minifinrl/core/pipeline.py`, `tests/test_classical.py`


**portfolio** (Δ vs E03)

| model | seeds | hist Sharpe | hist CAGR | synth Sharpe | wins | beats ref | turnover/day | costs |
|---|---|---|---|---|---|---|---|---|
| equal_weight | 0 | 1.50 | 31.8% | 1.09 | 10/20 | - | 0.9% | 1.6% |
| min_variance | 0 | 0.87 | 15.4% | 0.96 | 6/20 | 6/20 | 0.6% | 0.6% |
| ppo | 3 | 1.34 | 29.5% | 0.97 | 1/20 | 1/20 | 6.5% | 10.4% |
| risk_parity | 0 | 1.49 | 26.4% | 1.08 | 3/20 | 8/20 | 0.3% | 0.4% |
| sac | 3 | -0.35 | -5.9% | 0.09 | 0/20 | 0/20 | 77.3% | 68.9% |

**trading** (Δ vs E03)

| model | seeds | hist Sharpe | hist CAGR | synth Sharpe | wins | beats ref | turnover/day | costs |
|---|---|---|---|---|---|---|---|---|
| buy_hold | 0 | 1.25 | 30.8% | 1.10 | 12/20 | - | 0.2% | 0.1% |
| ppo | 3 | 1.02 | 23.2% | 0.95 | 5/20 | 7/20 | 0.5% | 0.2% |
| sac | 3 | 0.72 | 18.1% | 0.94 | 3/20 | 3/20 | 5.9% | 4.6% |

**Conclusion.** Risk parity ties 1/N on Sharpe (historical 1.49 vs 1.50, synthetic 1.08 vs 1.09) with less risk (vol 13.3% vs 16.3%, max drawdown -16% vs -19.5%, costs 0.42% vs 1.61%) and beats 1/N on 8/20 paths. Minimum variance is the least risky (vol 12.2%, drawdown -13%) but underweighted the 2023-26 tech rally (historical Sharpe 0.87, CAGR 15.4%); on synthetic paths it beats 1/N on 6/20. Synthetic wins now spread 1/N 10, min-var 6, risk parity 3, PPO 1, SAC 0. Classical risk-based methods beat 1/N on 30-40% of paths; PPO on 1/20. The DRL agents remain the weakest non-degenerate strategies.


---

## E05: TD3 matched to SAC: is the churn the entropy bonus?

*2026-10-06T06:39:05+00:00, commit `571029a-dirty`, data `f555b4c0286c`, results sha256 `d9b8c3237eca`, compared with E04*

**Change.** Added TD3 with every shared hyperparameter equal to SAC (batch 256, buffer 100k, lr 3e-4, 1000 warm-up steps, 50k steps, 3 seeds, both apps). Only the objective and exploration differ. Everything else as E04.

**Definition.**

```
SAC: maximise E[ sum_t gamma^t ( r_t + alpha * H(pi(.|s_t)) ) ], stochastic policy, alpha auto-tuned
TD3: maximise E[ sum_t gamma^t r_t ],  deterministic mu(s); explore with a_t = mu(s_t) + N(0, (0.1 * half_range)^2)
     twin critics, policy_delay = 2, target smoothing noise 0.2 clipped at 0.5 (SB3 defaults)
turnover_t = sum_i |w_new,i - w_drift,i| (portfolio) or traded notional / value (trading); evaluation is deterministic for both
```

**Hypothesis.** SAC's entropy bonus rewards ever-changing actions, which in a softmax allocation means churning weights. TD3, identical except for that objective, should turn over much less and pay far lower costs.

**Files.** `minifinrl/core/agents/sb3_wrapper.py`, `minifinrl/core/configs/settings.py`, `minifinrl/core/pipeline.py`


**portfolio** (Δ vs E04)

| model | seeds | hist Sharpe | hist CAGR | synth Sharpe | wins | beats ref | turnover/day | costs |
|---|---|---|---|---|---|---|---|---|
| equal_weight | 0 | 1.50 | 31.8% | 1.09 | 10/20 | - | 0.9% | 1.6% |
| min_variance | 0 | 0.87 | 15.4% | 0.96 | 6/20 | 6/20 | 0.6% | 0.6% |
| ppo | 3 | 1.34 | 29.5% | 0.97 | 1/20 | 1/20 | 6.5% | 10.4% |
| risk_parity | 0 | 1.49 | 26.4% | 1.08 | 3/20 | 8/20 | 0.3% | 0.4% |
| sac | 3 | -0.35 | -5.9% | 0.09 | 0/20 | 0/20 | 77.3% | 68.9% |
| td3 | 3 | 1.15 | 26.9% | 0.79 | 0/20 | 0/20 | 22.2% | 31.5% |

**trading** (Δ vs E04)

| model | seeds | hist Sharpe | hist CAGR | synth Sharpe | wins | beats ref | turnover/day | costs |
|---|---|---|---|---|---|---|---|---|
| buy_hold | 0 | 1.25 | 30.8% | 1.10 | 8/20 | - | 0.2% | 0.1% |
| ppo | 3 | 1.02 | 23.2% | 0.95 | 5/20 | 7/20 | 0.5% | 0.2% |
| sac | 3 | 0.72 | 18.1% | 0.94 | 3/20 | 3/20 | 5.9% | 4.6% |
| td3 | 3 | 1.19 | 26.7% | 0.88 | 4/20 | 5/20 | 3.8% | 3.0% |

**Conclusion.** Largely confirmed in portfolio: turnover 77%/day -> 22%/day (3.5x less), costs 69% -> 32% of capital, Sharpe -0.35 -> 1.15. Not fully: TD3 still churns 3x PPO, because its deterministic actions saturate (83% of portfolio actions at the +-10 bound), flipping between concentrated allocations. Trading: TD3 is the best agent on history (1.19; seed 2 at 1.50 beats buy-and-hold's 1.25) but the worst agent on synthetic paths (0.88), with seeds spanning 0.79-1.50, so its historical edge does not survive resampling. No agent beats the passive or classical baselines; ranking among agents is unstable.


---

## E07: Walk-forward: 7 test years, and does following last year's winner work?

*2026-10-06T06:54:46+00:00, commit `571029a-dirty`, data `d65b235e7dc6`, results sha256 `3bcc9bca21fe`, compared with E05*

**Change.** Replaced the single 2023-26 test window with 7 walk-forward folds: train on the 5 calendar years before each test year Y, test on Y (2019..2024, last fold 2025-01-01..2026-06-30). PPO retrained per fold (3 seeds, 50k steps, both apps), all baselines run per fold, rf = DTB3. Then a selection study on the chained out-of-sample daily returns 2020-2026. Universe = the original 8 stocks. Workspace results/experiments/E07/.

**Definition.**

```
fold Y: train [Y-5-01-01, Y-1-12-31], test [Y-01-01, Y-12-31]   (features from trailing data only)
strategy daily return = median over seeds;  fold Sharpe on excess over r_f,t
rules (chained over folds 2..7):
  static:s        always s
  follow_winner   fold k uses argmax_s Sharpe_s(fold k-1)
  mix_classical   r_t = (r_EW + r_RP + r_MV) / 3   (inter-strategy rebalancing cost ignored)
  hindsight_best  fold k uses argmax_s Sharpe_s(fold k)   (unattainable: regret yardstick)
persistence: Spearman rho(Sharpe ranks fold k, fold k+1)
```

**Hypothesis.** (1) The winning strategy depends on the market year, so '1/N wins' was partly a 2023-26 artefact. (2) Because rankings do not persist, picking last year's winner does no better than a fixed rule.

**Files.** `minifinrl/core/walkforward.py`, `minifinrl/engine.py`, `minifinrl/system.py`, `minifinrl/interfaces/cli.py`, `tests/test_walkforward.py`


**portfolio** (Δ vs E05)

| model | seeds | hist Sharpe | hist CAGR | synth Sharpe | wins | beats ref | turnover/day | costs |
|---|---|---|---|---|---|---|---|---|
| equal_weight | 0 | 1.70 (+0.20) | 31.6% | - | - | - | 0.9% | 0.3% |
| min_variance | 0 | 0.74 (-0.13) | 14.0% | - | - | - | 0.9% | 0.2% |
| ppo | 3 | 1.63 (+0.29) | 30.6% | - | - | - | 7.0% | 2.1% |
| risk_parity | 0 | 1.47 (-0.02) | 25.5% | - | - | - | 0.4% | 0.1% |

**trading** (Δ vs E05)

| model | seeds | hist Sharpe | hist CAGR | synth Sharpe | wins | beats ref | turnover/day | costs |
|---|---|---|---|---|---|---|---|---|
| buy_hold | 0 | 1.62 (+0.37) | 31.1% | - | - | - | 0.4% | 0.1% |
| ppo | 3 | 0.95 (-0.07) | 25.0% | - | - | - | 0.6% | 0.2% |

**Conclusion.** Both confirmed. (1) Winners depend on the year: 1/N wins the bull years (2019, 2020, 2021, 2023, 2024), minimum variance wins the 2022 bear (-0.13 vs 1/N -0.84) and 2025-26 (1.65 vs 0.95). Rank persistence flips sign: rho = +1.0, +0.8, -0.2, -0.8, +1.0, -0.8. (2) Following last year's winner is the worst rule except static min-var: Sharpe 0.69 vs 0.88 for always-risk-parity and 0.85 for always-1/N. Best implementable rule = always risk parity (0.88, 20.0%/yr); perfect hindsight would give 1.19, so knowing the regime in advance is worth ~0.3 Sharpe, and nothing here captures it. PPO is never the sole best in any fold (ties 1/N in 2024), chained 0.77; in trading, buy-and-hold beats PPO in all 7 years (chained 0.89 vs 0.47). Trading persistence is undefined (2 strategies). All rules share a ~-31% max drawdown (the 2020 crash).


---

## E06: Cross-asset universe: 9 sector ETFs + Treasuries + gold

*2026-10-06T08:11:21+00:00, commit `571029a-dirty`, data `d65b235e7dc6`, results sha256 `ebf9fa2d9494`, compared with E05*

**Change.** Same pipeline and settings as E05 (rf = DTB3, drift-aware costs, all baselines, PPO/SAC/TD3 x 3 seeds, 50k steps, 20 synthetic paths) on a different universe: XLB XLE XLF XLI XLK XLP XLU XLV XLY TLT IEF GLD. All trade since before 2014 (XLC, XLRE excluded), so no gaps and no survivorship bias. Isolated workspace results/experiments/E06/. Run paused at 15/18 models and resumed; state verified clean before resuming.

**Definition.**

```
universe U = {9 SPDR sectors} + {TLT, IEF} + {GLD},  N = 12
obs: trading 1 + N + N + 7N = 109,  portfolio N + 7N = 96   (was 73 / 64)
everything else identical to E05; validation: 37,680 rows = 3,140 days x 12, 0 errors
```

**Hypothesis.** With assets of very different risk (bonds vs equity sectors vs gold), 1/N is a poor benchmark, so risk parity and minimum variance should beat it clearly; agents get a more varied allocation problem.

**Files.** `minifinrl/core/configs/tickers.py`, `minifinrl/system.py`, `minifinrl/interfaces/cli.py`


**portfolio** (Δ vs E05)

| model | seeds | hist Sharpe | hist CAGR | synth Sharpe | wins | beats ref | turnover/day | costs |
|---|---|---|---|---|---|---|---|---|
| equal_weight | 0 | 0.93 (-0.57) | 14.5% | 0.68 (-0.41) | 14/20 | - | 0.7% | 0.7% |
| min_variance | 0 | 0.44 (-0.42) | 7.3% | -0.07 (-1.03) | 3/20 | 4/20 | 0.6% | 0.6% |
| ppo | 3 | 0.85 (-0.49) | 13.5% | 0.62 (-0.36) | 0/20 | 0/20 | 5.7% | 6.1% |
| risk_parity | 0 | 0.83 (-0.66) | 11.9% | 0.63 (-0.45) | 3/20 | 6/20 | 0.3% | 0.2% |
| sac | 3 | -0.46 (-0.11) | -2.7% | -0.51 (-0.60) | 0/20 | 0/20 | 75.3% | 54.3% |
| td3 | 3 | 0.26 (-0.89) | 6.9% | 0.14 (-0.65) | 0/20 | 0/20 | 25.3% | 21.2% |

**trading** (Δ vs E05)

| model | seeds | hist Sharpe | hist CAGR | synth Sharpe | wins | beats ref | turnover/day | costs |
|---|---|---|---|---|---|---|---|---|
| buy_hold | 0 | 0.88 (-0.37) | 14.4% | 0.66 (-0.44) | 3/20 | - | 0.2% | 0.1% |
| ppo | 3 | 0.94 (-0.08) | 13.9% | 0.60 (-0.35) | 5/20 | 9/20 | 0.5% | 0.3% |
| sac | 3 | 0.35 (-0.37) | 7.2% | 0.76 (-0.18) | 9/20 | 14/20 | 4.3% | 2.5% |
| td3 | 3 | 0.53 (-0.66) | 12.5% | 0.57 (-0.31) | 3/20 | 6/20 | 4.0% | 2.4% |

**Conclusion.** Refuted for 2023-26, for a specific and checkable reason. 1/N still has the best Sharpe (hist 0.93, synth 0.68, wins 14/20); risk parity 0.83 and min-var 0.44, despite lower risk (vol 8.7% and 6.4% vs 10.3%; max drawdown -8.6% / -8.3% vs -11.3%). Cause: risk-based weights overweight low-vol Treasuries, and over 2023-26 TLT returned -0.3%/yr and IEF 3.0%/yr, below the ~4.5% T-bill: bonds lost to cash, so lower risk came at a return cost larger than the risk saved. Agents: trading PPO edges buy-and-hold on history (0.94 vs 0.88, the first agent to beat the reference on history) but not on synthetic paths (0.60 vs 0.66). Trading SAC beats buy-and-hold on 14/20 synthetic paths yet is worst on history (0.35): binomial p = 0.058 one-sided, not significant. Portfolio: no agent beats 1/N; the SAC/TD3 churn finding from E05 replicates (turnover 75% vs 25%/day, costs 54% vs 21%). All Sharpes are lower than E05 (ETFs: lower volatility but much lower excess return).


---

## E08: LLM bias classifier against the rules baseline

*2026-10-06T11:28:12+00:00, commit `a2f5e62-dirty`, data ``, results sha256 ``*

**Change.** First evaluation of the bias classifiers on 50 hand-labelled StockTwits posts (labelled by the author, single labeller). Rules baseline (keyword patterns, rules-v1) against an LLM classifier through aip (Gemini 3.5 flash-lite, SMALL tier, temperature 0) whose instructions are the definitions and four rules in corpus/LABELLING.md. No labelled text appears in the prompt. Signals whose quoted evidence is not in the post are dropped. Backtest results are unchanged and not part of this entry.

**Definition.**

```
per label, over the 50 texts as yes/no: precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN)
Cohen kappa = (p_o - p_e)/(1 - p_e), p_e from the two sides' yes-rates
macro = mean over labels with a defined value; exact match = predicted label set equals the reference set
```

**Hypothesis.** Keyword rules miss most bias signals in informal posts; an LLM given the written definitions should reach moderate agreement with the labels.

**Files.** `minifinrl/adapters/aip_bias.py`, `minifinrl/bias_eval.py`, `minifinrl/system.py`, `corpus/LABELLING.md`


**Conclusion.** Confirmed. Rules: macro F1 0.04, macro kappa 0.02 (finds 2 of 19 overconfidence labels and nothing else). LLM: macro F1 0.46, macro kappa 0.42, exact match 0.54, no failures, no evidence dropped, total cost about 1.3 cents for 50 posts. Per label the LLM does best on herding (kappa 0.65) and fomo (0.54), is conservative on overconfidence (precision 0.78, recall 0.37) and finds no anchoring (0 of 5): four of those five are the 2018-analogy and future-target cases, where the written rule (anchoring needs a past price reference) and the labels differ. It labels three sarcastic posts that the rules say are not biases (st-0066, st-0079, st-0125). Revenge trading has no reference examples and loss aversion one, so those scores are not meaningful. With one labeller and 50 texts these are indicative only; a second labeller would give the human-agreement ceiling.


---
