# Provenance

This project reuses design and, in places, near-verbatim logic from
[FinRL](https://github.com/AI4Finance-Foundation/FinRL) (AI4Finance
Foundation), trimmed down to a two-algorithm, two-application benchmark.
Each adapted module carries a `PROVENANCE` note at its top pointing to the
specific FinRL source file and describing what changed and why.

- `minifinrl/core/meta/data.py`: adapted from `finrl/meta/preprocessor/yahoodownloader.py`
- `minifinrl/core/meta/features.py`: adapted from `finrl/meta/preprocessor/preprocessors.py`
- `minifinrl/core/envs/stock_trading.py`: adapted from `finrl/meta/env_stock_trading/env_stocktrading.py`
  and `finrl/applications/stock_trading/`
- `minifinrl/core/envs/portfolio_allocation.py`: adapted from `finrl/meta/env_portfolio_allocation/env_portfolio.py`
- `minifinrl/core/agents/sb3_wrapper.py`: analogous to `finrl/agents/stablebaselines3/models.py`, rewritten smaller
- `minifinrl/core/meta/synthetic.py`, `minifinrl/core/eval/*`: original to this project (no direct FinRL analog)

If you use this code, cite FinRL:

```
Liu, X.-Y., Yang, H., Chen, Q., Zhang, R., Yang, L., Xiao, B., & Wang, C. D.
(2020). FinRL: A deep reinforcement learning library for automated stock
trading in quantitative finance. arXiv:2011.09607.
```
