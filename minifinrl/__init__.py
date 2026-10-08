"""mini-FinRL: deep-RL trading research, and a trade review built on it.

One package per feature, each with the same shape (see ARCHITECTURE.md):

    market     prices, cache, validation, symbols, trading calendar
    regime     the market regime model (HMM), the luck test
    research   deep-RL environments, agents, backtests, experiments
    sentiment  bias signals in trading text, and their evaluation
    review     the trade review: reading a trade, the luck check, explanation
    orders     declared and always refused: paper analysis only
    platform   settings, logging, errors, the capability registry
    interfaces CLI, HTTP API, website
"""

__version__ = "0.2.0"
