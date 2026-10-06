"""
Observation scaling shared by both environments.

PROVENANCE: original to this project. FinRL feeds raw values to the policy:
cash ~1e6, prices 15-800, CCI +-580, moving averages at price level. On
those inputs the 5k-step trading SAC collapsed to one constant action
(PLAN_SHIPPABLE.md F9). Scaling is done inside the env, deterministically,
so nothing extra has to be saved next to a model (unlike VecNormalize's
running statistics) and a backtest reproduces exactly.

Every rule uses only same-day values (the indicator and that day's close),
so scaling adds no look-ahead. Target: typical values within about +-1.5,
measured on the 2014-2026 panel (p1..p99 in the comments).

An indicator with no rule raises, so adding one to INDICATORS forces a
decision here instead of silently feeding it unscaled.
"""

from __future__ import annotations

import re
from typing import Callable

import numpy as np

# Recorded in every model card. Change it whenever a rule below changes:
# a policy trained under one scaling must not be fed another.
OBS_SCALING = "scale-v1"

# (pattern on the stockstats name, rule(values, close) -> scaled)
_RULES: list[tuple[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]] = [
    # moving averages at price level -> relative distance from close.
    # sma30/close - 1 is -0.12..0.145 (p1..p99); x10 -> about +-1.5
    (r"^close_\d+_(sma|ema)$", lambda x, c: (x / c - 1.0) * 10.0),
    # macd is in price units -> relative to close. -0.052..0.043; x20 -> about +-1
    (r"^macd[sh]?$", lambda x, c: x / c * 20.0),
    # bounded 0..100 oscillators, centred. rsi_30 35..73 -> -0.3..0.47
    (r"^rsi_\d+$", lambda x, c: (x - 50.0) / 50.0),
    # cci -240..259 -> about +-1.3
    (r"^cci_\d+$", lambda x, c: x / 200.0),
    # dx 0.25..53 -> 0..1.06
    (r"^dx_\d+$", lambda x, c: x / 50.0),
    # Mahalanobis distance, heavy right tail (median 4.8, max 343) -> log. 0..5.8
    (r"^turbulence$", lambda x, c: np.log1p(np.maximum(x, 0.0))),
]


def rule_for(name: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    for pattern, rule in _RULES:
        if re.match(pattern, name):
            return rule
    raise ValueError(f"no scaling rule for feature '{name}': add one to envs/scaling.py")


def scale_features(features: np.ndarray, prices: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """(T, N, F) raw features + (T, N) closes -> (T, N, F) scaled."""
    if features.shape[2] != len(feature_names):
        raise ValueError(f"{features.shape[2]} feature columns but {len(feature_names)} names: {feature_names}")
    out = np.empty_like(features, dtype=np.float64)
    for f, name in enumerate(feature_names):
        out[:, :, f] = rule_for(name)(features[:, :, f].astype(np.float64), prices.astype(np.float64))
    if not np.isfinite(out).all():
        raise ValueError("scaled features contain NaN/inf (zero or negative close?)")
    return out


def default_feature_names(n_features: int) -> list[str]:
    """INDICATORS, plus 'turbulence' when meta.features.to_array appended it."""
    from minifinrl.core.configs.settings import INDICATORS

    if n_features == len(INDICATORS):
        return list(INDICATORS)
    if n_features == len(INDICATORS) + 1:
        return list(INDICATORS) + ["turbulence"]
    raise ValueError(f"{n_features} feature columns don't match INDICATORS ({len(INDICATORS)}) [+ turbulence]")
