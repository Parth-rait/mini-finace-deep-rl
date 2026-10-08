"""The Engine through its registry, end to end on random-walk data:
fetch -> fit -> train -> backtest -> report, then every read capability."""

from __future__ import annotations

import pytest

from minifinrl.platform.capabilities import CapabilityForbidden, CapabilityRegistry, CapabilityUnavailable
from minifinrl.research.evaluation.report import wilson
from minifinrl.engine import Engine, EngineConfig


def test_wilson_interval():
    lo, hi = wilson(10, 20)
    assert lo == pytest.approx(0.2993, abs=1e-4) and hi == pytest.approx(0.7007, abs=1e-4)
    assert wilson(0, 20)[0] == 0.0 and wilson(20, 20)[1] == 1.0


def test_reads_before_any_results_are_unavailable(rw_engine):
    reg = CapabilityRegistry.from_engine(rw_engine)
    with pytest.raises(CapabilityUnavailable, match="no backtest results"):
        reg.invoke("rank_stability", {"app": "trading"})
    with pytest.raises(CapabilityUnavailable, match="no market data"):
        reg.invoke("market_snapshot", {})


def test_no_bias_classifier_is_unavailable():
    reg = CapabilityRegistry.from_engine(Engine(EngineConfig(), bias=None))
    with pytest.raises(CapabilityUnavailable, match="no bias classifier"):
        reg.invoke("classify_biases", {"text": "all in, can't lose"})


def test_place_order_is_refused(rw_engine):
    with pytest.raises(CapabilityForbidden):
        CapabilityRegistry.from_engine(rw_engine).invoke("place_order", {"ticker": "AAA", "side": "buy", "quantity": 1})


def test_full_pipeline_through_capabilities(rw_engine, tmp_path):
    reg = CapabilityRegistry.from_engine(rw_engine)

    fetched = reg.invoke("fetch_data", {}, interface="cli")
    assert fetched.ok, fetched.errors
    assert reg.invoke("fit_hmm", {"n_paths": 2, "length": 40}, interface="cli").paths == 2
    trained = reg.invoke("train", {"app": "trading", "model": "ppo", "seed": 0, "timesteps": 256}, interface="cli")
    assert trained.model_path.endswith("trading_ppo_seed0.zip")
    run = reg.invoke("run_backtests", {"paths": "both"}, interface="cli")
    # baselines: trading buy_hold, portfolio 1/N + min-var + risk parity = 4, x (1 historical + 2 synthetic),
    # + trading ppo x 3
    assert run.rows == 4 * 3 + 3 and not run.skipped

    report = reg.invoke("report", {}, interface="cli")
    trading = next(a for a in report.apps if a.app == "trading")
    assert {r.model for r in trading.historical} == {"buy_hold", "ppo"}
    portfolio = next(a for a in report.apps if a.app == "portfolio")
    assert {r.model for r in portfolio.historical} == {"equal_weight", "min_variance", "risk_parity"}

    res = reg.invoke("backtest_results", {"app": "trading", "path_kind": "synthetic"})
    assert len(res.rows) == 4 and all(r.path.startswith("path_") for r in res.rows)
    assert all(r.distinct_actions is not None for r in res.rows)

    rs = reg.invoke("rank_stability", {"app": "trading"})
    assert rs.n_paths == 2 and sum(m.wins for m in rs.models) == 2
    assert all(m.ci_low <= m.win_rate <= m.ci_high for m in rs.models)

    snap = reg.invoke("market_snapshot", {"days": 3})
    assert len(snap.bars) == 9 and snap.as_of < "2026-06-30"
    with pytest.raises(CapabilityUnavailable, match="not in the configured universe"):
        reg.invoke("market_snapshot", {"tickers": ["ZZZ"]})

    h = reg.invoke("health", {})
    assert h.models == ["trading_ppo_seed0"] and h.results_rows == run.rows
    assert h.manifest_hash == fetched.data_hash and h.last_bar == snap.as_of
    assert "place_order" in h.capabilities and h.bias_classifier == "rules-v1"


def test_classify_biases_rules(rw_engine):
    out = CapabilityRegistry.from_engine(rw_engine).invoke(
        "classify_biases", {"text": "Lost big yesterday, doubling down to win it back. Can't lose."}
    )
    labels = {s.label.value for s in out.signals}
    assert labels == {"revenge_trading", "overconfidence"}
    assert all(s.evidence in "Lost big yesterday, doubling down to win it back. Can't lose." for s in out.signals)
