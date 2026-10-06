"""
The Engine: the one facade over the finance core.

Everything the system can do is one typed method here, declared with
@capability. The CLI, the API, the agent and the gate all reach the core
through these methods (via the registry in capabilities.py) and through
nothing else, so a feature added here shows up everywhere at once, and
the finance logic has exactly one front door.

The Engine receives its collaborators (config, and port implementations
like the bias classifier); it never constructs adapters or reads the
environment. That is system.py's job.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import minifinrl
from minifinrl.capabilities import CapabilityForbidden, CapabilityRegistry, CapabilityUnavailable, capability
from minifinrl.core import pipeline
from minifinrl.core.configs.logging_config import get_logger
from minifinrl.core.agents.registry import ModelRegistry
from minifinrl.core.configs.settings import HMM_N_PATHS, HMM_PATH_LENGTH, TOTAL_TIMESTEPS, TRAIN_START
from minifinrl.core import experiments, walkforward
from minifinrl.core.meta.crosscheck import crosscheck
from minifinrl.core.eval.report import rank_stability as rank_table
from minifinrl.core.luck.luck_test import LuckTestError, luck_test
from minifinrl.core.meta.panel import build_panel
from minifinrl.core.meta.synthetic import RegimeSyntheticGenerator
from minifinrl.core.eval.report import wilson  # noqa: F401  (re-exported for callers)
from minifinrl.core.meta.market_data import MANIFEST_DIR, latest_manifest, load_market_data
from minifinrl.core.meta.panel import DataSpec
from minifinrl.core.meta.providers import get_price_provider
from minifinrl.core.meta.store import PriceStore, market_today
from minifinrl.ports import BiasClassification, BiasClassifier
from minifinrl.schemas import (
    AppReport,
    BacktestQuery,
    BacktestResults,
    BacktestRow,
    BacktestRunIn,
    BacktestRunOut,
    Bar,
    CollapsedRow,
    Empty,
    FetchIn,
    FetchOut,
    FitIn,
    FitOut,
    CrosscheckIn,
    CrosscheckOut,
    ExperimentIn,
    ExperimentOut,
    WalkForwardIn,
    WalkForwardOut,
    Health,
    LuckTestIn,
    LuckTestOut,
    ModelInfo,
    ModelsOut,
    ModelSummary,
    ModelWinRate,
    OrderIn,
    OrderOut,
    RankStability,
    RankStabilityIn,
    ReportOut,
    Snapshot,
    SnapshotIn,
    TickerAgreement,
    SummaryRow,
    TextIn,
    TrainIn,
    TrainOut,
    WinRateRow,
)

log = get_logger(__name__)


@dataclass(frozen=True)
class EngineConfig:
    profile: str = "research"
    mode: str = "research"
    spec: DataSpec = field(default_factory=DataSpec)
    paths: pipeline.Paths = field(default_factory=pipeline.Paths)
    manifest_dir: Path = MANIFEST_DIR
    experiment_dir: Path = experiments.EXPERIMENT_DIR
    experiments_md: Path = experiments.EXPERIMENTS_MD
    today: str | None = None  # pinned in tests/ci; None = real market date


def _none_if_nan(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


class Engine:
    LUCK_CACHE_SIZE = 32  # fitted regime models, one per entry day (~1 s to fit each)

    def __init__(self, cfg: EngineConfig, *, bias: BiasClassifier | None = None):
        self.cfg = cfg
        self.bias = bias
        self._luck_fits: OrderedDict[str, RegimeSyntheticGenerator] = OrderedDict()
        self._luck_lock = threading.Lock()

    # ---- helpers (not capabilities) ------------------------------------------------

    def _today(self) -> str:
        return self.cfg.today or market_today()

    def _store(self) -> PriceStore:
        return PriceStore(get_price_provider(self.cfg.spec.price_provider), self.cfg.spec.store_root, today=self.cfg.today)

    def _results(self) -> pd.DataFrame:
        try:
            return pipeline.load_results(self.cfg.paths)
        except FileNotFoundError as exc:
            raise CapabilityUnavailable(f"{exc}") from exc

    # ---- read ---------------------------------------------------------------------------

    @capability("health", Empty, Health, effect="read", budget_ms=500)
    def health(self, req: Empty) -> Health:
        """What is loaded and how fresh it is: data, models, results, capabilities."""
        store = self._store()
        last = [b for t in self.cfg.spec.tickers if (b := store.last_bar(t))]
        last_bar = min(last) if len(last) == len(self.cfg.spec.tickers) else None  # the stalest ticker
        stale = int(np.busday_count(np.datetime64(last_bar, "D"), np.datetime64(self._today(), "D"))) if last_bar else None
        manifest = latest_manifest(self.cfg.manifest_dir)
        entries = ModelRegistry(self.cfg.paths.model_dir).list()
        results = Path(self.cfg.paths.results_path)
        return Health(
            version=minifinrl.__version__,
            profile=self.cfg.profile,
            mode=self.cfg.mode,
            last_bar=last_bar,
            staleness_bdays=stale,
            manifest_hash=manifest["data_hash"] if manifest else None,
            models=[e.model_id for e in entries],
            legacy_models=[e.model_id for e in entries if e.legacy],
            results_rows=int(sum(1 for _ in results.open()) - 1) if results.exists() else 0,
            bias_classifier=self.bias.name if self.bias else None,
            capabilities=CapabilityRegistry.from_engine(self).names(),
        )

    @capability("market_snapshot", SnapshotIn, Snapshot, effect="read", budget_ms=1000)
    def market_snapshot(self, req: SnapshotIn) -> Snapshot:
        """The last N daily bars per ticker from the local store. Never calls the provider."""
        tickers = req.tickers or list(self.cfg.spec.tickers)
        unknown = sorted(set(tickers) - set(self.cfg.spec.tickers))
        if unknown:
            raise CapabilityUnavailable(f"tickers {unknown} are not in the configured universe")
        panel = self._store().get(tickers, "1900-01-01", "9999-12-31", refresh=False)
        if panel.empty:
            raise CapabilityUnavailable("no market data on disk: run fetch_data first")
        keep = sorted(panel["date"].unique())[-req.days:]
        panel = panel[panel["date"].isin(keep)]
        return Snapshot(as_of=keep[-1], bars=[Bar(**r) for r in panel.to_dict("records")])

    @capability("backtest_results", BacktestQuery, BacktestResults, effect="read", budget_ms=1000)
    def backtest_results(self, req: BacktestQuery) -> BacktestResults:
        """Stored metrics from the last backtest run, with median/IQR Sharpe per model. Never trains."""
        df = self._results()
        df = df[df["app"] == req.app]
        if req.model:
            df = df[df["model"] == req.model]
        kinds = {"historical": df["path"] == "historical", "synthetic": df["path"] != "historical"}
        if req.path_kind != "all":
            df = df[kinds[req.path_kind]]
        if df.empty:
            raise CapabilityUnavailable(f"no backtest rows for {req.model_dump(exclude_none=True)}")

        summary = []
        for kind in ("historical", "synthetic"):
            sub = df[df["path"] == "historical"] if kind == "historical" else df[df["path"] != "historical"]
            for model, g in sub.groupby("model"):
                s = g["sharpe"]
                summary.append(ModelSummary(
                    model=model, path_kind=kind, n=len(g), sharpe_median=float(s.median()),
                    sharpe_q25=float(s.quantile(0.25)), sharpe_q75=float(s.quantile(0.75)),
                ))
        fields = BacktestRow.model_fields
        rows = [
            BacktestRow(**{k: _none_if_nan(v) for k, v in r.items() if k in fields})
            for r in df.astype(object).to_dict("records")
        ]
        collapsed = pipeline.collapsed_policies(df)
        return BacktestResults(
            rows=rows, summary=summary,
            collapsed=[f"{r.model}/{int(r.seed) if not pd.isna(r.seed) else '-'}" for r in collapsed.itertuples()],
        )

    @capability("rank_stability", RankStabilityIn, RankStability, effect="read", budget_ms=1000)
    def rank_stability(self, req: RankStabilityIn) -> RankStability:
        """Share of synthetic market paths each model wins, with a 95% interval. The research question, as data."""
        df = self._results()
        synth = df[(df["app"] == req.app) & (df["path"] != "historical")]
        if synth.empty:
            raise CapabilityUnavailable(f"no synthetic-path results for app '{req.app}'")
        ref = pipeline.REFERENCE_BASELINE[req.app]
        table = rank_table(synth.to_dict("records"), path_col="path", model_col="model", metric=req.metric,
                           baseline=ref if (synth["model"] == ref).any() else "baseline")
        models = [
            ModelWinRate(model=r["model"], wins=r["wins"], win_rate=r["win_rate"], ci_low=r["ci_low"],
                         ci_high=r["ci_high"], beats_baseline=_none_if_nan(r["beats_baseline"]))
            for r in table.astype(object).where(table.notna(), None).to_dict("records")
        ]
        return RankStability(app=req.app, metric=req.metric, n_paths=int(table["n_paths"].iloc[0]), models=models)

    @capability("list_models", Empty, ModelsOut, effect="read", budget_ms=500)
    def list_models(self, req: Empty) -> ModelsOut:
        """Saved policies with their metadata cards: what each was trained on, for how long, on which data."""
        out = []
        for e in ModelRegistry(self.cfg.paths.model_dir).list():
            c = e.card
            out.append(ModelInfo(model_id=e.model_id, legacy=e.legacy) if c is None else ModelInfo(
                model_id=e.model_id, legacy=False, app=c.spec.app, algo=c.algo, seed=c.seed,
                timesteps=c.timesteps, obs_scaling=c.spec.obs_scaling, data_hash=c.data_hash,
                created_at=c.created_at, train_seconds=c.train_seconds,
            ))
        return ModelsOut(models=out)

    # ---- compute --------------------------------------------------------------------------

    def _fit_for_entry(self, prices: pd.DataFrame, vix: pd.Series) -> RegimeSyntheticGenerator:
        key = str(prices.index[-1])  # the entry day: the fit only ever sees data up to it
        with self._luck_lock:
            if key in self._luck_fits:
                self._luck_fits.move_to_end(key)
                return self._luck_fits[key]
        gen = RegimeSyntheticGenerator().fit(prices, vix)
        with self._luck_lock:
            self._luck_fits[key] = gen
            while len(self._luck_fits) > self.LUCK_CACHE_SIZE:
                self._luck_fits.popitem(last=False)
        return gen

    @capability("luck_test", LuckTestIn, LuckTestOut, effect="compute", budget_ms=3000)
    def luck_test(self, req: LuckTestIn) -> LuckTestOut:
        """Was one trade's outcome inside the normal range of luck, given the market regime on its entry day?"""
        fp = build_panel(self.cfg.spec)
        close = fp.panel.pivot(index="date", columns="tic", values="close")
        try:
            r = luck_test(close, fp.vix, req.ticker, req.date, req.direction, req.horizon_days,
                          n_paths=req.n_paths, seed=req.seed, fit=self._fit_for_entry)
        except LuckTestError as exc:
            raise CapabilityUnavailable(str(exc)) from exc
        return LuckTestOut(**vars(r))

    # ---- llm (through a port) -------------------------------------------------------------

    @capability("classify_biases", TextIn, BiasClassification, effect="llm", budget_ms=8000)
    def classify_biases(self, req: TextIn) -> BiasClassification:
        """Behavioural-bias signals in a piece of trading text, with the quoted evidence. Describes the text, not the person."""
        if self.bias is None:
            raise CapabilityUnavailable(f"no bias classifier configured in profile '{self.cfg.profile}'")
        return self.bias.classify(req.text)

    # ---- forbidden ----------------------------------------------------------------------------

    @capability("place_order", OrderIn, OrderOut, effect="forbidden")
    def place_order(self, req: OrderIn) -> OrderOut:
        """Send an order to a broker. Declared so it can be refused explicitly: this system gives paper analysis only."""
        raise CapabilityForbidden("place_order is never executed")  # the registry refuses before reaching here

    # ---- batch (CLI only) -------------------------------------------------------------------

    @capability("fetch_data", FetchIn, FetchOut, effect="batch")
    def fetch_data(self, req: FetchIn) -> FetchOut:
        """Fetch through the incremental store, validate, and write a provenance manifest."""
        spec = self.cfg.spec
        md = load_market_data(
            list(spec.tickers), req.start or TRAIN_START, req.end, mode=req.mode or self.cfg.mode,
            price_provider=spec.price_provider, vix_provider=spec.vix_provider,
            refresh=spec.refresh and not req.offline, strict=False, store_root=spec.store_root,
            today=self.cfg.today, manifest_dir=self.cfg.manifest_dir,
        )
        m = md.manifest
        return FetchOut(
            ok=md.report.ok, data_hash=m["data_hash"], rows=m["panel"]["rows"], requested=m["requested"],
            providers=m["providers"], errors=md.report.errors, warnings=md.report.warnings,
        )

    @capability("fit_hmm", FitIn, FitOut, effect="batch")
    def fit_hmm(self, req: FitIn) -> FitOut:
        """Fit the regime model on the full history and sample synthetic market paths."""
        written = pipeline.fit_and_sample(
            self.cfg.spec, self.cfg.paths, n_paths=req.n_paths or HMM_N_PATHS, length=req.length or HMM_PATH_LENGTH
        )
        return FitOut(paths=len(written), directory=str(self.cfg.paths.synthetic_dir))

    @capability("train", TrainIn, TrainOut, effect="batch")
    def train(self, req: TrainIn) -> TrainOut:
        """Train one (app, model, seed) on the training window and save the policy."""
        steps = req.timesteps or TOTAL_TIMESTEPS
        out = pipeline.train(req.app, req.model, req.seed, steps, self.cfg.spec, self.cfg.paths)
        return TrainOut(model_path=str(out), timesteps=steps)

    @capability("run_backtests", BacktestRunIn, BacktestRunOut, effect="batch")
    def run_backtests(self, req: BacktestRunIn) -> BacktestRunOut:
        """Backtest every saved model and the baselines on the test window and/or synthetic paths."""
        df = pipeline.run_backtests(req.paths, self.cfg.spec, self.cfg.paths, risk_free=req.risk_free, baselines=req.baselines)
        return BacktestRunOut(rows=len(df), results_path=str(self.cfg.paths.results_path),
                              skipped=df.attrs.get("skipped", []))

    @capability("crosscheck_prices", CrosscheckIn, CrosscheckOut, effect="batch")
    def crosscheck_prices(self, req: CrosscheckIn) -> CrosscheckOut:
        """Do two price providers agree on daily returns over the same history? Run before trusting a new source."""
        spec = self.cfg.spec
        panels = []
        for name in (req.reference, req.candidate):
            store = PriceStore(get_price_provider(name), spec.store_root, today=self.cfg.today)
            panels.append(store.get(list(spec.tickers), req.start, req.end, refresh=spec.refresh))
        rows = crosscheck(panels[0], panels[1], list(spec.tickers))
        return CrosscheckOut(reference=req.reference, candidate=req.candidate,
                             agrees=all(r["agrees"] for r in rows), tickers=[TickerAgreement(**r) for r in rows])

    @capability("walk_forward", WalkForwardIn, WalkForwardOut, effect="batch")
    def walk_forward(self, req: WalkForwardIn) -> WalkForwardOut:
        """Retrain per fold on the preceding years, test each following year, and compare strategy-selection rules out of sample."""
        import json

        folds = walkforward.make_folds(req.first_test_year, req.last_test_year, train_years=req.train_years,
                                       final_end=self.cfg.spec.end)
        _metrics, returns = walkforward.run_walk_forward(
            self.cfg.spec, self.cfg.paths, folds, apps=req.apps, models=req.models, seeds=req.seeds,
            timesteps=req.timesteps, log=log,
        )
        names = [f.name for f in folds]
        result = walkforward.study(returns, req.apps, names)
        Path(self.cfg.paths.results_path).with_name("walkforward_study.json").write_text(json.dumps(result, indent=2, default=float))
        return WalkForwardOut(folds=names, rows=len(_metrics), results_path=str(self.cfg.paths.results_path), study=result)

    @capability("record_experiment", ExperimentIn, ExperimentOut, effect="batch")
    def record_experiment(self, req: ExperimentIn) -> ExperimentOut:
        """Log a result-affecting change: its definition, hypothesis, and a frozen copy of the current results."""
        manifest = latest_manifest(self.cfg.manifest_dir)
        rec = experiments.record(
            **req.model_dump(), results_path=Path(self.cfg.paths.results_path), model_dir=Path(self.cfg.paths.model_dir),
            data_hash=manifest["data_hash"] if manifest else None,
            directory=self.cfg.experiment_dir, markdown=self.cfg.experiments_md,
        )
        return ExperimentOut(id=rec.id, results_sha256=rec.results_sha256, models=len(rec.models), log=str(self.cfg.experiments_md))

    @capability("report", Empty, ReportOut, effect="batch")
    def report(self, req: Empty) -> ReportOut:
        """Median/IQR Sharpe per model and synthetic-path win rates, per app."""
        df = self._results()
        collapsed = [
            CollapsedRow(app=r.app, model=r.model, seed=None if pd.isna(r.seed) else int(r.seed), path=r.path)
            for r in pipeline.collapsed_policies(df).itertuples()
        ]
        apps = []
        for app, t in pipeline.report_tables(df).items():
            def rows(key, cls):
                if key not in t:
                    return None
                frame = t[key].astype(object).where(t[key].notna(), None)
                return [cls(**r) for r in frame.to_dict("records")]

            apps.append(AppReport(
                app=app, historical=rows("historical", SummaryRow), synthetic=rows("synthetic", SummaryRow),
                win_rate=rows("win_rate", WinRateRow),
            ))
        return ReportOut(collapsed=collapsed, apps=apps)

