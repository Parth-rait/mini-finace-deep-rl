"""
Research capabilities: reading recorded results (read effect) and running
the deep-RL study (batch effect, CLI only).

The training stack (torch, stable-baselines3) is imported inside the batch
methods only, so serving results and the website never loads it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from minifinrl.market.dataset import latest_manifest
from minifinrl.platform.capabilities import CapabilityUnavailable, capability
from minifinrl.platform.log import get_logger
from minifinrl.platform.schemas import Empty
from minifinrl.regime.settings import HMM_N_PATHS, HMM_PATH_LENGTH
from minifinrl.research import results
from minifinrl.research.agents.registry import ModelRegistry
from minifinrl.research.evaluation.report import rank_stability as rank_table
from minifinrl.research.schemas import (
    AppReport,
    BacktestQuery,
    BacktestResults,
    BacktestRow,
    BacktestRunIn,
    BacktestRunOut,
    CollapsedRow,
    ExperimentIn,
    ExperimentOut,
    FitIn,
    FitOut,
    ModelInfo,
    ModelsOut,
    ModelSummary,
    ModelWinRate,
    RankStability,
    RankStabilityIn,
    ReportOut,
    ResearchSummary,
    SummaryRow,
    TrainIn,
    TrainOut,
    WalkForwardIn,
    WalkForwardOut,
    WinRateRow,
)
from minifinrl.research.settings import TOTAL_TIMESTEPS

log = get_logger(__name__)


def _none_if_nan(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


class ResearchService:
    def __init__(self, cfg):
        self.cfg = cfg

    def _results(self) -> pd.DataFrame:
        try:
            return results.load_results(self.cfg.paths)
        except FileNotFoundError as exc:
            raise CapabilityUnavailable(f"{exc}") from exc

    # ---- read ---------------------------------------------------------------------------

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
        collapsed = results.collapsed_policies(df)
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
        ref = results.REFERENCE_BASELINE[req.app]
        table = rank_table(synth.to_dict("records"), path_col="path", model_col="model", metric=req.metric,
                           baseline=ref if (synth["model"] == ref).any() else "baseline")
        models = [
            ModelWinRate(model=r["model"], wins=r["wins"], win_rate=r["win_rate"], ci_low=r["ci_low"],
                         ci_high=r["ci_high"], beats_baseline=_none_if_nan(r["beats_baseline"]))
            for r in table.astype(object).where(table.notna(), None).to_dict("records")
        ]
        return RankStability(app=req.app, metric=req.metric, n_paths=int(table["n_paths"].iloc[0]), models=models)

    @capability("research_summary", Empty, ResearchSummary, effect="read", budget_ms=1000)
    def research_summary(self, req: Empty) -> ResearchSummary:
        """The recorded research results: strategy comparison, walk-forward study and bias classifier evaluation."""
        log_ = {r["id"]: r for r in results.load_log(self.cfg.experiment_dir)}
        if "E05" not in log_:
            raise CapabilityUnavailable("no recorded experiments yet: see EXPERIMENTS.md")
        base = Path(self.cfg.experiment_dir)

        def read(path: Path):
            return json.loads(path.read_text()) if path.exists() else None

        clf = None
        if (base / "E08").exists():
            clf = {k: read(base / "E08" / f"{k}_eval.json") for k in ("rules", "aip")}
        mood = None
        if (base / "E09").exists():
            mood = {k: read(base / "E09" / f"{k}_eval.json") for k in ("rules", "aip")}
        return ResearchSummary(
            experiments=[{"id": r["id"], "title": r["title"], "conclusion": r["conclusion"]} for r in log_.values()],
            main=log_["E05"]["summary"], etf=log_.get("E06", {}).get("summary"),
            walk_forward=read(base / "E07" / "walkforward_study.json"), classifier=clf, mood=mood,
        )

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

    # ---- batch (CLI only) -------------------------------------------------------------

    @capability("fit_hmm", FitIn, FitOut, effect="batch")
    def fit_hmm(self, req: FitIn) -> FitOut:
        """Fit the regime model on the full history and sample synthetic market paths."""
        from minifinrl.research import pipeline

        written = pipeline.fit_and_sample(
            self.cfg.spec, self.cfg.paths, n_paths=req.n_paths or HMM_N_PATHS, length=req.length or HMM_PATH_LENGTH
        )
        return FitOut(paths=len(written), directory=str(self.cfg.paths.synthetic_dir))

    @capability("train", TrainIn, TrainOut, effect="batch")
    def train(self, req: TrainIn) -> TrainOut:
        """Train one (app, model, seed) on the training window and save the policy."""
        from minifinrl.research import pipeline

        steps = req.timesteps or TOTAL_TIMESTEPS
        out = pipeline.train(req.app, req.model, req.seed, steps, self.cfg.spec, self.cfg.paths)
        return TrainOut(model_path=str(out), timesteps=steps)

    @capability("run_backtests", BacktestRunIn, BacktestRunOut, effect="batch")
    def run_backtests(self, req: BacktestRunIn) -> BacktestRunOut:
        """Backtest every saved model and the baselines on the test window and/or synthetic paths."""
        from minifinrl.research import pipeline

        df = pipeline.run_backtests(req.paths, self.cfg.spec, self.cfg.paths, risk_free=req.risk_free, baselines=req.baselines)
        return BacktestRunOut(rows=len(df), results_path=str(self.cfg.paths.results_path),
                              skipped=df.attrs.get("skipped", []))

    @capability("walk_forward", WalkForwardIn, WalkForwardOut, effect="batch")
    def walk_forward(self, req: WalkForwardIn) -> WalkForwardOut:
        """Retrain per fold on the preceding years, test each following year, and compare strategy-selection rules out of sample."""
        from minifinrl.research import walkforward

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
        from minifinrl.research import experiments

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
            for r in results.collapsed_policies(df).itertuples()
        ]
        apps = []
        for app, t in results.report_tables(df).items():
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
