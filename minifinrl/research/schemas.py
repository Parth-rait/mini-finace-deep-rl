"""Inputs and outputs of the research capabilities."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from minifinrl.platform.schemas import Input

App = Literal["trading", "portfolio"]

Algo = Literal["ppo", "sac", "td3"]


class BacktestQuery(Input):
    app: App
    model: str | None = Field(default=None, description="ppo | sac | baseline; default all")
    path_kind: Literal["historical", "synthetic", "all"] = "all"


class BacktestRow(BaseModel):
    app: str
    model: str
    seed: int | None
    path: str
    final_value: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    distinct_actions: int | None = None
    clip_share: float | None = None
    turnover: float | None = None
    cost_frac: float | None = None


class ModelSummary(BaseModel):
    model: str
    path_kind: Literal["historical", "synthetic"]
    n: int
    sharpe_median: float
    sharpe_q25: float
    sharpe_q75: float


class BacktestResults(BaseModel):
    rows: list[BacktestRow]
    summary: list[ModelSummary]
    collapsed: list[str] = Field(description="'model/seed' pairs whose policy made one distinct action every step")


class RankStabilityIn(Input):
    app: App
    metric: Literal["sharpe", "sortino", "cagr", "final_value"] = "sharpe"


class ModelWinRate(BaseModel):
    model: str
    wins: int
    win_rate: float
    ci_low: float = Field(description="95% Wilson interval")
    ci_high: float
    beats_baseline: int | None = Field(description="paths where this model's seed-median beat the baseline")


class RankStability(BaseModel):
    app: str
    metric: str
    n_paths: int
    models: list[ModelWinRate]
    method: str = "per path, each model's median over seeds; one winner per path"


class ModelInfo(BaseModel):
    model_id: str
    legacy: bool = Field(description="no metadata card: trained before versioned models, refused by backtests")
    app: str | None = None
    algo: str | None = None
    seed: int | None = None
    timesteps: int | None = None
    obs_scaling: str | None = None
    data_hash: str | None = None
    created_at: str | None = None
    train_seconds: float | None = None


class ModelsOut(BaseModel):
    models: list[ModelInfo]


class ResearchSummary(BaseModel):
    experiments: list[dict] = Field(description="id, title and conclusion of every recorded experiment")
    main: dict = Field(description="E05: app -> strategy -> metrics on the 8-stock universe")
    etf: dict | None = Field(default=None, description="E06: the same on the 12-ETF universe")
    walk_forward: dict | None = Field(default=None, description="E07: Sharpe per test year and selection rules")
    classifier: dict | None = Field(default=None, description="E08: rules vs LLM bias classifier")


class FitIn(Input):
    n_paths: int | None = Field(default=None, ge=1, le=1000, description="default HMM_N_PATHS")
    length: int | None = Field(default=None, ge=20, le=5000, description="default HMM_PATH_LENGTH")


class FitOut(BaseModel):
    paths: int
    directory: str


class TrainIn(Input):
    app: App
    model: Algo
    seed: int = Field(ge=0)
    timesteps: int | None = Field(default=None, ge=100, description="default TOTAL_TIMESTEPS")


class TrainOut(BaseModel):
    model_path: str
    timesteps: int


class BacktestRunIn(Input):
    paths: Literal["historical", "synthetic", "both"] = "both"
    risk_free: Literal["dtb3", "zero"] = Field(default="dtb3", description="Sharpe excess over 3m T-bill (E03) or over 0")
    baselines: list[str] | None = Field(default=None, description="baseline names to run; default all for each app")


class BacktestRunOut(BaseModel):
    rows: int
    results_path: str
    skipped: list[str] = Field(default_factory=list, description="models refused by the metadata check, with the reason")


class SummaryRow(BaseModel):
    model: str
    sharpe_median: float
    sharpe_q25: float
    sharpe_q75: float


class WinRateRow(BaseModel):
    model: str
    wins: int
    n_paths: int
    win_rate: float
    ci_low: float
    ci_high: float
    beats_baseline: int | None


class AppReport(BaseModel):
    app: str
    historical: list[SummaryRow] | None
    synthetic: list[SummaryRow] | None
    win_rate: list[WinRateRow] | None


class CollapsedRow(BaseModel):
    app: str
    model: str
    seed: int | None
    path: str


class WalkForwardIn(Input):
    first_test_year: int = Field(default=2019, ge=2015, le=2026)
    last_test_year: int = Field(default=2025, ge=2015, le=2026)
    train_years: int = Field(default=5, ge=1, le=10)
    apps: list[App] = Field(default_factory=lambda: ["trading", "portfolio"])
    models: list[Algo] = Field(default_factory=lambda: ["ppo"])
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    timesteps: int = Field(default=50_000, ge=100)


class WalkForwardOut(BaseModel):
    folds: list[str]
    rows: int
    results_path: str
    study: dict = Field(description="per app: selection rules (chained out-of-sample), rank persistence, Sharpe per fold")


class ExperimentIn(Input):
    id: str = Field(pattern=r"^E\d{2,3}$", description="permanent id, e.g. E03")
    title: str = Field(min_length=3, max_length=120)
    change: str = Field(description="what changed, in one or two sentences")
    math: str = Field(description="the exact definition (formulas) of the change")
    hypothesis: str
    conclusion: str = Field(description="what the numbers say, written after looking at them")
    files: list[str] = Field(default_factory=list)
    compare_to: str | None = Field(default=None, description="earlier experiment id to show deltas against")


class ExperimentOut(BaseModel):
    id: str
    results_sha256: str | None
    models: int
    log: str


class ReportOut(BaseModel):
    collapsed: list[CollapsedRow]
    apps: list[AppReport]
