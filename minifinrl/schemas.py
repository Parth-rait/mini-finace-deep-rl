"""
Input/output models for every Engine capability. Nothing but pydantic
here, so the API, the agent's tool schemas and the CLI's flags are all
generated from the same definitions.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

App = Literal["trading", "portfolio"]
Algo = Literal["ppo", "sac", "td3"]


class _In(BaseModel):
    model_config = ConfigDict(extra="forbid")  # unknown fields are a 422, not silently ignored


class Empty(_In):
    pass


# ---- health ----------------------------------------------------------------------


class Health(BaseModel):
    version: str
    profile: str
    mode: str
    last_bar: str | None = Field(description="newest date in the price store, None if empty")
    staleness_bdays: int | None
    manifest_hash: str | None = Field(description="data_hash of the latest provenance manifest")
    models: list[str]
    legacy_models: list[str] = Field(description="models without a metadata card (will be refused)")
    results_rows: int
    bias_classifier: str | None
    capabilities: list[str]


# ---- market data --------------------------------------------------------------------


class SnapshotIn(_In):
    tickers: list[str] | None = Field(default=None, description="default: the configured universe")
    days: int = Field(default=5, ge=1, le=60)


class Bar(BaseModel):
    date: str
    tic: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class Snapshot(BaseModel):
    as_of: str
    bars: list[Bar]


# ---- results (read from the last backtest run) ----------------------------------------


class BacktestQuery(_In):
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


class RankStabilityIn(_In):
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


# ---- models ---------------------------------------------------------------------------


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


# ---- luck test ------------------------------------------------------------------------


class LuckTestIn(_In):
    ticker: str
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="entry day (YYYY-MM-DD); snapped to the next trading day")
    direction: Literal["long", "short"] = "long"
    horizon_days: int = Field(default=20, ge=1, le=60, description="trading days held")
    n_paths: int = Field(default=1000, ge=200, le=5000)
    seed: int = Field(default=0, ge=0)


class LuckTestOut(BaseModel):
    ticker: str
    direction: str
    entry_date: str
    exit_date: str
    horizon_days: int
    realized_return: float
    synthetic_median: float
    band_low: float = Field(description="5th percentile of outcomes the regime model considers plausible")
    band_high: float = Field(description="95th percentile")
    prob_profit: float = Field(description="share of synthetic outcomes above 0; near 0.5 means the model has no view")
    percentile: float = Field(description="where the realized outcome falls among synthetic outcomes, 0-100")
    verdict: Literal["unusually_bad", "within_luck_range", "unusually_good"]
    regime: str = Field(description="most likely market regime on the entry day, from data up to that day only")
    regime_probs: dict[str, float]
    n_paths: int
    seed: int
    fit_start: str
    fit_end: str
    note: str = ("Locates the outcome within the range of luck under a regime model with no predictive "
                 "signal. Not a forecast, not a measure of skill, not investment advice.")


# ---- trade review ------------------------------------------------------------------------


_DATE = r"^\d{4}-\d{2}-\d{2}$"


class ReviewIn(_In):
    """The form fields are the trade; `text` is optional context (your reasoning,
    or the whole trade in words, which is parsed to fill fields left empty)."""

    text: str = Field(default="", max_length=2000, description="your reasoning, or the trade in your own words")
    ticker: str | None = Field(default=None, max_length=12, description="fills or overrides what was read from the text")
    date: str | None = Field(default=None, pattern=_DATE, description="buy (or short) date")
    sell_date: str | None = Field(default=None, pattern=_DATE, description="sell (or cover) date")
    still_holding: bool | None = Field(default=None, description="measure up to the latest close")
    direction: Literal["long", "short"] | None = None
    horizon_days: int | None = Field(default=None, ge=1, le=260, description="trading days held, if no sell date")


class ParseIn(_In):
    text: str = Field(min_length=1, max_length=2000)


class ParseOut(BaseModel):
    ticker: str | None
    name: str | None = Field(description="the listing's name in the symbol directory")
    date: str | None
    sell_date: str | None
    still_holding: bool | None
    direction: str | None
    horizon_days: int | None
    reasoning: str | None
    notes: list[str] = Field(default_factory=list, description="anything that needs checking before the review")
    parser: str


class SymbolIn(_In):
    q: str = Field(min_length=1, max_length=60)
    limit: int = Field(default=8, ge=1, le=25)


class SymbolHit(BaseModel):
    symbol: str
    name: str
    exchange: str
    etf: bool


class SymbolsOut(BaseModel):
    results: list[SymbolHit]
    note: str | None = None


class TradingDaysIn(_In):
    date: str = Field(pattern=_DATE, description="buy date")
    sell_date: str | None = Field(default=None, pattern=_DATE, description="sell date; empty means still holding")


class TradingDaysOut(BaseModel):
    entry_date: str | None = Field(description="first trading day on or after the buy date")
    exit_date: str | None = Field(description="last trading day on or before the sell date, or the latest close")
    days: int | None = Field(description="trading days between them, counted on the exchange calendar")
    notes: list[str] = Field(default_factory=list)


class Understood(BaseModel):
    ticker: str | None
    company: str | None
    date: str | None
    direction: str | None
    horizon_days: int | None
    reasoning: str | None
    sell_date: str | None = None
    still_holding: bool | None = None
    name: str | None = Field(default=None, description="the listing's name in the symbol directory")
    entry_date: str | None = Field(default=None, description="first trading day on or after the date")
    exit_date: str | None = None
    parser: str = Field(description="which parser read the text")


class BiasSpan(BaseModel):
    label: str
    evidence: str
    confidence: float
    start: int | None = Field(description="character offsets of the evidence in the reasoning, for highlighting")
    end: int | None


class Surface(BaseModel):
    days: list[int]
    returns: list[float]
    density: list[list[float]] = Field(description="[day][return bucket] probability; each row sums to 1")
    p05: list[float]
    p50: list[float]
    p95: list[float]
    realized: list[float] = Field(description="your position's directional return after each day held")


class MarketContext(BaseModel):
    spy_return: float | None = Field(description="S&P 500 ETF over the same days")
    universe_return: float | None = Field(description="equal weight of the research stocks over the same days")


class ReviewOut(BaseModel):
    status: Literal["ok", "needs_input", "unsupported", "rejected"]
    messages: list[str] = Field(default_factory=list, description="what could not be done, and why")
    missing: list[str] = Field(default_factory=list, description="fields to fill in before the review can run")
    suggestions: list[str] = Field(default_factory=list)
    understood: Understood
    what_you_did: str | None = None
    outcome: LuckTestOut | None = None
    surface: Surface | None = None
    biases: list[BiasSpan] = Field(default_factory=list)
    bias_source: str | None = None
    market: MarketContext | None = None
    explanation: str | None = None
    explanation_source: Literal["llm", "template"] | None = None
    disclaimer: str = "Paper analysis of a past trade. Not investment advice, not a forecast, not a judgement of the person."


class ResearchSummary(BaseModel):
    experiments: list[dict] = Field(description="id, title and conclusion of every recorded experiment")
    main: dict = Field(description="E05: app -> strategy -> metrics on the 8-stock universe")
    etf: dict | None = Field(default=None, description="E06: the same on the 12-ETF universe")
    walk_forward: dict | None = Field(default=None, description="E07: Sharpe per test year and selection rules")
    classifier: dict | None = Field(default=None, description="E08: rules vs LLM bias classifier")


# ---- text -> bias signals (port-backed) ---------------------------------------------------


class TextIn(_In):
    text: str = Field(min_length=3, max_length=4000)


# ---- forbidden -------------------------------------------------------------------------


class OrderIn(_In):
    ticker: str
    side: Literal["buy", "sell"]
    quantity: float = Field(gt=0)


class OrderOut(BaseModel):
    placed: bool


# ---- batch (CLI only) ----------------------------------------------------------------------


class FetchIn(_In):
    mode: Literal["research", "live"] | None = Field(default=None, description="default: the profile's mode")
    start: str | None = None
    end: str | None = Field(default=None, description="exclusive; default TEST_END (research) or today (live)")
    offline: bool = Field(default=False, description="validate what's on disk, no network")


class FetchOut(BaseModel):
    ok: bool
    data_hash: str
    rows: int
    requested: dict
    providers: dict
    errors: list[str]
    warnings: list[str]


class FitIn(_In):
    n_paths: int | None = Field(default=None, ge=1, le=1000, description="default HMM_N_PATHS")
    length: int | None = Field(default=None, ge=20, le=5000, description="default HMM_PATH_LENGTH")


class FitOut(BaseModel):
    paths: int
    directory: str


class TrainIn(_In):
    app: App
    model: Algo
    seed: int = Field(ge=0)
    timesteps: int | None = Field(default=None, ge=100, description="default TOTAL_TIMESTEPS")


class TrainOut(BaseModel):
    model_path: str
    timesteps: int


class BacktestRunIn(_In):
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


class CrosscheckIn(_In):
    candidate: str = Field(description="price provider to test, e.g. tiingo or alpaca")
    reference: str = "yahoo"
    start: str = Field(default="2023-01-01", pattern=r"^\d{4}-\d{2}-\d{2}$")
    end: str = Field(default="2026-06-30", pattern=r"^\d{4}-\d{2}-\d{2}$", description="exclusive")


class TickerAgreement(BaseModel):
    ticker: str
    ref_days: int
    cand_days: int
    overlap_days: int
    missing_in_cand: int
    extra_in_cand: int
    max_abs_diff: float | None
    p99_abs_diff: float | None = Field(description="99th percentile of |daily return difference|")
    return_corr: float | None
    flagged_days: list[dict] = Field(default_factory=list, description="days where |r_ref - r_cand| > 50 bps, largest first")
    agrees: bool


class CrosscheckOut(BaseModel):
    reference: str
    candidate: str
    agrees: bool
    tickers: list[TickerAgreement]
    rule: str = "p99 |r_ref - r_cand| <= 10 bps on overlapping days, and < 1% of reference days missing"


class WalkForwardIn(_In):
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


class BiasEvalIn(_In):
    labels_path: str = Field(default="corpus/bias_labels.jsonl", description="JSONL of hand-labelled texts (corpus/LABELLING.md)")


class LabelScore(BaseModel):
    label: str
    support: int = Field(description="texts with this label in the reference")
    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float | None
    f1: float | None
    kappa_model: float | None = Field(description="Cohen's kappa, classifier vs reference labels")
    kappa_humans: float | None = Field(default=None, description="Cohen's kappa, labeller A vs labeller B")


class BiasEvalOut(BaseModel):
    classifier: str
    texts: int
    failures: list[str] = Field(default_factory=list, description="ids where the classifier gave no valid answer (scored as no prediction)")
    double_labelled: int
    exact_match: float
    macro_f1: float | None
    macro_kappa_model: float | None
    macro_kappa_humans: float | None
    per_label: list[LabelScore]


class ExperimentIn(_In):
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
