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
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import minifinrl
from minifinrl import bias_eval, review
from minifinrl.capabilities import CapabilityForbidden, CapabilityRegistry, CapabilityUnavailable, capability
from minifinrl.core import pipeline
from minifinrl.core.configs.logging_config import get_logger
from minifinrl.core.agents.registry import ModelRegistry
from minifinrl.core.configs.settings import HMM_N_PATHS, HMM_PATH_LENGTH, ROOT, TOTAL_TIMESTEPS, TRAIN_START
from minifinrl.core import experiments, walkforward
from minifinrl.core.meta.crosscheck import crosscheck
from minifinrl.core.eval.report import rank_stability as rank_table
from minifinrl.core.luck.luck_test import LuckTestError, luck_test, outcome_surface
from minifinrl.core.meta import symbols
from minifinrl.core.meta import validate as checks
from minifinrl.core.meta.providers import ProviderError, get_vix_provider
from minifinrl.core.meta.store import SeriesStore
from minifinrl.core.meta.panel import build_panel
from minifinrl.core.meta.synthetic import MIN_TICKER_RETURNS, RegimeSyntheticGenerator
from minifinrl.core.eval.report import wilson  # noqa: F401  (re-exported for callers)
from minifinrl.core.meta.market_data import MANIFEST_DIR, latest_manifest, load_market_data
from minifinrl.core.meta.panel import DataSpec
from minifinrl.core.meta.providers import get_price_provider
from minifinrl.core.meta.store import PriceStore, market_today
from minifinrl.ports import (
    BiasClassification,
    BiasClassifier,
    BudgetExhausted,
    ClassificationFailed,
    ParsedTrade,
    TradeExplainer,
    TradeParser,
)
from minifinrl.schemas import (
    AppReport,
    BacktestQuery,
    BiasEvalIn,
    BiasEvalOut,
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
    BiasSpan,
    Health,
    LuckTestIn,
    MarketContext,
    ResearchSummary,
    ReviewIn,
    ReviewOut,
    Surface,
    Understood,
    LuckTestOut,
    ModelInfo,
    ModelsOut,
    ModelSummary,
    ModelWinRate,
    OrderIn,
    OrderOut,
    ParseIn,
    ParseOut,
    SymbolHit,
    SymbolIn,
    SymbolsOut,
    TradingDaysIn,
    TradingDaysOut,
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
    symbols_dir: Path = symbols.CACHE


def _none_if_nan(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


class Engine:
    LUCK_CACHE_SIZE = 32  # fitted regime models, one per entry day (~1 s to fit each)

    def __init__(self, cfg: EngineConfig, *, bias: BiasClassifier | None = None,
                 bias_fallback: BiasClassifier | None = None, parser: TradeParser | None = None,
                 parser_fallback: TradeParser | None = None, explainer: TradeExplainer | None = None):
        self.cfg = cfg
        self.bias = bias
        self.bias_fallback = bias_fallback
        self.parser = parser
        self.parser_fallback = parser_fallback
        self.explainer = explainer
        self._luck_fits: OrderedDict[str, RegimeSyntheticGenerator] = OrderedDict()
        self._luck_lock = threading.Lock()
        self._symbols: tuple[str, list[symbols.Listing]] | None = None

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

    @capability("research_summary", Empty, ResearchSummary, effect="read", budget_ms=1000)
    def research_summary(self, req: Empty) -> ResearchSummary:
        """The recorded research results: strategy comparison, walk-forward study and bias classifier evaluation."""
        import json

        log_ = {r["id"]: r for r in experiments.load_log(self.cfg.experiment_dir)}
        if "E05" not in log_:
            raise CapabilityUnavailable("no recorded experiments yet: see EXPERIMENTS.md")
        base = Path(self.cfg.experiment_dir)

        def read(path: Path):
            return json.loads(path.read_text()) if path.exists() else None

        clf = None
        if (base / "E08").exists():
            clf = {k: read(base / "E08" / f"{k}_eval.json") for k in ("rules", "aip")}
        return ResearchSummary(
            experiments=[{"id": r["id"], "title": r["title"], "conclusion": r["conclusion"]} for r in log_.values()],
            main=log_["E05"]["summary"], etf=log_.get("E06", {}).get("summary"),
            walk_forward=read(base / "E07" / "walkforward_study.json"), classifier=clf,
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

    # ---- trade review ---------------------------------------------------------------------------

    def _review_fit(self, universe: list[str]):
        def fit(prices: pd.DataFrame, vix: pd.Series) -> RegimeSyntheticGenerator:
            key = f"{prices.index[-1]}|{','.join(prices.columns)}"
            with self._luck_lock:
                if key in self._luck_fits:
                    self._luck_fits.move_to_end(key)
                    return self._luck_fits[key]
            gen = RegimeSyntheticGenerator().fit(prices, vix, market_columns=universe)
            with self._luck_lock:
                self._luck_fits[key] = gen
                while len(self._luck_fits) > self.LUCK_CACHE_SIZE:
                    self._luck_fits.popitem(last=False)
            return gen
        return fit

    def _closes(self, tickers: list[str], end: str) -> pd.DataFrame:
        spec = self.cfg.spec
        panel = self._store().get(tickers, spec.start, end, refresh=spec.refresh)
        if panel.empty:
            return pd.DataFrame()
        return panel.pivot(index="date", columns="tic", values="close")

    def _market_closes(self, today: str) -> tuple[pd.DataFrame, list[str]]:
        """The research universe's closes and its trading days (days every
        universe ticker traded): the calendar every holding window is counted on."""
        universe = list(self.cfg.spec.tickers)
        try:
            closes = self._closes(universe, today)
        except ProviderError as exc:
            raise CapabilityUnavailable(f"price data is unavailable right now: {exc}") from exc
        if closes.empty:
            raise CapabilityUnavailable("price data is unavailable right now")
        return closes, list(closes.index[closes[universe].notna().all(axis=1)])

    def _listings(self) -> list[symbols.Listing] | None:
        """The US symbol directory, reloaded once a day. None if it can't be had:
        the review then runs without the directory checks."""
        today = self._today()
        if self._symbols is None or self._symbols[0] != today:
            try:
                self._symbols = (today, symbols.load(self.cfg.symbols_dir, refresh=self.cfg.spec.refresh))
            except ProviderError as exc:
                log.warning("symbol directory unavailable: %s", exc)
                return None
        return self._symbols[1]

    def _parse(self, text: str, today: str, overrides: dict) -> tuple[ParsedTrade, str, list[str]]:
        notes, primary, name = [], None, None
        if self.parser is not None and text.strip():
            try:
                primary, name = self.parser.parse(text, today), self.parser.name
            except (ClassificationFailed, BudgetExhausted, ProviderError) as exc:
                notes.append(f"The language model could not read the trade ({type(exc).__name__}); used the simple parser instead.")
            except Exception as exc:  # network errors from the model provider surface as many types
                log.warning("LLM parser failed: %s: %s", type(exc).__name__, exc)
                notes.append("The language model was unavailable; used the simple parser instead.")
        fallback = (self.parser_fallback.parse(text, today) if self.parser_fallback else ParsedTrade())
        name = name or (self.parser_fallback.name if self.parser_fallback else "form")
        parsed = review.merge(primary, fallback, overrides)
        # dates that can't be read without guessing are dropped, whatever the parsers made of them
        problems = review.date_problems(text)
        dropped = False
        for f in ("date", "sell_date"):
            if problems and f not in overrides and getattr(parsed, f):
                setattr(parsed, f, None)
                dropped = True
        if problems and (dropped or "date" not in overrides):
            notes += [f"Dates: {p}." for p in problems]
            parsed.unclear = [u for u in parsed.unclear if "date" not in u.lower()]  # said once, above
        return parsed, name, notes

    def _check_symbol(self, symbol: str, company: str | None, from_form: bool) -> tuple[str, str | None, list[str], str | None]:
        """(symbol, listing name, notes, reason it can't be reviewed). Follows
        renames, refuses delisted symbols, and corrects a parser that matched
        only part of the name ("apple hospitality" read as AAPL)."""
        notes = []
        if symbol in symbols.DELISTED:
            return symbol, None, notes, (f"{symbol} no longer trades: {symbols.DELISTED[symbol]}. Its old prices aren't "
                                         "available from the data source, so the trade can't be checked.")
        if symbol in symbols.RENAMED:
            new = symbols.RENAMED[symbol]
            notes.append(f"{symbol} now trades as {new}, with the same price history; reviewed {new}.")
            symbol = new
        listings = self._listings()
        if not listings:
            return symbol, None, notes, None
        hit = symbols.lookup(listings, symbol)
        words = [w for w in re.split(r"\W+", (company or "").lower()) if w and w not in ("inc", "the", "shares", "stock")]
        if words and hit and not from_form:
            name_words = re.split(r"\W+", hit.name.lower())
            matched = [w for w in words if any(nw.startswith(w) for nw in name_words)]
            if matched and len(matched) < len(words):
                top = symbols.search(listings, company, 1)
                if top and top[0].symbol != symbol and all(
                        any(nw.startswith(w) for nw in re.split(r"\W+", top[0].name.lower())) for w in words):
                    notes.append(f"Read '{company}' as {top[0].symbol} ({top[0].name}), not {symbol} ({hit.name}).")
                    symbol, hit = top[0].symbol, top[0]
        return symbol, (hit.name if hit else None), notes, None

    def _suggest(self, text: str) -> list[str]:
        listings = self._listings()
        if listings and text:
            return [f"{l.name} ({l.symbol})" for l in symbols.search(listings, text, 4)]
        return review.suggest(text)

    @capability("search_symbols", SymbolIn, SymbolsOut, effect="read", budget_ms=1500)
    def search_symbols(self, req: SymbolIn) -> SymbolsOut:
        """Find a US-listed stock or ETF by symbol or company name."""
        listings = self._listings()
        if listings is None:
            raise CapabilityUnavailable("the symbol directory is unavailable right now")
        q, note, first = req.q.strip(), None, []
        sym = symbols.yahoo_symbol(q)
        if review.is_bond_query(q):
            note, first = review.BOND_MESSAGE, [symbols.lookup(listings, s) for s in review.BOND_ETFS]
        elif review.scope_problem(q, q if "." in q else None) or review.scope_problem(q, None):
            note = review.FOREIGN_MESSAGE
        elif sym in symbols.RENAMED:
            note, first = f"{sym} now trades as {symbols.RENAMED[sym]}.", [symbols.lookup(listings, symbols.RENAMED[sym])]
        elif sym in symbols.DELISTED:
            note = f"{sym} no longer trades: {symbols.DELISTED[sym]}."
        hits = [h for h in first if h] + symbols.search(listings, q, req.limit)
        hits = list(dict.fromkeys(hits))[: req.limit]
        if not hits and note is None:
            note = f"No US-listed stock or ETF matches '{q}'."
        return SymbolsOut(results=[SymbolHit(**vars(h)) for h in hits], note=note)

    @capability("trading_days", TradingDaysIn, TradingDaysOut, effect="read", budget_ms=3000)
    def trading_days(self, req: TradingDaysIn) -> TradingDaysOut:
        """Trading days between a buy date and a sell date (or the latest close), counted on the exchange calendar."""
        today = self._today()
        _, dates = self._market_closes(today)
        try:
            w = review.trading_window(dates, today, req.date, req.sell_date, still_holding=not req.sell_date, max_days=None)
        except review.WindowProblem as exc:
            return TradingDaysOut(entry_date=None, exit_date=None, days=None, notes=[exc.message])
        if w.days > review.MAX_HORIZON:
            w.notes.append(f"The luck check covers holds of up to {review.MAX_HORIZON} trading days (about a year).")
        return TradingDaysOut(entry_date=dates[w.i], exit_date=dates[w.j], days=w.days, notes=w.notes)

    @capability("parse_trade", ParseIn, ParseOut, effect="llm", budget_ms=10000)
    def parse_trade(self, req: ParseIn) -> ParseOut:
        """Read a trade written in words into form fields, for the person to check before the review."""
        parsed, parser_name, notes = self._parse(req.text, self._today(), {})
        notes += [f"Unclear: {u}" for u in parsed.unclear]
        symbol, problem, _ = review.resolve_ticker(parsed.ticker, parsed.company)
        name = None
        if problem == "crypto":
            notes.append(review.CRYPTO_MESSAGE)
        elif msg := review.scope_problem(f"{req.text} {parsed.company or ''}", parsed.ticker if symbol else None):
            notes.append(msg)
        elif symbol:
            symbol, name, more, refusal = self._check_symbol(symbol, parsed.company, from_form=False)
            notes += more + ([refusal] if refusal else [])
        if review.mentions_options(req.text):
            notes.append(review.OPTIONS_MESSAGE)
        return ParseOut(ticker=symbol, name=name, date=parsed.date, sell_date=parsed.sell_date,
                        still_holding=parsed.still_holding, direction=parsed.direction, horizon_days=parsed.horizon_days,
                        reasoning=parsed.reasoning, notes=notes, parser=parser_name)

    def _classify(self, reasoning: str) -> tuple[list[BiasSpan], str | None, str | None]:
        if len(reasoning.split()) < 3:
            return [], None, "No reasoning was given, so there was nothing to check for bias signals."
        result, source = None, None
        for clf in (self.bias, self.bias_fallback):
            if clf is None:
                continue
            try:
                result, source = clf.classify(reasoning), clf.name
                break
            except Exception as exc:  # LLM down or out of budget: fall back to the rules classifier
                log.warning("bias classifier %s failed: %s: %s", clf.name, type(exc).__name__, exc)
        if result is None:
            return [], None, "No bias classifier was available."
        if source != (self.bias.name if self.bias else source):
            source += " (fallback: the language model was unavailable)"
        spans = []
        for sig in result.signals:
            span = review.evidence_span(reasoning, sig.evidence)
            spans.append(BiasSpan(label=sig.label.value, evidence=sig.evidence, confidence=sig.confidence,
                                  start=span[0] if span else None, end=span[1] if span else None))
        note = None if spans else "No bias signals were found in your reasoning."
        return spans, source, note

    @capability("review_trade", ReviewIn, ReviewOut, effect="llm", budget_ms=20000)
    def review_trade(self, req: ReviewIn) -> ReviewOut:
        """Check whether a trade's outcome was luck, point out bias signals in the reasoning, and put it in market context."""
        today = self._today()
        overrides = {k: v for k, v in req.model_dump().items() if k != "text" and v is not None}
        form_complete = set(overrides) >= {"ticker", "date", "direction"} and bool(
            overrides.keys() & {"sell_date", "still_holding", "horizon_days"})
        if form_complete:  # the form carries the trade, so the text is only the reasoning
            parsed, parser_name, messages = ParsedTrade(**overrides, reasoning=req.text.strip() or None), "form", []
        else:
            parsed, parser_name, messages = self._parse(req.text, today, overrides)
        und = Understood(**{k: getattr(parsed, k) for k in ("ticker", "company", "date", "direction", "horizon_days",
                                                            "reasoning", "sell_date", "still_holding")},
                         parser=parser_name)
        messages += [f"Unclear: {u}" for u in parsed.unclear]

        def stop(status: str, *msgs: str, **extra) -> ReviewOut:
            return ReviewOut(status=status, messages=messages + list(msgs), understood=und, **extra)

        symbol, problem, suggestions = review.resolve_ticker(parsed.ticker, parsed.company)
        if problem == "crypto":
            return stop("unsupported", review.CRYPTO_MESSAGE)
        if msg := review.scope_problem(f"{req.text} {parsed.company or ''}", parsed.ticker if symbol else None):
            return stop("unsupported", msg)
        if symbol:
            symbol, und.name, more, refusal = self._check_symbol(symbol, parsed.company, from_form="ticker" in overrides)
            messages += more
            if refusal:
                und.ticker = symbol
                return stop("rejected", refusal)
        und.ticker = symbol
        missing = [f for f in review.REQUIRED if getattr(und, f) in (None, "")]
        if "date" in missing and und.sell_date and und.horizon_days:
            missing.remove("date")  # counted back from the sell date
        if not (und.sell_date or und.still_holding or und.horizon_days):
            missing.append("sell_date")
        if missing:
            if not symbol and parsed.company:
                suggestions = self._suggest(parsed.company)
                listings = self._listings()
                words = [w for w in re.split(r"\W+", parsed.company.lower()) if w]
                if listings and words and not any(all(w in h.name.lower() for w in words)
                                                  for h in symbols.search(listings, parsed.company, 8)):
                    messages.append(f"No US-listed stock or ETF is named '{parsed.company}'. " + review.FOREIGN_MESSAGE)
            return stop("needs_input", "Some details of the trade are missing. Fill them in to run the review.",
                        missing=missing, suggestions=suggestions)
        if und.direction not in ("long", "short"):
            return stop("needs_input", f"Direction '{und.direction}' should be long or short.", missing=["direction"])
        if review.mentions_options(req.text):
            messages.append(review.OPTIONS_MESSAGE)

        universe = list(self.cfg.spec.tickers)
        closes, dates = self._market_closes(today)  # the market reference must load, or nothing can be judged
        try:
            w = review.trading_window(dates, today, und.date, und.sell_date, und.still_holding, und.horizon_days)
        except review.WindowProblem as exc:
            return stop(exc.status, exc.message, missing=exc.missing)
        messages += w.notes
        entry, exit_ = dates[w.i], dates[w.j]
        und.date = und.date or entry
        und.horizon_days = w.days
        if symbol not in universe:
            # an unknown symbol and a provider hiccup both come back empty from
            # Yahoo, so a failure here is reported as "no history", not an outage
            try:
                extra = self._closes([symbol], today)
            except ProviderError:
                extra = pd.DataFrame()
            if symbol not in extra.columns or extra[symbol].dropna().empty:
                return stop("rejected", f"No price history found for '{symbol}'. Check the symbol, or try again "
                            "later if the data source is busy.", suggestions=self._suggest(parsed.company or symbol))
            closes = closes.join(extra[[symbol]], how="outer")
        own = closes[symbol].dropna()
        if own.index[0] > entry:
            return stop("rejected", f"{symbol}'s price history starts on {own.index[0]}, after the buy date {entry}. "
                        "Check the date, or whether the symbol belonged to a different company then.")
        if exit_ not in own.index:
            return stop("rejected", f"There is no {symbol} price on {exit_} (its data ends {own.index[-1]}).")
        n_hist = int((own.index <= entry).sum()) - 1
        if n_hist < MIN_TICKER_RETURNS:
            return stop("rejected", f"{symbol} had only {n_hist} trading days of history before {entry}. The luck check "
                        f"needs at least {MIN_TICKER_RETURNS} to learn how the stock moves.")
        if symbol != "SPY":
            try:  # context only: without it the review still runs
                spy_px = self._closes(["SPY"], today)
                closes = closes.join(spy_px[["SPY"]], how="left") if "SPY" in spy_px.columns else closes
            except ProviderError:
                messages.append("The S&P 500 comparison is unavailable right now.")
        target = own.reset_index().rename(columns={symbol: "close"}).assign(tic=symbol)
        target[["open", "high", "low"]] = target[["close"]].values.repeat(3, axis=1)
        target["volume"] = 0
        problems = checks.check_prices(target) + checks.check_gaps(target)
        if problems:
            return stop("rejected", f"The price data for {symbol} failed quality checks: {'; '.join(problems)}")
        # Daily moves above 50% are almost always bad data in the large-cap research
        # universe, but real for meme and small-cap stocks (AMC rose ~301% on
        # 2021-01-27). For a reviewed trade they are a warning, not a rejection.
        moves = own.pct_change()
        big = moves.abs()[moves.abs() > 0.5]
        if len(big):
            worst = big.idxmax()
            messages.append(
                f"{symbol} has {len(big)} daily move{'s' if len(big) > 1 else ''} above 50% in its history (largest "
                f"{review.pct(float(moves[worst]), 0)} on {worst}). The luck check assumes "
                "ordinary daily moves, so read its range with care for a stock like this."
            )

        market_cols = universe + [symbol] if symbol not in universe else universe
        close = closes.loc[dates, market_cols]  # a recent listing is NaN before its first day; the fit allows that
        vix = SeriesStore(get_vix_provider(self.cfg.spec.vix_provider), key="VIX", root=self.cfg.spec.store_root,
                          today=self.cfg.today).get(self.cfg.spec.start, today, refresh=self.cfg.spec.refresh)
        fit = self._review_fit(universe)
        try:
            r = luck_test(close, vix, symbol, entry, und.direction, w.days, n_paths=1000, seed=0, fit=fit)
            surf = outcome_surface(close, vix, symbol, entry, und.direction, w.days, n_paths=1000, seed=0, fit=fit)
        except LuckTestError as exc:
            return stop("rejected", f"The luck check can't run for this trade: {exc}.")
        und.entry_date, und.exit_date = r.entry_date, r.exit_date
        verb = "bought" if und.direction == "long" else "shorted"
        if und.sell_date:
            what = (f"You {verb} {symbol} on {r.entry_date} and {'sold' if und.direction == 'long' else 'covered'} "
                    f"on {r.exit_date}, {w.days} trading days later.")
        elif und.still_holding:
            what = (f"You {verb} {symbol} on {r.entry_date} and still hold it; measured to the latest close on "
                    f"{r.exit_date}, {w.days} trading days later.")
        else:
            what = (f"You {verb} {symbol} on {r.entry_date} and held it for {w.days} trading days, "
                    f"closing on {r.exit_date}.")

        spans, bias_source, bias_note = self._classify(und.reasoning or "")
        spy = None
        if "SPY" in closes.columns and symbol != "SPY":
            s_ = closes["SPY"].dropna()
            if r.entry_date in s_.index and r.exit_date in s_.index:
                spy = float(s_[r.exit_date] / s_[r.entry_date] - 1)
        uni = closes.loc[dates, universe]
        uni_ret = float((uni.loc[r.exit_date] / uni.loc[r.entry_date] - 1).mean())

        verdict = {"unusually_good": "unusually good, better than the model's normal range of luck",
                   "unusually_bad": "unusually bad, worse than the model's normal range of luck",
                   "within_luck_range": "within the normal range of luck"}[r.verdict]
        # plain-English keys: the model copies wording from the facts, so no
        # code-style names may appear here (an earlier version leaked "spy_return")
        facts = {review.F_WHAT: what,
                 review.F_OUTCOME: {review.F_RETURN: review.pct(r.realized_return), review.F_LOW: review.pct(r.band_low),
                                    review.F_HIGH: review.pct(r.band_high), review.F_PCT: review.ordinal(r.percentile),
                                    review.F_VERDICT: verdict, review.F_REGIME: r.regime},
                 review.F_BIASES: [{"bias": b.label.replace("_", " "), "your words": b.evidence} for b in spans]}
        if bias_note:
            facts[review.F_BIAS_NOTE] = bias_note
        if spy is not None:
            facts[review.F_SPY] = review.pct(spy)
        explanation, source = review.template_explanation(facts), "template"
        if self.explainer is not None:
            # the model writes {placeholders} instead of numbers; the code fills them in
            slots = {"ticker": symbol, "entry_date": r.entry_date, "exit_date": r.exit_date,
                     "days": str(w.days), "your_return": review.pct(r.realized_return),
                     "low": review.pct(r.band_low), "high": review.pct(r.band_high),
                     "percentile": review.ordinal(r.percentile), "research_stocks_return": review.pct(uni_ret)}
            if spy is not None:
                slots["spy_return"] = review.pct(spy)
            text = ""
            try:
                text = self.explainer.explain({**facts, "placeholders": dict(slots)})
                filled = review.fill_slots(text, slots, [b.evidence for b in spans], und.reasoning or req.text)
                bad = review.unsupported_numbers(filled, {**facts, "s": slots})  # second, independent check
                if bad:
                    log.warning("explanation rejected: numbers %s are not computed values", bad)
                else:
                    explanation, source = filled, "llm"
            except review.SlotError as exc:
                log.warning("explanation rejected: %s | text: %.300s", exc, text)
            except Exception as exc:  # model unavailable: the template explanation stands
                log.warning("explainer failed: %s: %s", type(exc).__name__, exc)
        return ReviewOut(
            status="ok", messages=messages, understood=und, what_you_did=what, outcome=LuckTestOut(**vars(r)),
            surface=Surface(**vars(surf)), biases=spans, bias_source=bias_source,
            market=MarketContext(spy_return=spy, universe_return=uni_ret), explanation=explanation,
            explanation_source=source,
        )

    # ---- llm (through a port) -------------------------------------------------------------

    @capability("classify_biases", TextIn, BiasClassification, effect="llm", budget_ms=8000)
    def classify_biases(self, req: TextIn) -> BiasClassification:
        """Behavioural-bias signals in a piece of trading text, with the quoted evidence. Describes the text, not the person."""
        if self.bias is None:
            raise CapabilityUnavailable(f"no bias classifier configured in profile '{self.cfg.profile}'")
        return self.bias.classify(req.text)

    # ---- batch: evaluation against hand labels --------------------------------------------------

    @capability("evaluate_biases", BiasEvalIn, BiasEvalOut, effect="batch")
    def evaluate_biases(self, req: BiasEvalIn) -> BiasEvalOut:
        """Score the configured bias classifier against hand-labelled texts: precision, recall, F1 and Cohen's kappa per bias."""
        if self.bias is None:
            raise CapabilityUnavailable(f"no bias classifier configured in profile '{self.cfg.profile}'")
        path = Path(req.labels_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise CapabilityUnavailable(f"no labels at {path}: see corpus/LABELLING.md")
        return BiasEvalOut(**bias_eval.evaluate(bias_eval.load_labels(path), self.bias))

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

