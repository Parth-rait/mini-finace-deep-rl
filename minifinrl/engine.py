"""
The Engine: builds one service per feature and hands them to the capability
registry. It holds no feature logic itself, only the system-wide `health`.

Each feature (market, regime, research, sentiment, review, plan, journal, orders) declares
its capabilities on its own service class, and the CLI, the API and the
agent's tools are generated from those declarations. Adding a feature means
adding its package and one line in `Engine.__init__`; nothing else changes.

The Engine receives its collaborators (config, and port implementations
like the bias classifier); it never constructs adapters or reads the
environment. That is system.py's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import minifinrl
from minifinrl.journal import settings as journal_settings
from minifinrl.journal.service import JournalService
from minifinrl.market import symbols
from minifinrl.market.dataset import MANIFEST_DIR, latest_manifest
from minifinrl.market.panel import DataSpec
from minifinrl.market.service import MarketService
from minifinrl.orders.service import OrdersService
from minifinrl.plan.service import PlanService
from minifinrl.platform.capabilities import CapabilityRegistry, capability
from minifinrl.platform.schemas import Empty, Health
from minifinrl.regime.service import RegimeService
from minifinrl.research import paths as research_paths
from minifinrl.research.agents.registry import ModelRegistry
from minifinrl.research.service import ResearchService
from minifinrl.review.ports import TradeExplainer, TradeParser
from minifinrl.review.service import ReviewService
from minifinrl.sentiment.ports import BiasClassifier, StateReader
from minifinrl.sentiment.service import SentimentService


@dataclass(frozen=True)
class EngineConfig:
    profile: str = "research"
    mode: str = "research"
    spec: DataSpec = field(default_factory=DataSpec)
    paths: research_paths.Paths = field(default_factory=research_paths.Paths)
    manifest_dir: Path = MANIFEST_DIR
    experiment_dir: Path = research_paths.EXPERIMENT_DIR
    experiments_md: Path = research_paths.EXPERIMENTS_MD
    today: str | None = None  # pinned in tests/ci; None = real market date
    symbols_dir: Path = symbols.CACHE
    database_url: str = journal_settings.DATABASE_URL  # the trade journal (SQLite file unless set)


class Engine:
    def __init__(self, cfg: EngineConfig, *, bias: BiasClassifier | None = None,
                 bias_fallback: BiasClassifier | None = None, parser: TradeParser | None = None,
                 parser_fallback: TradeParser | None = None, explainer: TradeExplainer | None = None,
                 state_reader: StateReader | None = None, state_fallback: StateReader | None = None):
        self.cfg = cfg
        self.market = MarketService(cfg)
        self.regime = RegimeService(cfg)
        self.research = ResearchService(cfg)
        self.sentiment = SentimentService(cfg, bias, bias_fallback, state_reader, state_fallback)
        self.review = ReviewService(cfg, self.market, self.regime, self.sentiment, parser=parser,
                                    parser_fallback=parser_fallback, explainer=explainer)
        self.plan = PlanService(cfg, self.market, self.regime, self.sentiment, self.review)
        self.journal = JournalService(cfg, self.review, self.plan)
        self.orders = OrdersService()
        self.services = (self.market, self.regime, self.research, self.sentiment, self.review, self.plan, self.journal,
                         self.orders)

    # what is plugged in, for health checks and tests
    @property
    def bias(self) -> BiasClassifier | None:
        return self.sentiment.bias

    @property
    def parser(self) -> TradeParser | None:
        return self.review.parser

    @property
    def explainer(self) -> TradeExplainer | None:
        return self.review.explainer

    @capability("health", Empty, Health, effect="read", budget_ms=500)
    def health(self, req: Empty) -> Health:
        """What is loaded and how fresh it is: data, models, results, capabilities."""
        store = self.market.price_store()
        last = [b for t in self.cfg.spec.tickers if (b := store.last_bar(t))]
        last_bar = min(last) if len(last) == len(self.cfg.spec.tickers) else None  # the stalest ticker
        today = self.market.today()
        stale = int(np.busday_count(np.datetime64(last_bar, "D"), np.datetime64(today, "D"))) if last_bar else None
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
