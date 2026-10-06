"""
One call from "which tickers, which dates" to a validated panel + VIX + a
provenance manifest.

PROVENANCE: original to this project. Wires meta/providers.py (where data
comes from), meta/store.py (incremental cache) and meta/validate.py (is it
fit to use), and records what was used: a manifest per load with the
provider, fetch time, per-ticker row counts and date ranges, and a SHA-256
of the exact panel. A saved model can then store the `data_hash` it was
trained on, and the service can refuse a model whose data no longer
matches (see PLAN_SHIPPABLE.md, step M1).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from minifinrl.core.configs.logging_config import get_logger
from minifinrl.core.configs.settings import (
    DATA_MODE,
    DATA_RAW,
    DATA_STORE,
    PRICE_PROVIDER,
    ROOT,
    TEST_END,
    VIX_PROVIDER,
)
from minifinrl.core.meta.providers import get_price_provider, get_vix_provider
from minifinrl.core.meta.store import PriceStore, SeriesStore, market_today
from minifinrl.core.meta.validate import ValidationReport, validate_panel, validate_series

log = get_logger(__name__)

MANIFEST_DIR = Path(DATA_RAW) / "manifests"
_VERSIONED = ["numpy", "pandas", "yfinance", "stockstats", "gymnasium", "stable-baselines3", "hmmlearn", "torch"]


@dataclass
class MarketData:
    panel: pd.DataFrame  # tidy [date, tic, open, high, low, close, volume], adjusted
    vix: pd.Series  # indexed by date string
    report: ValidationReport
    manifest: dict


def resolve_end(end: str | None, mode: str = DATA_MODE) -> str:
    """research: TEST_END unless given. live: today (exclusive), i.e. up to
    and including the last completed session."""
    if end is not None:
        return end
    if mode == "live":
        return market_today()
    if mode == "research":
        return TEST_END
    raise ValueError(f"unknown DATA_MODE '{mode}', expected 'research' or 'live'")


def frame_hash(df: pd.DataFrame | pd.Series) -> str:
    """SHA-256 of a canonical CSV rendering. Floats at full repr precision,
    so any change to any value changes the hash."""
    return hashlib.sha256(df.to_csv(index=True).encode()).hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5, check=True
        )
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None


def _versions() -> dict[str, str | None]:
    out = {}
    for pkg in _VERSIONED:
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def build_manifest(
    panel: pd.DataFrame, vix: pd.Series, *, tickers: list[str], start: str, end: str, mode: str,
    price_provider: str, vix_provider: str, report: ValidationReport,
) -> dict:
    per_tic = {
        t: {"rows": int(len(g)), "first": g["date"].min(), "last": g["date"].max()}
        for t, g in panel.groupby("tic")
    }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": mode,
        "requested": {"tickers": list(tickers), "start": start, "end_exclusive": end},
        "providers": {"prices": price_provider, "vix": vix_provider},
        "panel": {"rows": int(len(panel)), "dates": int(panel["date"].nunique()), "per_ticker": per_tic},
        "vix": {"rows": int(len(vix)), "first": vix.index.min() if len(vix) else None,
                "last": vix.index.max() if len(vix) else None},
        "data_hash": frame_hash(panel),
        "vix_hash": frame_hash(vix),
        "validation": report.to_dict(),
        "git_commit": _git_commit(),
        "versions": _versions(),
    }


def write_manifest(manifest: dict, directory: Path | None = None) -> Path:
    directory = Path(directory) if directory is not None else MANIFEST_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp = manifest["created_at"].replace(":", "").replace("-", "")[:15]
    path = directory / f"{stamp}_{manifest['data_hash'][:12]}.json"
    text = json.dumps(manifest, indent=2, default=str)
    path.write_text(text)
    (directory / "latest.json").write_text(text)
    return path


def latest_manifest(directory: Path | str | None = None) -> dict | None:
    """The most recent manifest written by load_market_data, or None."""
    p = Path(directory if directory is not None else MANIFEST_DIR) / "latest.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_market_data(
    tickers: list[str],
    start: str,
    end: str | None = None,
    *,
    mode: str = DATA_MODE,
    price_provider: str = PRICE_PROVIDER,
    vix_provider: str = VIX_PROVIDER,
    refresh: bool = True,
    strict: bool = True,
    store_root: Path | str = DATA_STORE,
    today: str | None = None,
    write: bool = True,
    manifest_dir: Path | str | None = None,
) -> MarketData:
    """Fetch (incrementally), validate and record. `strict=True` raises
    DataValidationError on any error; the manifest is written first either
    way, so a failed load still leaves a record of what was wrong.
    `refresh=False` reads only what is already on disk (the service path)."""
    end = resolve_end(end, mode)
    prices = PriceStore(get_price_provider(price_provider), store_root, today=today)
    vix_store = SeriesStore(get_vix_provider(vix_provider), key="VIX", root=store_root, today=today)

    panel = prices.get(tickers, start, end, refresh=refresh)
    vix = vix_store.get(start, end, refresh=refresh)

    live_today = (today or market_today()) if mode == "live" else None
    report = validate_panel(panel, tickers, start, end, live_today=live_today)
    vix_report = validate_series(vix, "VIX", start, end)
    report.errors += vix_report.errors
    report.warnings += vix_report.warnings

    manifest = build_manifest(
        panel, vix, tickers=tickers, start=start, end=end, mode=mode,
        price_provider=price_provider, vix_provider=vix_provider, report=report,
    )
    if write:
        path = write_manifest(manifest, manifest_dir)
        log.info("market data: %d rows, hash %s, manifest %s", len(panel), manifest["data_hash"][:12], path.name)
    if strict:
        report.raise_for_errors()
    return MarketData(panel=panel, vix=vix, report=report, manifest=manifest)
