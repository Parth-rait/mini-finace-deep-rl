"""
Experiment log: every result-affecting change is recorded with its exact
mathematical definition, the hypothesis it tests, and the numbers before
and after.

PROVENANCE: original to this project. Each record captures, automatically:
git commit, data_hash, the model cards in use, a frozen copy of the raw
backtest results (results/experiments/<id>/backtest_results.csv + its
SHA-256), and a per-app/per-model summary. Records are appended to
results/experiments/log.jsonl (machine-readable) and EXPERIMENTS.md (for
people). Results are never overwritten: a later run cannot erase the
evidence behind an earlier conclusion.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from minifinrl.core.agents.registry import ModelRegistry
from minifinrl.core.configs.settings import RESULTS, ROOT
from minifinrl.core.eval.report import rank_stability
from minifinrl.core.pipeline import REFERENCE_BASELINE

EXPERIMENT_DIR = RESULTS / "experiments"
EXPERIMENTS_MD = ROOT / "EXPERIMENTS.md"

_HEADER = """# Experiment log

Every change that can move a result is recorded here: what changed, its exact
definition, the hypothesis, and the numbers it produced. Written by the
`record_experiment` capability (`python -m minifinrl record-experiment ...`);
the machine-readable twin is `results/experiments/log.jsonl`, and each entry's
raw backtest rows are frozen in `results/experiments/<id>/`.

Sharpe = annualised mean / std of daily returns (excess over the stated
risk-free rate). "Synthetic" = 20 regime-model paths of 500 days. Wins and
"beats ref" use each model's median over seeds per path; the reference
baseline is fully invested buy-and-hold (trading) and daily 1/N (portfolio).
"""


@dataclass
class ExperimentRecord:
    id: str
    title: str
    change: str
    math: str
    hypothesis: str
    conclusion: str
    files: list[str]
    compare_to: str | None
    created_at: str
    git_commit: str | None
    data_hash: str | None
    models: list[dict]
    results_sha256: str | None
    summary: dict = field(default_factory=dict)  # app -> model -> metrics


def _git() -> str | None:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
        return head.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None


def summarize_results(df: pd.DataFrame) -> dict:
    """app -> model -> {hist_sharpe, synth_sharpe, wins, n_paths, beats_ref,
    turnover, cost_frac, seeds}: medians over seeds (and paths)."""
    out: dict = {}
    # rows written before baselines were named (E01 and earlier) say "baseline"
    df = df.copy()
    legacy = df["model"] == "baseline"
    df.loc[legacy, "model"] = df.loc[legacy, "app"].map(REFERENCE_BASELINE)
    for app, sub in df.groupby("app"):
        hist, synth = sub[sub["path"] == "historical"], sub[sub["path"] != "historical"]
        ref = REFERENCE_BASELINE.get(app)
        wins = rank_stability(synth.to_dict("records"), "path", "model", baseline=ref).set_index("model") if not synth.empty else None
        models = {}
        for model, g in sub.groupby("model"):
            h, s = hist[hist["model"] == model], synth[synth["model"] == model]
            row = {
                "seeds": int(g["seed"].nunique()) if g["seed"].notna().any() else 0,
                "hist_sharpe": float(h["sharpe"].median()) if len(h) else None,
                "hist_cagr": float(h["cagr"].median()) if len(h) else None,
                "synth_sharpe": float(s["sharpe"].median()) if len(s) else None,
                "turnover": float(g["turnover"].median()) if "turnover" in g and g["turnover"].notna().any() else None,
                "cost_frac": float(h["cost_frac"].median()) if "cost_frac" in h and h["cost_frac"].notna().any() else None,
            }
            if wins is not None and model in wins.index:
                w = wins.loc[model]
                row.update(wins=int(w["wins"]), n_paths=int(w["n_paths"]),
                           beats_ref=None if pd.isna(w["beats_baseline"]) else int(w["beats_baseline"]))
            models[model] = row
        out[app] = models
    return out


def load_log(directory: Path = EXPERIMENT_DIR) -> list[dict]:
    p = Path(directory) / "log.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def _fmt(v, kind: str) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if kind == "pct":
        return f"{v:.1%}"
    if kind == "int":
        return str(int(v))
    return f"{v:.2f}"


def _delta(now, before) -> str:
    if now is None or before is None:
        return ""
    d = now - before
    return f" ({'+' if d >= 0 else ''}{d:.2f})" if abs(d) >= 0.005 else ""


def to_markdown(rec: ExperimentRecord, previous: dict | None) -> str:
    lines = [
        f"\n## {rec.id}: {rec.title}\n",
        f"*{rec.created_at}, commit `{rec.git_commit}`, data `{(rec.data_hash or '')[:12]}`, "
        f"results sha256 `{(rec.results_sha256 or '')[:12]}`"
        + (f", compared with {rec.compare_to}" if rec.compare_to else "") + "*\n",
        f"**Change.** {rec.change}\n",
        f"**Definition.**\n\n```\n{rec.math.strip()}\n```\n",
        f"**Hypothesis.** {rec.hypothesis}\n",
    ]
    if rec.files:
        lines.append("**Files.** " + ", ".join(f"`{f}`" for f in rec.files) + "\n")
    prev = (previous or {}).get("summary", {})
    for app, models in rec.summary.items():
        lines.append(f"\n**{app}** (Δ vs {rec.compare_to})\n" if rec.compare_to else f"\n**{app}**\n")
        lines.append("| model | seeds | hist Sharpe | hist CAGR | synth Sharpe | wins | beats ref | turnover/day | costs |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for model, m in sorted(models.items()):
            p = prev.get(app, {}).get(model, {})
            wins = f"{m['wins']}/{m['n_paths']}" if m.get("wins") is not None else "-"
            beats = f"{m['beats_ref']}/{m['n_paths']}" if m.get("beats_ref") is not None else "-"
            lines.append(
                f"| {model} | {m['seeds']} | {_fmt(m['hist_sharpe'], 'f')}{_delta(m['hist_sharpe'], p.get('hist_sharpe'))} "
                f"| {_fmt(m['hist_cagr'], 'pct')} | {_fmt(m['synth_sharpe'], 'f')}{_delta(m['synth_sharpe'], p.get('synth_sharpe'))} "
                f"| {wins} | {beats} | {_fmt(m['turnover'], 'pct')} | {_fmt(m['cost_frac'], 'pct')} |"
            )
    lines.append(f"\n**Conclusion.** {rec.conclusion}\n")
    lines.append("\n---")
    return "\n".join(lines)


def record(
    *, id: str, title: str, change: str, math: str, hypothesis: str, conclusion: str,
    files: list[str], compare_to: str | None, results_path: Path, model_dir: Path, data_hash: str | None,
    directory: Path = EXPERIMENT_DIR, markdown: Path = EXPERIMENTS_MD,
) -> ExperimentRecord:
    directory = Path(directory)
    if any(r["id"] == id for r in load_log(directory)):
        raise ValueError(f"experiment {id} is already recorded; ids are permanent, use a new one")
    previous = next((r for r in load_log(directory) if r["id"] == compare_to), None) if compare_to else None
    if compare_to and previous is None:
        raise ValueError(f"compare_to={compare_to} is not in the log")

    snap_dir = directory / id
    snap_dir.mkdir(parents=True, exist_ok=True)
    sha, summary = None, {}
    if Path(results_path).exists():
        shutil.copy2(results_path, snap_dir / "backtest_results.csv")
        sha = hashlib.sha256(Path(results_path).read_bytes()).hexdigest()
        summary = summarize_results(pd.read_csv(results_path))
    models = [
        {"id": e.model_id, "legacy": e.legacy, **({} if e.card is None else {
            "algo": e.card.algo, "timesteps": e.card.timesteps, "env_version": e.card.spec.env_version,
            "obs_scaling": e.card.spec.obs_scaling, "data_hash": e.card.data_hash[:12]})}
        for e in ModelRegistry(model_dir).list()
    ]
    rec = ExperimentRecord(
        id=id, title=title, change=change, math=math, hypothesis=hypothesis, conclusion=conclusion,
        files=files, compare_to=compare_to, created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=_git(), data_hash=data_hash, models=models, results_sha256=sha, summary=summary,
    )
    (snap_dir / "record.json").write_text(json.dumps(asdict(rec), indent=2))
    with (directory / "log.jsonl").open("a") as f:
        f.write(json.dumps(asdict(rec)) + "\n")
    md = Path(markdown)
    if not md.exists():
        md.write_text(_HEADER)
    with md.open("a") as f:
        f.write(to_markdown(rec, previous) + "\n")
    return rec
