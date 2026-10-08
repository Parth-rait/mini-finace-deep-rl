"""
Versioned model artifacts: a metadata sidecar next to every saved policy,
and a load that refuses a policy whose inputs no longer match.

PROVENANCE: original to this project. SB3's load() only checks array
shapes, so a policy trained on other tickers, another indicator list, or
unscaled observations (every model before 30 Sep) loads fine with the
same shape and silently misreads its input. Here each `<app>_<algo>_seed<n>.zip`
gets a `.json` card recording what it was trained on, and loading compares
that card with what the environment will actually feed it.

What must match (an error if not): app/env class, algo, ticker order,
feature names, observation scaling version, observation and action
shapes. What should match (a warning in research, an error when the caller
asks for strict data): the data_hash of the training panel.
"""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from minifinrl.platform.log import get_logger
from minifinrl.platform.settings import ROOT

log = get_logger(__name__)

CARD_VERSION = 1


class ModelMismatch(ValueError):
    """The saved policy was trained on inputs that differ from what the
    environment provides. Retrain, or fix the config."""


@dataclass(frozen=True)
class ModelSpec:
    """What a policy's inputs are. Built from a live env when loading,
    and frozen into the card when saving."""

    app: str
    env_class: str
    tickers: tuple[str, ...]
    feature_names: tuple[str, ...]
    obs_scaling: str
    obs_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    env_version: str = ""  # "" on cards written before envs were versioned -> treated as "<app>-v1"

    @classmethod
    def from_env(cls, app: str, env) -> "ModelSpec":
        e = getattr(env, "unwrapped", env)
        return cls(
            app=app,
            env_class=type(e).__name__,
            tickers=tuple(e.tickers),
            feature_names=tuple(e.feature_names),
            obs_scaling=getattr(e, "obs_scaling", "none"),
            obs_shape=tuple(int(x) for x in e.observation_space.shape),
            action_shape=tuple(int(x) for x in e.action_space.shape),
            env_version=getattr(e, "env_version", f"{app}-v1"),
        )


@dataclass(frozen=True)
class ModelCard:
    card_version: int
    algo: str
    seed: int
    timesteps: int
    spec: ModelSpec
    train_start: str
    train_end: str
    data_hash: str
    created_at: str
    git_commit: str | None
    versions: dict[str, str | None] = field(default_factory=dict)
    train_seconds: float | None = None
    params: dict = field(default_factory=dict)  # algorithm hyperparameters as passed to SB3

    @property
    def model_id(self) -> str:
        return f"{self.spec.app}_{self.algo}_seed{self.seed}"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "ModelCard":
        d = json.loads(text)
        s = d.pop("spec")
        # cards from before env versioning describe the original (v1) dynamics
        s.setdefault("env_version", f"{s['app']}-v1")
        spec = ModelSpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in s.items()})
        return cls(spec=spec, **d)


def card_path(model_zip: str | Path) -> Path:
    return Path(model_zip).with_suffix(".json")


def _git_commit() -> str | None:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=5)
        return head.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None


def _versions() -> dict[str, str | None]:
    out = {}
    for pkg in ("stable-baselines3", "torch", "gymnasium", "numpy", "pandas"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def make_card(*, algo: str, seed: int, timesteps: int, spec: ModelSpec, train_start: str, train_end: str,
              data_hash: str, train_seconds: float | None = None, params: dict | None = None) -> ModelCard:
    return ModelCard(
        card_version=CARD_VERSION, algo=algo, seed=seed, timesteps=timesteps, spec=spec,
        train_start=train_start, train_end=train_end, data_hash=data_hash,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=_git_commit(), versions=_versions(), train_seconds=train_seconds, params=dict(params or {}),
    )


def write_card(model_zip: str | Path, card: ModelCard) -> Path:
    p = card_path(model_zip)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(card.to_json())
    return p


def read_card(model_zip: str | Path) -> ModelCard | None:
    p = card_path(model_zip)
    return ModelCard.from_json(p.read_text()) if p.exists() else None


def mismatches(card: ModelCard, expect: ModelSpec, algo: str) -> list[str]:
    """Every way the card disagrees with the expected inputs. Ticker order
    is compared exactly: the observation is laid out per ticker."""
    out = []
    if card.algo != algo:
        out.append(f"algo: trained {card.algo}, loading as {algo}")
    for name in ("app", "env_class", "env_version", "tickers", "feature_names", "obs_scaling", "obs_shape", "action_shape"):
        trained, now = getattr(card.spec, name), getattr(expect, name)
        if trained != now:
            out.append(f"{name}: trained {list(trained) if isinstance(trained, tuple) else trained}, "
                       f"env provides {list(now) if isinstance(now, tuple) else now}")
    return out


def verify(model_zip: str | Path, expect: ModelSpec, algo: str, *, data_hash: str | None = None,
           strict_data: bool = False) -> ModelCard:
    """Raise ModelMismatch unless the policy at `model_zip` was trained on
    the inputs `expect` describes. Returns its card."""
    card = read_card(model_zip)
    if card is None:
        raise ModelMismatch(
            f"{Path(model_zip).name} has no metadata card: it predates versioned models "
            "(and observation scaling). Retrain it."
        )
    problems = mismatches(card, expect, algo)
    if problems:
        raise ModelMismatch(f"{card.model_id} does not match this environment: " + "; ".join(problems))
    if data_hash is not None and card.data_hash != data_hash:
        msg = (f"{card.model_id} was trained on data {card.data_hash[:12]}, current data is {data_hash[:12]} "
               "(re-fetched or changed; Yahoo adjusted closes also drift ~1e-6 between downloads)")
        if strict_data:
            raise ModelMismatch(msg)
        log.warning(msg)
    return card


@dataclass(frozen=True)
class ModelEntry:
    model_id: str
    path: Path
    card: ModelCard | None  # None = legacy model without a card

    @property
    def legacy(self) -> bool:
        return self.card is None


class ModelRegistry:
    """The saved policies in a directory, with their cards."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)

    def list(self) -> list[ModelEntry]:
        if not self.model_dir.exists():
            return []
        return [ModelEntry(p.stem, p, read_card(p)) for p in sorted(self.model_dir.glob("*.zip"))]

    def get(self, app: str, algo: str, seed: int) -> ModelEntry | None:
        p = self.model_dir / f"{app}_{algo}_seed{seed}.zip"
        return ModelEntry(p.stem, p, read_card(p)) if p.exists() else None
