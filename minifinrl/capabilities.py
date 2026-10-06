"""
The capability registry: declare a thing the system can do once, and the
CLI, the API, the agent's tools and the gate's coverage check all come
from that one declaration (PLAN_SHIPPABLE.md section 1.3).

    class Engine:
        @capability("rank_stability", RankStabilityIn, RankStabilityOut, effect="read")
        def rank_stability(self, req: RankStabilityIn) -> RankStabilityOut: ...

Effect classes carry the exposure and safety rules, so an interface never
decides them ad hoc:

    read       disk only, fast                API, agent, CLI
    compute    bounded CPU, time budget       API, agent, CLI
    llm        may call a model, costs money  API, agent, CLI
    batch      long-running (train, fetch)    CLI only, never inside a request
    forbidden  declared, never executed       listed, so refusals are testable
"""

from __future__ import annotations

import inspect
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, ValidationError

from minifinrl.core.configs.logging_config import get_logger
from minifinrl.core.meta.providers import ProviderError
from minifinrl.core.meta.validate import DataValidationError
from minifinrl.ports import BudgetExhausted

log = get_logger(__name__)

Effect = Literal["read", "compute", "llm", "batch", "forbidden"]
EFFECTS: tuple[str, ...] = ("read", "compute", "llm", "batch", "forbidden")
INTERFACE_EFFECTS: dict[str, frozenset[str]] = {
    "cli": frozenset({"read", "compute", "llm", "batch"}),
    "api": frozenset({"read", "compute", "llm"}),
    "agent": frozenset({"read", "compute", "llm"}),
}
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


# ---- errors: what went wrong, independent of how an interface reports it ----------


class CapabilityError(RuntimeError):
    kind = "bug"


class CapabilityForbidden(CapabilityError, PermissionError):
    kind = "forbidden"


class CapabilityUnavailable(CapabilityError):
    """A dependency the capability needs isn't configured or isn't ready
    (no bias classifier in this profile, no data on disk yet)."""

    kind = "unavailable"


class CapabilityNotFound(CapabilityError, LookupError):
    kind = "not_found"


def error_kind(exc: BaseException) -> str:
    """bad_input | forbidden | not_found | unavailable | budget | bug.
    Interfaces map these to their own vocabulary (HTTP status, exit code,
    tool error), so the mapping is decided once, here."""
    if isinstance(exc, CapabilityError):
        return exc.kind
    if isinstance(exc, ValidationError):
        return "bad_input"
    if isinstance(exc, (ProviderError, DataValidationError)):
        return "unavailable"
    if isinstance(exc, BudgetExhausted):
        return "budget"
    if isinstance(exc, (FileNotFoundError, KeyError)):
        return "not_found"
    return "bug"


# ---- declaration -------------------------------------------------------------------


@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    input: type[BaseModel]
    output: type[BaseModel]
    effect: str
    budget_ms: int | None
    summary: str
    method: str


def capability(
    name: str,
    input: type[BaseModel],
    output: type[BaseModel],
    *,
    effect: Effect,
    budget_ms: int | None = None,
) -> Callable[[Callable], Callable]:
    """Mark an Engine method as a capability. Checked at import time, so a
    malformed declaration fails before anything runs."""
    if not _NAME.match(name):
        raise ValueError(f"capability name '{name}' must be snake_case")
    if effect not in EFFECTS:
        raise ValueError(f"capability '{name}': effect '{effect}' not in {EFFECTS}")
    for role, model in (("input", input), ("output", output)):
        if not (inspect.isclass(model) and issubclass(model, BaseModel)):
            raise TypeError(f"capability '{name}': {role} must be a pydantic model, got {model!r}")

    def mark(fn: Callable) -> Callable:
        doc = inspect.getdoc(fn) or ""
        fn.__capability__ = CapabilitySpec(
            name=name, input=input, output=output, effect=effect, budget_ms=budget_ms,
            summary=doc.split("\n\n")[0].replace("\n", " "), method=fn.__name__,
        )
        return fn

    return mark


# ---- registry ----------------------------------------------------------------------


class CapabilityRegistry:
    def __init__(self, target: Any, specs: dict[str, CapabilitySpec]):
        self._target = target
        self._specs = specs

    @classmethod
    def from_engine(cls, engine: Any) -> "CapabilityRegistry":
        specs: dict[str, CapabilitySpec] = {}
        for klass in reversed(type(engine).__mro__):
            for attr, fn in vars(klass).items():
                spec = getattr(fn, "__capability__", None)
                if spec is None:
                    continue
                if spec.name in specs and specs[spec.name].method != attr:
                    raise ValueError(f"duplicate capability name '{spec.name}' ({specs[spec.name].method}, {attr})")
                specs[spec.name] = spec
        if not specs:
            raise ValueError(f"{type(engine).__name__} declares no capabilities")
        return cls(engine, dict(sorted(specs.items())))

    # -- introspection --------------------------------------------------------

    def names(self, interface: str | None = None) -> list[str]:
        return [s.name for s in self.list(interface)]

    def list(self, interface: str | None = None) -> list[CapabilitySpec]:
        """All specs, or those an interface may execute. `forbidden` ones are
        included for every interface: they are listed so a caller gets an
        explicit refusal instead of 'no such thing'."""
        if interface is None:
            return list(self._specs.values())
        allowed = INTERFACE_EFFECTS[interface]
        return [s for s in self._specs.values() if s.effect in allowed or s.effect == "forbidden"]

    def get(self, name: str) -> CapabilitySpec:
        if name not in self._specs:
            raise CapabilityNotFound(f"no capability '{name}' (have: {', '.join(self._specs)})")
        return self._specs[name]

    def specs(self, interface: str | None = None) -> list[dict[str, Any]]:
        """JSON-able descriptions: what an LLM tool list or an API index shows."""
        return [
            {
                "name": s.name, "summary": s.summary, "effect": s.effect, "budget_ms": s.budget_ms,
                "input_schema": s.input.model_json_schema(), "output_schema": s.output.model_json_schema(),
            }
            for s in self.list(interface)
        ]

    # -- execution ---------------------------------------------------------------

    def invoke(self, name: str, payload: BaseModel | dict[str, Any] | None = None, *, interface: str | None = None) -> BaseModel:
        """Validate input, refuse what the effect class forbids, run, check the
        output type, and log time against the budget."""
        spec = self.get(name)
        if spec.effect == "forbidden":
            raise CapabilityForbidden(f"'{name}' is forbidden: this system never executes it")
        if interface is not None and spec.effect not in INTERFACE_EFFECTS[interface]:
            raise CapabilityForbidden(f"'{name}' ({spec.effect}) is not available through the {interface} interface")

        if isinstance(payload, spec.input):
            req = payload
        else:
            req = spec.input.model_validate(payload or {})

        t0 = time.perf_counter()
        out = getattr(self._target, spec.method)(req)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if not isinstance(out, spec.output):
            raise CapabilityError(f"'{name}' returned {type(out).__name__}, declared {spec.output.__name__}")
        if spec.budget_ms is not None and elapsed_ms > spec.budget_ms:
            log.warning("capability %s took %.0f ms, budget %d ms", name, elapsed_ms, spec.budget_ms)
        else:
            log.debug("capability %s took %.0f ms", name, elapsed_ms)
        return out

    # -- adapters for the agent's tool guard ---------------------------------------

    def callables(self, interface: str = "agent") -> dict[str, Callable[..., dict]]:
        """{name: fn(**json_args) -> json_result} for aip.guards.ToolGuard, which
        calls `registry[name](**args)` with plain dicts. Forbidden capabilities
        are absent, so the guard refuses them as 'does not exist'."""
        allowed = INTERFACE_EFFECTS[interface]

        def bind(name: str) -> Callable[..., dict]:
            return lambda **kw: self.invoke(name, kw, interface=interface).model_dump(mode="json")

        return {s.name: bind(s.name) for s in self._specs.values() if s.effect in allowed}

    def schemas(self, interface: str = "agent") -> dict[str, type[BaseModel]]:
        allowed = INTERFACE_EFFECTS[interface]
        return {s.name: s.input for s in self._specs.values() if s.effect in allowed}
