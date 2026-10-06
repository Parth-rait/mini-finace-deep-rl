"""The registry: declaration checks, effect rules, the ToolGuard shape."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from minifinrl.capabilities import (
    CapabilityError,
    CapabilityForbidden,
    CapabilityNotFound,
    CapabilityRegistry,
    CapabilityUnavailable,
    capability,
    error_kind,
)
from minifinrl.core.meta.providers import ProviderError
from minifinrl.ports import BudgetExhausted


class In(BaseModel):
    x: int


class Out(BaseModel):
    y: int


class Toy:
    @capability("double", In, Out, effect="read")
    def double(self, req: In) -> Out:
        """Twice x."""
        return Out(y=2 * req.x)

    @capability("slow_job", In, Out, effect="batch")
    def slow_job(self, req: In) -> Out:
        return Out(y=req.x)

    @capability("launch", In, Out, effect="forbidden")
    def launch(self, req: In) -> Out:
        raise AssertionError("must never run")

    @capability("liar", In, Out, effect="read")
    def liar(self, req: In):
        return {"y": 1}


@pytest.fixture
def reg():
    return CapabilityRegistry.from_engine(Toy())


def test_declaration_checks():
    with pytest.raises(ValueError, match="snake_case"):
        capability("Bad-Name", In, Out, effect="read")
    with pytest.raises(ValueError, match="effect"):
        capability("x", In, Out, effect="write")
    with pytest.raises(TypeError, match="pydantic"):
        capability("x", dict, Out, effect="read")


def test_duplicate_names_rejected():
    class Dup:
        @capability("same", In, Out, effect="read")
        def a(self, req):
            ...

        @capability("same", In, Out, effect="read")
        def b(self, req):
            ...

    with pytest.raises(ValueError, match="duplicate"):
        CapabilityRegistry.from_engine(Dup())


def test_invoke_validates_and_runs(reg):
    assert reg.invoke("double", {"x": 3}).y == 6
    assert reg.invoke("double", In(x=4)).y == 8
    with pytest.raises(ValidationError):
        reg.invoke("double", {"x": "three"})


def test_forbidden_never_runs(reg):
    with pytest.raises(CapabilityForbidden):
        reg.invoke("launch", {"x": 1})


def test_batch_not_reachable_from_api_or_agent(reg):
    assert reg.invoke("slow_job", {"x": 1}, interface="cli").y == 1
    for interface in ("api", "agent"):
        with pytest.raises(CapabilityForbidden, match="not available"):
            reg.invoke("slow_job", {"x": 1}, interface=interface)


def test_interface_listings(reg):
    assert reg.names("api") == ["double", "launch", "liar"]  # forbidden listed, batch not
    assert "slow_job" in reg.names("cli")


def test_output_type_enforced(reg):
    with pytest.raises(CapabilityError, match="returned dict"):
        reg.invoke("liar", {"x": 1})


def test_unknown_name(reg):
    with pytest.raises(CapabilityNotFound):
        reg.invoke("nope")


def test_toolguard_shape(reg):
    """What aip.guards.ToolGuard.call(name, args, registry, schemas) expects:
    plain-dict callables and pydantic schemas; forbidden and batch absent."""
    calls, schemas = reg.callables(), reg.schemas()
    assert set(calls) == {"double", "liar"} and set(schemas) == set(calls)
    assert calls["double"](x=5) == {"y": 10}
    assert schemas["double"] is In


def test_specs_are_json_schema(reg):
    s = {d["name"]: d for d in reg.specs("agent")}
    assert s["double"]["summary"] == "Twice x."
    assert s["double"]["input_schema"]["properties"]["x"]["type"] == "integer"


@pytest.mark.parametrize("exc,kind", [
    (CapabilityForbidden("x"), "forbidden"), (CapabilityUnavailable("x"), "unavailable"),
    (CapabilityNotFound("x"), "not_found"), (ProviderError("x"), "unavailable"),
    (BudgetExhausted("x"), "budget"), (ZeroDivisionError(), "bug"),
])
def test_error_kinds(exc, kind):
    assert error_kind(exc) == kind
