"""
HTTP API, generated from the capability registry.

    uvicorn minifinrl.interfaces.api:create_app --factory --port 8000
    python -m minifinrl.interfaces.api              # same, on 127.0.0.1:8000
    open http://localhost:8000/                     # the website
    curl -s localhost:8000/api | jq                 # index of routes
    curl -s -X POST localhost:8000/luck-test -H 'content-type: application/json' \\
         -d '{"ticker": "META", "date": "2023-02-01", "horizon_days": 5}' | jq

One route per capability the API may run (read, compute, llm): POST with the
capability's input model as the JSON body, plus GET for read capabilities that
take no input. Batch jobs (fetch, train, backtest) get no route: they never
run inside a request. place_order gets a route that always answers 403, so
the refusal is explicit and testable.

Errors are classified once (capabilities.error_kind) and mapped here:
bad_input 422, forbidden 403, not_found 404, unavailable 503 + Retry-After,
budget 429 + Retry-After. Anything else is a bug: 500 with a request id, the
traceback goes to the log, not the response.

Configuration comes from the environment so the same factory serves any
setup: MINIFINRL_PROFILE (research | live | ci), MINIFINRL_UNIVERSE,
MINIFINRL_WORKSPACE, MINIFINRL_BIAS (rules | aip), PORT.
"""

from __future__ import annotations

import contextvars
import inspect
import logging
import os
import time
import uuid

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import minifinrl
from minifinrl.capabilities import CapabilitySpec, error_kind
from minifinrl.system import System, build_system

log = logging.getLogger("mini_finrl.api")

WEB = Path(__file__).resolve().parent / "web"
STATUS = {"bad_input": 422, "forbidden": 403, "not_found": 404, "unavailable": 503, "budget": 429}
RETRY_AFTER_S = 30
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


def route_path(spec: CapabilitySpec) -> str:
    return "/" + spec.name.replace("_", "-")


def _endpoint(system: System, spec: CapabilitySpec):
    """A handler whose signature FastAPI can read: one body parameter typed
    as the capability's input model, or none for an empty input."""
    registry = system.capabilities

    def run(req: BaseModel):
        try:
            return registry.invoke(spec.name, req, interface="api")
        except Exception as exc:  # classified once, in capabilities.error_kind
            kind = error_kind(exc)
            rid = _request_id.get()
            if kind not in STATUS:
                log.exception("request %s: %s failed", rid, spec.name)
                raise HTTPException(500, {"error": "internal error", "request_id": rid}) from exc
            headers = {"Retry-After": str(RETRY_AFTER_S)} if kind in ("unavailable", "budget") else None
            raise HTTPException(STATUS[kind], {"error": kind, "message": str(exc), "request_id": rid}, headers=headers) from exc

    if spec.input.model_fields:
        def endpoint(req):
            return run(req)

        endpoint.__signature__ = inspect.Signature(
            [inspect.Parameter("req", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=spec.input)]
        )
    else:
        def endpoint():
            return run(spec.input())
    return endpoint


def _refusal(spec: CapabilitySpec):
    def endpoint():
        raise HTTPException(403, {"error": "forbidden", "request_id": _request_id.get(),
                                  "message": f"{spec.name} is never executed by this service"})
    return endpoint


def create_app(system: System | None = None) -> FastAPI:
    """Build the app. Pass a System (tests do); otherwise one is built from
    the MINIFINRL_* environment variables, once, here."""
    if system is None:
        extra = {"bias_backend": os.environ["MINIFINRL_BIAS"]} if os.environ.get("MINIFINRL_BIAS") else {}
        system = build_system(
            os.environ.get("MINIFINRL_PROFILE", "research"),
            universe=os.environ.get("MINIFINRL_UNIVERSE") or None,
            workspace=os.environ.get("MINIFINRL_WORKSPACE") or None,
            **extra,
        )
    app = FastAPI(title="mini-FinRL", version=minifinrl.__version__,
                  description="Paper analysis only: backtests, rank stability, luck test. No order execution.")
    app.state.system = system

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = uuid.uuid4().hex[:12]
        token = _request_id.set(rid)
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            _request_id.reset(token)
        response.headers["X-Request-Id"] = rid
        response.headers["X-Duration-Ms"] = f"{(time.perf_counter() - t0) * 1000:.0f}"
        return response

    routes = []
    for spec in system.capabilities.list("api"):
        path = route_path(spec)
        if spec.effect == "forbidden":
            app.add_api_route(path, _refusal(spec), methods=["POST"], name=spec.name, summary=spec.summary,
                              status_code=403, responses={403: {"description": "always refused"}})
            routes.append({"path": path, "methods": ["POST"], "effect": spec.effect, "summary": spec.summary})
            continue
        methods = ["POST"] + (["GET"] if spec.effect == "read" and not spec.input.model_fields else [])
        for method in methods:  # one route per method, so each gets its own OpenAPI operation id
            app.add_api_route(path, _endpoint(system, spec), methods=[method], name=f"{spec.name}_{method.lower()}",
                              operation_id=f"{spec.name}_{method.lower()}", summary=spec.summary,
                              response_model=spec.output)
        routes.append({"path": path, "methods": methods, "effect": spec.effect, "summary": spec.summary})

    @app.get("/", include_in_schema=False)
    def site():
        return FileResponse(WEB / "index.html")

    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/api", summary="Index of routes")
    def index():
        return {"service": "mini-FinRL", "version": minifinrl.__version__, "profile": system.config.profile,
                "tickers": list(system.config.engine.spec.tickers), "routes": routes,
                "note": "Paper analysis only. Nothing here places orders or gives investment advice."}

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))
