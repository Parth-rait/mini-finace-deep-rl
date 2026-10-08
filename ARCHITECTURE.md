# Architecture

The code is organised by feature. Each feature is a package with the same
shape, features depend on each other in one direction only, and every
capability the system offers is declared once and shows up in the CLI, the
HTTP API and the website automatically. The rules below are checked by
`tests/test_architecture.py`, so they hold without anyone having to remember
them.

## Layout

```
minifinrl/
  platform/     shared: settings (paths), logging, errors, base schemas, the capability registry
  market/       prices, validated cache, symbol directory, trading calendar, panel and indicators
  regime/       the market regime model (HMM) and the luck test
  research/     deep-RL environments, agents, backtests, walk-forward, experiment log
  sentiment/    bias signals in trading text, and their evaluation
  review/       the trade review
  orders/       declared and always refused (paper analysis only)
  engine.py     builds one service per feature; the system-wide health check
  system.py     composition root: picks adapters (rules or LLM) from the profile
  interfaces/   cli.py, api.py, web/
```

## The shape of a feature

```
<feature>/
  __init__.py   one paragraph: what this feature is responsible for
  service.py    <Feature>Service: its capabilities, as @capability methods (thin)
  schemas.py    the inputs and outputs of those capabilities (pydantic)
  settings.py   its tunables (optional)
  ports.py      interfaces for swappable parts, e.g. a classifier (optional)
  adapters/     implementations that call outside services such as an LLM (optional)
  *.py          the logic itself, as plain functions and classes
```

Services stay thin: they read the request, call the feature's logic and build
the response. The logic lives in ordinary modules that can be tested without
the registry, the API or a model.

## Dependency rules

```
platform  <-  market  <-  regime  <-  research
platform  <-  sentiment
platform  <-  market, regime  <-  review
platform  <-  orders
features  <-  engine.py  <-  system.py  <-  interfaces/
```

1. A feature imports only `platform`, itself, and the features listed for it in
   `FEATURES` in `tests/test_architecture.py`. No cycles.
2. A feature never imports another feature's `service` or `adapters`. When one
   feature needs another at run time (the review needs prices and the regime
   model), the Engine passes the service in.
3. Only `adapters/` packages import LLM libraries (aip). Only `interfaces/`
   imports web libraries. Only `research/` imports the training stack (torch,
   stable-baselines3, gymnasium), and only inside batch methods, so the API and
   website start without it (about 180 MB of memory instead of 340 MB).
4. No module passes 400 lines. When one gets close, split it by responsibility,
   the way the review is split into `intake`, `dates`, `resolve`, `trade` and
   `explain`.

## Recipes

**Add a capability to an existing feature.** Add the input and output models to
the feature's `schemas.py` and a method to its service:

```python
@capability("my_thing", MyThingIn, MyThingOut, effect="read", budget_ms=1000)
def my_thing(self, req: MyThingIn) -> MyThingOut:
    """One sentence: this becomes the CLI help and the API summary."""
```

The CLI flag set, the HTTP route and the agent tool schema are generated from
it. The effect decides where it may run: `read` and `compute` everywhere,
`llm` everywhere (it may cost money), `batch` on the CLI only, `forbidden`
never.

**Add a feature.** Create the package with the shape above, add its
dependencies to `FEATURES` in `tests/test_architecture.py`, and add one line
in `Engine.__init__` constructing its service (passing any services it needs).
Nothing else changes.

**Swap a model** (for example Gemini for Ollama). Write a new class in the
feature's `adapters/` that satisfies its `ports.py` protocol, and choose it in
`system.py`. Feature logic and services stay as they are.

**Tune a setting.** Each feature's `settings.py` holds its own tunables; paths
shared by everything are in `platform/settings.py`.

## Errors

A capability either returns its output or raises. Interfaces map the error
kind to their own vocabulary (HTTP status, exit code) in one place,
`platform.capabilities.error_kind`. A feature tags its own exception classes
with `error_kind = "unavailable"` (or `budget`, `not_found`, `bad_input`), so
the interfaces never need to know about them.
