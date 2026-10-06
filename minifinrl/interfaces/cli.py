"""
Command line, generated from the capability registry.

    python -m minifinrl list
    python -m minifinrl train --app trading --model ppo --seed 0 --timesteps 5000
    python -m minifinrl rank-stability --app trading
    python -m minifinrl health --profile ci

Every capability the CLI may run (read, compute, llm, batch) is a
subcommand; its flags come from the capability's input model, so a new
Engine method is on the command line with no code here. Output is JSON,
except where a renderer below reproduces a script's classic printout.

Exit codes: 0 ok · 1 ran, but the result reports ok=false (e.g. data failed
validation) · 2 a dependency is unavailable (provider down, no data/results)
· 64 bad input · 66 not found · 75 budget exhausted · 77 forbidden.
Bugs are not caught: you get the traceback.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
import typing
from typing import Any, Callable, Literal

import pandas as pd
from pydantic import BaseModel

from minifinrl.capabilities import error_kind
from minifinrl.system import PROFILES, build_system

EXIT = {"bad_input": 64, "not_found": 66, "unavailable": 2, "budget": 75, "forbidden": 77}


# ---- flags from a pydantic model ------------------------------------------------------------


def _unwrap_optional(ann: Any) -> Any:
    if typing.get_origin(ann) in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return ann


def _add_flags(parser: argparse.ArgumentParser, model: type[BaseModel]) -> None:
    for name, f in model.model_fields.items():
        flag = "--" + name.replace("_", "-")
        ann = _unwrap_optional(f.annotation)
        origin = typing.get_origin(ann)
        kw: dict[str, Any] = {"dest": name, "help": f.description}
        if ann is bool:
            parser.add_argument(flag, action="store_true", **kw)
            continue
        if origin is Literal:
            kw["choices"] = list(typing.get_args(ann))
        elif origin is list:
            kw["nargs"] = "+"
            item = typing.get_args(ann)[0]
            if typing.get_origin(item) is Literal:  # e.g. list[Literal["trading", "portfolio"]]
                kw["choices"] = list(typing.get_args(item))
            else:
                kw["type"] = item
        elif ann in (int, float, str):
            kw["type"] = ann
        if f.is_required():
            kw["required"] = True
        else:
            kw["default"] = argparse.SUPPRESS  # leave it to the model's default
        parser.add_argument(flag, **kw)


# ---- renderers: classic printouts for capabilities that had a script ------------------------


def _render_report(out: BaseModel) -> None:
    """Byte-for-byte what scripts/05_report.py used to print."""
    d = out.model_dump()
    if d["collapsed"]:
        print("\nWARNING: collapsed policies (one distinct action every step; metrics reflect a fixed rule, not learning):")
        print(pd.DataFrame(d["collapsed"]).to_string(index=False))
    for app in d["apps"]:
        print(f"\n=== {app['app']} ===")
        if app["historical"] is not None:
            print("historical sharpe by model (median over seeds):")
            print(pd.DataFrame(app["historical"]).to_string(index=False))
        if app["synthetic"] is not None:
            print("\nsynthetic sharpe by model (median over seeds & paths):")
            print(pd.DataFrame(app["synthetic"]).to_string(index=False))
            print("\nsynthetic path win rate by model:")
            print(pd.DataFrame(app["win_rate"]).to_string(index=False))


RENDERERS: dict[str, Callable[[BaseModel], None]] = {"report": _render_report}


# ---- entry point -------------------------------------------------------------------------------


class _Parser(argparse.ArgumentParser):
    """argparse exits 2 on bad flags, which would read as 'unavailable'.
    Subparsers inherit this class, so every usage error is 64."""

    def error(self, message: str):
        self.print_usage(sys.stderr)
        self.exit(EXIT["bad_input"], f"{self.prog}: error: {message}\n")


def _parser(registry) -> argparse.ArgumentParser:
    p = _Parser(prog="minifinrl", description="mini-FinRL capabilities")
    p.add_argument("--profile", choices=sorted(PROFILES), default="research")
    p.add_argument("--json", action="store_true", help="always print JSON, even where a renderer exists")
    p.add_argument("--universe", default=None, help="ticker universe from configs/tickers.UNIVERSES")
    p.add_argument("--workspace", default=None, help="experiment id: isolate models/paths/results under results/experiments/<id>/")
    sub = p.add_subparsers(dest="capability", required=True, metavar="CAPABILITY")
    sub.add_parser("list", help="list every capability and its effect class")
    for spec in registry.list("cli"):
        if spec.effect == "forbidden":
            continue
        sp = sub.add_parser(spec.name.replace("_", "-"), aliases=[spec.name], help=spec.summary)
        sp.add_argument("--profile", choices=sorted(PROFILES), default=argparse.SUPPRESS)
        sp.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
        sp.add_argument("--universe", default=argparse.SUPPRESS)
        sp.add_argument("--workspace", default=argparse.SUPPRESS)
        _add_flags(sp, spec.input)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # profile must be known before the system (and so the parser) is built
    pre = _Parser(add_help=False)
    pre.add_argument("--profile", choices=sorted(PROFILES), default="research")
    pre.add_argument("--universe", default=None)
    pre.add_argument("--workspace", default=None)
    known = pre.parse_known_args(argv)[0]

    system = build_system(known.profile, universe=known.universe, workspace=known.workspace)
    registry = system.capabilities
    args = vars(_parser(registry).parse_args(argv))
    name = args.pop("capability").replace("-", "_")
    as_json = args.pop("json", False)
    for k in ("profile", "universe", "workspace"):
        args.pop(k, None)

    if name == "list":
        for s in registry.list():
            print(f"{s.name:18s} {s.effect:9s} {s.summary}")
        return 0

    try:
        out = registry.invoke(name, args, interface="cli")
    except Exception as exc:  # classified once, in capabilities.error_kind
        kind = error_kind(exc)
        if kind == "bug":
            raise
        print(f"{name}: {kind}: {exc}", file=sys.stderr)
        return EXIT[kind]

    if name in RENDERERS and not as_json:
        RENDERERS[name](out)
    else:
        print(json.dumps(out.model_dump(mode="json"), indent=2))
    return 1 if getattr(out, "ok", True) is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
