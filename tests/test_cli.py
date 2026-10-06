"""The generated CLI: flags from schemas, exit codes, the report renderer."""

from __future__ import annotations

import json

import pytest

import minifinrl.interfaces.cli as cli
from minifinrl.capabilities import CapabilityRegistry
from minifinrl.system import System, SystemConfig


@pytest.fixture
def rw_cli(monkeypatch, rw_engine):
    system = System(SystemConfig.from_profile("research"), rw_engine, CapabilityRegistry.from_engine(rw_engine))
    monkeypatch.setattr(cli, "build_system", lambda profile, **kw: system)
    return system


def test_list(capsys):
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "place_order" in out and "forbidden" in out and "train" in out


def test_bad_input_is_64(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["rank-stability", "--app", "nope"])
    assert e.value.code == 64
    assert cli.main(["train", "--app", "trading", "--model", "ppo", "--seed", "0", "--timesteps", "5"]) == 64


def test_forbidden_has_no_subcommand():
    with pytest.raises(SystemExit) as e:
        cli.main(["place-order", "--ticker", "A", "--side", "buy", "--quantity", "1"])
    assert e.value.code == 64


def test_unavailable_is_2(rw_cli, capsys):
    assert cli.main(["rank-stability", "--app", "trading"]) == 2
    assert "unavailable" in capsys.readouterr().err


def test_json_output_and_list_flags(rw_cli, capsys):
    assert cli.main(["classify-biases", "--text", "The whole sub is buying, all in."]) == 0
    out = json.loads(capsys.readouterr().out)
    assert {s["label"] for s in out["signals"]} == {"herding", "overconfidence"}


def test_ok_false_exits_1(rw_cli, capsys):
    # live mode, data ending a year before 'today' (2027-01-01) -> stale -> invalid
    assert cli.main(["fetch-data", "--mode", "live", "--end", "2026-01-01"]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_snake_case_alias(rw_cli, capsys):
    assert cli.main(["fetch_data"]) == 0


def test_list_of_choices_flag(rw_cli, capsys, monkeypatch):
    """--apps trading portfolio: a list of Literal values (failed before the fix)."""
    seen = {}
    monkeypatch.setattr(rw_cli.capabilities, "invoke", lambda name, args, interface: seen.update(args) or (_ for _ in ()).throw(SystemExit(0)))
    with pytest.raises(SystemExit):
        cli.main(["walk-forward", "--apps", "trading", "portfolio", "--seeds", "0", "1"])
    assert seen["apps"] == ["trading", "portfolio"] and seen["seeds"] == [0, 1]
    with pytest.raises(SystemExit) as e:
        cli.main(["walk-forward", "--apps", "crypto"])
    assert e.value.code == 64
