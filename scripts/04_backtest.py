"""Evaluate every saved (app, model, seed) combination, plus a fixed
non-RL baseline per app (buy-and-hold for trading, equal-weight for
portfolio allocation), against the historical test window and/or the
synthetic paths. Writes one row of metrics per (app, model, seed, path)
to results/logs/backtest_results.csv.

Thin shell: same as `python -m minifinrl run-backtests ...` (all flags pass through).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # works without pip install

from minifinrl.interfaces.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["run_backtests", *sys.argv[1:]]))
