"""Turn raw backtest rows into the rank-stability summary this project
exists to produce: does the model that wins on real history keep winning
across synthetic paths, or was it luck?

Thin shell: same as `python -m minifinrl report ...` (all flags pass through).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # works without pip install

from minifinrl.interfaces.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["report", *sys.argv[1:]]))
