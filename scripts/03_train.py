"""Train one (app, model, seed) combination on the historical training
window and save the resulting policy to results/models/.

Thin shell: same as `python -m minifinrl train ...` (all flags pass through).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # works without pip install

from minifinrl.interfaces.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
