"""Compute features on the full historical panel, cache it, then fit the
regime HMM on it and sample synthetic paths to data/synthetic/.

Thin shell: same as `python -m minifinrl fit-hmm ...` (all flags pass through).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # works without pip install

from minifinrl.interfaces.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["fit_hmm", *sys.argv[1:]]))
