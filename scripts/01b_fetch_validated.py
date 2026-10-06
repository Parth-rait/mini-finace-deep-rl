"""Fetch through the incremental store, validate, and write a provenance
manifest. The new-data-layer counterpart of 01_download_data.py, which is
left as-is so the existing results stay reproducible.

Thin shell: same as `python -m minifinrl fetch-data ...` (all flags pass through).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # works without pip install

from minifinrl.interfaces.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["fetch_data", *sys.argv[1:]]))
