"""Where things live on disk, and I/O limits shared by every feature.
Feature-specific settings live in each feature's own settings.py."""

import os
from pathlib import Path

# Where data/ and results/ live. In a checkout: the repo root (the folder
# holding pyproject.toml, 2 levels above this file's package). Installed
# from git, this file sits in site-packages, so set MINIFINRL_HOME; without
# it the current directory is used rather than writing into site-packages.
_CHECKOUT_ROOT = Path(__file__).resolve().parents[2]
if os.environ.get("MINIFINRL_HOME"):
    ROOT = Path(os.environ["MINIFINRL_HOME"]).resolve()
elif (_CHECKOUT_ROOT / "pyproject.toml").exists() or (_CHECKOUT_ROOT / ".git").exists():
    ROOT = _CHECKOUT_ROOT
else:
    ROOT = Path.cwd()
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SYNTHETIC = ROOT / "data" / "synthetic"
RESULTS = ROOT / "results"
MODEL_DIR = RESULTS / "models"
FIGURE_DIR = RESULTS / "figures"
LOG_DIR = RESULTS / "logs"

HTTP_TIMEOUT_S = 20
HTTP_RETRIES = 3
