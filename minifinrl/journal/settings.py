"""Where the journal is stored, and its limits."""

import os

from minifinrl.platform.settings import ROOT

# SQLite file by default; any SQLAlchemy URL works, e.g. a Neon Postgres URL
# (postgresql+psycopg://...), which needs the [postgres] extra.
DATABASE_URL = os.environ.get("MINIFINRL_DATABASE_URL") or f"sqlite:///{ROOT / 'data' / 'journal.db'}"

MAX_TRADES_PER_PROFILE = 2000
