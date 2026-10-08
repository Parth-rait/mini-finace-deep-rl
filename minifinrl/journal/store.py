"""
Journal storage with SQLAlchemy Core, so one implementation serves SQLite
(a file, the default) and Postgres (Neon, when deployed).

Two tables. A profile is just an ID and when it was made; every trade row
carries its profile ID, and every read or write is scoped by it, so one ID
can never see another's trades. Tables are created on first use.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    JSON,
    Column,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Engine as SqlEngine

metadata = MetaData()

profiles = Table(
    "profiles", metadata,
    Column("id", String(24), primary_key=True),
    Column("created_at", String(32), nullable=False),
    Column("account_size", Float),  # optional, to show plan risk as a share of the account
)

trades = Table(
    "trades", metadata,
    Column("id", String(32), primary_key=True),
    Column("profile_id", String(24), ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("created_at", String(32), nullable=False),
    Column("updated_at", String(32), nullable=False),
    Column("status", String(16), nullable=False),  # planned | open | closed
    Column("ticker", String(16), nullable=False),
    Column("name", String(200)),
    Column("direction", String(8), nullable=False),  # long | short
    Column("buy_date", String(10)),
    Column("sell_date", String(10)),
    Column("price_paid", Float),
    Column("price_sold", Float),
    Column("quantity", Float),
    Column("reasoning", Text),
    Column("state", JSON),  # what was read from the reasoning when the trade was logged
    Column("result", JSON),  # the latest review of the trade
    # the plan, frozen when saved: what the trade is later checked against
    Column("plan_entry", Float),
    Column("plan_stop", Float),
    Column("plan_target", Float),
    Column("plan_amount", Float),
    Column("plan_horizon", Integer),
    Column("plan", JSON),  # the plan check: risk, odds, range and nudges at the time
)

TRADE_FIELDS = [c.name for c in trades.columns]


def _add_missing_columns(engine: SqlEngine) -> None:
    """Upgrade a journal made by an earlier version: add any column the tables
    gained since (always nullable, so existing rows stay valid)."""
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    with engine.begin() as c:
        for table in metadata.sorted_tables:
            have = {col["name"] for col in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name not in have:
                    c.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {col.name} {col.type.compile(dialect=engine.dialect)}'))


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JournalStore:
    def __init__(self, url: str):
        self.url = url
        self._engine: SqlEngine | None = None
        self._lock = threading.Lock()

    def _db(self) -> SqlEngine:
        with self._lock:
            if self._engine is None:
                kwargs = {}
                if self.url.startswith("sqlite"):
                    path = self.url.split("///", 1)[-1]
                    if path and path != ":memory:":
                        Path(path).parent.mkdir(parents=True, exist_ok=True)
                    kwargs["connect_args"] = {"check_same_thread": False}
                self._engine = create_engine(self.url, future=True, pool_pre_ping=True, **kwargs)
                if self.url.startswith("sqlite"):
                    from sqlalchemy import event

                    @event.listens_for(self._engine, "connect")
                    def _fk_on(dbapi_conn, _):  # SQLite enforces ON DELETE CASCADE only when asked
                        dbapi_conn.execute("PRAGMA foreign_keys=ON")
                metadata.create_all(self._engine)
                _add_missing_columns(self._engine)
            return self._engine

    # ---- profiles -------------------------------------------------------------------------

    def create_profile(self, profile_id: str) -> dict:
        row = {"id": profile_id, "created_at": now()}
        with self._db().begin() as c:
            c.execute(insert(profiles).values(**row))
        return row

    def profile(self, profile_id: str) -> dict | None:
        with self._db().connect() as c:
            r = c.execute(select(profiles).where(profiles.c.id == profile_id)).mappings().first()
        return dict(r) if r else None

    def set_account_size(self, profile_id: str, account_size: float | None) -> dict | None:
        with self._db().begin() as c:
            c.execute(update(profiles).where(profiles.c.id == profile_id).values(account_size=account_size))
        return self.profile(profile_id)

    def delete_profile(self, profile_id: str) -> int:
        """Deletes the profile and all its trades. Returns how many trades went."""
        with self._db().begin() as c:
            n = c.execute(delete(trades).where(trades.c.profile_id == profile_id)).rowcount
            c.execute(delete(profiles).where(profiles.c.id == profile_id))
        return n

    # ---- trades ----------------------------------------------------------------------------

    def count_trades(self, profile_id: str) -> int:
        with self._db().connect() as c:
            return c.execute(select(func.count()).select_from(trades).where(trades.c.profile_id == profile_id)).scalar_one()

    def add_trade(self, row: dict) -> dict:
        stamp = now()
        row = {**row, "created_at": stamp, "updated_at": stamp}
        with self._db().begin() as c:
            c.execute(insert(trades).values(**row))
        return row

    def trades(self, profile_id: str) -> list[dict]:
        q = select(trades).where(trades.c.profile_id == profile_id).order_by(trades.c.created_at.desc(), trades.c.id)
        with self._db().connect() as c:
            return [dict(r) for r in c.execute(q).mappings()]

    def trade(self, profile_id: str, trade_id: str) -> dict | None:
        q = select(trades).where(trades.c.profile_id == profile_id, trades.c.id == trade_id)
        with self._db().connect() as c:
            r = c.execute(q).mappings().first()
        return dict(r) if r else None

    def update_trade(self, profile_id: str, trade_id: str, **fields) -> dict | None:
        fields["updated_at"] = now()
        with self._db().begin() as c:
            c.execute(update(trades).where(trades.c.profile_id == profile_id, trades.c.id == trade_id).values(**fields))
        return self.trade(profile_id, trade_id)

    def delete_trade(self, profile_id: str, trade_id: str) -> int:
        with self._db().begin() as c:
            return c.execute(delete(trades).where(trades.c.profile_id == profile_id, trades.c.id == trade_id)).rowcount
