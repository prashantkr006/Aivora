"""Web-app SQLite schema.

Kept in a separate database file (``data/db/webapp.sqlite``) so the
existing ``aivora.sqlite`` (which the model + backtest read) is
untouched.  A single-file store is fine at this scale — expected
concurrency is at most a handful of Streamlit sessions.

Every write path is idempotent and every read path is scoped by
``user_id``.  The user-scoping is enforced at the SQL layer here
so the higher-level modules physically can't leak another user's
rows.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ..utils.config import get_config
from ..utils.logger import get_logger

log = get_logger(__name__)


# =============================================================
#  Schema DDL
# =============================================================
SCHEMA_USERS = """
CREATE TABLE IF NOT EXISTS users (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    email          TEXT    UNIQUE NOT NULL,
    password_hash  TEXT    NOT NULL,
    display_name   TEXT,
    is_admin       INTEGER DEFAULT 0,
    created_at     TEXT    NOT NULL,
    last_login     TEXT
);
"""

SCHEMA_USER_BROKERS = """
CREATE TABLE IF NOT EXISTS user_brokers (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id                INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    broker                 TEXT    NOT NULL CHECK(broker IN ('ZERODHA','DHAN')),
    client_id              TEXT,
    api_key_enc            TEXT,
    api_secret_enc         TEXT,
    access_token_enc       TEXT,
    totp_secret_enc        TEXT,
    password_enc           TEXT,
    token_updated_at       TEXT,
    is_active              INTEGER DEFAULT 1,
    UNIQUE(user_id, broker)
);
"""

SCHEMA_USER_PORTFOLIOS = """
CREATE TABLE IF NOT EXISTS user_portfolios (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    mode             TEXT    NOT NULL CHECK(mode IN ('paper','live')),
    initial_capital  REAL    NOT NULL,
    current_capital  REAL    NOT NULL,
    peak_capital     REAL    NOT NULL,
    master_switch    INTEGER DEFAULT 0,
    settings_json    TEXT,
    last_data_update TEXT,
    last_signal_time TEXT,
    UNIQUE(user_id, mode)
);
"""

SCHEMA_USER_TRADES = """
CREATE TABLE IF NOT EXISTS user_trades (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id            INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    mode               TEXT    NOT NULL CHECK(mode IN ('paper','live')),
    trade_id           TEXT    NOT NULL,
    entry_time         TEXT    NOT NULL,
    exit_time          TEXT,
    symbol             TEXT    NOT NULL,
    side               TEXT    NOT NULL,
    strike             REAL,
    lots               INTEGER,
    lot_size           INTEGER,
    entry_premium      REAL,
    exit_premium       REAL,
    current_premium    REAL,
    entry_spot         REAL,
    exit_spot          REAL,
    entry_order_id     TEXT,
    exit_order_id      TEXT,
    horizon_close_time TEXT,
    gross_pnl          REAL,
    costs              REAL,
    realized_pnl       REAL,
    unrealized_pnl     REAL,
    exit_reason        TEXT,
    UNIQUE(user_id, trade_id)
);
"""

SCHEMA_USER_EVENTS = """
CREATE TABLE IF NOT EXISTS user_events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ts       TEXT    NOT NULL,
    level    TEXT    NOT NULL,
    msg      TEXT    NOT NULL
);
"""

SCHEMA_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_user_brokers_user ON user_brokers(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_user_trades_user  ON user_trades(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_user_trades_mode  ON user_trades(user_id, mode);",
    "CREATE INDEX IF NOT EXISTS idx_user_events_user  ON user_events(user_id, ts);",
]


# =============================================================
#  Connection
# =============================================================
def default_db_path() -> Path:
    """Where the webapp DB lives, relative to the repo."""
    return get_config().paths["db_dir"] / "webapp.sqlite"


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """SQLite connection with sane defaults for a multi-session UI."""
    db_path = Path(db_path) if db_path else default_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


_INIT_DONE: set = set()


def init_db(db_path: Path | None = None) -> None:
    """Create every table + index if they don't exist.

    Idempotent AND quiet: the first call for a given DB path logs
    once at INFO; subsequent calls in the same process are silent.
    This keeps the terminal readable — Streamlit's auto-refresh
    used to spam this line every 30 seconds.
    """
    resolved = str(db_path or default_db_path())
    with connect(db_path) as conn:
        for stmt in (SCHEMA_USERS, SCHEMA_USER_BROKERS, SCHEMA_USER_PORTFOLIOS,
                     SCHEMA_USER_TRADES, SCHEMA_USER_EVENTS):
            conn.execute(stmt)
        for idx in SCHEMA_INDEXES:
            conn.execute(idx)
    if resolved not in _INIT_DONE:
        log.info("webapp DB ready at %s", resolved)
        _INIT_DONE.add(resolved)
