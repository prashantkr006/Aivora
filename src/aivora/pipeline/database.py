"""Thin SQLite persistence layer.

We use SQLite because the dataset is small (years of 5-min
candles for two indices fit comfortably in <500 MB) and a single
file is easier to ship around than a server.

Two tables:

* ``spot_futures`` — one row per (symbol, 5-min timestamp).
* ``options_chain`` — one row per (symbol, datetime, strike, type).

Both tables enforce uniqueness on the natural key so upserts are
idempotent.  Use :func:`upsert_spot_futures` / :func:`upsert_option_chain`
for write paths; they handle dedup automatically.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

from ..utils.config import get_config
from ..utils.logger import get_logger

log = get_logger(__name__)


SCHEMA_SPOT_FUTURES = """
CREATE TABLE IF NOT EXISTS spot_futures (
    datetime    TEXT    NOT NULL,
    symbol      TEXT    NOT NULL,
    spot_open   REAL,
    spot_high   REAL,
    spot_low    REAL,
    spot_close  REAL,
    fut_open    REAL,
    fut_high    REAL,
    fut_low     REAL,
    fut_close   REAL,
    volume      REAL,
    is_filled   INTEGER DEFAULT 0,
    PRIMARY KEY (symbol, datetime)
);
"""

SCHEMA_OPTIONS = """
CREATE TABLE IF NOT EXISTS options_chain (
    datetime    TEXT    NOT NULL,
    symbol      TEXT    NOT NULL,
    strike      REAL    NOT NULL,
    type        TEXT    NOT NULL CHECK(type IN ('CE','PE')),
    ltp         REAL,
    oi          REAL,
    iv          REAL,
    PRIMARY KEY (symbol, datetime, strike, type)
);
"""

SCHEMA_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_spot_dt    ON spot_futures(datetime);",
    "CREATE INDEX IF NOT EXISTS idx_opt_dt     ON options_chain(datetime);",
    "CREATE INDEX IF NOT EXISTS idx_opt_symbol ON options_chain(symbol);",
]


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with safe defaults.

    WAL mode lets readers run while the pipeline writes, which
    matters once we add an intraday snapshot loop.
    """
    cfg = get_config()
    db_path = Path(db_path) if db_path else cfg.paths["sqlite_path"]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | None = None) -> None:
    """Create tables and indexes if they don't exist."""
    with connect(db_path) as conn:
        conn.execute(SCHEMA_SPOT_FUTURES)
        conn.execute(SCHEMA_OPTIONS)
        for stmt in SCHEMA_INDEXES:
            conn.execute(stmt)
    log.info("Database initialised at %s", db_path or get_config().paths["sqlite_path"])


def upsert_spot_futures(df: pd.DataFrame, db_path: Path | None = None) -> int:
    """INSERT OR REPLACE on the canonical spot_futures schema.

    Returns the number of rows written.
    """
    if df.empty:
        return 0
    cols = [
        "datetime", "symbol", "spot_open", "spot_high", "spot_low", "spot_close",
        "fut_open", "fut_high", "fut_low", "fut_close", "volume", "is_filled",
    ]
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = None
    out = out[cols]
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    out["is_filled"] = out["is_filled"].fillna(False).astype(int)
    # sqlite3 can bind None/NaN but not pandas' NA sentinel (nor NaT) —
    # normalise every missing-value flavour to plain None before insert.
    out = out.astype(object).where(pd.notnull(out), None)

    sql = f"""
        INSERT OR REPLACE INTO spot_futures
            ({', '.join(cols)})
        VALUES ({', '.join(['?'] * len(cols))})
    """
    with connect(db_path) as conn:
        conn.executemany(sql, out.itertuples(index=False, name=None))
    log.info("upsert_spot_futures: wrote %d rows", len(out))
    return len(out)


def upsert_option_chain(df: pd.DataFrame, db_path: Path | None = None) -> int:
    """Persist ATM (or any) option snapshots.

    Expects ``df`` with columns ``datetime, symbol, strike, type,
    ltp, oi, iv`` — exactly what the historical CSV decomposes to
    after we split CE / PE rows.
    """
    if df.empty:
        return 0
    cols = ["datetime", "symbol", "strike", "type", "ltp", "oi", "iv"]
    out = df[cols].copy()
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    out = out.astype(object).where(pd.notnull(out), None)
    sql = f"""
        INSERT OR REPLACE INTO options_chain
            ({', '.join(cols)})
        VALUES ({', '.join(['?'] * len(cols))})
    """
    with connect(db_path) as conn:
        conn.executemany(sql, out.itertuples(index=False, name=None))
    log.info("upsert_option_chain: wrote %d rows", len(out))
    return len(out)


def last_loaded_timestamp(symbol: str, db_path: Path | None = None) -> pd.Timestamp | None:
    """Return the most recent timestamp stored for ``symbol``.

    Used by the pipeline to skip already-ingested ranges and keep
    daily updates cheap.
    """
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT MAX(datetime) FROM spot_futures WHERE symbol = ?", (symbol,)
        ).fetchone()
    if not row or row[0] is None:
        return None
    return pd.to_datetime(row[0])


def load_spot_futures(db_path: Path | None = None) -> pd.DataFrame:
    """Read the entire spot_futures table back into a dataframe."""
    with connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM spot_futures", conn)
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values(["symbol", "datetime"]).reset_index(drop=True)


def load_option_chain(db_path: Path | None = None) -> pd.DataFrame:
    """Read the options table back, pivoted into CE / PE columns.

    The wide layout matches what the feature engineer expects.
    """
    with connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM options_chain", conn)
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"])
    wide = df.pivot_table(
        index=["datetime", "symbol"],
        columns="type",
        values=["ltp", "oi", "iv"],
        aggfunc="last",
    )
    wide.columns = [f"{t.lower()}_{v}" for v, t in wide.columns]
    return wide.reset_index()


def load_spot_futures_since(
    cutoff: pd.Timestamp,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Read only ``spot_futures`` rows with ``datetime >= cutoff``.

    Used by the live tick to avoid loading the full 184k-row table on
    every 5-minute refresh. A 45+ calendar-day cutoff is enough for
    every rolling-window feature to compute identical values at the
    latest row (the longest lookback is ``vol_regime_pct``'s 1500-row
    window ≈ 20 trading days).
    """
    cutoff_str = pd.Timestamp(cutoff).isoformat()
    with connect(db_path) as conn:
        df = pd.read_sql(
            "SELECT * FROM spot_futures WHERE datetime >= ?",
            conn, params=(cutoff_str,),
        )
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values(["symbol", "datetime"]).reset_index(drop=True)


def load_option_chain_since(
    cutoff: pd.Timestamp,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Same shape as :func:`load_option_chain`, but only rows with
    ``datetime >= cutoff``. Used by the live tick alongside
    :func:`load_spot_futures_since`.
    """
    cutoff_str = pd.Timestamp(cutoff).isoformat()
    with connect(db_path) as conn:
        df = pd.read_sql(
            "SELECT * FROM options_chain WHERE datetime >= ?",
            conn, params=(cutoff_str,),
        )
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"])
    wide = df.pivot_table(
        index=["datetime", "symbol"],
        columns="type",
        values=["ltp", "oi", "iv"],
        aggfunc="last",
    )
    wide.columns = [f"{t.lower()}_{v}" for v, t in wide.columns]
    return wide.reset_index()
