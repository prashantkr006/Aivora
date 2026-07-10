"""Database round-trip test using a temporary SQLite file."""

from __future__ import annotations

import pandas as pd
import pytest

from aivora.pipeline import database


def test_upsert_and_load(tmp_path):
    db = tmp_path / "test.sqlite"
    database.init_db(db)

    df = pd.DataFrame({
        "datetime": pd.date_range("2024-01-02 09:15", periods=3, freq="5min"),
        "symbol": "NIFTY",
        "spot_open": [22000, 22010, 22020],
        "spot_high": [22010, 22020, 22030],
        "spot_low":  [21990, 22000, 22010],
        "spot_close":[22005, 22015, 22025],
        "fut_open":  [22001, 22011, 22021],
        "fut_high":  [22011, 22021, 22031],
        "fut_low":   [21991, 22001, 22011],
        "fut_close": [22006, 22016, 22026],
        "volume":    [10_000, 12_000, 11_000],
    })
    assert database.upsert_spot_futures(df, db) == 3
    # Re-upserting must be idempotent.
    assert database.upsert_spot_futures(df, db) == 3

    out = database.load_spot_futures(db)
    assert len(out) == 3
    assert database.last_loaded_timestamp("NIFTY", db) == df["datetime"].max()


def test_load_empty(tmp_path):
    db = tmp_path / "empty.sqlite"
    database.init_db(db)
    assert database.load_spot_futures(db).empty
    assert database.load_option_chain(db).empty
    assert database.last_loaded_timestamp("NIFTY", db) is None
