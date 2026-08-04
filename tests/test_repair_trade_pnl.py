"""Tests for the trade-repair script.

This edits financial records, so the tests care less about the happy path
than about the script refusing to act: it must never rewrite a row it is
not certain about, and must never write at all without --apply.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

SCHEMA = """
CREATE TABLE user_trades (
    id INTEGER PRIMARY KEY, user_id INTEGER, mode TEXT, trade_id TEXT,
    entry_time TEXT, exit_time TEXT, symbol TEXT, side TEXT, strike REAL,
    lots INTEGER, lot_size INTEGER, entry_premium REAL, exit_premium REAL,
    current_premium REAL, entry_spot REAL, exit_spot REAL,
    entry_order_id TEXT, exit_order_id TEXT, horizon_close_time TEXT,
    gross_pnl REAL, costs REAL, realized_pnl REAL, unrealized_pnl REAL,
    exit_reason TEXT, tradingsymbol TEXT
);
CREATE TABLE user_portfolios (
    user_id INTEGER, mode TEXT, initial_capital REAL, current_capital REAL,
    peak_capital REAL, master_switch INTEGER
);
CREATE TABLE user_events (
    id INTEGER PRIMARY KEY, user_id INTEGER, mode TEXT, ts TEXT,
    level TEXT, msg TEXT
);
"""


@pytest.fixture()
def env(tmp_path):
    """A repo-shaped sandbox: config + an isolated DB the script will find."""
    data = tmp_path / "data"
    (data / "db").mkdir(parents=True)
    db = data / "db" / "webapp.sqlite"

    conn = sqlite3.connect(db)
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT INTO user_portfolios VALUES (27,'live',51000.0,51000.0,51000.0,0)"
    )
    # One trade, force-closed by the emergency square-off at a bogus mark.
    conn.execute(
        "INSERT INTO user_trades (id,user_id,mode,trade_id,entry_time,exit_time,"
        "symbol,side,strike,lots,lot_size,entry_premium,exit_premium,"
        "gross_pnl,costs,realized_pnl,exit_reason) VALUES "
        "(1,27,'live','t1','2026-08-04T10:00:00','2026-08-04T14:00:00',"
        "'BANKNIFTY','CE',57700,1,30,736.05,722.00,-421.50,0.0,-421.50,'emergency')"
    )
    conn.commit()
    conn.close()
    return {"tmp": tmp_path, "db": db}


def _run(env, spec, *extra):
    spec_path = env["tmp"] / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-m", "scripts.repair_trade_pnl",
         "--user", "27", "--mode", "live", "--date", "2026-08-04",
         "--spec", str(spec_path), "--db", str(env["db"]), *extra],
        cwd=REPO, capture_output=True, text=True,
    )


def _rows(env):
    c = sqlite3.connect(env["db"])
    c.row_factory = sqlite3.Row
    r = c.execute("SELECT * FROM user_trades WHERE id=1").fetchone()
    p = c.execute("SELECT * FROM user_portfolios").fetchone()
    c.close()
    return r, p


GOOD = [{"symbol": "BANKNIFTY", "side": "CE", "strike": 57700, "pnl": 3139.50}]


# -------------------------------------------------------------------
#  Refusing to act
# -------------------------------------------------------------------
def test_dry_run_writes_nothing(env):
    r = _run(env, GOOD)
    assert r.returncode == 0, r.stderr
    assert "DRY RUN" in r.stdout
    row, pf = _rows(env)
    assert row["realized_pnl"] == -421.50
    assert pf["current_capital"] == 51000.0


def test_a_spec_matching_nothing_stops_everything(env):
    r = _run(env, [{"symbol": "NIFTY", "side": "PE", "strike": 24650,
                    "pnl": -107.25}])
    assert r.returncode == 1
    assert "matched 0 trades" in r.stdout


def test_one_bad_spec_blocks_the_whole_batch(env):
    """All-or-nothing: a partial repair leaves the book half wrong."""
    r = _run(env, GOOD + [{"symbol": "NIFTY", "side": "PE",
                           "strike": 24650, "pnl": -107.25}], "--apply")
    assert r.returncode == 1
    row, _ = _rows(env)
    assert row["realized_pnl"] == -421.50, "nothing may be written"


def test_an_open_trade_is_never_rewritten(env):
    sqlite3.connect(env["db"]).execute(
        "UPDATE user_trades SET exit_time=NULL"
    ).connection.commit()
    r = _run(env, GOOD, "--apply")
    assert r.returncode == 1
    assert "still open" in r.stdout


def test_a_trade_closed_normally_is_protected(env):
    """Only the rows damaged by the emergency close may be touched."""
    con = sqlite3.connect(env["db"])
    con.execute("UPDATE user_trades SET exit_reason='take_profit'")
    con.commit()
    r = _run(env, GOOD, "--apply")
    assert r.returncode == 1
    assert "expected one of" in r.stdout


# -------------------------------------------------------------------
#  The repair itself
# -------------------------------------------------------------------
def test_apply_derives_the_exit_price_from_the_brokers_pnl(env):
    r = _run(env, GOOD, "--apply")
    assert r.returncode == 0, r.stdout + r.stderr
    row, _ = _rows(env)
    # 736.05 + 3139.50/30
    assert row["exit_premium"] == pytest.approx(840.70)
    assert row["gross_pnl"] == pytest.approx(3139.50)
    assert row["exit_reason"] == "closed_at_broker"


def test_slippage_is_not_charged_by_default(env):
    """Slippage models bid-ask crossing in the backtest.  A real fill already
    happened at a real price, so charging it again invents a cost."""
    _run(env, GOOD, "--apply")
    without = _rows(env)[0]["costs"]
    assert without > 0
    assert _rows(env)[0]["realized_pnl"] == pytest.approx(3139.50 - without)

    # Same trade with the model switched on: the gap is exactly the slippage.
    con = sqlite3.connect(env["db"])
    con.execute("UPDATE user_trades SET exit_reason='emergency'")
    con.commit()
    con.close()
    _run(env, GOOD, "--include-slippage", "--apply")
    with_slip = _rows(env)[0]["costs"]
    turnover = (736.05 + 840.70) * 30
    assert with_slip - without == pytest.approx(turnover * 0.001, rel=1e-3)


def test_capital_is_rebuilt_from_initial_plus_realised(env):
    _run(env, GOOD, "--apply")
    row, pf = _rows(env)
    assert pf["current_capital"] == pytest.approx(51000.0 + row["realized_pnl"])


def test_the_database_is_backed_up_before_writing(env):
    _run(env, GOOD, "--apply")
    backups = list(env["db"].parent.glob("webapp.backup-*.sqlite"))
    assert len(backups) == 1
    old = sqlite3.connect(backups[0]).execute(
        "SELECT realized_pnl FROM user_trades WHERE id=1"
    ).fetchone()[0]
    assert old == -421.50, "the backup must hold the pre-repair values"


def test_net_mode_treats_the_figure_as_already_net_of_charges(env):
    _run(env, GOOD, "--pnl", "net", "--apply")
    row, _ = _rows(env)
    assert row["realized_pnl"] == pytest.approx(3139.50, abs=1.0)
    assert row["gross_pnl"] > 3139.50
