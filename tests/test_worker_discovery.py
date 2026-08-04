"""Which portfolios the worker schedules a tick for.

The bug these pin: the worker only registered a job for portfolios with
master_switch=1.  Turning the switch off is meant to stop new *entries* —
run_user_tick runs the position tracker first and only then returns
"paused (master switch OFF) ... open trade(s) monitored".  With no job
registered that code never ran, so an open live position left behind an
off switch had no stop-loss, no trailing stop and no horizon exit
watching it.
"""

from __future__ import annotations

import importlib
import sqlite3

import pytest


@pytest.fixture()
def worker(tmp_path, monkeypatch):
    """run_worker wired to a throwaway DB with the two tables it reads."""
    db_path = tmp_path / "webapp.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE user_portfolios (
            user_id INTEGER, mode TEXT, master_switch INTEGER
        );
        CREATE TABLE user_trades (
            user_id INTEGER, mode TEXT, trade_id TEXT, exit_time TEXT
        );
        """
    )
    conn.commit()
    conn.close()

    mod = importlib.import_module("scripts.run_worker")
    monkeypatch.setattr(mod.webapp_db, "init_db", lambda: None)
    monkeypatch.setattr(
        mod.webapp_db, "connect", lambda: sqlite3.connect(db_path)
    )
    mod._ACTIVE.clear()
    mod._db = db_path          # convenience handle for the tests
    return mod


def _portfolio(mod, user_id, mode, switch):
    with sqlite3.connect(mod._db) as c:
        c.execute("INSERT INTO user_portfolios VALUES (?,?,?)",
                  (user_id, mode, switch))


def _trade(mod, user_id, mode, trade_id, exit_time=None):
    with sqlite3.connect(mod._db) as c:
        c.execute("INSERT INTO user_trades VALUES (?,?,?,?)",
                  (user_id, mode, trade_id, exit_time))


def test_switch_on_is_scheduled(worker):
    _portfolio(worker, 1, "live", 1)
    assert worker._active_portfolios() == {(1, "live"): True}


def test_switch_off_and_flat_is_not_scheduled(worker):
    _portfolio(worker, 1, "live", 0)
    assert worker._active_portfolios() == {}


def test_switch_off_with_open_position_is_still_scheduled(worker):
    """The whole point — an open position keeps its exits alive."""
    _portfolio(worker, 1, "live", 0)
    _trade(worker, 1, "live", "t1")
    assert worker._active_portfolios() == {(1, "live"): False}


def test_closed_trade_does_not_keep_a_paused_portfolio_alive(worker):
    _portfolio(worker, 1, "live", 0)
    _trade(worker, 1, "live", "t1", exit_time="2026-08-04T14:00:00")
    assert worker._active_portfolios() == {}


def test_open_position_in_one_mode_does_not_wake_the_other(worker):
    _portfolio(worker, 1, "live", 0)
    _portfolio(worker, 1, "paper", 0)
    _trade(worker, 1, "live", "t1")
    assert worker._active_portfolios() == {(1, "live"): False}


def test_another_users_open_position_does_not_wake_mine(worker):
    _portfolio(worker, 1, "live", 0)
    _portfolio(worker, 2, "live", 0)
    _trade(worker, 2, "live", "t1")
    assert worker._active_portfolios() == {(2, "live"): False}


def test_switch_on_wins_over_being_flat(worker):
    """master_switch=1 with no position is 'trading', not 'monitoring'."""
    _portfolio(worker, 1, "live", 1)
    _trade(worker, 1, "live", "t1")
    assert worker._active_portfolios()[(1, "live")] is True


class _Scheduler:
    def __init__(self):
        self.jobs = {}

    def add_job(self, fn, args, trigger, id, **kw):
        self.jobs[id] = args

    def remove_job(self, id):
        del self.jobs[id]


def test_sync_unregisters_only_once_the_position_is_closed(worker):
    """A paused portfolio keeps ticking until its last trade exits."""
    sched = _Scheduler()
    _portfolio(worker, 1, "live", 0)
    _trade(worker, 1, "live", "t1")

    worker._sync(sched)
    assert "tick-1-live" in sched.jobs

    with sqlite3.connect(worker._db) as c:
        c.execute("UPDATE user_trades SET exit_time = '2026-08-04T14:00:00'")
    worker._sync(sched)
    assert sched.jobs == {}
