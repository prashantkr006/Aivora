"""Tests for restating paper trades at the real lot size.

Paper took its contract size from config.yaml, which said NIFTY 75 and
BANKNIFTY 15 long after NSE had moved to 65 and 30.  So every paper P&L is
the P&L of a position that could not have been held.

As with the other repair script, these tests care most about it refusing
to act: it edits financial records and must never write without --apply,
never touch a trade already sized correctly, and never touch live.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
from tests.test_repair_trade_pnl import SCHEMA  # noqa: E402


@pytest.fixture()
def env(tmp_path):
    db = tmp_path / "webapp.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT INTO user_portfolios "
        "(user_id, mode, initial_capital, current_capital, peak_capital, master_switch) "
        "VALUES (27,'paper',100000.0,101000.0,101000.0,1)"
    )
    rows = [
        # (id, mode, symbol, lot_size, entry, exit, gross, costs, realized, exit_time)
        (1, 'paper', 'BANKNIFTY', 15, 700.0, 750.0, 750.0, 100.0, 650.0,
         '2026-07-08T11:00:00'),
        (2, 'paper', 'NIFTY', 75, 100.0, 90.0, -750.0, 60.0, -810.0,
         '2026-07-09T11:00:00'),
        (3, 'paper', 'NIFTY', 65, 50.0, 55.0, 325.0, 20.0, 305.0,
         '2026-07-10T11:00:00'),          # already correct
        (4, 'paper', 'BANKNIFTY', 15, 600.0, None, None, None, None, None),  # open
    ]
    for i, mode, sym, lot, entry, exit_, gross, costs, real, xt in rows:
        conn.execute(
            "INSERT INTO user_trades (id,user_id,mode,trade_id,entry_time,exit_time,"
            "symbol,side,strike,lots,lot_size,entry_premium,exit_premium,"
            "current_premium,gross_pnl,costs,realized_pnl,unrealized_pnl,exit_reason) "
            "VALUES (?,27,?,?,'2026-07-08T09:30:00',?,?,'CE',100,1,?,?,?,?,?,?,?,0.0,'x')",
            (i, mode, f"t{i}", xt, sym, lot, entry, exit_,
             exit_ if exit_ is not None else entry, gross, costs, real),
        )
    # A LIVE trade with the same stale-looking size — must never be touched.
    conn.execute(
        "INSERT INTO user_trades (id,user_id,mode,trade_id,entry_time,exit_time,"
        "symbol,side,strike,lots,lot_size,entry_premium,exit_premium,"
        "gross_pnl,costs,realized_pnl,exit_reason) VALUES "
        "(9,27,'live','t9','2026-08-04T09:15:00','2026-08-04T10:00:00',"
        "'BANKNIFTY','CE',57700,1,30,736.05,840.70,3139.5,62.74,3076.76,'x')"
    )
    conn.commit()
    conn.close()
    return {"db": db}


def _run(env, *extra):
    return subprocess.run(
        [sys.executable, "-m", "scripts.repair_lot_sizes",
         "--user", "27", "--mode", "paper", "--db", str(env["db"]), *extra],
        cwd=REPO, capture_output=True, text=True,
    )


def _trade(env, tid):
    c = sqlite3.connect(env["db"])
    c.row_factory = sqlite3.Row
    r = c.execute("SELECT * FROM user_trades WHERE id=?", (tid,)).fetchone()
    c.close()
    return r


def _capital(env):
    c = sqlite3.connect(env["db"])
    v = c.execute(
        "SELECT current_capital FROM user_portfolios WHERE mode='paper'"
    ).fetchone()[0]
    c.close()
    return v


# -------------------------------------------------------------------
#  Refusing to act
# -------------------------------------------------------------------
def test_dry_run_writes_nothing(env):
    r = _run(env)
    assert r.returncode == 0, r.stderr
    assert "DRY RUN" in r.stdout
    assert _trade(env, 1)["lot_size"] == 15
    assert _capital(env) == 101000.0


def test_dry_run_and_apply_together_is_refused(env):
    assert _run(env, "--dry-run", "--apply").returncode == 2
    assert _trade(env, 1)["lot_size"] == 15


def test_live_trades_are_never_touched(env):
    """Live always read the size from Kite — it was never wrong."""
    _run(env, "--apply")
    live = _trade(env, 9)
    assert live["lot_size"] == 30
    assert live["realized_pnl"] == 3076.76


def test_a_trade_already_at_the_right_size_is_left_alone(env):
    before = dict(_trade(env, 3))
    _run(env, "--apply")
    assert dict(_trade(env, 3)) == before


def test_open_trades_are_skipped_by_default(env):
    _run(env, "--apply")
    assert _trade(env, 4)["lot_size"] == 15
    assert "--include-open" in _run(env).stdout


# -------------------------------------------------------------------
#  The restatement
# -------------------------------------------------------------------
def test_banknifty_pnl_doubles(env):
    """15 -> 30 on the same prices."""
    _run(env, "--apply", "--keep-costs")
    t = _trade(env, 1)
    assert t["lot_size"] == 30
    assert t["gross_pnl"] == pytest.approx((750.0 - 700.0) * 1 * 30)
    assert t["realized_pnl"] == pytest.approx(1500.0 - 100.0)


def test_nifty_loss_shrinks(env):
    """75 -> 65 makes the same price move a smaller number."""
    _run(env, "--apply", "--keep-costs")
    t = _trade(env, 2)
    assert t["gross_pnl"] == pytest.approx((90.0 - 100.0) * 65)


def test_prices_and_times_are_never_moved(env):
    """Only the multiplier was wrong; the fills are what they are."""
    before = {k: _trade(env, 1)[k] for k in
              ("entry_premium", "exit_premium", "entry_time", "exit_time", "lots")}
    _run(env, "--apply")
    after = {k: _trade(env, 1)[k] for k in before}
    assert after == before


def test_costs_are_recomputed_on_the_real_turnover_by_default(env):
    """Charges scale with turnover, which scales with lot size — the
    recorded figure was computed on a position half the real size."""
    _run(env, "--apply")
    t = _trade(env, 1)
    assert t["costs"] != 100.0
    assert t["costs"] > 100.0, "double the turnover should cost more, not less"
    assert t["realized_pnl"] == pytest.approx(t["gross_pnl"] - t["costs"])


def test_keep_costs_honours_the_recorded_figure(env):
    _run(env, "--apply", "--keep-costs")
    assert _trade(env, 1)["costs"] == 100.0


def test_include_open_restates_unrealised_not_realised(env):
    _run(env, "--apply", "--include-open")
    t = _trade(env, 4)
    assert t["lot_size"] == 30
    assert t["realized_pnl"] == 0.0, "an open trade has realised nothing"
    assert t["unrealized_pnl"] is not None


def test_capital_is_rebuilt_from_initial_plus_realised(env):
    _run(env, "--apply")
    c = sqlite3.connect(env["db"])
    realised = c.execute(
        "SELECT SUM(realized_pnl) FROM user_trades "
        "WHERE mode='paper' AND exit_time IS NOT NULL"
    ).fetchone()[0]
    c.close()
    assert _capital(env) == pytest.approx(100000.0 + realised)


def test_the_database_is_backed_up_before_writing(env):
    _run(env, "--apply")
    backups = list(env["db"].parent.glob("webapp.backup-*.sqlite"))
    assert len(backups) == 1
    old = sqlite3.connect(backups[0]).execute(
        "SELECT lot_size FROM user_trades WHERE id=1"
    ).fetchone()[0]
    assert old == 15


def test_running_twice_changes_nothing_the_second_time(env):
    _run(env, "--apply")
    after_first = dict(_trade(env, 1))
    second = _run(env, "--apply")
    assert "Nothing to do" in second.stdout
    assert dict(_trade(env, 1)) == after_first


def test_the_summary_shows_the_total_change(env):
    out = _run(env).stdout
    assert "TOTAL" in out
    assert "old P&L" in out and "new P&L" in out


# -------------------------------------------------------------------
#  A stale config must not make this silently do nothing
# -------------------------------------------------------------------
# First production run printed "32 already correct" against a container
# whose own config.yaml was still stale: it compared a wrong number to a
# wrong number and they matched. The override exists so the correct size
# can be stated outright, and the header shows where each one came from.

def test_the_sizes_it_will_use_are_printed_with_their_source(env):
    out = _run(env).stdout
    assert "Sizes :" in out
    assert "config.yaml" in out


def test_an_override_beats_the_config(env):
    out = _run(env, "--lot-size", "BANKNIFTY=45").stdout
    assert "BANKNIFTY=45 (override)" in out
    assert "15 -> 45" in out


def test_an_override_applies(env):
    _run(env, "--lot-size", "BANKNIFTY=45", "--apply", "--keep-costs")
    t = _trade(env, 1)
    assert t["lot_size"] == 45
    assert t["gross_pnl"] == pytest.approx((750.0 - 700.0) * 45)


def test_a_malformed_override_is_refused(env):
    for bad in ("BANKNIFTY", "BANKNIFTY=0", "BANKNIFTY=abc", "BANKNIFTY=-3"):
        r = _run(env, "--lot-size", bad)
        assert r.returncode == 2, bad
    assert _trade(env, 1)["lot_size"] == 15
