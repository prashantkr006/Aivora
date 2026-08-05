"""The trailing stop has to survive between ticks.

The tracker computed peak_premium and trailing_sl_price on every tick and
handed them to update_open_marks, which wrote only current_premium and
unrealized_pnl and dropped the rest. Nothing carried forward, so each tick
restarted from the entry price: the trail could never rise, and the stop
could never fire — it needs to be set on one tick and breached on a later
one.

Production evidence: the same line, at the same price, every five minutes
for an hour —

    09:20  Trailing SL updated to Rs 734.35 (+0%) - BANKNIFTY CE
    09:25  Trailing SL updated to Rs 734.35 (+0%) - BANKNIFTY CE
    09:30  Trailing SL updated to Rs 734.35 (+0%) - BANKNIFTY CE

A trail that only ever rises should log once.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import pytest

from aivora.live import trailing_sl as tsl
from aivora.live.position_tracker import _decide_exit, _step_trailing_sl
from aivora.webapp import portfolios, users


@pytest.fixture()
def pf():
    u = users.register(f"trail_{time.time()}@example.com", "TestPassword_123")
    p = portfolios.UserPortfolio(u.id, "paper")
    p.set_initial_capital(100_000.0)
    yield p


def _open(pf, entry=700.0):
    from aivora.live.portfolio import Trade, make_trade_id

    t = Trade(
        trade_id=make_trade_id(),
        entry_time=datetime(2026, 8, 5, 9, 15).isoformat(timespec="seconds"),
        symbol="BANKNIFTY", side="CE", strike=57700.0,
        lots=1, lot_size=30, entry_premium=entry,
        current_premium=entry, entry_spot=57700.0,
    )
    pf.open_trade(t)
    return t.trade_id


def _trade(pf, tid):
    return next(t for t in pf.load()["trades"] if t["trade_id"] == tid)


# -------------------------------------------------------------------
#  Persistence
# -------------------------------------------------------------------
def test_peak_survives_a_tick(pf):
    tid = _open(pf)
    pf.update_open_marks({tid: {
        "current_premium": 800.0, "unrealized_pnl": 3000.0,
        "peak_premium": 800.0,
    }})
    assert _trade(pf, tid)["peak_premium"] == 800.0


def test_the_trail_survives_a_tick(pf):
    tid = _open(pf)
    pf.update_open_marks({tid: {
        "current_premium": 800.0, "unrealized_pnl": 3000.0,
        "peak_premium": 800.0, "trailing_sl_price": 770.0,
    }})
    assert _trade(pf, tid)["trailing_sl_price"] == 770.0


def test_a_patch_without_trail_leaves_the_stored_one_alone(pf):
    """The tracker omits trailing_sl_price while the trail is dormant. That
    must not wipe a trail set earlier."""
    tid = _open(pf)
    pf.update_open_marks({tid: {
        "current_premium": 800.0, "unrealized_pnl": 3000.0,
        "peak_premium": 800.0, "trailing_sl_price": 770.0,
    }})
    pf.update_open_marks({tid: {
        "current_premium": 790.0, "unrealized_pnl": 2700.0,
        "peak_premium": 800.0,
    }})
    t = _trade(pf, tid)
    assert t["trailing_sl_price"] == 770.0
    assert t["current_premium"] == 790.0


def test_a_closed_trade_is_not_marked(pf):
    tid = _open(pf)
    pf.close_trade(trade_id=tid, exit_time=datetime(2026, 8, 5, 10, 15),
                   exit_premium=750.0, exit_reason="horizon",
                   gross_pnl=1500.0, costs=50.0)
    pf.update_open_marks({tid: {"current_premium": 999.0,
                                "unrealized_pnl": 0.0}})
    assert _trade(pf, tid)["current_premium"] != 999.0


def test_an_unknown_field_is_reported_not_swallowed(pf, caplog):
    """Silently dropping a field is how this bug lasted this long."""
    import logging

    tid = _open(pf)
    with caplog.at_level(logging.WARNING):
        pf.update_open_marks({tid: {"current_premium": 800.0,
                                    "unrealized_pnl": 0.0,
                                    "nonsense_field": 1.0}})
    assert any("nonsense_field" in r.message for r in caplog.records)


# -------------------------------------------------------------------
#  What persistence buys: a trail that rises, and a stop that fires
# -------------------------------------------------------------------
def test_the_trail_rises_across_ticks_and_never_falls(pf):
    tid = _open(pf, entry=700.0)
    seen = []
    for premium in (700.0, 780.0, 900.0, 850.0, 800.0):
        t = _trade(pf, tid)
        trail = _step_trailing_sl(t, premium, pf)
        marks = {"current_premium": premium, "unrealized_pnl": 0.0,
                 "peak_premium": t["peak_premium"]}
        if trail is not None:
            marks["trailing_sl_price"] = trail
        pf.update_open_marks({tid: marks})
        seen.append(trail)

    assert seen[0] is None, "dormant below +10%"
    risen = [s for s in seen if s is not None]
    assert risen == sorted(risen), f"trail must never fall: {risen}"
    assert _trade(pf, tid)["peak_premium"] == 900.0, "peak must hold at the high"


def test_the_stop_can_finally_fire(pf):
    """Set on one tick, breached on a later one — impossible before."""
    tid = _open(pf, entry=700.0)

    t = _trade(pf, tid)
    trail = _step_trailing_sl(t, 900.0, pf)          # tick 1: peak, trail set
    pf.update_open_marks({tid: {
        "current_premium": 900.0, "unrealized_pnl": 0.0,
        "peak_premium": t["peak_premium"], "trailing_sl_price": trail,
    }})
    assert trail is not None

    t = _trade(pf, tid)                              # tick 2: reload from DB
    assert t["trailing_sl_price"] == trail, "the trail must come back"
    reason = _decide_exit(t, datetime(2026, 8, 5, 9, 25), trail - 1.0,
                          {"take_profit_pct": 0.60, "stop_loss_pct": 0.30})
    assert reason == "trailing_stop"


def test_without_persistence_the_stop_could_never_fire():
    """Pins the old behaviour as the bug it was: a trail read back as None
    disables the check entirely."""
    assert tsl.would_stop_here(500.0, None) is False


def test_the_fixed_stop_still_applies_while_the_trail_is_dormant():
    """The one mercy of the old bug — positions were not unprotected."""
    trade = {"entry_premium": 700.0, "trailing_sl_price": None}
    assert _decide_exit(trade, datetime(2026, 8, 5, 9, 25), 480.0,
                        {"take_profit_pct": 0.60,
                         "stop_loss_pct": 0.30}) == "stop_loss"
