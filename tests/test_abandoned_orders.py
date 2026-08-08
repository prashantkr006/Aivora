"""An entry order we gave up on must not stay live at the broker.

2026-08-07, from the live event log:

    10:15:25  CLOSE 6f4e8aa9 reason=horizon
    10:15:26  CLOSE 9070ddd9 reason=horizon
    10:15:47  LIVE order 260807170261993 not filled: status=OPEN
    10:20:33  OPEN BANKNIFTY PE strike=57900 @ 615.75

open_live_trade waited 20 seconds, saw status=OPEN, logged it and returned
None — recording nothing and leaving a live LIMIT order sitting on the
exchange. That order can fill at any point in the session. When it did,
the account held a position AiVora had never heard of, and the next tick
opened a second BANKNIFTY put on top of it, because the book looked empty.

Only the tracked one was ever exited. The other sat until the broker's own
intraday square-off, at whatever price that happened to be, with no stop
and no target — and twice the intended money was at risk in the meantime.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from aivora.live import live_executor
from aivora.live.live_executor import _abandon, _untracked_qty

HELD = "NFO:BANKNIFTY26AUG57900PE"


class _Portfolio:
    def __init__(self, trades=None):
        self._trades = trades or []
        self.logs = []

    def load(self):
        return {"mode": "live", "trades": self._trades, "settings": {},
                "current_capital": 60000.0}

    def append_log(self, msg, level="info"):
        self.logs.append((level, msg))


class _Kite:
    def __init__(self, after_cancel="CANCELLED", cancel_raises=None,
                 status_raises=None, positions=None):
        self.after_cancel = after_cancel
        self.cancel_raises = cancel_raises
        self.status_raises = status_raises
        self._positions = positions if positions is not None else []
        self.cancelled = []

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        if self.cancel_raises:
            raise self.cancel_raises

    def order_status(self, order_id):
        if self.status_raises:
            raise self.status_raises
        return {"status": self.after_cancel, "average_price": 615.75}

    def positions(self):
        return {"net": self._positions, "day": self._positions}


def _booked(qty=30, symbol=HELD):
    return {"trade_id": "t1", "symbol": "BANKNIFTY", "side": "PE",
            "lots": 1, "lot_size": qty, "tradingsymbol": symbol,
            "entry_premium": 588.0}


def _pos(qty, sym="BANKNIFTY26AUG57900PE"):
    return {"exchange": "NFO", "tradingsymbol": sym, "quantity": qty}


# -------------------------------------------------------------------
#  Abandoning an order means cancelling it
# -------------------------------------------------------------------
def test_an_unfilled_order_is_cancelled():
    pf, k = _Portfolio(), _Kite()
    assert _abandon(pf, k, "260807170261993", "OPEN") is None
    assert k.cancelled == ["260807170261993"], "the order must come off the book"


def test_the_log_says_it_was_cancelled():
    pf, k = _Portfolio(), _Kite()
    _abandon(pf, k, "X1", "OPEN")
    assert any("cancelled" in m for _, m in pf.logs)


def test_a_fill_that_beats_the_cancel_is_recorded_not_dropped():
    """Cancelling races the fill. If it filled anyway, we own it — and an
    untracked position is far worse than an unexpected one."""
    pf, k = _Portfolio(), _Kite(after_cancel="COMPLETE")
    final = _abandon(pf, k, "X1", "OPEN")
    assert final is not None
    assert final["average_price"] == 615.75


def test_a_refused_cancel_still_checks_the_final_state():
    """Kite refuses to cancel an order that is already gone. Which way it
    went is the whole question."""
    pf = _Portfolio()
    k = _Kite(after_cancel="COMPLETE", cancel_raises=RuntimeError("too late"))
    assert _abandon(pf, k, "X1", "OPEN") is not None


def test_an_unconfirmable_order_is_escalated():
    """Cancel failed and the status cannot be read — the one case where we
    genuinely do not know what the account holds."""
    pf = _Portfolio()
    k = _Kite(cancel_raises=RuntimeError("net"), status_raises=RuntimeError("net"))
    assert _abandon(pf, k, "X1", "OPEN") is None
    msg = " ".join(m for _, m in pf.logs)
    assert "CHECK YOUR BROKER" in msg


def test_open_live_trade_abandons_rather_than_returning_bare():
    import inspect

    src = inspect.getsource(live_executor.open_live_trade)
    assert "_abandon(" in src
    assert 'not filled: status=' not in src, "the bare give-up path is gone"


# -------------------------------------------------------------------
#  Never stack on a position the book cannot see
# -------------------------------------------------------------------
def test_a_position_the_book_knows_about_is_not_counted_as_stray():
    pf = _Portfolio([_booked()])
    assert _untracked_qty(pf, _Kite(positions=[_pos(30)]), HELD) == 0


def test_a_position_the_book_does_not_know_about_is_found():
    assert _untracked_qty(_Portfolio(), _Kite(positions=[_pos(30)]), HELD) == 30


def test_only_the_excess_counts():
    """One tracked lot plus one stray — the stray is what matters."""
    pf = _Portfolio([_booked()])
    assert _untracked_qty(pf, _Kite(positions=[_pos(60)]), HELD) == 30


def test_a_different_contract_is_not_confused_with_this_one():
    k = _Kite(positions=[_pos(30, "BANKNIFTY26AUG57800PE")])
    assert _untracked_qty(_Portfolio(), k, HELD) == 0


def test_an_unreachable_broker_does_not_block_trading():
    """A stale read must not stop entries outright; every other guard still
    applies."""
    class _Broken:
        def positions(self):
            raise RuntimeError("timeout")

    assert _untracked_qty(_Portfolio(), _Broken(), HELD) == 0


def test_entry_refuses_when_the_broker_holds_something_untracked():
    import inspect

    src = inspect.getsource(live_executor.open_live_trade)
    assert "_untracked_qty(" in src
    assert src.index("_untracked_qty(") < src.index("place_limit_buy"), (
        "check the account before adding to it"
    )


# -------------------------------------------------------------------
#  And say so on every tick, not just at entry
# -------------------------------------------------------------------
def test_reconcile_reports_a_position_the_book_does_not_have():
    from aivora.live.reconcile import _report_orphans

    pf = _Portfolio()
    _report_orphans(pf, {HELD: {"quantity": 30}}, {})
    msg = " ".join(m for _, m in pf.logs)
    assert HELD in msg and "not being managed" in msg


def test_reconcile_stays_quiet_when_the_two_agree():
    from aivora.live.reconcile import _report_orphans

    pf = _Portfolio()
    _report_orphans(pf, {HELD: {"quantity": 30}}, {HELD: [_booked()]})
    assert pf.logs == []


def test_reconcile_still_never_opens_a_trade():
    """It reports the stray; inventing a trade for it is not its job — the
    position may simply be the user's own."""
    import inspect

    from aivora.live import reconcile

    src = inspect.getsource(reconcile._report_orphans)
    assert "open_trade" not in src
    assert "close_trade" not in src
