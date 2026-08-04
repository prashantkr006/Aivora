"""Keeping AiVora's book in step with the broker account.

The incident: AiVora opened a live trade, the user exited it by hand in
Kite, and AiVora never noticed.  The position stayed "open" in the book —
so it kept being marked, its symbol stayed blocked for new entries, and
when the user hit emergency square-off AiVora sent a sell order for a
contract it no longer held.  The rejection is what crashed the dashboard.

Reconciliation only ever *closes* what the broker reports flat.  It must
never invent a trade, and must change nothing when the broker's answer is
missing or ambiguous — closing the whole book on a failed API call would
be far worse than the bug it fixes.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from aivora.live.reconcile import EXIT_REASON, reconcile_live_book

NOW = datetime(2026, 8, 4, 14, 30)
HELD = "NFO:BANKNIFTY26AUG57700CE"


class _Portfolio:
    def __init__(self, trades, mode="live"):
        self._trades = trades
        self._mode = mode
        self.logs = []
        self.closed = []

    def load(self):
        return {"mode": self._mode, "trades": self._trades,
                "settings": {}, "current_capital": 51000.0}

    def close_trade(self, trade_id, exit_time, exit_premium, exit_reason,
                    gross_pnl, costs, **kw):
        self.closed.append({
            "trade_id": trade_id, "exit_premium": exit_premium,
            "exit_reason": exit_reason, "gross_pnl": gross_pnl,
        })
        for t in self._trades:
            if t["trade_id"] == trade_id:
                t["exit_time"] = exit_time.isoformat()

    def append_log(self, msg, level="info"):
        self.logs.append((level, msg))


class _Kite:
    def __init__(self, rows=None, raises=None):
        self._rows = rows if rows is not None else []
        self._raises = raises
        self.calls = 0

    def positions(self):
        self.calls += 1
        if self._raises:
            raise self._raises
        return {"net": self._rows, "day": self._rows}


def _trade(trade_id="t1", **over):
    t = {
        "trade_id": trade_id, "symbol": "BANKNIFTY", "side": "CE",
        "strike": 57700.0, "entry_premium": 736.05, "entry_spot": 57650.0,
        "current_premium": 800.0, "lots": 1, "lot_size": 30,
        "tradingsymbol": HELD,
    }
    t.update(over)
    return t


def _row(qty, **over):
    r = {"exchange": "NFO", "tradingsymbol": "BANKNIFTY26AUG57700CE",
         "quantity": qty, "sell_price": 817.0, "last_price": 810.0}
    r.update(over)
    return r


# -------------------------------------------------------------------
#  The incident itself
# -------------------------------------------------------------------
def test_position_exited_at_the_broker_is_closed_in_the_book():
    pf = _Portfolio([_trade()])
    n = reconcile_live_book(pf, _Kite([_row(0)]), NOW)
    assert n == 1
    assert pf.closed[0]["exit_reason"] == EXIT_REASON


def test_it_books_the_price_the_broker_actually_sold_at():
    pf = _Portfolio([_trade()])
    reconcile_live_book(pf, _Kite([_row(0, sell_price=817.0)]), NOW)
    assert pf.closed[0]["exit_premium"] == 817.0
    # 1 lot x 30 x (817.00 - 736.05)
    assert pf.closed[0]["gross_pnl"] == pytest.approx(2428.5)


def test_the_user_is_told_it_was_not_AiVora_that_closed_it():
    pf = _Portfolio([_trade()])
    reconcile_live_book(pf, _Kite([_row(0)]), NOW)
    assert any("your broker" in m for _, m in pf.logs)


# -------------------------------------------------------------------
#  Must not touch what is genuinely open
# -------------------------------------------------------------------
def test_position_still_held_is_left_alone():
    pf = _Portfolio([_trade()])
    assert reconcile_live_book(pf, _Kite([_row(30)]), NOW) == 0
    assert pf.closed == []


def test_broker_holding_more_than_us_is_left_alone():
    """A manual position on the same contract is the user's business."""
    pf = _Portfolio([_trade()])
    assert reconcile_live_book(pf, _Kite([_row(60)]), NOW) == 0
    assert pf.closed == []


def test_partial_exit_is_flagged_not_guessed():
    pf = _Portfolio([_trade(lots=2)])       # book expects 60
    assert reconcile_live_book(pf, _Kite([_row(30)]), NOW) == 0
    assert pf.closed == []
    assert any("partially exited" in m for _, m in pf.logs)


# -------------------------------------------------------------------
#  Ambiguity must never close the book
# -------------------------------------------------------------------
def test_a_failed_positions_call_closes_nothing():
    """An API error and a flat account must not look the same."""
    pf = _Portfolio([_trade()])
    assert reconcile_live_book(pf, _Kite(raises=RuntimeError("token")), NOW) == 0
    assert pf.closed == []
    assert any("not reconciled" in m for _, m in pf.logs)


def test_contract_missing_from_the_broker_is_flagged_not_closed():
    pf = _Portfolio([_trade()])
    assert reconcile_live_book(pf, _Kite([]), NOW) == 0
    assert pf.closed == []
    assert any("does not list it" in m for _, m in pf.logs)


def test_trade_without_a_recorded_contract_is_flagged_not_closed():
    """Re-deriving the strike could match the wrong option and close a
    position that is genuinely open."""
    pf = _Portfolio([_trade(tradingsymbol=None)])
    assert reconcile_live_book(pf, _Kite([_row(0)]), NOW) == 0
    assert pf.closed == []
    assert any("no contract recorded" in m for _, m in pf.logs)


def test_zero_prices_never_book_an_exit_at_zero():
    pf = _Portfolio([_trade()])
    reconcile_live_book(pf, _Kite([_row(0, sell_price=0, last_price=0)]), NOW)
    assert pf.closed[0]["exit_premium"] == 800.0    # our last mark


# -------------------------------------------------------------------
#  Cheap and inert when there is nothing to do
# -------------------------------------------------------------------
def test_paper_mode_never_calls_the_broker():
    k = _Kite([_row(0)])
    assert reconcile_live_book(_Portfolio([_trade()], mode="paper"), k, NOW) == 0
    assert k.calls == 0


def test_flat_book_never_calls_the_broker():
    k = _Kite([_row(0)])
    assert reconcile_live_book(_Portfolio([]), k, NOW) == 0
    assert k.calls == 0


def test_closed_trades_are_ignored():
    k = _Kite([_row(0)])
    pf = _Portfolio([_trade(exit_time="2026-08-04T12:00:00")])
    assert reconcile_live_book(pf, k, NOW) == 0
    assert k.calls == 0


def test_one_broker_call_per_tick_regardless_of_book_size():
    k = _Kite([_row(0)])
    pf = _Portfolio([_trade("t1"), _trade("t2"), _trade("t3")])
    reconcile_live_book(pf, k, NOW)
    assert k.calls == 1


def test_multiple_book_trades_on_one_contract_close_together():
    """The broker nets them into one row, so they stand or fall together."""
    pf = _Portfolio([_trade("t1"), _trade("t2")])
    assert reconcile_live_book(pf, _Kite([_row(0)]), NOW) == 2


def test_two_book_trades_on_one_contract_are_summed_before_comparing():
    """2 x 30 in the book vs 30 at the broker is a partial exit, not a match."""
    pf = _Portfolio([_trade("t1"), _trade("t2")])
    assert reconcile_live_book(pf, _Kite([_row(30)]), NOW) == 0
    assert any("partially exited" in m for _, m in pf.logs)


# -------------------------------------------------------------------
#  Wiring
# -------------------------------------------------------------------
def test_the_tick_reconciles_before_it_reads_the_book():
    import inspect

    from aivora.webapp import trading_engine

    src = inspect.getsource(trading_engine.run_user_tick)
    assert "reconcile_live_book" in src
    assert src.index("reconcile_live_book") < src.index("_tracker.tick"), (
        "reconciling after the tracker lets it act on a phantom position"
    )


def test_the_panic_button_reconciles_before_placing_exit_orders():
    import inspect

    import app.multi_user_app as m

    # Drop the docstring — it names close_live_trade too.
    body = inspect.getsource(m._emergency_square_off).split('"""', 2)[-1]
    assert "reconcile_live_book" in body
    assert body.index("reconcile_live_book(") < body.index("close_live_trade("), (
        "selling a position the broker no longer holds is what broke this"
    )
