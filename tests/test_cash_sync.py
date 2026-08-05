"""Keeping the book's capital in line with the money actually in the account.

AiVora derived capital purely from its own trades — initial + realised. Cash
added or withdrawn in Kite was invisible, so the moment money moved outside
AiVora the two numbers parted and stayed parted, because nothing looked.

These tests care most about when the sync must NOT run. Overwriting capital
from a figure that is not comparable is worse than leaving it stale: with a
position open the premium is already spent and margin is blocked, and F&O
profit does not reach the balance until it settles.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from aivora.live.cash_sync import MIN_DELTA, sync_capital_from_broker

MORNING = datetime(2026, 8, 5, 7, 30)


class _Portfolio:
    def __init__(self, capital=51000.0, mode="live", trades=None,
                 synced_at=None, flows=0.0):
        self._state = {
            "mode": mode,
            "initial_capital": 51000.0,
            "current_capital": capital,
            "external_flows": flows,
            "cash_synced_at": synced_at,
            "trades": trades or [],
            "settings": {},
        }
        self.logs = []
        self.applied = []

    def load(self):
        return dict(self._state)

    def set_broker_cash(self, cash, when):
        delta = float(cash) - float(self._state["current_capital"])
        self._state["external_flows"] += delta
        self._state["current_capital"] = float(cash)
        self._state["cash_synced_at"] = when.isoformat(timespec="seconds")
        self.applied.append(delta)
        return delta

    def append_log(self, msg, level="info"):
        self.logs.append((level, msg))


class _Kite:
    def __init__(self, cash=51000.0, raises=None):
        self._cash = cash
        self._raises = raises
        self.calls = 0

    def available_funds(self):
        self.calls += 1
        if self._raises:
            raise self._raises
        return self._cash


def _open_trade():
    return {"trade_id": "t1", "symbol": "NIFTY", "side": "PE",
            "entry_premium": 50.0, "lots": 1, "lot_size": 65}


# -------------------------------------------------------------------
#  The point of the feature
# -------------------------------------------------------------------
def test_a_withdrawal_made_in_kite_is_picked_up():
    pf = _Portfolio(capital=51000.0)
    delta = sync_capital_from_broker(pf, _Kite(cash=41000.0), MORNING)
    assert delta == pytest.approx(-10000.0)
    assert pf.load()["current_capital"] == 41000.0


def test_a_deposit_made_in_kite_is_picked_up():
    pf = _Portfolio(capital=51000.0)
    assert sync_capital_from_broker(pf, _Kite(cash=76000.0), MORNING) == 25000.0
    assert pf.load()["current_capital"] == 76000.0


def test_the_move_is_booked_as_an_external_flow():
    """Not folded into initial_capital, so 'what I started with' stays put
    and return-on-capital does not jump when cash moves."""
    pf = _Portfolio(capital=51000.0)
    sync_capital_from_broker(pf, _Kite(cash=61000.0), MORNING)
    s = pf.load()
    assert s["initial_capital"] == 51000.0
    assert s["external_flows"] == pytest.approx(10000.0)


def test_the_user_is_told_what_moved_and_which_way():
    pf = _Portfolio(capital=51000.0)
    sync_capital_from_broker(pf, _Kite(cash=41000.0), MORNING)
    msg = " ".join(m for _, m in pf.logs)
    assert "withdrawn from" in msg
    assert "51,000" in msg and "41,000" in msg


# -------------------------------------------------------------------
#  When it must not run
# -------------------------------------------------------------------
def test_an_open_position_blocks_the_sync():
    """Premium is spent and margin is blocked — the two numbers are not
    comparable, and syncing would book a withdrawal that never happened."""
    pf = _Portfolio(capital=51000.0, trades=[_open_trade()])
    k = _Kite(cash=29000.0)
    assert sync_capital_from_broker(pf, k, MORNING) is None
    assert k.calls == 0, "must not even ask the broker"
    assert pf.load()["current_capital"] == 51000.0


def test_paper_is_never_synced_to_a_real_account():
    """Paper capital is imaginary on purpose."""
    k = _Kite(cash=41000.0)
    pf = _Portfolio(capital=100000.0, mode="paper")
    assert sync_capital_from_broker(pf, k, MORNING) is None
    assert k.calls == 0


def test_it_runs_once_a_day():
    pf = _Portfolio(capital=51000.0, synced_at="2026-08-05T07:30:00")
    k = _Kite(cash=41000.0)
    assert sync_capital_from_broker(pf, k, MORNING) is None
    assert k.calls == 0


def test_a_sync_from_yesterday_does_not_block_today():
    pf = _Portfolio(capital=51000.0, synced_at="2026-08-04T07:30:00")
    assert sync_capital_from_broker(pf, _Kite(cash=41000.0), MORNING) == -10000.0


# -------------------------------------------------------------------
#  Bad numbers must never overwrite good ones
# -------------------------------------------------------------------
def test_an_unreachable_broker_leaves_capital_alone():
    pf = _Portfolio(capital=51000.0)
    assert sync_capital_from_broker(
        pf, _Kite(raises=RuntimeError("token expired")), MORNING) is None
    assert pf.load()["current_capital"] == 51000.0
    assert any("not synced" in m for _, m in pf.logs)


def test_a_zero_balance_is_refused():
    """Zerodha reports 0 in some fields on the day a deposit lands. Zeroing
    the book on that is far worse than leaving it stale."""
    pf = _Portfolio(capital=51000.0)
    assert sync_capital_from_broker(pf, _Kite(cash=0.0), MORNING) is None
    assert pf.load()["current_capital"] == 51000.0


def test_rounding_noise_is_not_reported_as_a_cash_movement():
    pf = _Portfolio(capital=51000.0)
    delta = sync_capital_from_broker(
        pf, _Kite(cash=51000.0 + MIN_DELTA / 2), MORNING)
    assert delta == 0.0
    assert not any("outside AiVora" in m for _, m in pf.logs)


def test_a_no_op_sync_still_records_that_it_ran():
    """Otherwise it would retry on every tick all day."""
    pf = _Portfolio(capital=51000.0)
    sync_capital_from_broker(pf, _Kite(cash=51000.0), MORNING)
    assert pf.load()["cash_synced_at"] is not None


# -------------------------------------------------------------------
#  Wiring
# -------------------------------------------------------------------
def test_the_token_refresh_syncs_capital():
    """The user's ask: every morning, as the token is added."""
    import inspect

    from scripts import auto_refresh_kite_tokens as m

    assert "_sync_capital" in inspect.getsource(m._refresh_one)
    assert "sync_capital_from_broker" in inspect.getsource(m._sync_capital)


def test_a_failed_sync_never_costs_the_user_their_token():
    import inspect

    from scripts import auto_refresh_kite_tokens as m

    src = inspect.getsource(m._refresh_one)
    after = src.split("_sync_capital", 1)[1]
    assert "except Exception" in after


def test_the_tick_syncs_too_for_tokens_added_by_hand():
    import inspect

    from aivora.webapp import trading_engine

    src = inspect.getsource(trading_engine.run_user_tick)
    assert "sync_capital_from_broker" in src
    assert src.index("sync_capital_from_broker") < src.index("_tracker.tick")
