"""Regression tests for how open positions are priced.

The bug these pin: _live_premium asked for the option that is at-the-money
*right now* instead of the contract the trade actually holds.  As the index
moved, the ATM strike drifted away from the strike we bought, so the tracker
priced a different option.

That value feeds _decide_exit, so take-profit, stop-loss and the trailing
stop were all judged against a contract we do not own.  Observed in
production: a position genuinely +₹2,435 was marked at +₹692, which kept the
target from firing.
"""

from __future__ import annotations

import pytest

from aivora.live.position_tracker import _live_premium


class _Kite:
    """Records what was asked for, and prices the two options differently."""

    HELD = "NFO:BANKNIFTY26AUG57700CE"

    def __init__(self):
        self.ltp_calls = []
        self.atm_calls = []

    def ltp_for(self, tradingsymbol):
        self.ltp_calls.append(tradingsymbol)
        return 830.0                      # the contract we actually hold

    def atm_option_symbols(self, symbol, spot):
        return {"CE": f"NFO:{symbol}ATM{int(spot)}CE",
                "PE": f"NFO:{symbol}ATM{int(spot)}PE"}

    def atm_option_quote(self, symbol, spot):
        self.atm_calls.append((symbol, spot))
        return {"ce_ltp": 770.0, "pe_ltp": 12.0}   # a *different* strike


def _trade(**over):
    t = {
        "symbol": "BANKNIFTY", "side": "CE", "strike": 57700.0,
        "entry_premium": 736.05, "entry_spot": 57650.0,
        "lots": 1, "lot_size": 30,
        "tradingsymbol": _Kite.HELD,
    }
    t.update(over)
    return t


def test_prices_the_contract_actually_held():
    k = _Kite()
    px = _live_premium(k, _trade(), spot_now=58200.0)   # index has moved
    assert px == 830.0
    assert k.ltp_calls == [_Kite.HELD]
    assert k.atm_calls == [], "must not re-derive a strike from the live spot"


def test_moving_spot_does_not_change_which_option_is_priced():
    """The whole point: the mark must not drift with the index."""
    k = _Kite()
    a = _live_premium(k, _trade(), spot_now=57650.0)
    b = _live_premium(k, _trade(), spot_now=59000.0)
    assert a == b == 830.0
    assert set(k.ltp_calls) == {_Kite.HELD}


def test_legacy_trade_without_tradingsymbol_uses_entry_spot():
    """Trades opened before the field existed must still resolve to the
    strike chosen at entry — never to the current at-the-money strike."""
    k = _Kite()
    px = _live_premium(k, _trade(tradingsymbol=None), spot_now=59000.0)
    assert px == 770.0                       # fell back to the ATM quote
    assert k.atm_calls == [("BANKNIFTY", 57650.0)], (
        "fallback must use the ENTRY spot, not the current spot"
    )


def test_put_side_uses_the_put_quote_on_the_legacy_path():
    k = _Kite()
    px = _live_premium(k, _trade(side="PE", tradingsymbol=None), spot_now=59000.0)
    assert px == 12.0


def test_exit_order_sells_the_held_contract():
    """close_live_trade must not re-derive the symbol when we recorded it."""
    import inspect

    from aivora.live import live_executor

    src = inspect.getsource(live_executor.close_live_trade)
    assert 'trade_dict.get("tradingsymbol")' in src, (
        "the exit must sell the contract we hold"
    )
