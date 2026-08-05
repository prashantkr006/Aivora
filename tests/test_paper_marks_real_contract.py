"""Paper has to price its exits off the market, not off a model.

Paper never asked what its option was worth. It priced every exit with
``theoretical_exit_premium`` — what the premium ought to be given the spot
move and the time elapsed. That model turns any favourable spot move into a
gain and barely charges for time decay, so paper does not shadow live, it
flatters it.

Measured on 2026-08-05, same signals, same lot sizes, same 1-lot sizing:

    paper  7 trades, 7 wins (100%),  +Rs 5,347
    live   7 trades, 5 wins  (71%),  +Rs 8,629

All seven paper trades had a favourable spot move, and the model scored all
seven as wins. The clearest single case is the 09:15 NIFTY put: spot moved
8.8 points its way over an hour, the model marked it +Rs 34, and the market
paid -Rs 286. Over that hour theta on a Rs 136 premium dwarfs the delta gain
from an 8.8-point move.

Paper is not a soft version of live — it is a compressed one. It rarely
shows a loss and it rarely shows a full win: live's exits on the shared
winners were all higher than paper's.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from aivora.live.position_tracker import _mark

NOW = datetime(2026, 8, 5, 10, 15)
HELD = "NFO:NIFTY26AUG24650PE"

SETTINGS = {"take_profit_pct": 0.60, "stop_loss_pct": 0.30}


class _Kite:
    def __init__(self, ltp=132.45, raises=None):
        self.ltp = ltp
        self._raises = raises
        self.quotes = []

    def ltp_for(self, tradingsymbol):
        self.quotes.append(tradingsymbol)
        if self._raises:
            raise self._raises
        return self.ltp

    def atm_option_symbols(self, symbol, spot):
        return {"CE": f"NFO:{symbol}ATMCE", "PE": f"NFO:{symbol}ATMPE",
                "lot_size": 65}


def _trade(**over):
    t = {
        "symbol": "NIFTY", "side": "PE", "strike": 24650.0,
        "entry_premium": 136.50, "entry_spot": 24625.40,
        "current_premium": 136.50, "lots": 1, "lot_size": 65,
        "entry_time": "2026-08-05T09:15:00",
        "tradingsymbol": HELD,
    }
    t.update(over)
    return t


# -------------------------------------------------------------------
#  The market wins over the model
# -------------------------------------------------------------------
def test_paper_marks_against_the_real_contract():
    k = _Kite(ltp=132.45)
    assert _mark(k, _trade(), 24616.60, NOW, SETTINGS) == 132.45
    assert k.quotes == [HELD]


def test_the_incident_case_now_shows_the_loss_it_was():
    """Spot moved 8.8 points the put's way; the market still paid less."""
    px = _mark(_Kite(ltp=132.45), _trade(), 24616.60, NOW, SETTINGS)
    pnl = (px - 136.50) * 1 * 65
    assert pnl < 0
    assert pnl == pytest.approx(-263.25)


def test_the_model_would_have_called_the_same_trade_a_win():
    """Pins why this mattered: on the same inputs the model disagrees with
    the market about the sign."""
    modelled = _mark(None, _trade(), 24616.60, NOW, SETTINGS)
    assert modelled > 136.50, "the model reads a favourable spot move as a gain"


# -------------------------------------------------------------------
#  The model stays as a fallback
# -------------------------------------------------------------------
def test_without_a_broker_it_falls_back_to_the_model():
    px = _mark(None, _trade(), 24616.60, NOW, SETTINGS)
    assert px > 0


def test_a_trade_with_no_contract_recorded_falls_back():
    """Opened before the contract was stored, or Kite could not resolve it."""
    k = _Kite()
    px = _mark(k, _trade(tradingsymbol=None), 24616.60, NOW, SETTINGS)
    assert k.quotes == []
    assert px > 0


def test_a_failed_quote_falls_back_rather_than_raising():
    """A missed tick must not stop the tracker mid-loop."""
    px = _mark(_Kite(raises=RuntimeError("rate limit")), _trade(),
               24616.60, NOW, SETTINGS)
    assert px > 0


# -------------------------------------------------------------------
#  Wiring
# -------------------------------------------------------------------
def test_paper_entries_record_the_contract():
    import inspect

    from aivora.live import paper_executor

    src = inspect.getsource(paper_executor.open_paper_trade)
    assert "atm_option_symbols" in src
    assert "tradingsymbol=tradingsymbol" in src


@pytest.mark.parametrize("module,fn", [
    ("aivora.webapp.trading_engine", "run_user_tick"),
    ("aivora.live.scheduler", "run_tick"),
])
def test_the_tracker_gets_the_broker_in_both_modes(module, fn):
    """Paper needs the quote even though it places no order."""
    import importlib
    import inspect

    src = inspect.getsource(getattr(importlib.import_module(module), fn))
    assert 'kite=kite if' not in src, "kite must not be withheld from paper"
    assert "kite=kite" in src


def test_orders_still_branch_on_the_portfolio_mode():
    """Handing paper a broker must not make paper place orders."""
    import inspect

    from aivora.live import position_tracker

    src = inspect.getsource(position_tracker.tick)
    assert "close_live_trade" in src and "close_paper_trade" in src
    assert 'live = state["mode"] == "live"' in src
