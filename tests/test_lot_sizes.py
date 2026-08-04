"""Where the contract size comes from.

config.yaml carried NIFTY 75 and BANKNIFTY 15 long after NSE had moved to
65 and 30, and nothing ever checked.  Every backtested BANKNIFTY P&L was
computed on half the real contract size, and one lot looked like 21% of a
51,000 account when it is really 43% — so the backtest never saw the
concentration that actually exists.

Live already asked Kite.  Paper did not, which meant paper was sizing a
different system from the one it was supposed to be shadowing.
"""

from __future__ import annotations

from datetime import date

import pytest

from aivora.live import lot_sizes
from aivora.live.lot_sizes import configured_lot_size, lot_size_for, reset_cache

DAY = date(2026, 8, 4)


@pytest.fixture(autouse=True)
def _clean():
    reset_cache()
    yield
    reset_cache()


class _Kite:
    """Answers with the exchange's real sizes, and counts the asking."""

    REAL = {"NIFTY": 65, "BANKNIFTY": 30}

    def __init__(self, raises=None):
        self.calls = 0
        self._raises = raises

    def lot_size(self, symbol):
        self.calls += 1
        if self._raises:
            raise self._raises
        return self.REAL[symbol]


# -------------------------------------------------------------------
#  The exchange is the source of truth
# -------------------------------------------------------------------
def test_it_uses_the_exchange_not_the_config():
    assert lot_size_for("BANKNIFTY", _Kite(), today=DAY) == 30


def test_the_answer_is_cached_for_the_day():
    k = _Kite()
    for _ in range(5):
        lot_size_for("NIFTY", k, today=DAY)
    assert k.calls == 1


def test_a_new_day_asks_again():
    k = _Kite()
    lot_size_for("NIFTY", k, today=DAY)
    lot_size_for("NIFTY", k, today=date(2026, 8, 5))
    assert k.calls == 2


# -------------------------------------------------------------------
#  Falling back must never stop trading
# -------------------------------------------------------------------
def test_without_a_broker_it_falls_back_to_config():
    """The backtest and the offline tests have nothing to ask."""
    assert lot_size_for("NIFTY", None, today=DAY) == configured_lot_size("NIFTY")


def test_a_broker_error_falls_back_rather_than_raising():
    """A stale lot size is bad; refusing to trade because the instruments
    dump timed out is worse."""
    got = lot_size_for("NIFTY", _Kite(raises=RuntimeError("timeout")), today=DAY)
    assert got == configured_lot_size("NIFTY")


def test_a_nonsense_answer_is_rejected():
    class _Zero:
        def lot_size(self, symbol):
            return 0

    assert lot_size_for("NIFTY", _Zero(), today=DAY) == configured_lot_size("NIFTY")


def test_a_failed_lookup_is_not_cached():
    """Otherwise one timeout would pin the fallback for the whole day."""
    lot_size_for("NIFTY", _Kite(raises=RuntimeError("boom")), today=DAY)
    k = _Kite()
    assert lot_size_for("NIFTY", k, today=DAY) == 65
    assert k.calls == 1


def test_an_unknown_symbol_is_an_error_not_a_guess():
    with pytest.raises(KeyError):
        configured_lot_size("FINNIFTY")


# -------------------------------------------------------------------
#  Drift can never be silent again
# -------------------------------------------------------------------
def test_disagreement_is_logged(caplog):
    import logging

    class _Drifted(_Kite):
        def lot_size(self, symbol):
            self.calls += 1
            return 99

    with caplog.at_level(logging.WARNING):
        assert lot_size_for("NIFTY", _Drifted(), today=DAY) == 99
    assert any("lot size drift" in r.message for r in caplog.records)


def test_the_config_now_matches_the_exchange():
    """The values NSE actually uses, seen on real fills on 2026-08-04."""
    assert configured_lot_size("NIFTY") == 65
    assert configured_lot_size("BANKNIFTY") == 30


# -------------------------------------------------------------------
#  Wiring
# -------------------------------------------------------------------
def test_paper_entries_resolve_the_lot_size():
    import inspect

    from aivora.live import paper_executor

    src = inspect.getsource(paper_executor.open_paper_trade)
    assert "lot_size_for(symbol, kite)" in src
    assert 'lot_size = int(inst["lot_size"])' not in src


@pytest.mark.parametrize("module,fn", [
    ("aivora.webapp.trading_engine", "run_user_tick"),
    ("aivora.live.scheduler", "run_tick"),
])
def test_paper_callers_hand_over_the_broker(module, fn):
    """Paper sizing on a different contract from live is not shadowing
    live — it is shadowing something else."""
    import importlib
    import inspect

    src = inspect.getsource(getattr(importlib.import_module(module), fn))
    assert "kite=kite" in src


def test_sizing_impact_is_what_the_incident_showed():
    """1 BANKNIFTY lot at 736.05 on a 51,000 account."""
    entry, capital = 736.05, 51_000.0
    stale, real = 15, 30
    assert entry * stale / capital == pytest.approx(0.2165, abs=1e-3)
    assert entry * real / capital == pytest.approx(0.4330, abs=1e-3)
