"""What a live close is allowed to charge.

Two defects met here.

The cost rates live under ``backtest.costs`` in config.yaml, but the live
closers passed the *portfolio settings* dict as ``cfg``.  That dict holds
none of those keys, so every rate quietly fell back to compute_round_trip's
literal defaults — the config file was not being read at all on the live
path.

And one of those defaults was ``slippage_pct``.  Slippage models crossing
the bid-ask spread, a guess the backtest must make because it never sends
an order.  A live fill happened at a real price with the spread already
inside it, so charging it again invents a fee the broker never took.
Measured on a real trade: ₹87.35 booked where ₹49.25 was actually charged.
"""

from __future__ import annotations

import inspect

import pytest

from aivora.backtest.costs import compute_round_trip, live_cost_cfg
from aivora.utils.config import get_config


def test_live_config_charges_no_slippage():
    assert live_cost_cfg()["slippage_pct"] == 0.0


def test_live_config_still_carries_the_real_charges():
    """Zeroing slippage must not zero the fees the broker does take."""
    cfg = live_cost_cfg()
    for key in ("brokerage_flat_per_order", "stt_pct_sell",
                "exchange_txn_pct", "gst_pct", "stamp_duty_pct_buy"):
        assert cfg[key] > 0, f"{key} went missing"


def test_it_reads_config_rather_than_relying_on_defaults():
    configured = get_config().raw["backtest"]["costs"]
    cfg = live_cost_cfg()
    for key, value in configured.items():
        if key == "slippage_pct":
            continue
        assert cfg[key] == value


def test_the_difference_is_exactly_the_slippage():
    """The production trade that exposed this: BANKNIFTY 57600 PE."""
    entry, exit_px, lots, lot_size = 637.4, 632.75, 1, 30
    modelled = dict(get_config().raw["backtest"]["costs"])

    with_slip = compute_round_trip(entry, exit_px, lots, lot_size, modelled).total
    real = compute_round_trip(entry, exit_px, lots, lot_size, live_cost_cfg()).total

    turnover = (entry + exit_px) * lots * lot_size
    assert with_slip == pytest.approx(87.35, abs=0.01)
    assert real == pytest.approx(49.25, abs=0.01)
    assert with_slip - real == pytest.approx(turnover * modelled["slippage_pct"])


def test_mutating_the_returned_config_cannot_poison_the_next_caller():
    live_cost_cfg()["stt_pct_sell"] = 999.0
    assert live_cost_cfg()["stt_pct_sell"] != 999.0


# -------------------------------------------------------------------
#  Wiring: every live close must use it
# -------------------------------------------------------------------
@pytest.mark.parametrize("where", ["live_executor", "reconcile"])
def test_live_closers_use_the_live_config(where):
    import importlib

    mod = importlib.import_module(f"aivora.live.{where}")
    fn = (mod.close_live_trade if where == "live_executor" else mod._close)
    src = inspect.getsource(fn)
    assert "live_cost_cfg()" in src
    assert 'cfg=portfolio.load()["settings"]' not in src
    assert "cfg=settings" not in src


def test_paper_keeps_charging_slippage():
    """Paper fills are imagined, so the spread has to be modelled — the
    whole point of the setting.  Removing it there would flatter paper."""
    from aivora.live import paper_executor

    src = inspect.getsource(paper_executor)
    assert "live_cost_cfg" not in src


def test_the_backtest_keeps_charging_slippage():
    from aivora.backtest import backtester

    src = inspect.getsource(backtester)
    assert "live_cost_cfg" not in src
