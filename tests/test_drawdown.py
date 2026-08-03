"""Regression tests for drawdown measurement.

The bug these pin: ``_summarise`` divided the equity dip by the running
peak while position sizing is fixed off a constant ``capital``.  Because
rupee P&L per trade does not grow with accumulated profit, the reported
drawdown shrank purely as a function of how well the strategy performed.
On the production config it understated the drawdown 12.1x
(-0.08% reported against -0.93% real).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aivora.backtest.backtester import _summarise


def _trades(pnls, start="2024-01-01"):
    ts = pd.date_range(start, periods=len(pnls), freq="D")
    return pd.DataFrame({"datetime": ts, "pnl": [float(p) for p in pnls]})


def test_drawdown_is_relative_to_capital_not_running_peak():
    """A fixed rupee loss must report the same percentage whether it happens
    early or after the account has grown."""
    capital = 100_000.0
    loss = -1_000.0

    early = _summarise(_trades([loss, 500, 500]), capital)
    # Same loss, but only after +200,000 of profit has accumulated.
    late = _summarise(_trades([100_000, 100_000, loss]), capital)

    assert np.isclose(early["max_drawdown"], loss / capital)
    assert np.isclose(late["max_drawdown"], loss / capital)
    # The headline symptom: identical risk must not look 3x smaller merely
    # because the equity curve grew beneath it.
    assert np.isclose(early["max_drawdown"], late["max_drawdown"])


def test_drawdown_reported_in_rupees_too():
    capital = 100_000.0
    s = _summarise(_trades([1_000, -2_500, 400]), capital)
    assert np.isclose(s["max_drawdown_rupees"], -2_500.0)
    assert np.isclose(s["max_drawdown"], -2_500.0 / capital)


def test_drawdown_stays_meaningful_when_equity_goes_negative():
    """Dividing by the peak produced values beyond -100% (observed: -215%)
    once cumulative equity turned negative."""
    capital = 10_000.0
    s = _summarise(_trades([-5_000, -5_000, -5_000]), capital)
    assert np.isclose(s["max_drawdown_rupees"], -15_000.0)
    assert np.isclose(s["max_drawdown"], -1.5)   # -150% of the base, exactly


def test_drawdown_consistent_with_the_other_metrics():
    """return_pct, Sharpe and monthly returns all divide by capital; the
    drawdown must use the same denominator or the summary is incoherent."""
    capital = 100_000.0
    pnls = [5_000, -3_000, 8_000, -1_000]
    s = _summarise(_trades(pnls), capital)
    assert np.isclose(s["return_pct"], sum(pnls) / capital)
    # worst dip is the -3,000 right after the +5,000 peak
    assert np.isclose(s["max_drawdown_rupees"], -3_000.0)
    assert np.isclose(s["max_drawdown"], -3_000.0 / capital)


def test_no_drawdown_when_monotonically_profitable():
    s = _summarise(_trades([100, 200, 300]), 100_000.0)
    assert np.isclose(s["max_drawdown"], 0.0)
    assert np.isclose(s["max_drawdown_rupees"], 0.0)
