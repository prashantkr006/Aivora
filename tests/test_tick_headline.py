"""Regression tests for the per-tick summary headline.

The bug these pin: the headline was built from every entry *attempt*,
ignoring whether the order actually filled.  A live order rejected by the
broker (observed in production: "No IPs configured for this app") was
therefore logged as "trade opened" on every retry while `user_trades` stayed
empty — the log claimed a position the account did not hold.
"""

from __future__ import annotations

import re

import pytest


def _headline(actions, hhmm="12:50"):
    """Mirror of the headline construction in run_user_tick."""
    def _label(a):
        side_txt = "CALL" if a.get("side") == "CE" else "PUT"
        return f"{a.get('symbol', '?')} {side_txt}"

    opened = [a for a in actions
              if a.get("entered_live") or a.get("entered_paper")]
    attempted = [a for a in actions if a not in opened]

    if opened:
        headline = "🚀 " + ", ".join(_label(a) for a in opened) + " — trade opened"
        if attempted:
            headline += ("  |  ⚠️ not filled: "
                         + ", ".join(_label(a) for a in attempted))
    elif attempted:
        headline = ("⚠️ " + ", ".join(_label(a) for a in attempted)
                    + " — entry attempted but NOT filled (see error above)")
    else:
        headline = f"✅ {hhmm} — checked, nothing to trade"
    return headline


def test_rejected_live_order_is_not_reported_as_opened():
    """The exact production case: order rejected, nothing held."""
    actions = [{"entered_live": False, "symbol": "BANKNIFTY", "side": "PE"}]
    h = _headline(actions)
    assert "trade opened" not in h
    assert "NOT filled" in h
    assert "BANKNIFTY PUT" in h


def test_successful_live_order_is_reported_as_opened():
    actions = [{"entered_live": True, "symbol": "BANKNIFTY", "side": "PE"}]
    h = _headline(actions)
    assert h.startswith("🚀")
    assert "BANKNIFTY PUT — trade opened" in h
    assert "NOT filled" not in h


def test_paper_entry_is_reported_as_opened():
    actions = [{"entered_paper": True, "symbol": "NIFTY", "side": "CE"}]
    h = _headline(actions)
    assert "NIFTY CALL — trade opened" in h


def test_mixed_fill_and_rejection_reports_both():
    actions = [
        {"entered_live": True, "symbol": "NIFTY", "side": "CE"},
        {"entered_live": False, "symbol": "BANKNIFTY", "side": "PE"},
    ]
    h = _headline(actions)
    assert "NIFTY CALL — trade opened" in h
    assert "not filled: BANKNIFTY PUT" in h


def test_no_actions_is_the_quiet_case():
    assert _headline([]) == "✅ 12:50 — checked, nothing to trade"


def test_headline_logic_matches_the_engine():
    """Guard against the engine drifting away from this mirrored logic."""
    import inspect

    from aivora.webapp import trading_engine

    src = inspect.getsource(trading_engine.run_user_tick)
    # The fix hinges on filtering by the entered_* flags before building
    # the headline; if that disappears the false-success bug is back.
    assert 'a.get("entered_live") or a.get("entered_paper")' in src
    assert "attempted" in src
