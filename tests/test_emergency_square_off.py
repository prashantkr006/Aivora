"""Regression tests for the emergency square-off button.

The bug these pin: the button marked every open trade closed in the database
without sending a single order to the broker.  In production the user hit it,
AiVora showed the position exited, and Zerodha still held it — they had to
exit manually.  Worse, once a trade is flagged closed the position tracker
stops monitoring it, so the live position was left with no stop loss, no
target and no horizon exit.
"""

from __future__ import annotations

import inspect
import re

import pytest


def _source():
    import app.multi_user_app as m
    return inspect.getsource(m._emergency_square_off)


def test_live_path_places_real_orders():
    """Live must go through close_live_trade, which only closes the book
    entry after the broker confirms a COMPLETE fill."""
    src = _source()
    assert "close_live_trade" in src, (
        "live square-off must send an exit order, not just update the DB"
    )


def test_live_path_does_not_close_the_book_directly():
    """portfolio.close_trade() must never be called on the live branch —
    that is exactly what detached the book from the broker."""
    src = _source()
    live_branch = src.split('if mode != "live":')[1]
    # the paper branch (immediately after the guard) may call close_trade;
    # everything after its `return` is the live path.
    paper, live = live_branch.split("return", 1)
    assert "close_trade(" in paper, "paper branch should still do bookkeeping"
    assert "portfolio.close_trade(" not in live, (
        "live branch must not mark trades closed itself"
    )


def test_missing_broker_credentials_warns_instead_of_faking_success():
    """With no usable Kite session the user must be told to exit manually —
    silently 'closing' positions is what caused the incident."""
    src = _source()
    assert "manually" in src.lower()
    # must bail out rather than fall through to bookkeeping
    assert "access_token" in src


def test_unfilled_positions_are_reported_not_hidden():
    """Anything that did not actually exit must be surfaced by symbol."""
    src = _source()
    assert "still_open" in src
    assert re.search(r"did NOT exit|still open", src), (
        "partial failures must be reported to the user"
    )


def test_paper_mode_still_works_without_a_broker():
    """Paper square-off must not require Kite credentials."""
    src = _source()
    paper = src.split('if mode != "live":')[1].split("return")[0]
    assert "broker_mod" not in paper
    assert "kite" not in paper.lower()


def test_button_delegates_to_the_helper():
    """The sidebar button must call the helper rather than re-implementing
    bookkeeping inline, which is how the bug survived unnoticed."""
    import app.multi_user_app as m

    src = inspect.getsource(m.sidebar_for)
    assert "_emergency_square_off(user, portfolio, mode)" in src


# -------------------------------------------------------------------
#  The handler must survive its own error path
# -------------------------------------------------------------------
# Second incident: close_live_trade raised, and the except block called
# portfolio.append_log — a method the DB-backed UserPortfolio did not have.
# The AttributeError escaped and took the entire dashboard down, at the
# exact moment the user was trying to exit a live position.

def test_user_portfolio_answers_to_append_log():
    """The engine layer calls append_log; UserPortfolio must accept it."""
    from aivora.webapp.portfolios import UserPortfolio

    assert hasattr(UserPortfolio, "append_log")


def test_append_log_forwards_to_log_event():
    from aivora.webapp.portfolios import UserPortfolio

    seen = []
    up = UserPortfolio.__new__(UserPortfolio)
    up.log_event = lambda msg, level="info": seen.append((msg, level))
    up.append_log("boom", "error")
    assert seen == [("boom", "error")]


def test_a_failing_log_cannot_kill_the_dashboard():
    """Even if logging itself explodes, the button must keep going."""
    src = _source()
    live = src.split('if mode != "live":')[1].split("return", 1)[1]
    handler = live.split("except Exception")[1]
    assert handler.count("try:") >= 1, (
        "the error handler's own logging must be guarded"
    )


def test_failure_reasons_reach_the_user():
    """Standing at the panic button, the user needs to know what is still
    exposed *and why* — not just a count."""
    src = _source()
    assert "failures" in src
    assert re.search(r"failures\.append", src)
    assert re.search(r"for f in failures", src), (
        "collected reasons must actually be rendered"
    )
