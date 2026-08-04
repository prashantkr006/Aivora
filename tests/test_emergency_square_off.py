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
