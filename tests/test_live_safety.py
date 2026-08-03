"""Regression tests for the live-order pre-flight check.

The bug these pin: ``assert_can_trade_live`` validated the *process
environment* credentials (KITE_API_KEY / KITE_ACCESS_TOKEN) while the
multi-user path places orders with per-user credentials decrypted from
``user_brokers``.  In production the container has no KITE_* variables, so
every live entry failed with "Kite credentials missing in .env" even though
the user's own Kite session was valid and had just served option quotes.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from aivora.live.safety import SafetyError, assert_can_trade_live
from aivora.utils.calendar import is_trading_day


class _Creds:
    def __init__(self, api_key: str = "", access_token: str = ""):
        self.api_key = api_key
        self.access_token = access_token


class _Kite:
    def __init__(self, creds):
        self.creds = creds


class _Portfolio:
    """Minimal stand-in exposing only what the safety check reads."""

    def __init__(self, now: datetime):
        msoo = now.hour * 60 + now.minute - (9 * 60 + 15)
        self._state = {
            "master_switch": True,
            "mode": "live",
            "initial_capital": 100_000.0,
            "trades": [],
            "settings": {
                # Window wide enough that the check passes whenever the test
                # runs during a session; the trading-day guard is handled by
                # skipping on non-trading days.
                "min_minutes_since_open": msoo - 5,
                "max_minutes_since_open": msoo + 5,
                "daily_loss_limit_pct": 0.05,
            },
        }

    def load(self):
        return self._state


def _portfolio_now():
    now = datetime.now()
    if not is_trading_day(now.date()):
        pytest.skip("safety check requires a trading day")
    return _Portfolio(now), now


def test_valid_per_user_creds_pass_even_with_empty_environment(monkeypatch):
    """The whole point of the fix: a valid per-user client must be accepted
    regardless of what the process environment contains."""
    monkeypatch.delenv("KITE_API_KEY", raising=False)
    monkeypatch.delenv("KITE_ACCESS_TOKEN", raising=False)

    portfolio, _ = _portfolio_now()
    kite = _Kite(_Creds(api_key="user_key", access_token="user_token"))

    # Must not raise a *credential* error.  Other environment-dependent
    # checks (frozen model files) may still fail; only credentials matter here.
    try:
        assert_can_trade_live(portfolio, kite)
    except SafetyError as exc:
        assert "credentials" not in str(exc).lower(), (
            f"valid per-user credentials were rejected: {exc}"
        )


def test_missing_per_user_creds_are_rejected(monkeypatch):
    """A user whose Kite token expired must still be blocked — even if the
    process environment happens to hold valid credentials."""
    monkeypatch.setenv("KITE_API_KEY", "env_key")
    monkeypatch.setenv("KITE_ACCESS_TOKEN", "env_token")

    portfolio, _ = _portfolio_now()
    kite = _Kite(_Creds(api_key="user_key", access_token=""))  # expired token

    with pytest.raises(SafetyError, match="(?i)credentials"):
        assert_can_trade_live(portfolio, kite)


def test_legacy_path_without_client_still_uses_environment(monkeypatch):
    """The single-user scheduler passes no client; that path must keep
    validating environment credentials rather than silently skipping."""
    monkeypatch.delenv("KITE_API_KEY", raising=False)
    monkeypatch.delenv("KITE_ACCESS_TOKEN", raising=False)

    portfolio, _ = _portfolio_now()

    with pytest.raises(SafetyError, match="(?i)credentials"):
        assert_can_trade_live(portfolio)


def test_executor_forwards_client_to_safety_check():
    """Both live order paths must pass their client through — if either
    reverts to the no-client form the credential bug returns."""
    import inspect

    from aivora.live import live_executor

    src = inspect.getsource(live_executor)
    assert "assert_can_trade_live(portfolio, kite)" in src
    assert "assert_can_exit_live(portfolio, kite)" in src
    assert "assert_can_trade_live(portfolio)" not in src.replace(
        "assert_can_trade_live(portfolio, kite)", ""
    )


# ── exit path must not inherit entry gates ────────────────────────
#
# These pin the second bug: close_live_trade used the entry pre-flight, so a
# breached loss cap / closed entry window / master switch OFF stranded open
# positions, and emergency_square_off was blocked too.

def _exit_portfolio(**overrides):
    now = datetime.now()
    if not is_trading_day(now.date()):
        pytest.skip("safety check requires a trading day")
    p = _Portfolio(now)
    p._state.update(overrides)
    return p


def _good_kite():
    return _Kite(_Creds(api_key="k", access_token="t"))


def test_exit_allowed_when_master_switch_off():
    from aivora.live.safety import assert_can_exit_live

    assert_can_exit_live(_exit_portfolio(master_switch=False), _good_kite())


def test_exit_allowed_outside_entry_window():
    from aivora.live.safety import assert_can_exit_live

    p = _exit_portfolio()
    # An entry window that has certainly closed.
    p._state["settings"]["min_minutes_since_open"] = 0
    p._state["settings"]["max_minutes_since_open"] = 1
    assert_can_exit_live(p, _good_kite())


def test_exit_allowed_after_daily_loss_cap_breached():
    """The most dangerous case: losses hit the cap, so closing the losing
    position is exactly what must still be possible."""
    from aivora.live.safety import assert_can_exit_live

    p = _exit_portfolio()
    p._state["trades"] = [{
        "exit_time": datetime.now().date().isoformat() + "T10:00:00",
        "realized_pnl": -99_000.0,
    }]
    assert_can_exit_live(p, _good_kite())


def test_exit_still_blocked_without_credentials():
    from aivora.live.safety import assert_can_exit_live

    with pytest.raises(SafetyError, match="(?i)credentials"):
        assert_can_exit_live(_exit_portfolio(), _Kite(_Creds("k", "")))


def test_exit_still_blocked_for_paper_portfolio():
    from aivora.live.safety import assert_can_exit_live

    with pytest.raises(SafetyError, match="(?i)live mode"):
        assert_can_exit_live(_exit_portfolio(mode="paper"), _good_kite())


def test_entry_still_enforces_all_gates():
    """The entry path must keep every gate it had — the fix must not have
    loosened new-risk controls."""
    p = _exit_portfolio(master_switch=False)
    with pytest.raises(SafetyError, match="(?i)master switch"):
        assert_can_trade_live(p, _good_kite())

    p2 = _exit_portfolio()
    p2._state["trades"] = [{
        "exit_time": datetime.now().date().isoformat() + "T10:00:00",
        "realized_pnl": -99_000.0,
    }]
    with pytest.raises(SafetyError, match="(?i)daily loss"):
        assert_can_trade_live(p2, _good_kite())
