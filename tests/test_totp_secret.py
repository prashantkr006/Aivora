"""Catching a bad TOTP seed when it is entered, not at 6:15am in a log.

A wrong value sat in the totp_secret field for at least three days. The
morning cron failed every one of them with pyotp's "Non-base32 digit found",
which says nothing to anyone, and the failure went only to a file nobody
reads. The user reconnected by hand each morning and concluded the
auto-login simply did not work.

Nothing about that was a hard bug. It was a wrong value that never got
checked, failing where nobody was looking.
"""

from __future__ import annotations

import pytest

from aivora.live.kite_auth import check_totp_secret, normalise_totp_secret

VALID = "JBSWY3DPEHPK3PXP"          # canonical base32 test seed


# -------------------------------------------------------------------
#  Normalising the common pastes
# -------------------------------------------------------------------
@pytest.mark.parametrize("raw", [
    VALID,
    VALID.lower(),                       # some apps show it lowercase
    "JBSW Y3DP EHPK 3PXP",               # spaced groups, as displayed
    "JBSW-Y3DP-EHPK-3PXP",               # hyphenated
    f"  {VALID}  ",
])
def test_a_tidy_paste_is_accepted(raw):
    assert normalise_totp_secret(raw) == VALID
    assert check_totp_secret(raw) is None


# -------------------------------------------------------------------
#  What actually went wrong
# -------------------------------------------------------------------
def test_the_production_value_is_rejected_with_a_reason():
    """15 chars, mixed case, containing digits outside base32."""
    why = check_totp_secret("Abcd0efgh1ijklm")
    assert why is not None
    assert "base32" in why
    assert "0" in why and "1" in why, "name the offending characters"


def test_the_six_digit_code_gets_its_own_message():
    """The most common mistake — pasting the code instead of the seed."""
    why = check_totp_secret("492817")
    assert why is not None
    assert "6-digit" in why and "seed" in why


def test_the_whole_otpauth_link_gets_its_own_message():
    why = check_totp_secret("otpauth://totp/Zerodha:AB1234?secret=JBSWY3DPEHPK3PXP")
    assert why is not None
    assert "secret=" in why


def test_empty_is_rejected():
    assert check_totp_secret("") is not None
    assert check_totp_secret(None) is not None


def test_the_reason_never_contains_the_secret():
    """An error message is not a place to print a credential."""
    secret = "Abcd0efgh1ijklm"
    assert secret not in (check_totp_secret(secret) or "")


# -------------------------------------------------------------------
#  Wiring
# -------------------------------------------------------------------
def test_the_login_flow_normalises_before_use():
    import inspect

    from aivora.live import kite_auth

    src = inspect.getsource(kite_auth.totp_auto_login)
    assert "normalise_totp_secret" in src
    assert 'totp_secret.replace(" ", "")' not in src


def test_the_morning_cron_checks_before_trying():
    import inspect

    from scripts import auto_refresh_kite_tokens as m

    src = inspect.getsource(m._refresh_one)
    assert "check_totp_secret" in src
    assert src.index("check_totp_secret") < src.index("totp_auto_login")


def test_a_failed_refresh_reaches_the_user():
    """The whole reason this lasted three days."""
    import inspect

    from scripts import auto_refresh_kite_tokens as m

    assert "_tell_the_user" in inspect.getsource(m.main)
    src = inspect.getsource(m._tell_the_user)
    assert "log_event" in src
    assert '"error"' in src


def test_the_profile_page_rejects_a_bad_seed_on_save():
    import inspect

    import app.multi_user_app as app_mod

    src = inspect.getsource(app_mod.profile_page)
    assert "check_totp_secret" in src
    assert "rejected" in src
