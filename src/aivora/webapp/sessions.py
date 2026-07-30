"""Persistent, HMAC-signed sessions for the Streamlit UI.

Design goals:

* Users stay logged in across browser refreshes.
* An attacker with the cookie can NOT extend expiry or impersonate
  another user without the server secret.
* No server-side session table — the cookie carries the user id
  plus an expiry stamp, signed with ``AIVORA_MASTER_KEY``.
* Cookie storage is best-effort — if the ``streamlit-cookies-
  controller`` package isn't available (older Streamlit) we fall
  back to ``st.session_state`` and refresh logs out, but nothing
  breaks.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from itsdangerous import BadSignature, SignatureExpired, TimestampSigner

from ..utils.logger import get_logger

log = get_logger(__name__)

_COOKIE_NAME = "aivora_session"
_MAX_AGE_SECONDS = 24 * 60 * 60


# =============================================================
#  Signed token
# =============================================================
def _signer() -> TimestampSigner:
    """Anchor sessions to the Fernet master key.

    We only need HMAC here (not encryption) — the payload is just
    a user id — but re-using the master key means we don't have
    to manage a second secret.
    """
    key = os.getenv("AIVORA_MASTER_KEY", "")
    if not key:
        raise RuntimeError("AIVORA_MASTER_KEY not set; cannot sign sessions.")
    return TimestampSigner(key, salt="aivora-session-v1")


def mint(user_id: int) -> str:
    """Return an opaque, signed token for ``user_id``."""
    signer = _signer()
    return signer.sign(str(int(user_id))).decode("ascii")


def verify(token: str, max_age_seconds: int = _MAX_AGE_SECONDS) -> Optional[int]:
    """Return the user_id if the token is valid, else None."""
    if not token:
        return None
    try:
        raw = _signer().unsign(token.encode("ascii"), max_age=max_age_seconds)
        return int(raw.decode("ascii"))
    except SignatureExpired:
        log.info("session expired")
        return None
    except (BadSignature, ValueError):
        log.warning("session cookie tampered / malformed")
        return None


# =============================================================
#  Streamlit integration
# =============================================================
def _cookies():
    """Return a CookieController when available, else None."""
    try:
        from streamlit_cookies_controller import CookieController  # type: ignore
    except Exception:  # pragma: no cover
        return None
    try:
        # Streamlit's built-in single-instance guard.
        import streamlit as st  # noqa: F401
        return CookieController(key="aivora-cookie")
    except Exception as exc:  # pragma: no cover
        log.warning("CookieController unavailable: %s", exc)
        return None


def _is_https() -> bool:
    """Best-effort check: is the dashboard served over HTTPS?

    Chrome and Firefox refuse cookies flagged ``secure=True`` unless
    the page was loaded over HTTPS, so localhost dev has to send
    ``secure=False``.  We read the dashboard URL from the env instead
    of the request (Streamlit doesn't expose scheme reliably behind a
    proxy).
    """
    return os.getenv("AIVORA_DASHBOARD_URL", "").lower().startswith("https://")


def install_from_cookie() -> Optional[int]:
    """If a valid session cookie exists, populate ``st.session_state``
    and return the user_id.  Called at the top of every page render.

    Note: ``streamlit-cookies-controller`` loads cookies asynchronously
    on the client side.  On the very first render after a browser
    refresh, ``ck.get(...)`` returns ``None`` because the JS component
    hasn't reported yet.  We give it one extra rerun to arrive before
    concluding the user isn't logged in — otherwise every hard refresh
    kicks the user back to the login page.
    """
    import streamlit as st

    # 1. If session_state already has the user id from a prior pass on
    #    this browser tab, trust it.  install_from_cookie may be called
    #    many times per render loop.
    uid_ss = st.session_state.get("user_id")
    if isinstance(uid_ss, int):
        return uid_ss

    # 2. Cookie path (survives refresh).
    ck = _cookies()
    if ck is not None:
        token = ck.get(_COOKIE_NAME)
        if isinstance(token, str) and token:
            uid = verify(token)
            if uid is not None:
                st.session_state["user_id"] = uid
                return uid
        # First-load: the controller's JS hasn't sent cookies yet.
        # Rerun ONCE so the next pass has them.  Guarded by a flag so
        # we never rerun again for the same tab (would infinite-loop
        # for genuinely-logged-out users).
        if not st.session_state.get("_ck_probed"):
            st.session_state["_ck_probed"] = True
            st.rerun()

    return None


def issue(user_id: int) -> None:
    """Set both the cookie and session_state after a successful login."""
    import streamlit as st

    st.session_state["user_id"] = int(user_id)
    # Once we know the user, drop the probe flag so a later logout that
    # revokes the cookie gets a fresh probe on the NEXT tab open.
    st.session_state.pop("_ck_probed", None)
    ck = _cookies()
    if ck is None:
        return
    token = mint(user_id)
    expires = datetime.now(timezone.utc) + timedelta(seconds=_MAX_AGE_SECONDS)
    try:
        ck.set(_COOKIE_NAME, token, expires=expires,
               secure=_is_https(), same_site="lax")
    except Exception as exc:
        log.warning("cookie set failed: %s", exc)


def revoke() -> None:
    """Clear the cookie + session_state on logout."""
    import streamlit as st

    for k in ("user_id", "mode", "_flash", "_page", "_ck_probed"):
        st.session_state.pop(k, None)
    ck = _cookies()
    if ck is None:
        return
    try:
        ck.remove(_COOKIE_NAME)
    except Exception as exc:
        log.warning("cookie remove failed: %s", exc)
