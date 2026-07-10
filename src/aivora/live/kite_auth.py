"""Kite Connect authentication — dashboard-friendly.

Two flows are exposed:

1. **Redirect flow** (default, no credentials stored).

   * :func:`login_url` returns Kite's official login URL.
   * The user opens it, logs in on Zerodha's page, and Zerodha
     redirects back to the app's registered redirect URL with
     ``?request_token=XXX`` appended.
   * :func:`exchange_request_token` swaps that ``request_token``
     (plus your API secret) for a 24-hour ``access_token`` and
     writes it to ``.env``.

2. **TOTP auto-login flow** (optional, one click, no browser).

   * :func:`totp_auto_login` performs the whole login by hitting
     Kite's internal endpoints directly, using ``pyotp`` for the
     six-digit code.  Requires the four extra secrets in ``.env``
     (``KITE_USER_ID``, ``KITE_PASSWORD``, ``KITE_TOTP_SECRET``,
     plus the existing ``KITE_API_KEY`` / ``KITE_API_SECRET``).

Neither flow stores the access token on disk beyond the ``.env``
file the rest of the app already reads.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import requests

from ..utils.config import get_config
from ..utils.logger import get_logger

log = get_logger(__name__)


# =============================================================
#  .env writer
# =============================================================
def _write_env(key: str, value: str) -> None:
    """Insert or replace ``key=value`` in the repo's ``.env`` file.

    Idempotent — safe to call every day.
    """
    env_path = Path(get_config().repo_root) / ".env"
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    prefix = f"{key}="
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            lines[i] = f"{prefix}{value}"
            break
    else:
        lines.append(f"{prefix}{value}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Also update the running process's env so the current session
    # picks up the new token without a restart.
    os.environ[key] = value
    log.info("Wrote %s to %s", key, env_path)


# =============================================================
#  Token status (used by the UI to render a nice indicator)
# =============================================================
@dataclass
class TokenStatus:
    present: bool
    length: int
    last_modified: Optional[datetime]
    hint: str


def token_status() -> TokenStatus:
    """Return whether an access_token is present and how fresh it is."""
    creds = get_config().kite_credentials()
    if not creds.access_token:
        return TokenStatus(
            present=False, length=0, last_modified=None,
            hint="No token in .env — click login below to generate one.",
        )
    env_path = Path(get_config().repo_root) / ".env"
    mtime = datetime.fromtimestamp(env_path.stat().st_mtime) if env_path.exists() else None
    age_h = ((datetime.now() - mtime).total_seconds() / 3600.0) if mtime else 99.0
    if age_h >= 24:
        hint = "Older than 24 hours — Kite tokens expire at 06:00 IST daily."
    elif age_h >= 6:
        hint = f"~{age_h:.1f} hours old. Still valid until next 06:00 IST."
    else:
        hint = f"Fresh ({age_h:.1f} h old)."
    return TokenStatus(
        present=True, length=len(creds.access_token),
        last_modified=mtime, hint=hint,
    )


# =============================================================
#  Flow 1: redirect (no stored password)
# =============================================================
def login_url() -> str:
    """Return the Kite login URL for the current API key.

    The user opens this in a browser, logs in with their normal
    Zerodha credentials + TOTP.  Kite then redirects to whatever
    redirect URL is registered on developers.kite.trade with
    ``?request_token=XXX&status=success`` appended.
    """
    creds = get_config().kite_credentials()
    if not creds.api_key:
        raise RuntimeError(
            "KITE_API_KEY missing in .env — add it before generating a login URL."
        )
    from kiteconnect import KiteConnect  # local import keeps startup light

    return KiteConnect(api_key=creds.api_key).login_url()


def extract_request_token(url_or_query: str) -> Optional[str]:
    """Pull ``request_token`` out of either a full URL or a query string.

    Handles the three shapes users typically paste in:

    * Full URL: ``https://your-app/?request_token=XYZ&status=success``
    * Query only: ``?request_token=XYZ&status=success``
    * Bare token: ``XYZ``  (kept as an escape hatch)
    """
    s = url_or_query.strip()
    if not s:
        return None
    # Bare token has no query separator.
    if "=" not in s and "?" not in s and len(s) > 12:
        return s
    try:
        parsed = urlparse(s)
        params = parse_qs(parsed.query or s.lstrip("?"))
        vals = params.get("request_token")
        return vals[0] if vals else None
    except Exception:
        return None


def exchange_request_token(request_token: str) -> str:
    """Exchange the one-shot ``request_token`` for a 24-hour access token.

    Uses ``KiteConnect.generate_session`` under the hood — no
    Zerodha login state is stored beyond the returned token,
    which we then write to ``.env``.
    """
    creds = get_config().kite_credentials()
    if not creds.api_key or not creds.api_secret:
        raise RuntimeError(
            "KITE_API_KEY and KITE_API_SECRET must both be in .env before exchanging."
        )
    from kiteconnect import KiteConnect

    kite = KiteConnect(api_key=creds.api_key)
    data = kite.generate_session(request_token.strip(), api_secret=creds.api_secret)
    access_token = data["access_token"]
    _write_env("KITE_ACCESS_TOKEN", access_token)
    if data.get("user_id"):
        _write_env("KITE_USER_ID", data["user_id"])
    log.info("Kite access token refreshed for user_id=%s", data.get("user_id"))
    return access_token


# =============================================================
#  Flow 2: TOTP auto-login (optional, one click, no browser)
# =============================================================
_LOGIN_URL = "https://kite.zerodha.com/api/login"
_TWOFA_URL = "https://kite.zerodha.com/api/twofa"
_CONNECT_URL_TEMPLATE = "https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"


def totp_auto_login(
    user_id: Optional[str] = None,
    password: Optional[str] = None,
    totp_secret: Optional[str] = None,
) -> str:
    """Silently perform the full Kite login using TOTP.

    Requires (either as arguments or via env vars):

    * ``KITE_USER_ID``
    * ``KITE_PASSWORD``
    * ``KITE_TOTP_SECRET``  — the seed from your Zerodha 2FA QR code
      (NOT the current 6-digit code).
    * ``KITE_API_KEY`` + ``KITE_API_SECRET``

    Returns the new access_token and writes it to ``.env`` as a
    side effect.  Raises ``RuntimeError`` on any missing input or
    Zerodha error.
    """
    try:
        import pyotp
    except ImportError as exc:
        raise RuntimeError(
            "pyotp not installed — run `pip install pyotp` (or reinstall requirements)."
        ) from exc

    user_id = user_id or os.getenv("KITE_USER_ID", "")
    password = password or os.getenv("KITE_PASSWORD", "")
    totp_secret = totp_secret or os.getenv("KITE_TOTP_SECRET", "")
    creds = get_config().kite_credentials()

    missing = [k for k, v in {
        "KITE_USER_ID": user_id, "KITE_PASSWORD": password,
        "KITE_TOTP_SECRET": totp_secret,
        "KITE_API_KEY": creds.api_key, "KITE_API_SECRET": creds.api_secret,
    }.items() if not v]
    if missing:
        raise RuntimeError(
            "TOTP auto-login needs these env vars: " + ", ".join(missing)
        )

    with requests.Session() as sess:
        # Step 1 — user_id + password → request_id.
        r1 = sess.post(_LOGIN_URL, data={"user_id": user_id, "password": password}, timeout=10)
        try:
            j1 = r1.json()
        except ValueError:
            raise RuntimeError(f"Login step 1 returned non-JSON: {r1.text[:200]}")
        if j1.get("status") != "success":
            raise RuntimeError(f"Login step 1 failed: {j1.get('message')}")
        request_id = j1["data"]["request_id"]

        # Step 2 — TOTP.
        totp_code = pyotp.TOTP(totp_secret.replace(" ", "")).now()
        r2 = sess.post(_TWOFA_URL, data={
            "user_id": user_id, "request_id": request_id,
            "twofa_value": totp_code, "twofa_type": "totp",
            "skip_session": "",
        }, timeout=10)
        try:
            j2 = r2.json()
        except ValueError:
            raise RuntimeError(f"TOTP step returned non-JSON: {r2.text[:200]}")
        if j2.get("status") != "success":
            raise RuntimeError(f"TOTP step failed: {j2.get('message')}")

        # Step 3 — hit /connect/login and capture the request_token
        # from the eventual redirect URL.
        r3 = sess.get(
            _CONNECT_URL_TEMPLATE.format(api_key=creds.api_key),
            allow_redirects=True, timeout=10,
        )
        # After all the redirects, the final URL contains request_token=.
        rq = extract_request_token(r3.url)
        if not rq:
            # Sometimes Kite embeds the token inside an HTML redirect script
            # instead of an HTTP 302 — grab it out of the body too.
            m = re.search(r"request_token=([A-Za-z0-9]+)", r3.text or "")
            rq = m.group(1) if m else None
        if not rq:
            raise RuntimeError(
                "Could not extract request_token from the connect flow. "
                "Check that the redirect URL registered on developers.kite.trade "
                "is reachable from this machine."
            )

    # Step 4 — reuse the redirect flow to swap request_token → access_token.
    return exchange_request_token(rq)
