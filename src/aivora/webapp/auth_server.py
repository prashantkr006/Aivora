"""FastAPI sidecar that handles Kite's OAuth redirect.

Streamlit can't register arbitrary HTTP routes, so we run this
tiny second process on port 8502 and register its `/kite/callback`
URL with Zerodha as the app's Redirect URL.  When Kite bounces
the user back with ``?request_token=…&state=…``:

    1. Decrypt the state param → user_id.
    2. Load the user's Kite api_key + api_secret from the
       encrypted store.
    3. Exchange the request_token for an access_token.
    4. Encrypt and store the access_token against that user.
    5. Redirect the browser back to the dashboard.

The server holds no other state — everything lives in the
existing SQLite DB.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the aivora package importable when this file is run as a script.
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import HTMLResponse, RedirectResponse  # noqa: E402

from aivora.utils.logger import get_logger  # noqa: E402
from aivora.webapp import brokers, oauth_state  # noqa: E402

log = get_logger(__name__)

app = FastAPI(title="AiVora auth sidecar")

# Where users should be sent after a successful login.  Overridable
# via env so Docker / VPS deployments can point somewhere real.
DASHBOARD_URL = os.getenv("AIVORA_DASHBOARD_URL", "http://localhost:8501")


@app.get("/health")
def health():
    return {"status": "ok", "service": "aivora-auth"}


@app.get("/kite/callback", response_class=HTMLResponse)
def kite_callback(request: Request):
    """Redirect target registered on developers.kite.trade.

    Zerodha appends ``request_token``, ``action`` and — critically —
    the ``state`` we set in :func:`kite_login_url`.
    """
    request_token = request.query_params.get("request_token")
    state = request.query_params.get("state")
    status = request.query_params.get("status")

    if status and status != "success":
        return _html_error(f"Kite returned status={status!r}")
    if not request_token or not state:
        return _html_error("Missing request_token or state in Kite callback.")

    user_id = oauth_state.consume(state)
    if user_id is None:
        raise HTTPException(status_code=400, detail="Invalid or expired state.")

    creds = brokers.get(user_id, "ZERODHA")
    if creds is None or not creds.api_key or not creds.api_secret:
        return _html_error(
            "Zerodha api_key / api_secret not on file for this user. "
            "Save them on the Profile page before connecting."
        )

    try:
        from kiteconnect import KiteConnect
    except ImportError:
        return _html_error("kiteconnect not installed on the server.")

    kite = KiteConnect(api_key=creds.api_key)
    try:
        data = kite.generate_session(request_token, api_secret=creds.api_secret)
        access_token = data["access_token"]
    except Exception as exc:  # noqa: BLE001
        log.exception("Kite generate_session failed for user_id=%s", user_id)
        return _html_error(f"Kite rejected the request_token: {exc}")

    brokers.upsert(
        user_id, "ZERODHA",
        client_id=data.get("user_id") or creds.client_id,
        access_token=access_token,
    )
    log.info("stored fresh Kite access_token for user_id=%s", user_id)

    return RedirectResponse(url=f"{DASHBOARD_URL}?kite_connected=1", status_code=303)


def _html_error(msg: str) -> HTMLResponse:
    body = (
        f"<html><body style='font-family:sans-serif;padding:2rem;max-width:640px;margin:auto'>"
        f"<h2>Login failed</h2><p>{msg}</p>"
        f"<p><a href='{DASHBOARD_URL}'>Return to dashboard</a></p>"
        f"</body></html>"
    )
    return HTMLResponse(content=body, status_code=400)


def kite_login_url(user_id: int) -> str:
    """Public helper for the Streamlit UI: returns the login URL
    with our ``state`` param embedded."""
    from kiteconnect import KiteConnect

    creds = brokers.get(user_id, "ZERODHA")
    if creds is None or not creds.api_key:
        raise RuntimeError("Save your Kite API key on the Profile page first.")
    state = oauth_state.issue(user_id)
    # KiteConnect.login_url() gives us the base; we append our state.
    base = KiteConnect(api_key=creds.api_key).login_url()
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}redirect_params=state%3D{state}"


def main() -> int:
    """Run the sidecar with uvicorn.  Invoked by scripts/run_auth_server.py."""
    import uvicorn

    port = int(os.getenv("AIVORA_AUTH_PORT", "8502"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
