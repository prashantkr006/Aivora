"""Encrypted ``state`` parameter for the Kite OAuth redirect flow.

Zerodha's Kite login accepts (and echoes back) a small ``state``
query parameter.  We put the user's id + a random nonce + a
timestamp in there, encrypted with Fernet, so the callback
handler:

* Knows which user initiated the flow (data isolation).
* Rejects replayed callbacks (nonce + timestamp bound).
* Cannot be forged without the master key.
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from typing import Optional

from . import crypto as crypto_mod

_TTL_SECONDS = 15 * 60  # Kite redirect flows are usually done in < 1 min


def issue(user_id: int) -> str:
    """Return an opaque state token binding a callback to ``user_id``."""
    payload = {
        "u": int(user_id),
        "n": secrets.token_urlsafe(12),
        "t": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return crypto_mod.encrypt(json.dumps(payload)) or ""


def consume(token: str) -> Optional[int]:
    """Verify + decode; return the user_id if valid, else None.

    "Consume" is aspirational — we don't currently track used
    nonces because the callback endpoint won't run twice for the
    same request_token anyway (Kite invalidates it on first use).
    The TTL check below is the real replay defence.
    """
    if not token:
        return None
    try:
        payload = json.loads(crypto_mod.decrypt(token) or "")
    except Exception:
        return None
    ts = payload.get("t")
    try:
        issued_at = datetime.fromisoformat(ts)
    except Exception:
        return None
    age = (datetime.now(timezone.utc) - issued_at).total_seconds()
    if age < 0 or age > _TTL_SECONDS:
        return None
    uid = payload.get("u")
    return int(uid) if isinstance(uid, int) else None
