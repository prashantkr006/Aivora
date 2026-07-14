"""Auto-refresh Kite access tokens for all webapp users with TOTP creds.

Zerodha access tokens expire ~06:00 IST daily. Instead of manual OAuth
every morning, this script:

  1. Reads every user from the webapp DB.
  2. Filters to those with complete TOTP creds
     (api_key + api_secret + user_id/client_id + password + totp_secret).
  3. Performs the TOTP auto-login using ``kite_auth.totp_auto_login``.
  4. Stores the freshly-issued access_token encrypted in the webapp DB.
  5. Logs a summary — refreshed vs skipped vs failed.

Meant to be run from cron every day at ~06:15 IST. On systems where
cron isn't available (e.g. Windows dev), run manually.

Usage:
    docker compose exec dashboard python -m scripts.auto_refresh_kite_tokens
    docker compose exec dashboard python -m scripts.auto_refresh_kite_tokens --user-id 27
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aivora.live import kite_auth  # noqa: E402
from aivora.utils.logger import get_logger  # noqa: E402
from aivora.webapp import brokers, db as webapp_db  # noqa: E402

log = get_logger("scripts.auto_refresh_kite_tokens")


def _refresh_one(user_id: int) -> Tuple[str, str]:
    """Refresh one user's Kite access token via TOTP.

    Returns (status, detail) where status is 'refreshed' / 'skipped' / 'failed'.
    """
    z = brokers.get(user_id, "ZERODHA")
    if z is None:
        return "skipped", "no ZERODHA broker record"
    missing = [k for k, v in {
        "api_key":      z.api_key,
        "api_secret":   z.api_secret,
        "client_id":    z.client_id,
        "password":     z.password,
        "totp_secret":  z.totp_secret,
    }.items() if not v]
    if missing:
        return "skipped", f"incomplete TOTP creds — missing: {', '.join(missing)}"

    # totp_auto_login currently reads KITE_* env vars — populate them
    # in this process for the duration of the call. Not thread-safe but
    # this script is a single-shot daily cron.
    os.environ["KITE_API_KEY"] = z.api_key
    os.environ["KITE_API_SECRET"] = z.api_secret
    os.environ["KITE_USER_ID"] = z.client_id
    os.environ["KITE_PASSWORD"] = z.password
    os.environ["KITE_TOTP_SECRET"] = z.totp_secret

    try:
        new_token = kite_auth.totp_auto_login()
    except Exception as exc:  # noqa: BLE001
        return "failed", str(exc)

    if not new_token:
        return "failed", "totp_auto_login returned empty token"

    brokers.upsert(user_id, "ZERODHA", access_token=new_token)
    return "refreshed", f"new token (len={len(new_token)})"


def _all_user_ids() -> List[int]:
    """Return user_ids that have any ZERODHA broker record."""
    with webapp_db.connect() as c:
        rows = c.execute(
            "SELECT DISTINCT user_id FROM user_brokers WHERE broker='ZERODHA'"
        ).fetchall()
    return [int(r[0]) for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", type=int, default=None,
                    help="Only refresh this one user's token (default: all)")
    args = ap.parse_args()

    webapp_db.init_db()

    users = [args.user_id] if args.user_id else _all_user_ids()
    log.info("Auto-refresh Kite tokens for %d user(s)", len(users))

    stats = {"refreshed": 0, "skipped": 0, "failed": 0}
    for uid in users:
        status, detail = _refresh_one(uid)
        stats[status] += 1
        log.info("user %s → %s: %s", uid, status.upper(), detail)

    log.info(
        "Done — refreshed=%d skipped=%d failed=%d",
        stats["refreshed"], stats["skipped"], stats["failed"],
    )
    # Exit non-zero if any failure so cron alerts you.
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
