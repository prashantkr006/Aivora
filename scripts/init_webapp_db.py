"""Bootstrap the multi-user webapp DB.

Idempotent — safe to re-run.  On first run it:

    1. Generates and writes ``AIVORA_MASTER_KEY`` into ``.env``
       (unless the env var is already set).
    2. Creates every table + index.
    3. Optionally creates the first admin user
       (``--admin-email`` / ``--admin-password``).

Usage::

    python -m scripts.init_webapp_db
    python -m scripts.init_webapp_db --admin-email you@x.com --admin-password 'S3cret!!'
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aivora.utils.logger import get_logger  # noqa: E402
from aivora.webapp import crypto as crypto_mod  # noqa: E402
from aivora.webapp import db as db_mod  # noqa: E402
from aivora.webapp import users as user_mod  # noqa: E402

log = get_logger("scripts.init_webapp_db")


def main() -> int:
    ap = argparse.ArgumentParser(description="Bootstrap the AiVora multi-user webapp DB")
    ap.add_argument("--admin-email", type=str, default=None)
    ap.add_argument("--admin-password", type=str, default=None,
                    help="Optional; interactively prompted if only --admin-email given.")
    ap.add_argument("--skip-master-key", action="store_true",
                    help="Do not touch .env; assume the caller managed AIVORA_MASTER_KEY.")
    args = ap.parse_args()

    # ---- master key ----
    if not args.skip_master_key:
        try:
            key = crypto_mod.install_master_key_to_env()
            print(f"AIVORA_MASTER_KEY ready (len={len(key)}).")
        except Exception as exc:
            log.error("Failed to install master key: %s", exc)
            return 2

    # ---- tables ----
    db_mod.init_db()
    print(f"Tables initialised at {db_mod.default_db_path()}")

    # ---- optional admin bootstrap ----
    if args.admin_email:
        password = args.admin_password
        if not password:
            password = getpass.getpass(f"Password for {args.admin_email}: ")
        try:
            u = user_mod.register(
                email=args.admin_email, password=password,
                display_name="Admin", is_admin=True,
            )
            print(f"Admin user created: id={u.id} email={u.email}")
        except ValueError as exc:
            if "UNIQUE" in str(exc) or "already" in str(exc).lower():
                print(f"Admin user {args.admin_email} already exists — skipping.")
            else:
                raise
    else:
        print("(No --admin-email supplied — no admin bootstrap.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
