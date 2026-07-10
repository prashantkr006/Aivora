"""Admin operations: activate / deactivate / delete users."""

from __future__ import annotations

from typing import Dict, List

from . import db as db_mod


def is_active(user_id: int) -> bool:
    """A user is 'active' unless deactivated_at is set.

    The column is added lazily so this module works on installs
    that predate the deactivation feature.
    """
    _ensure_column()
    with db_mod.connect() as conn:
        row = conn.execute(
            "SELECT deactivated_at FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return bool(row) and row["deactivated_at"] is None


def set_active(user_id: int, active: bool) -> None:
    _ensure_column()
    from datetime import datetime, timezone

    ts = None if active else datetime.now(timezone.utc).isoformat(timespec="seconds")
    with db_mod.connect() as conn:
        conn.execute(
            "UPDATE users SET deactivated_at = ? WHERE id = ?", (ts, user_id),
        )


def list_with_status() -> List[Dict]:
    _ensure_column()
    with db_mod.connect() as conn:
        rows = conn.execute(
            """
            SELECT u.id, u.email, u.display_name, u.is_admin,
                   u.created_at, u.last_login, u.deactivated_at,
                   GROUP_CONCAT(b.broker) AS brokers
            FROM users u
            LEFT JOIN user_brokers b ON b.user_id = u.id
            GROUP BY u.id
            ORDER BY u.id
            """
        ).fetchall()
    return [
        {
            "id": r["id"], "email": r["email"],
            "display_name": r["display_name"],
            "is_admin": bool(r["is_admin"]),
            "created_at": r["created_at"],
            "last_login": r["last_login"],
            "deactivated_at": r["deactivated_at"],
            "brokers": r["brokers"] or "",
            "active": r["deactivated_at"] is None,
        }
        for r in rows
    ]


def _ensure_column() -> None:
    """Add ``users.deactivated_at`` if the current DB doesn't have it.

    Idempotent — safe to call on every request; SQLite's schema
    check is essentially free.
    """
    with db_mod.connect() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
        if "deactivated_at" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN deactivated_at TEXT")
