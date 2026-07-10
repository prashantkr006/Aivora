"""User model + password auth + session state.

Uses bcrypt via the ``bcrypt`` library.  Sessions live in
Streamlit's ``st.session_state`` — good enough for this iteration.
A subsequent pass can migrate to signed HTTP-only cookies for
persistence across browser refreshes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import bcrypt

from ..utils.logger import get_logger
from . import db as db_mod

log = get_logger(__name__)


# =============================================================
#  Data class
# =============================================================
@dataclass
class User:
    id: int
    email: str
    display_name: Optional[str]
    is_admin: bool
    created_at: str
    last_login: Optional[str]


# =============================================================
#  Password hashing
# =============================================================
def hash_password(plain: str) -> str:
    if not plain or len(plain) < 8:
        raise ValueError("Password must be at least 8 characters.")
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def _validate_email(email: str) -> str:
    """Strict RFC-5321 check via ``email_validator``.  Returns
    the normalised form."""
    from email_validator import EmailNotValidError, validate_email

    try:
        v = validate_email(email, check_deliverability=False)
    except EmailNotValidError as exc:
        raise ValueError(f"Invalid email: {exc}") from exc
    return v.normalized


# =============================================================
#  CRUD
# =============================================================
def register(
    email: str, password: str,
    display_name: Optional[str] = None,
    is_admin: bool = False,
) -> User:
    """Create a fresh user.  Raises ``ValueError`` on duplicate."""
    email = _validate_email(email)
    pwd_hash = hash_password(password)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with db_mod.connect() as conn:
        try:
            cur = conn.execute(
                """
                INSERT INTO users (email, password_hash, display_name, is_admin, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (email, pwd_hash, display_name, int(is_admin), now),
            )
        except Exception as exc:
            raise ValueError(f"Could not create user: {exc}") from exc
        uid = int(cur.lastrowid)
    log.info("Registered user id=%d email=%s admin=%s", uid, email, is_admin)
    return get_by_id(uid)


def authenticate(email: str, password: str) -> Optional[User]:
    """Return the user if the password matches, else None.

    We deliberately do NOT distinguish "user not found" vs.
    "wrong password" to callers — enumeration protection.
    """
    email = email.strip().lower()
    with db_mod.connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE lower(email) = ?", (email,)
        ).fetchone()
    if row is None:
        return None
    if not verify_password(password, row["password_hash"]):
        return None
    with db_mod.connect() as conn:
        conn.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), row["id"]),
        )
    return _row_to_user(row)


def change_password(user_id: int, old_password: str, new_password: str) -> None:
    with db_mod.connect() as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    if row is None:
        raise ValueError("User not found.")
    if not verify_password(old_password, row["password_hash"]):
        raise ValueError("Old password is incorrect.")
    new_hash = hash_password(new_password)
    with db_mod.connect() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user_id),
        )
    log.info("Password changed for user_id=%d", user_id)


def get_by_id(user_id: int) -> User:
    with db_mod.connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        raise ValueError(f"User id={user_id} not found.")
    return _row_to_user(row)


def list_users() -> List[User]:
    with db_mod.connect() as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY id").fetchall()
    return [_row_to_user(r) for r in rows]


def count_users() -> int:
    with db_mod.connect() as conn:
        row = conn.execute("SELECT COUNT(*) FROM users").fetchone()
    return int(row[0])


def _row_to_user(row) -> User:
    return User(
        id=int(row["id"]),
        email=row["email"],
        display_name=row["display_name"],
        is_admin=bool(row["is_admin"]),
        created_at=row["created_at"],
        last_login=row["last_login"],
    )
