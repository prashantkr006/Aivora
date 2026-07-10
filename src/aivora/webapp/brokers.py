"""Per-user encrypted broker credentials.

Every secret column (``api_key_enc``, ``api_secret_enc``,
``access_token_enc``, ``totp_secret_enc``, ``password_enc``) is
stored as Fernet ciphertext and only decrypted in memory at the
moment it's needed.  Callers never see the ciphertext directly —
they always get a plaintext ``BrokerCredentials`` back.

Two brokers per user: ``ZERODHA`` and ``DHAN``.  A user can have
one, both, or neither.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from ..utils.logger import get_logger
from . import crypto as crypto_mod
from . import db as db_mod

log = get_logger(__name__)

_BROKERS = ("ZERODHA", "DHAN")


# =============================================================
#  Public data class (always plaintext — never leaks ciphertext)
# =============================================================
@dataclass
class BrokerCredentials:
    broker: str
    client_id: Optional[str]
    api_key: Optional[str]
    api_secret: Optional[str]
    access_token: Optional[str]
    totp_secret: Optional[str]
    password: Optional[str]
    token_updated_at: Optional[str]
    is_active: bool

    def has_data_creds(self) -> bool:
        """Enough to fetch market data / place orders on Kite."""
        if self.broker == "ZERODHA":
            return bool(self.api_key and self.access_token)
        if self.broker == "DHAN":
            return bool(self.client_id and self.access_token)
        return False

    def has_totp_creds(self) -> bool:
        return bool(self.api_key and self.api_secret and self.password and self.totp_secret)


# =============================================================
#  Read
# =============================================================
def get(user_id: int, broker: str) -> Optional[BrokerCredentials]:
    """Return the user's credentials for one broker, decrypted."""
    _assert_broker(broker)
    with db_mod.connect() as conn:
        row = conn.execute(
            "SELECT * FROM user_brokers WHERE user_id = ? AND broker = ?",
            (user_id, broker),
        ).fetchone()
    if row is None:
        return None
    return _row_to_creds(row)


def list_for_user(user_id: int) -> List[BrokerCredentials]:
    with db_mod.connect() as conn:
        rows = conn.execute(
            "SELECT * FROM user_brokers WHERE user_id = ? ORDER BY broker",
            (user_id,),
        ).fetchall()
    return [_row_to_creds(r) for r in rows]


# =============================================================
#  Write — one function that ALWAYS goes through Fernet
# =============================================================
def upsert(
    user_id: int,
    broker: str,
    *,
    client_id: Optional[str] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    access_token: Optional[str] = None,
    totp_secret: Optional[str] = None,
    password: Optional[str] = None,
    is_active: Optional[bool] = None,
) -> None:
    """Store or update broker creds.  Every ``*_enc`` column is
    Fernet-encrypted on write.

    Passing ``None`` means "leave this field alone".  Passing an
    empty string means "clear it".
    """
    _assert_broker(broker)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    existing = get(user_id, broker)
    if existing is None:
        # Fresh row — treat every None as blank so we insert a
        # complete record.  ``encrypt`` handles None + "".
        with db_mod.connect() as conn:
            conn.execute(
                """
                INSERT INTO user_brokers
                    (user_id, broker, client_id,
                     api_key_enc, api_secret_enc, access_token_enc,
                     totp_secret_enc, password_enc,
                     token_updated_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id, broker, client_id or "",
                    crypto_mod.encrypt(api_key or ""),
                    crypto_mod.encrypt(api_secret or ""),
                    crypto_mod.encrypt(access_token or ""),
                    crypto_mod.encrypt(totp_secret or ""),
                    crypto_mod.encrypt(password or ""),
                    now if access_token else None,
                    int(True if is_active is None else is_active),
                ),
            )
        log.info("Created broker creds for user_id=%d broker=%s", user_id, broker)
        return

    # Partial update — merge over the existing row.
    updates: List[str] = []
    params: List = []

    def _add(col: str, value):
        updates.append(f"{col} = ?")
        params.append(value)

    if client_id is not None:
        _add("client_id", client_id)
    if api_key is not None:
        _add("api_key_enc", crypto_mod.encrypt(api_key))
    if api_secret is not None:
        _add("api_secret_enc", crypto_mod.encrypt(api_secret))
    if access_token is not None:
        _add("access_token_enc", crypto_mod.encrypt(access_token))
        _add("token_updated_at", now)
    if totp_secret is not None:
        _add("totp_secret_enc", crypto_mod.encrypt(totp_secret))
    if password is not None:
        _add("password_enc", crypto_mod.encrypt(password))
    if is_active is not None:
        _add("is_active", int(is_active))

    if not updates:
        return

    params.extend([user_id, broker])
    with db_mod.connect() as conn:
        conn.execute(
            f"UPDATE user_brokers SET {', '.join(updates)} "
            f"WHERE user_id = ? AND broker = ?",
            params,
        )
    log.info("Updated broker creds for user_id=%d broker=%s fields=%s",
             user_id, broker, [u.split(" ")[0] for u in updates])


def clear(user_id: int, broker: str) -> None:
    """Remove all stored creds for a broker."""
    _assert_broker(broker)
    with db_mod.connect() as conn:
        conn.execute(
            "DELETE FROM user_brokers WHERE user_id = ? AND broker = ?",
            (user_id, broker),
        )
    log.warning("Cleared broker creds for user_id=%d broker=%s", user_id, broker)


# =============================================================
#  Helpers
# =============================================================
def _assert_broker(broker: str) -> None:
    if broker not in _BROKERS:
        raise ValueError(f"broker must be one of {_BROKERS}, got {broker!r}")


def _row_to_creds(row) -> BrokerCredentials:
    return BrokerCredentials(
        broker=row["broker"],
        client_id=row["client_id"] or None,
        api_key=crypto_mod.decrypt(row["api_key_enc"]),
        api_secret=crypto_mod.decrypt(row["api_secret_enc"]),
        access_token=crypto_mod.decrypt(row["access_token_enc"]),
        totp_secret=crypto_mod.decrypt(row["totp_secret_enc"]),
        password=crypto_mod.decrypt(row["password_enc"]),
        token_updated_at=row["token_updated_at"],
        is_active=bool(row["is_active"]),
    )
