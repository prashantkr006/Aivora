"""Secret encryption for the multi-user store.

Uses Fernet (AES-128-CBC + HMAC-SHA256) — the recommended
symmetric primitive from ``cryptography``.  A single master key
lives in ``AIVORA_MASTER_KEY`` in ``.env``; every stored broker
secret is encrypted with it.

Key rotation and secret-manager integration are deliberately out
of scope for this iteration — the master key is loaded once from
env at import time, decrypts on demand, and never lands on disk.

If ``AIVORA_MASTER_KEY`` is missing we **refuse to encrypt**.  A
blank key would silently make every stored secret readable to
anyone with the DB file; failing loudly is safer.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

from ..utils.config import get_config
from ..utils.logger import get_logger

log = get_logger(__name__)

_KEY_ENV = "AIVORA_MASTER_KEY"


class CryptoError(RuntimeError):
    pass


def _load_master_key() -> bytes:
    key = os.getenv(_KEY_ENV, "")
    if not key:
        raise CryptoError(
            f"{_KEY_ENV} not set. Run "
            "`python -m scripts.init_webapp_db --generate-master-key` "
            "or add a 44-char Fernet key manually."
        )
    # Fernet accepts urlsafe-b64 32-byte keys — 44 chars including padding.
    if len(key) != 44:
        raise CryptoError(
            f"{_KEY_ENV} must be a 44-char urlsafe-base64 Fernet key. "
            "Regenerate via `generate_master_key()`."
        )
    return key.encode("ascii")


def generate_master_key() -> str:
    """Return a fresh, URL-safe base64-encoded 32-byte Fernet key."""
    return Fernet.generate_key().decode("ascii")


def _cipher() -> Fernet:
    return Fernet(_load_master_key())


def encrypt(plain: Optional[str]) -> Optional[str]:
    """Encrypt an optional plaintext.  ``None`` and ``""`` pass through."""
    if plain is None or plain == "":
        return None
    return _cipher().encrypt(plain.encode("utf-8")).decode("utf-8")


def decrypt(cipher: Optional[str]) -> Optional[str]:
    """Decrypt an optional stored ciphertext.  Returns None on missing."""
    if cipher is None or cipher == "":
        return None
    try:
        return _cipher().decrypt(cipher.encode("utf-8")).decode("utf-8")
    except InvalidToken as exc:
        raise CryptoError(
            "Decryption failed — the master key doesn't match the ciphertext. "
            "Was AIVORA_MASTER_KEY rotated or lost?"
        ) from exc


def install_master_key_to_env(env_path: Optional[Path] = None) -> str:
    """Generate a new master key and write it to ``.env``.

    Idempotent: if a key is already present, returns the existing one.
    """
    env_path = env_path or (Path(get_config().repo_root) / ".env")
    existing_lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    for line in existing_lines:
        if line.startswith(f"{_KEY_ENV}="):
            existing = line.split("=", 1)[1].strip()
            if existing:
                return existing
    new_key = generate_master_key()
    lines = existing_lines[:]
    replaced = False
    for i, line in enumerate(lines):
        if line.startswith(f"{_KEY_ENV}="):
            lines[i] = f"{_KEY_ENV}={new_key}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{_KEY_ENV}={new_key}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ[_KEY_ENV] = new_key
    log.warning(
        "Generated a new %s and wrote it to %s. BACK THIS UP — losing "
        "it means every stored secret becomes unreadable.", _KEY_ENV, env_path,
    )
    return new_key
