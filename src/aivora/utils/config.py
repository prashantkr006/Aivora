"""Project-wide configuration loader.

Reads ``config.yaml`` once and exposes a frozen ``Config`` object
that other modules import.  Environment variables (loaded from a
local ``.env`` file via python-dotenv) override file values for
secrets like the Kite API key.

The loader is intentionally tiny — we treat config as plain data
and read it from a single canonical location at import time so
that all downstream modules share the same view of the world.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

# Walk up from this file (src/aivora/utils/config.py) to the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_FILE = _REPO_ROOT / "config.yaml"
_ENV_FILE = _REPO_ROOT / ".env"

# Loading once at module import is enough — the .env may contain
# credentials needed before any function in this file is called.
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)


@dataclass(frozen=True)
class KiteCredentials:
    api_key: str
    api_secret: str
    access_token: str
    user_id: str


@dataclass(frozen=True)
class DhanCredentials:
    """Credentials for the DhanHQ REST API.

    Only ``client_id`` + ``access_token`` are required — data
    endpoints additionally require an active Data APIs subscription
    on the account itself (not something these credentials control).
    """

    client_id: str
    access_token: str


@dataclass(frozen=True)
class Config:
    """Container exposing config sections as plain dictionaries.

    We could model each section as a typed dataclass, but YAML keys
    map cleanly to dicts and dataclass nesting balloons quickly.
    Callers should treat returned dicts as read-only.
    """

    raw: Dict[str, Any]
    repo_root: Path

    # ---- convenience accessors ----
    @property
    def project(self) -> Dict[str, Any]:
        return self.raw["project"]

    @property
    def paths(self) -> Dict[str, Path]:
        # Resolve every path relative to the repo root so the project
        # works regardless of the user's current working directory.
        return {k: (self.repo_root / v) for k, v in self.raw["paths"].items()}

    @property
    def instruments(self) -> list[Dict[str, Any]]:
        return self.raw["instruments"]

    @property
    def market(self) -> Dict[str, Any]:
        return self.raw["market"]

    @property
    def zerodha(self) -> Dict[str, Any]:
        return self.raw["zerodha"]

    @property
    def dhan(self) -> Dict[str, Any]:
        return self.raw["dhan"]

    @property
    def features(self) -> Dict[str, Any]:
        return self.raw["features"]

    @property
    def labels(self) -> Dict[str, Any]:
        return self.raw["labels"]

    @property
    def model(self) -> Dict[str, Any]:
        return self.raw["model"]

    @property
    def backtest(self) -> Dict[str, Any]:
        return self.raw["backtest"]

    @property
    def goals(self) -> Dict[str, Any]:
        return self.raw.get("goals", {})

    def kite_credentials(self) -> KiteCredentials:
        """Return Kite credentials sourced from environment variables."""
        return KiteCredentials(
            api_key=os.getenv("KITE_API_KEY", ""),
            api_secret=os.getenv("KITE_API_SECRET", ""),
            access_token=os.getenv("KITE_ACCESS_TOKEN", ""),
            user_id=os.getenv("KITE_USER_ID", ""),
        )

    def dhan_credentials(self) -> DhanCredentials:
        """Return DhanHQ credentials sourced from environment variables."""
        return DhanCredentials(
            client_id=os.getenv("DHAN_CLIENT_ID", ""),
            access_token=os.getenv("DHAN_ACCESS_TOKEN", ""),
        )


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Load and cache the project configuration."""
    if not _CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {_CONFIG_FILE}. "
            "Create one from the template in the repo root."
        )
    with _CONFIG_FILE.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, repo_root=_REPO_ROOT)
