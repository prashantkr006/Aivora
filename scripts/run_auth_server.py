"""Launcher for the OAuth callback sidecar (FastAPI on port 8502)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aivora.webapp.auth_server import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
