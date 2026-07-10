"""Launcher for the Streamlit dashboard.

Runs Streamlit as a child process so users don't need to remember
the full command line.  Fails fast if Streamlit isn't installed
so the error message is clearer than a raw ImportError.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    app = root / "app" / "streamlit_app.py"
    if not app.exists():
        print(f"streamlit_app.py not found at {app}", file=sys.stderr)
        return 2
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(
            "streamlit is not installed. Run:\n"
            "    pip install -r requirements.txt\n",
            file=sys.stderr,
        )
        return 3
    cmd = [sys.executable, "-m", "streamlit", "run", str(app),
           "--server.headless", "true"]
    return subprocess.call(cmd, cwd=str(root))


if __name__ == "__main__":
    raise SystemExit(main())
