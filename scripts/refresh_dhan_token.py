"""Helper for the DhanHQ 24-hour access token.

Dhan access tokens expire 24 hours after generation. The simplest
workflow for an individual trader:

    1. Open https://web.dhan.co/
    2. Profile -> Access DhanHQ APIs -> Generate Access Token
    3. Copy the token and paste it when prompted by this script.

The script writes the token (and optionally your client id) into
``.env`` so the rest of the project picks it up automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aivora.utils.config import get_config  # noqa: E402
from aivora.utils.logger import get_logger  # noqa: E402

log = get_logger("scripts.refresh_dhan_token")


def _write_env(env_path: Path, key: str, value: str) -> None:
    """Insert or replace ``key=value`` in the .env file."""
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    prefix = f"{key}="
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            lines[i] = f"{prefix}{value}"
            break
    else:
        lines.append(f"{prefix}{value}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Updated {key} in {env_path}")


def main() -> int:
    cfg = get_config()
    env_path = Path(cfg.repo_root) / ".env"

    print(
        "\n1. Open https://web.dhan.co/  ->  Profile  ->  Access DhanHQ APIs\n"
        "2. Click 'Generate Access Token' and copy it.\n"
    )
    client_id = input("DHAN_CLIENT_ID (leave blank to keep current): ").strip()
    token = input("DHAN_ACCESS_TOKEN: ").strip()

    if not token:
        print("No token provided — aborting.")
        return 1
    if client_id:
        _write_env(env_path, "DHAN_CLIENT_ID", client_id)
    _write_env(env_path, "DHAN_ACCESS_TOKEN", token)
    print("Done. Token is valid for 24 hours.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
