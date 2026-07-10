"""One-off helper to mint a daily Kite ``access_token``.

Kite's access tokens expire every day at 06:00 IST.  Run this
script once each morning, paste the ``request_token`` from the
redirected URL, and the script will write the new token back to
``.env``.

This script is interactive — do NOT schedule it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aivora.utils.config import get_config  # noqa: E402
from aivora.utils.logger import get_logger  # noqa: E402

log = get_logger("scripts.generate_access_token")


def main() -> int:
    cfg = get_config()
    creds = cfg.kite_credentials()
    if not creds.api_key or not creds.api_secret:
        log.error("Set KITE_API_KEY and KITE_API_SECRET in .env first.")
        return 1

    from kiteconnect import KiteConnect

    kite = KiteConnect(api_key=creds.api_key)
    print("\n1. Open the URL below in a browser and complete login:")
    print("   ", kite.login_url())
    print(
        "\n2. After login Zerodha will redirect to your registered URL "
        "with ?request_token=XYZ in the query string."
    )
    request_token = input("3. Paste the request_token here: ").strip()

    data = kite.generate_session(request_token, api_secret=creds.api_secret)
    new_token = data["access_token"]
    print("✓ access_token obtained.")

    env_path = Path(cfg.repo_root) / ".env"
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    replaced = False
    for i, line in enumerate(lines):
        if line.startswith("KITE_ACCESS_TOKEN="):
            lines[i] = f"KITE_ACCESS_TOKEN={new_token}"
            replaced = True
            break
    if not replaced:
        lines.append(f"KITE_ACCESS_TOKEN={new_token}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ Wrote KITE_ACCESS_TOKEN to {env_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
