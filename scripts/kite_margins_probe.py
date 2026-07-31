"""Diagnostic: dump the raw Kite margins response for a given user
so we can see exactly which field carries the "available for trading"
number Zerodha's own dashboard shows.

Usage (from the VPS):
    docker compose exec dashboard python -m scripts.kite_margins_probe
    docker compose exec dashboard python -m scripts.kite_margins_probe --user-id 27
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aivora.live.kite_client import KiteClient  # noqa: E402
from aivora.utils.config import KiteCredentials  # noqa: E402
from aivora.webapp import brokers, db as webapp_db  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", type=int, default=27)
    args = ap.parse_args()

    webapp_db.init_db()
    z = brokers.get(args.user_id, "ZERODHA")
    if not z or not z.access_token or not z.api_key:
        print(f"ERROR: user {args.user_id} has no live Kite token")
        return 1

    kite = KiteClient(creds=KiteCredentials(
        api_key=z.api_key, api_secret=z.api_secret or "",
        access_token=z.access_token, user_id=z.client_id or "",
    ))

    print("=" * 70)
    print(f"Kite margins for user {args.user_id} (client_id={z.client_id})")
    print("=" * 70)

    # Raw response — equity segment
    try:
        eq = kite._call(kite._client().margins, segment="equity")
        print("\n[equity segment]")
        print(json.dumps(eq, indent=2, default=str))
    except Exception as exc:  # noqa: BLE001
        print(f"equity margins call failed: {exc}")
        return 2

    # Show what our helper would pick
    picked = kite.available_funds()
    print("\n" + "=" * 70)
    print(f"available_funds() picks: rs {picked:,.2f}")
    print("=" * 70)

    # Detailed field breakdown
    avail = (eq.get("available") or {}) if isinstance(eq, dict) else {}
    print("\nField breakdown under `available`:")
    for k in ("live_balance", "cash", "opening_balance",
              "collateral", "intraday_payin"):
        print(f"  {k:>18} = {avail.get(k)!r}")
    print(f"  {'net (top)':>18} = {eq.get('net')!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
