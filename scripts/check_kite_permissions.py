"""Diagnose why Kite is throwing ``Insufficient permission for that call``.

Reads Kite creds for a given user out of the multi-user webapp DB,
then runs three targeted probes so you can see *exactly* which
capability is missing:

    1. ``kite.profile()``       — proves the token is valid at all.
    2. ``kite.margins()``       — proves the base plan is active.
    3. ``kite.historical_data`` — the Historical Data add-on.

Each probe prints ``OK`` or the raw Kite error message.  If (1) fails,
your token expired.  If (1) passes but (3) fails, the Historical
API add-on hasn't been provisioned for that api_key yet.

Usage::

    python -m scripts.check_kite_permissions --email you@yourdomain.com
"""

from __future__ import annotations

import argparse
import getpass
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aivora.webapp import brokers, crypto, users  # noqa: E402

crypto.install_master_key_to_env()


def _probe(label: str, fn):
    print(f"\n[{label}]")
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL -> {type(exc).__name__}: {exc}")
        return False
    print(f"  OK   -> {result}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True, help="Your login email in the multi-user app")
    args = ap.parse_args()

    password = getpass.getpass(f"Password for {args.email}: ")
    u = users.authenticate(args.email, password)
    if u is None:
        print("Authentication failed — wrong email/password.")
        return 2

    zer = brokers.get(u.id, "ZERODHA")
    if zer is None or not zer.access_token:
        print("No Zerodha broker credentials on file for this user.")
        return 3

    print(f"user_id={u.id} api_key={zer.api_key[:6]}... "
          f"client_id={zer.client_id} token_updated_at={zer.token_updated_at}")

    try:
        from kiteconnect import KiteConnect
    except ImportError:
        print("kiteconnect not installed.")
        return 4

    kite = KiteConnect(api_key=zer.api_key)
    kite.set_access_token(zer.access_token)

    # Probe 1: token validity.
    profile_ok = _probe("profile()  — token validity",
                        lambda: {k: kite.profile().get(k)
                                 for k in ("user_id", "user_name", "email")})

    # Probe 2: base plan still active.
    _probe("margins() — base plan",
           lambda: {"available_cash": kite.margins("equity")
                    .get("available", {}).get("cash")})

    # Probe 3: Historical Data add-on.
    # NIFTY 50 spot instrument_token = 256265.  Ask for last 30 minutes.
    now = datetime.now()
    hist_ok = _probe("historical_data() — Historical add-on",
                     lambda: {"rows": len(kite.historical_data(
                         256265, now - timedelta(minutes=30), now, "5minute",
                     ))})

    print("\n" + "=" * 50)
    if not profile_ok:
        print("Your access_token is invalid — re-login via Profile → Connect Zerodha.")
    elif not hist_ok:
        print("Base plan works, Historical add-on does NOT.")
        print("Fix on Zerodha's side:")
        print("  1. https://developers.kite.trade/apps  ->  select your app")
        print("     ->  scroll to 'Subscriptions'  ->  confirm 'Historical Data' is ACTIVE")
        print("  2. If it's active but still failing:")
        print("     - Wait 30-60 minutes (subscription changes propagate on the hour).")
        print("     - Re-do OAuth to mint a NEW access_token (permissions are")
        print("       baked into the token at issue time on some plans).")
        print("     - Confirm the api_key printed above is the SAME app that")
        print("       has the Historical add-on (some users have multiple apps).")
    else:
        print("Everything is green — the multi-user tick should stop failing on the next 5-min mark.")
    return 0 if hist_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
