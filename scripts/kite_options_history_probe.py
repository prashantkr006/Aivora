"""Probe how far back Kite's Historical Data API can serve options bars.

Kite's NFO instruments dump only contains ACTIVE contracts. For each
active NIFTY/BANKNIFTY option we pick a representative near-ATM strike
and ask historical_data for a big range going far into the past. The
API returns whatever bars actually exist for that contract — the
earliest returned bar is the oldest date Kite has for it.

We then compare the OVERALL oldest date across all active contracts
with our current options_chain minimum date. If Kite has anything
older, we report it; otherwise we confirm the 5-year Dhan-backfilled
history we already have is the best possible.

Usage:
    docker compose exec dashboard python -m scripts.kite_options_history_probe
    # OR locally with a valid Kite session in webapp DB:
    python -m scripts.kite_options_history_probe --user-id 27
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from aivora.live.kite_client import KiteClient  # noqa: E402
from aivora.pipeline import database  # noqa: E402
from aivora.utils.config import KiteCredentials  # noqa: E402
from aivora.utils.logger import get_logger  # noqa: E402
from aivora.webapp import brokers, db as webapp_db  # noqa: E402

log = get_logger("scripts.kite_options_history_probe")


def _kite_from_user(user_id: int) -> KiteClient:
    webapp_db.init_db()
    z = brokers.get(user_id, "ZERODHA")
    if not z or not (z.api_key and z.access_token):
        raise RuntimeError(f"user {user_id} has no live Kite token")
    return KiteClient(creds=KiteCredentials(
        api_key=z.api_key, api_secret=z.api_secret or "",
        access_token=z.access_token, user_id=z.client_id or "",
    ))


def probe_contract(kite: KiteClient, token: int, name: str,
                   probe_from: datetime, probe_to: datetime) -> Optional[datetime]:
    """Query Kite historical for one contract and return its earliest bar."""
    try:
        raw = kite._call(
            kite._client().historical_data,
            token, probe_from, probe_to, "day", False, False,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("  %s: historical_data failed — %s", name, exc)
        return None
    if not raw:
        return None
    earliest = min(pd.Timestamp(bar["date"]).tz_localize(None) for bar in raw)
    log.info("  %s: %d bars, earliest = %s", name, len(raw), earliest.date())
    return earliest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", type=int, default=27)
    args = ap.parse_args()

    kite = _kite_from_user(args.user_id)
    nfo = kite.instruments("NFO")
    log.info("NFO dump has %d instruments", len(nfo))

    # 1) What we currently have.
    opts = database.load_option_chain()
    opts["datetime"] = pd.to_datetime(opts["datetime"])
    our_min = opts["datetime"].min()
    our_max = opts["datetime"].max()
    log.info("Our options_chain: %s → %s  (%d rows)",
             our_min.date(), our_max.date(), len(opts))

    # 2) Pick sample contracts to probe — nearest 3 expiries per symbol,
    #    one strike near current spot each.
    now = datetime.now()
    probe_from = datetime(2018, 1, 1)   # ask for 8 years — we'll see what actually returns
    probe_to = now
    results: List[Dict] = []

    for symbol in ["NIFTY", "BANKNIFTY"]:
        sym_opts = nfo[(nfo["name"] == symbol) & (nfo["instrument_type"] == "CE")].copy()
        if sym_opts.empty:
            continue
        expiries = sorted(sym_opts["expiry"].unique())[:5]  # first 5 expiries
        # Get a rough current spot from our own data
        latest_spot = float(
            database.load_spot_futures()
            .query("symbol == @symbol")
            .sort_values("datetime")
            .iloc[-1]["spot_close"]
        )
        step = 50 if symbol == "NIFTY" else 100
        atm = int(round(latest_spot / step) * step)

        log.info("\n=== %s (approx spot %.0f, ATM %d) ===", symbol, latest_spot, atm)
        for exp in expiries:
            match = sym_opts[(sym_opts["expiry"] == exp) & (sym_opts["strike"] == float(atm))]
            if match.empty:
                # fallback: any strike closest to ATM
                sym_opts["strike_gap"] = (sym_opts["strike"] - atm).abs()
                match = sym_opts[sym_opts["expiry"] == exp].nsmallest(1, "strike_gap")
            if match.empty:
                continue
            row = match.iloc[0]
            token = int(row["instrument_token"])
            name = f"{symbol} {exp} {int(row['strike'])} CE"
            earliest = probe_contract(kite, token, name, probe_from, probe_to)
            if earliest is not None:
                results.append({
                    "symbol": symbol, "expiry": exp,
                    "strike": int(row["strike"]), "earliest": earliest,
                })

    # 3) Verdict
    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"Our current options_chain min : {our_min.date()}")
    if results:
        overall_earliest = min(r["earliest"] for r in results)
        print(f"Kite oldest bar (across probes): {overall_earliest.date()}")
        if overall_earliest.date() < our_min.date():
            gain_days = (our_min.date() - overall_earliest.date()).days
            print(f"→ Kite has {gain_days} MORE days of history than we do. Backfill worth it.")
        else:
            gain_days = (overall_earliest.date() - our_min.date()).days
            print(f"→ Our data starts {gain_days} days EARLIER than Kite can serve.")
            print(f"  Kite historical options can't extend our history further.")
            print(f"  (Kite's NFO dump only has currently-active contracts;")
            print(f"   expired weeklies/monthlies from 2021-2025 aren't queryable.)")
    else:
        print("→ Kite returned no historical bars for any probed contract.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
