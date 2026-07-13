"""Backfill options_chain for a specific date range using Kite Historical.

For each (date, symbol):
  1. Read spot_futures rows on that date.
  2. Compute ATM strike per 5-min bar.
  3. Group unique (expiry, strike, side) combos to minimise API calls.
  4. Query Kite historical_data(oi=True) for each unique instrument.
  5. Distribute LTP + OI back to the 5-min bars and upsert.

Limitations:
  * Kite's NFO instruments dump only lists CURRENTLY ACTIVE contracts.
    Options that already expired can't be queried through Kite Historical
    — those days will be reported as UNAVAILABLE and left for Option C
    (Dhan gap-fill) to handle.

Usage:
  python -m scripts.kite_backfill_options --from 2026-07-03 --to 2026-07-13
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, time as dt_time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from aivora.live.kite_client import KiteClient  # noqa: E402
from aivora.pipeline import database  # noqa: E402
from aivora.utils.calendar import is_trading_day  # noqa: E402
from aivora.utils.config import KiteCredentials, get_config  # noqa: E402
from aivora.utils.logger import get_logger  # noqa: E402
from aivora.webapp import brokers  # noqa: E402
from aivora.webapp import db as webapp_db  # noqa: E402

log = get_logger("scripts.kite_backfill_options")


def _kite_from_user(user_id: int) -> KiteClient:
    """Build a KiteClient with a user's live token from the webapp DB.

    .env tokens go stale every 6am IST; the dashboard keeps user 27's
    token fresh via the OAuth loop, so use that instead of the env.
    """
    webapp_db.init_db()
    z = brokers.get(user_id, "ZERODHA")
    if not z or not (z.api_key and z.access_token):
        raise RuntimeError(
            f"user {user_id} has no live Kite token in webapp DB — "
            "re-connect via Profile → Zerodha (OAuth) first"
        )
    creds = KiteCredentials(
        api_key=z.api_key,
        api_secret=z.api_secret or "",
        access_token=z.access_token,
        user_id=z.client_id or "",
    )
    return KiteClient(creds=creds)


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _nearest_expiry_for_date(nfo_symbol_rows: pd.DataFrame,
                             target: date) -> Optional[date]:
    """Return the earliest expiry >= target from the NFO dump for the symbol."""
    exps = sorted({e for e in nfo_symbol_rows["expiry"].unique() if e >= target})
    return exps[0] if exps else None


def _atm_strike(spot: float, step: int) -> int:
    return int(round(spot / step) * step)


def _lookup_token(
    nfo_symbol_rows: pd.DataFrame,
    expiry: date, strike: int, side: str,
) -> Optional[int]:
    """Return instrument_token or None if the contract isn't in the dump."""
    m = (
        (nfo_symbol_rows["expiry"] == expiry)
        & (nfo_symbol_rows["strike"] == float(strike))
        & (nfo_symbol_rows["instrument_type"] == side)
    )
    hit = nfo_symbol_rows.loc[m]
    if hit.empty:
        return None
    return int(hit.iloc[0]["instrument_token"])


def backfill_one_day(kite: KiteClient, spot_df: pd.DataFrame,
                     target: date, symbol: str, step: int,
                     nfo_sym: pd.DataFrame) -> Tuple[List[dict], List[str]]:
    """Return (rows_for_upsert, warnings). rows may be empty if expired."""
    warnings: List[str] = []
    day_spot = spot_df[
        (spot_df["symbol"] == symbol) & (spot_df["datetime"].dt.date == target)
    ].sort_values("datetime").reset_index(drop=True)
    if day_spot.empty:
        warnings.append(f"{symbol} {target}: no spot rows — skipping")
        return [], warnings

    expiry = _nearest_expiry_for_date(nfo_sym, target)
    if expiry is None:
        warnings.append(f"{symbol} {target}: NO active expiry found (all expired) — skipping")
        return [], warnings

    # Determine unique (strike, side) combos for the day
    day_spot["atm"] = day_spot["spot_close"].apply(lambda s: _atm_strike(float(s), step))
    unique_strikes = sorted(day_spot["atm"].unique())

    # Query each (strike, side) once, distribute back
    from_dt = datetime.combine(target, dt_time(9, 15))
    to_dt = datetime.combine(target, dt_time(15, 30))

    combined_rows: List[dict] = []
    for strike in unique_strikes:
        for side in ("CE", "PE"):
            token = _lookup_token(nfo_sym, expiry, strike, side)
            if token is None:
                warnings.append(
                    f"{symbol} {target}: token missing "
                    f"(expiry={expiry} strike={strike} {side}) — skipping"
                )
                continue
            try:
                raw = kite._call(
                    kite._client().historical_data,
                    token, from_dt, to_dt, "5minute", False, True,
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append(
                    f"{symbol} {target}: historical_data failed for "
                    f"{expiry}/{strike}/{side}: {exc}"
                )
                continue
            if not raw:
                continue
            for bar in raw:
                bar_ts = pd.Timestamp(bar["date"]).tz_localize(None)
                # Only keep rows where this strike was actually the ATM
                # at that bar (spot ATM tracking).
                spot_match = day_spot.loc[day_spot["datetime"] == bar_ts]
                if spot_match.empty or int(spot_match.iloc[0]["atm"]) != strike:
                    continue
                combined_rows.append({
                    "datetime": bar_ts,
                    "symbol": symbol,
                    "strike": float(strike),
                    "type": side,
                    "ltp": float(bar.get("close", 0.0) or 0.0),
                    "oi": float(bar.get("oi", 0.0) or 0.0),
                    "iv": None,  # Kite historical doesn't expose IV
                })
    return combined_rows, warnings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_", type=_parse_date, required=True)
    ap.add_argument("--to", dest="to_", type=_parse_date, required=True)
    ap.add_argument("--symbols", nargs="+", default=None,
                    help="Limit to specific symbols (default = all in config)")
    ap.add_argument("--user-id", type=int, default=27,
                    help="Webapp user id whose live Kite token we should borrow (default: 27)")
    args = ap.parse_args()

    cfg = get_config()
    instruments = cfg.instruments
    if args.symbols:
        instruments = [i for i in instruments if i["symbol"] in args.symbols]

    # Load spot on the target range once
    spot = database.load_spot_futures()
    spot["datetime"] = pd.to_datetime(spot["datetime"])
    mask = (
        (spot["datetime"].dt.date >= args.from_)
        & (spot["datetime"].dt.date <= args.to_)
    )
    spot_window = spot.loc[mask]

    kite = _kite_from_user(args.user_id)
    log.info("Loading NFO instrument dump …")
    nfo = kite.instruments("NFO")

    all_warnings: List[str] = []
    grand_rows: List[dict] = []
    trading_days = [
        d for d in pd.date_range(args.from_, args.to_)
        if is_trading_day(d.date())
    ]
    log.info("Backfilling %d trading day(s) x %d symbols …",
             len(trading_days), len(instruments))

    t0 = time.perf_counter()
    for inst in instruments:
        sym = inst["symbol"]
        step = int(inst["strike_step"])
        nfo_sym = nfo[nfo["name"] == sym].copy()
        for d_ts in trading_days:
            d = d_ts.date()
            rows, warns = backfill_one_day(kite, spot_window, d, sym, step, nfo_sym)
            grand_rows.extend(rows)
            all_warnings.extend(warns)
            log.info("  %s %s → %d rows", sym, d, len(rows))

    elapsed = time.perf_counter() - t0
    log.info("Fetched %d option rows in %.1fs", len(grand_rows), elapsed)

    if grand_rows:
        df = pd.DataFrame(grand_rows)
        n = database.upsert_option_chain(df)
        log.info("Upserted %d rows into options_chain", n)
    else:
        log.warning("No rows fetched.")

    if all_warnings:
        log.warning("=== %d warning(s) ===", len(all_warnings))
        for w in all_warnings:
            log.warning("  %s", w)

    # Final verification
    opts_after = database.load_option_chain()
    opts_after["datetime"] = pd.to_datetime(opts_after["datetime"])
    print("\n=== options_chain coverage after backfill ===")
    for inst in instruments:
        sym = inst["symbol"]
        for d_ts in trading_days:
            d = d_ts.date()
            n = int(((opts_after["symbol"] == sym) & (opts_after["datetime"].dt.date == d)).sum())
            print(f"  {sym:<10s} {d}: {n:>4d} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
