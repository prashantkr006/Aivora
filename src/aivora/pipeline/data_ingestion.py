"""Data ingestion — DhanHQ only.

Thin helper layer over :class:`DhanClient` for the **daily-update**
path. Multi-month backfill lives in ``DhanClient.load_full_historical_data``
(``dhan_client.py``); this module only covers incremental, live-adjacent
pulls:

* :func:`fetch_recent_spot` — pull the last few trading days of
  5-minute OHLCV for a spot index.

* :func:`record_atm_option_snapshot` — capture a single ATM CE/PE
  snapshot from the live option chain (not the expired-options
  endpoint — this is today's live chain).

* :data:`SPOT_SCHEMA` / :data:`OPTION_SCHEMA` — the canonical column
  sets the rest of the pipeline expects.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd

from ..utils.calendar import previous_trading_day
from ..utils.logger import get_logger
from .dhan_client import DhanClient

log = get_logger(__name__)

# Canonical schema written to the ``spot_futures`` table. Futures
# columns are kept for schema compatibility with the historical
# loader's output, but are populated as NaN — the model only uses spot.
SPOT_SCHEMA = [
    "datetime", "symbol",
    "spot_open", "spot_high", "spot_low", "spot_close",
    "fut_open", "fut_high", "fut_low", "fut_close",
    "volume",
]

# Long-format option-snapshot schema written to the
# ``options_chain`` table.
OPTION_SCHEMA = ["datetime", "symbol", "strike", "type", "ltp", "oi", "iv"]


# =============================================================
#  Live / daily — Dhan helpers
# =============================================================
def fetch_recent_spot(
    client: DhanClient,
    instrument: Dict,
    days_back: int = 2,
    interval_minutes: int = 5,
) -> pd.DataFrame:
    """Pull the last ``days_back`` trading days of spot OHLCV from Dhan.

    Returns a dataframe in the :data:`SPOT_SCHEMA` so it can be
    upserted directly into ``spot_futures``. Futures columns are NaN
    because Dhan's intraday endpoint covers the spot index directly —
    the model only consumes spot OHLC anyway.
    """
    today = datetime.now()
    start_date = previous_trading_day((today - timedelta(days=days_back + 4)).date())
    start_dt = datetime.combine(start_date, datetime.min.time())

    candles = client.spot_intraday(
        security_id=str(instrument["dhan_security_id"]),
        exchange_segment=instrument["dhan_segment"],
        instrument_type=instrument["dhan_instrument_type"],
        from_dt=start_dt,
        to_dt=today,
        interval_minutes=interval_minutes,
    )

    if candles.empty:
        log.warning("No Dhan candles for %s", instrument["symbol"])
        return pd.DataFrame(columns=SPOT_SCHEMA)

    candles = candles.rename(
        columns={
            "open": "spot_open", "high": "spot_high",
            "low": "spot_low", "close": "spot_close",
        }
    )
    candles["symbol"] = instrument["symbol"]
    for col in SPOT_SCHEMA:
        if col not in candles.columns:
            candles[col] = pd.NA
    return candles[SPOT_SCHEMA]


def record_atm_option_snapshot(
    client: DhanClient,
    instrument: Dict,
    snapshot_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """Record a single live ATM CE+PE snapshot in :data:`OPTION_SCHEMA`.

    Designed to be called every 5 minutes during market hours from a
    background recorder. Returns a dataframe ready for
    :func:`aivora.pipeline.database.upsert_option_chain`.
    """
    snap = client.atm_option_snapshot(
        under_security_id=str(instrument["dhan_security_id"]),
        under_segment=instrument["dhan_segment"],
        strike_step=int(instrument["strike_step"]),
    )
    ts = snapshot_time or datetime.now().replace(second=0, microsecond=0)
    rows = []
    for side, ltp_key, oi_key, iv_key in (
        ("CE", "ce_ltp", "ce_oi", "ce_iv"),
        ("PE", "pe_ltp", "pe_oi", "pe_iv"),
    ):
        rows.append({
            "datetime": ts,
            "symbol": instrument["symbol"],
            "strike": snap["atm_strike"],
            "type": side,
            "ltp": snap.get(ltp_key),
            "oi": snap.get(oi_key),
            "iv": snap.get(iv_key),
        })
    return pd.DataFrame(rows, columns=OPTION_SCHEMA)
