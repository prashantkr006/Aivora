"""Top-level ETL orchestrator (DhanHQ-only).

Three high-level entry points:

* :func:`run_historical_load` — multi-month cold start. Pulls spot
  intraday + per-expiry ATM option intraday from DhanHQ for the
  configured date range, writes everything to the SQLite store, and
  rebuilds the training Parquet.

* :func:`run_daily_update` — incremental: fetch the last day or two
  from DhanHQ, append, rebuild features.

* :func:`build_training_dataset` — re-run feature engineering against
  whatever is currently in the DB and write the Parquet.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from ..utils.config import get_config
from ..utils.logger import get_logger
from . import data_cleaning, data_ingestion, database, feature_engineering
from .dhan_client import DhanClient

log = get_logger(__name__)


# =============================================================
#  Helpers
# =============================================================
def _split_combined_history(combined: pd.DataFrame):
    """Split the wide historical dataframe into the two DB tables.

    The historical loader returns one row per (symbol, datetime) with
    both spot OHLC and ATM CE/PE columns. We persist the spot half to
    ``spot_futures`` and the option half (in long form) to
    ``options_chain``, which keeps the daily-update path — which
    writes to the same tables — schema-compatible.
    """
    spot_cols = [
        "datetime", "symbol",
        "spot_open", "spot_high", "spot_low", "spot_close",
        "volume",
    ]
    spot = combined[[c for c in spot_cols if c in combined.columns]].copy()
    for c in ["fut_open", "fut_high", "fut_low", "fut_close"]:
        spot[c] = pd.NA

    rows = []
    for side, ltp, oi, iv, strike in (
        ("CE", "ce_ltp", "ce_oi", "ce_iv", "ce_strike"),
        ("PE", "pe_ltp", "pe_oi", "pe_iv", "pe_strike"),
    ):
        if ltp not in combined.columns:
            continue
        chunk = combined[["datetime", "symbol", ltp, oi, iv]].copy()
        chunk["strike"] = combined[strike] if strike in combined.columns else pd.NA
        chunk.columns = ["datetime", "symbol", "ltp", "oi", "iv", "strike"]
        chunk["type"] = side
        chunk = chunk.dropna(subset=["ltp"], how="all")
        rows.append(chunk[["datetime", "symbol", "strike", "type", "ltp", "oi", "iv"]])

    options = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return spot, options


def _parse_iso_date(value, fallback: date) -> date:
    if value is None or value == "":
        return fallback
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


# =============================================================
#  Orchestrator
# =============================================================
def run_historical_load(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    combined: Optional[pd.DataFrame] = None,
) -> Path:
    """Cold start from DhanHQ.

    1. Pull multi-month history via :py:meth:`DhanClient.load_full_historical_data`
       — unless ``combined`` is already provided (e.g. by a caller that
       fetched it for its own summary/logging), in which case that
       dataframe is reused instead of fetching a second time.
    2. Clean it.
    3. Upsert into ``spot_futures`` and ``options_chain``.
    4. Build the training Parquet.

    Returns the path to the Parquet file.
    """
    cfg = get_config()
    log.info("=== run_historical_load ===")
    database.init_db()

    hist_cfg = cfg.raw.get("historical", {})
    start_date = _parse_iso_date(start_date, _parse_iso_date(
        hist_cfg.get("start_date"), date(2022, 1, 1)
    ))
    end_date = _parse_iso_date(end_date, _parse_iso_date(
        hist_cfg.get("end_date"), date.today()
    ))
    log.info("Historical range: %s -> %s", start_date, end_date)

    if combined is None:
        client = DhanClient()
        combined = client.load_full_historical_data(start_date, end_date)

    cleaned = data_cleaning.clean(combined)

    spot, options = _split_combined_history(cleaned)
    database.upsert_spot_futures(spot)
    if not options.empty:
        database.upsert_option_chain(options)

    return build_training_dataset()


def run_daily_update(record_options: bool = True) -> Path:
    """Incremental update — pull new candles from DhanHQ and rebuild.

    Steps:
        1. For each configured instrument, look up the latest stored
           timestamp and pull spot candles from Dhan.
        2. (Optional) Snapshot the live ATM option chain.
        3. Clean, upsert, then rebuild the training Parquet.

    If Dhan is unreachable the function logs and re-raises so the
    cron job picks it up.
    """
    cfg = get_config()
    log.info("=== run_daily_update ===")
    database.init_db()

    client = DhanClient()
    interval = int(cfg.market.get("candle_interval_minutes", 5))

    appended_spot = 0
    appended_opts = 0
    for inst in cfg.instruments:
        symbol = inst["symbol"]
        last_ts = database.last_loaded_timestamp(symbol)
        log.info("%s — last stored timestamp: %s", symbol, last_ts)

        # ---- spot candles ----
        new_df = data_ingestion.fetch_recent_spot(
            client, inst, days_back=2, interval_minutes=interval
        )
        if new_df.empty:
            log.info("No new candles for %s", symbol)
        else:
            cleaned = data_cleaning.clean(new_df)
            appended_spot += database.upsert_spot_futures(cleaned)

        # ---- ATM option snapshot ----
        # Limited by Dhan's 1-req-per-3-sec option-chain throttle —
        # for denser intraday OI/IV history, run a standalone 5-minute
        # recorder process.
        if record_options:
            try:
                opt_rows = data_ingestion.record_atm_option_snapshot(client, inst)
                if not opt_rows.empty:
                    appended_opts += database.upsert_option_chain(opt_rows)
            except Exception as exc:
                log.warning(
                    "ATM option snapshot failed for %s (%s) — continuing without it.",
                    symbol, exc,
                )

    log.info(
        "run_daily_update: appended spot=%d rows, options=%d rows",
        appended_spot, appended_opts,
    )
    return build_training_dataset()


def build_training_dataset() -> Path:
    """Re-run feature engineering against the database and write Parquet."""
    cfg = get_config()
    out_path: Path = cfg.paths["parquet_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    spot = database.load_spot_futures()
    if spot.empty:
        raise RuntimeError(
            "spot_futures table is empty — run `--mode historical` first."
        )
    opts = database.load_option_chain()

    # Merge wide-format options back onto spot data.
    if not opts.empty:
        merged = pd.merge(spot, opts, on=["datetime", "symbol"], how="left")
    else:
        merged = spot
        for c in ("ce_ltp", "pe_ltp", "ce_oi", "pe_oi", "ce_iv"):
            merged[c] = pd.NA
    merged = merged.rename(columns={"ce_iv": "iv"})

    feats = feature_engineering.engineer_features(merged)
    feats.to_parquet(out_path, index=False)
    log.info("build_training_dataset: wrote %s (rows=%d)", out_path, len(feats))
    return out_path
