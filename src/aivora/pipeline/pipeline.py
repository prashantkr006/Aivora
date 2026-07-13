"""Top-level ETL orchestrator (Kite-only).

Live tick (``webapp.trading_engine.MarketDataCache.refresh_if_stale``)
handles all data flows now — pulls spot candles via Kite every 5-min,
snapshots ATM CE/PE via Kite alongside, and upserts into the two SQLite
tables. This module keeps just one helper:

* :func:`build_training_dataset` — re-runs feature engineering against
  whatever is currently in the DB and writes the Parquet used by
  ``scripts/freeze_model`` and the walk-forward tooling.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils.config import get_config
from ..utils.logger import get_logger
from . import database, feature_engineering

log = get_logger(__name__)


def build_training_dataset() -> Path:
    """Re-run feature engineering against the database and write Parquet."""
    cfg = get_config()
    out_path: Path = cfg.paths["parquet_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    spot = database.load_spot_futures()
    if spot.empty:
        raise RuntimeError(
            "spot_futures table is empty — start the live tick "
            "(dashboard → Trading control → START) to populate."
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
