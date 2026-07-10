"""Cleaning rules applied between ingestion and feature engineering.

Goals:
    * Normalise timestamps to a tz-naive 5-minute grid.
    * Drop non-trading days using the NSE calendar.
    * Fill small gaps (≤3 candles) and flag the rest.
    * Clip extreme outliers in OI / LTP columns so a single
      misprint doesn't poison the gradient-boosted model.

Every step logs how many rows it touched so issues are traceable
back to a specific dataset and run.
"""

from __future__ import annotations

from datetime import time
from typing import Iterable

import numpy as np
import pandas as pd

from ..utils.calendar import is_trading_day
from ..utils.logger import get_logger

log = get_logger(__name__)

# Columns that can spike spuriously; we winsorise them per-symbol
# instead of dropping rows so the time index stays contiguous.
WINSORISE_COLS = ["ce_oi", "pe_oi", "ce_ltp", "pe_ltp", "iv", "volume"]


def _round_to_5min(ts: pd.Series) -> pd.Series:
    """Snap timestamps to the nearest 5-minute boundary.

    Upstream data sources occasionally return candles a second or
    two off the wall-clock boundary; snapping keeps spot and option
    series aligned for downstream merges.
    """
    return ts.dt.floor("5min")


def drop_non_trading_days(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows whose date is a weekend or NSE holiday."""
    before = len(df)
    dates = df["datetime"].dt.date
    mask = dates.map(is_trading_day)
    out = df.loc[mask].copy()
    log.info("drop_non_trading_days: removed %d rows", before - len(out))
    return out


def restrict_to_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only candles whose timestamp is inside 09:15 – 15:30."""
    open_t, close_t = time(9, 15), time(15, 30)
    t = df["datetime"].dt.time
    mask = (t >= open_t) & (t <= close_t)
    out = df.loc[mask].copy()
    dropped = (~mask).sum()
    if dropped:
        log.info("restrict_to_session: dropped %d off-session rows", dropped)
    return out


def winsorise(df: pd.DataFrame, lower: float = 0.001, upper: float = 0.999) -> pd.DataFrame:
    """Clip the configured columns at the per-symbol quantile cut-offs.

    We clip rather than drop so the time series remains gap-free.
    The default 0.1 %/99.9 % bounds remove obvious tape errors
    without distorting the bulk of the distribution.
    """
    out = df.copy()
    for sym, grp in out.groupby("symbol"):
        for col in WINSORISE_COLS:
            if col not in out.columns:
                continue
            series = grp[col].astype(float)
            if series.dropna().empty:
                continue
            lo, hi = series.quantile([lower, upper])
            mask = grp.index
            clipped = series.clip(lower=lo, upper=hi)
            n_clipped = int((series != clipped).sum())
            if n_clipped:
                log.debug("winsorise %s.%s: clipped %d", sym, col, n_clipped)
            out.loc[mask, col] = clipped
    return out


def fill_small_gaps(
    df: pd.DataFrame,
    max_consecutive: int = 3,
    price_cols: Iterable[str] = (
        "spot_open", "spot_high", "spot_low", "spot_close",
        "fut_open", "fut_high", "fut_low", "fut_close",
    ),
) -> pd.DataFrame:
    """Forward-fill ≤``max_consecutive`` missing candles per symbol.

    A boolean ``is_filled`` column is added so the model can learn
    that filled candles are slightly less trustworthy.
    """
    out = df.sort_values(["symbol", "datetime"]).copy()
    out["is_filled"] = False
    for sym, grp in out.groupby("symbol"):
        idx = grp.index
        for col in price_cols:
            if col not in out.columns:
                continue
            series = out.loc[idx, col]
            is_na = series.isna()
            if not is_na.any():
                continue
            # Identify runs of NaNs and only fill those that are short.
            run_id = (~is_na).cumsum()
            run_len = is_na.groupby(run_id).transform("sum")
            fill_mask = is_na & (run_len <= max_consecutive)
            if fill_mask.any():
                filled = series.ffill()
                out.loc[idx[fill_mask], col] = filled[fill_mask]
                out.loc[idx[fill_mask], "is_filled"] = True
    return out


def drop_unfillable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose essential spot columns are still NaN."""
    essentials = ["spot_open", "spot_high", "spot_low", "spot_close"]
    before = len(df)
    out = df.dropna(subset=essentials).copy()
    log.info(
        "drop_unfillable_rows: dropped %d rows missing %s",
        before - len(out),
        essentials,
    )
    return out


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the last occurrence per (symbol, datetime).

    The ingest step may pull overlapping windows from Kite and we
    always prefer the freshest value.
    """
    before = len(df)
    out = (
        df.sort_values(["symbol", "datetime"])
        .drop_duplicates(subset=["symbol", "datetime"], keep="last")
        .reset_index(drop=True)
    )
    log.info("deduplicate: dropped %d duplicate rows", before - len(out))
    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """End-to-end cleaning pipeline.  Idempotent — safe to re-run."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["datetime"] = _round_to_5min(df["datetime"])

    df = drop_non_trading_days(df)
    df = restrict_to_session(df)
    df = deduplicate(df)
    df = fill_small_gaps(df)
    df = drop_unfillable_rows(df)
    df = winsorise(df)

    # Convert numeric strings → floats.  ``errors='coerce'`` quietly
    # nukes garbage which we then forward-fill or drop.
    for col in [
        "spot_open", "spot_high", "spot_low", "spot_close",
        "fut_open", "fut_high", "fut_low", "fut_close",
        "ce_ltp", "pe_ltp", "ce_oi", "pe_oi", "iv", "volume",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)

    log.info("clean: final shape=%s", df.shape)
    return df.reset_index(drop=True)
