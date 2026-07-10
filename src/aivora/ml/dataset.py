"""Load the training Parquet and produce chronological train/val/test splits.

We split by *timestamp* rather than by row index so that all
rows of a given day end up in the same split — leaking even
a single intra-day candle from test into train inflates metrics
on a high-frequency dataset like this one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from ..pipeline.feature_engineering import feature_columns
from ..utils.config import get_config
from ..utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Splits:
    """Container holding chronological train / validation / test sets."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_cols: List[str]
    meta_test: pd.DataFrame  # symbol + datetime + spot_close, for backtesting


def load_dataset(parquet_path: Path | None = None) -> pd.DataFrame:
    """Read the training Parquet."""
    cfg = get_config()
    path = Path(parquet_path) if parquet_path else cfg.paths["parquet_path"]
    if not path.exists():
        raise FileNotFoundError(
            f"Training Parquet missing at {path}. "
            "Run aivora.pipeline.pipeline.build_training_dataset() first."
        )
    df = pd.read_parquet(path)
    log.info("load_dataset: %s rows from %s", len(df), path)
    return df


def make_splits(df: pd.DataFrame | None = None) -> Splits:
    """Chronological train / val / test split.

    Uses the timestamp percentiles from the config (``train_fraction``,
    ``validation_fraction``).  The remainder is the held-out test
    fold used by the backtest module.
    """
    cfg = get_config()
    df = load_dataset() if df is None else df
    # Drop rows without labels — needed for training since horizon-
    # forward returns are unknown for the freshest candles.  Live
    # inference reads the parquet directly and doesn't come through
    # here, so it still sees those latest rows.
    df = df[df["label"].notna()].copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    train_frac = cfg.model["train_fraction"]
    val_frac = cfg.model["validation_fraction"]

    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val :]

    feat_cols = feature_columns(df)
    log.info(
        "make_splits: train=%d, val=%d, test=%d, features=%d",
        len(train_df), len(val_df), len(test_df), len(feat_cols),
    )

    return Splits(
        X_train=train_df[feat_cols].astype(np.float32),
        y_train=train_df["label"].astype(int),
        X_val=val_df[feat_cols].astype(np.float32),
        y_val=val_df["label"].astype(int),
        X_test=test_df[feat_cols].astype(np.float32),
        y_test=test_df["label"].astype(int),
        feature_cols=feat_cols,
        # Carry option prices + session offset onto the meta test frame so
        # the backtester can use real premiums and time-of-day filters
        # when they exist.  Missing columns are added as NaN.
        meta_test=_build_meta_test(test_df),
    )


def _build_meta_test(test_df: pd.DataFrame) -> pd.DataFrame:
    keep = ["datetime", "symbol", "spot_close", "fwd_return",
            "ce_ltp", "pe_ltp", "minutes_since_open", "vol_regime_pct"]
    meta = pd.DataFrame(index=test_df.index)
    for col in keep:
        meta[col] = test_df[col] if col in test_df.columns else np.nan
    return meta.reset_index(drop=True)


def walk_forward_folds(
    X: pd.DataFrame, y: pd.Series, n_splits: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window walk-forward folds.

    Yields ``(train_idx, val_idx)`` tuples where the validation
    window immediately follows the training window — the standard
    setup for time-series cross-validation.
    """
    n = len(X)
    fold_size = n // (n_splits + 1)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        val_end = train_end + fold_size
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, min(val_end, n))
        if len(val_idx) == 0:
            break
        folds.append((train_idx, val_idx))
    return folds
