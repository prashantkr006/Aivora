"""Freeze the current-month binary UP/DOWN pair for live inference.

The winning backtest strategy (variant 18) is walk-forward - a
new model is trained monthly on the trailing 12 months.  For LIVE
use we snapshot the CURRENT month's fold:

    1. Load the training Parquet.
    2. Take the last 13 months: 12 for train, 1 for validation.
    3. Train two binary boosters (UP-vs-rest, DOWN-vs-rest) using
       the same feature and class-weight rules as the walk-forward
       loop.
    4. Save to ``models/current_up.pkl`` and ``models/current_down.pkl``.

Live inference (``aivora.live.inference.LiveInference``) auto-reloads
these files whenever their mtime changes.

Usage::

    python -m scripts.freeze_model                  # last 13 months of parquet
    python -m scripts.freeze_model --train-months 12 --val-months 1
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aivora.ml import binary as bin_mod  # noqa: E402
from aivora.ml.dataset import Splits  # noqa: E402
from aivora.pipeline.feature_engineering import feature_columns  # noqa: E402
from aivora.utils.config import get_config  # noqa: E402
from aivora.utils.logger import get_logger  # noqa: E402

log = get_logger("scripts.freeze_model")


def _last_n_months_slice(df: pd.DataFrame, train_months: int, val_months: int):
    ts = pd.to_datetime(df["datetime"])
    months = sorted(ts.dt.to_period("M").unique())
    need = train_months + val_months
    if len(months) < need:
        raise RuntimeError(
            f"Only {len(months)} months in parquet; need {need}. "
            "Run the historical + daily update first."
        )
    val_end = months[-1].to_timestamp() + pd.offsets.MonthBegin(1)
    val_start = months[-val_months].to_timestamp()
    train_start = months[-need].to_timestamp()

    train_mask = (ts >= train_start) & (ts < val_start)
    val_mask = (ts >= val_start) & (ts < val_end)
    return train_mask, val_mask


def main() -> int:
    ap = argparse.ArgumentParser(description="Freeze the current-month binary model pair")
    ap.add_argument("--train-months", type=int, default=12)
    ap.add_argument("--val-months", type=int, default=1)
    args = ap.parse_args()

    cfg = get_config()
    parquet = cfg.paths["parquet_path"]
    if not parquet.exists():
        log.error("Training parquet missing at %s", parquet)
        return 2

    df = pd.read_parquet(parquet)
    # Training needs valid labels — filter NaN-label rows out here so
    # the label column can be cast to int downstream.  Live inference
    # reads the same parquet but keeps NaN-label rows (fresh candles).
    df = df[df["label"].notna()].sort_values("datetime").reset_index(drop=True)
    log.info("Loaded parquet: rows=%d span=%s -> %s",
             len(df), df["datetime"].min(), df["datetime"].max())

    train_mask, val_mask = _last_n_months_slice(df, args.train_months, args.val_months)
    feat_cols = feature_columns(df)
    log.info("Train rows=%d val rows=%d features=%d",
             int(train_mask.sum()), int(val_mask.sum()), len(feat_cols))

    splits = Splits(
        X_train=df.loc[train_mask, feat_cols].astype(np.float32),
        y_train=df.loc[train_mask, "label"].astype(int),
        X_val=df.loc[val_mask, feat_cols].astype(np.float32),
        y_val=df.loc[val_mask, "label"].astype(int),
        X_test=pd.DataFrame(),
        y_test=pd.Series(dtype=int),
        feature_cols=feat_cols,
        meta_test=pd.DataFrame(),
    )

    up, dn = bin_mod.train_binary_pair(splits)

    out_dir = cfg.paths["models_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    up_path = out_dir / "current_up.pkl"
    dn_path = out_dir / "current_down.pkl"
    joblib.dump(up, up_path)
    joblib.dump(dn, dn_path)
    log.info("Wrote %s and %s", up_path.name, dn_path.name)

    meta_path = out_dir / "current_model.json"
    import json
    meta_path.write_text(json.dumps({
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "train_months": args.train_months,
        "val_months": args.val_months,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_features": len(feat_cols),
        "feature_columns": feat_cols,
        "parquet_span": [str(df["datetime"].min()), str(df["datetime"].max())],
    }, indent=2))
    log.info("Wrote %s", meta_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
