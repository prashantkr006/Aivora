"""Walk-forward evaluation.

The single-shot ``train_frac / val_frac / test_frac`` split we've
been using gives us a test window that's only 15 % of the total
history — with the 4-year AiVora dataset that's roughly two
months of trading, which is nowhere near enough to make honest
statements about monthly returns.

This module runs the model over a rolling window: retrain on the
previous ~12 months, evaluate on the next ~1 month, roll forward,
concatenate.  The result is 12+ months of *out-of-sample* trades
that can be aggregated month-by-month.

Two entry points:

* :func:`walk_forward_predict` — returns concatenated
  ``(probs, meta_test)`` across every fold.  Downstream code
  passes this to the existing :func:`aivora.backtest.backtester.run_backtest`.

* :func:`walk_forward_backtest` — full pipeline: predict + backtest
  + summary + goal check, mirroring the iteration orchestrator's
  API.  Useful when you want walk-forward metrics with the same
  variant knobs used elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..backtest.backtester import run_backtest
from ..pipeline.feature_engineering import feature_columns
from ..utils.config import get_config
from ..utils.logger import get_logger
from . import train as train_mod
from .dataset import Splits

log = get_logger(__name__)


# =============================================================
#  Fold generation
# =============================================================
@dataclass
class Fold:
    """One walk-forward fold."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_end: pd.Timestamp


def _month_index(ts: pd.Series) -> pd.Series:
    return ts.dt.to_period("M")


def make_folds(
    df: pd.DataFrame,
    train_months: int = 12,
    val_months: int = 1,
    test_months: int = 1,
    step_months: int = 1,
    min_train_rows: int = 500,
) -> List[Fold]:
    """Generate rolling folds by *calendar month*.

    Anchoring on months rather than row counts keeps each fold's
    monthly-return calculation sensible even when data density
    varies (e.g. holidays).
    """
    ts = pd.to_datetime(df["datetime"])
    months = sorted(ts.dt.to_period("M").unique())
    if len(months) < train_months + val_months + test_months:
        raise ValueError(
            f"Not enough months ({len(months)}) for train+val+test"
            f" = {train_months + val_months + test_months}"
        )

    folds: List[Fold] = []
    i = train_months
    # Strict `<` because we index months[i + val_months + test_months].
    while i + val_months + test_months < len(months):
        train_start = months[i - train_months].to_timestamp()
        train_end = months[i].to_timestamp()
        val_end = months[i + val_months].to_timestamp()
        test_end = months[i + val_months + test_months].to_timestamp()
        folds.append(Fold(
            train_start=train_start,
            train_end=train_end,
            val_end=val_end,
            test_end=test_end,
        ))
        i += step_months

    return folds


# =============================================================
#  Fast per-fold refit
# =============================================================
def _fit_one_fold(
    fold: Fold,
    df: pd.DataFrame,
    feat_cols: List[str],
    params: Dict,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> Tuple[lgb.Booster, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Fit on train, monitor on val, return trained model + test slice.

    Uses the class-weight helper from ``train.py`` so the folds
    stay consistent with the single-shot training we already have.
    """
    ts = pd.to_datetime(df["datetime"])
    train_mask = (ts >= fold.train_start) & (ts < fold.train_end)
    val_mask = (ts >= fold.train_end) & (ts < fold.val_end)
    test_mask = (ts >= fold.val_end) & (ts < fold.test_end)

    X_tr = df.loc[train_mask, feat_cols].astype(np.float32)
    y_tr = df.loc[train_mask, "label"].astype(int)
    X_va = df.loc[val_mask, feat_cols].astype(np.float32)
    y_va = df.loc[val_mask, "label"].astype(int)
    df_te = df.loc[test_mask].reset_index(drop=True)
    X_te = df_te[feat_cols].astype(np.float32)
    y_te = df_te["label"].astype(int)

    if len(X_tr) < 100 or len(X_va) < 20 or len(X_te) < 20:
        raise ValueError(
            f"Fold {fold.train_start} → {fold.test_end}: "
            f"insufficient data (train={len(X_tr)}, val={len(X_va)}, test={len(X_te)})"
        )

    dtrain = lgb.Dataset(
        X_tr, label=y_tr,
        weight=train_mod._class_weights(y_tr.values),
    )
    dval = lgb.Dataset(
        X_va, label=y_va,
        weight=train_mod._class_weights(y_va.values),
        reference=dtrain,
    )
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return booster, X_te, y_te, df_te


# =============================================================
#  Public entry points
# =============================================================
def walk_forward_predict(
    df: pd.DataFrame,
    params: Optional[Dict] = None,
    train_months: int = 12,
    val_months: int = 1,
    test_months: int = 1,
    step_months: int = 1,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Run walk-forward CV and return concatenated predictions.

    Returns
    -------
    probs : ndarray of shape (n_test_rows_total, 3)
    meta  : DataFrame with the columns the backtester needs
            (``datetime``, ``symbol``, ``spot_close``, ``fwd_return``,
             optional ``ce_ltp``/``pe_ltp``/``minutes_since_open``)
    """
    cfg = get_config()
    df = df.sort_values("datetime").reset_index(drop=True)
    feat_cols = feature_columns(df)

    # Use the default LightGBM params from config unless the caller
    # provides Optuna-tuned ones.  We deliberately don't run Optuna
    # inside every fold — that would inflate cost 10× for little gain.
    default_params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": cfg.model["metric"],
        "boosting_type": "gbdt",
        "verbosity": -1,
        "random_state": cfg.model["random_state"],
        "num_leaves": 64,
        "max_depth": 8,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "learning_rate": 0.03,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
    }
    params = {**default_params, **(params or {})}

    folds = make_folds(
        df,
        train_months=train_months,
        val_months=val_months,
        test_months=test_months,
        step_months=step_months,
    )
    log.info("walk_forward: %d folds", len(folds))

    all_probs: List[np.ndarray] = []
    all_meta: List[pd.DataFrame] = []
    keep_cols = ["datetime", "symbol", "spot_close", "fwd_return",
                 "ce_ltp", "pe_ltp", "minutes_since_open", "vol_regime_pct"]
    for i, fold in enumerate(folds, start=1):
        log.info(
            "walk_forward fold %d/%d — train %s → %s | test %s → %s",
            i, len(folds),
            fold.train_start.date(), fold.train_end.date(),
            fold.val_end.date(), fold.test_end.date(),
        )
        try:
            booster, X_te, _, df_te = _fit_one_fold(
                fold, df, feat_cols, params,
                num_boost_round=2000,
                early_stopping_rounds=cfg.model["early_stopping_rounds"],
            )
        except ValueError as exc:
            log.warning("  skipped: %s", exc)
            continue
        probs = booster.predict(X_te)
        meta = pd.DataFrame(index=df_te.index)
        for c in keep_cols:
            meta[c] = df_te[c] if c in df_te.columns else np.nan
        all_probs.append(np.asarray(probs))
        all_meta.append(meta.reset_index(drop=True))

    if not all_probs:
        raise RuntimeError("walk_forward produced no test folds — check input date range.")

    probs = np.vstack(all_probs)
    meta = pd.concat(all_meta, ignore_index=True)
    log.info("walk_forward: total test rows = %d over %d folds",
             len(meta), len(all_probs))
    return probs, meta


def walk_forward_backtest(
    df: pd.DataFrame,
    backtest_overrides: Optional[Dict[str, Any]] = None,
    name: str = "walk_forward",
    params: Optional[Dict] = None,
    **fold_kwargs: Any,
) -> Dict[str, Any]:
    """Walk-forward predict + backtest, returning the same shape
    as :func:`aivora.backtest.backtester.run_backtest`.

    Extra ``fold_kwargs`` are forwarded to :func:`walk_forward_predict`
    (``train_months``, ``val_months``, ``test_months``, ``step_months``).
    """
    probs, meta = walk_forward_predict(df, params=params, **fold_kwargs)
    # Build a "Splits"-like shim so run_backtest doesn't care that this
    # isn't a single-shot split — the backtester only reads ``meta_test``.
    splits_shim = Splits(
        X_train=pd.DataFrame(),
        y_train=pd.Series(dtype=int),
        X_val=pd.DataFrame(),
        y_val=pd.Series(dtype=int),
        X_test=pd.DataFrame(),
        y_test=pd.Series(dtype=int),
        feature_cols=[],
        meta_test=meta,
    )
    return run_backtest(probs, splits_shim, overrides=backtest_overrides, name=name)
