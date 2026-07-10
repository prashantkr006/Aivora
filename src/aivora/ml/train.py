"""LightGBM training + Optuna hyperparameter search.

Why LightGBM?
    * Handles mixed-scale features and missing values natively,
      removing the need for a scaling step.
    * Fast enough to fit hundreds of trials on a laptop.
    * The 3-class softmax objective maps directly onto our
      {FLAT, DOWN, UP} target.

Why Optuna?
    * Tree-Parzen Estimator is sample-efficient compared to grid
      search and pairs naturally with early stopping for time-
      series problems.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import f1_score, log_loss

from ..utils.config import get_config
from ..utils.logger import get_logger
from .dataset import Splits, make_splits, walk_forward_folds

log = get_logger(__name__)

# Mute Optuna's verbose default — our own logger handles updates.
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================
#  Hyperparameter search space
# =============================================================
def _suggest_params(trial: optuna.Trial) -> Dict:
    """Search space for the LightGBM classifier.

    Ranges are intentionally conservative — the dataset isn't
    huge and overfit-prone configs (deep trees + low regularisation)
    rarely transfer out of sample on 5-min returns.
    """
    return {
        "objective": "multiclass",
        "num_class": 3,
        "metric": get_config().model["metric"],
        "boosting_type": "gbdt",
        "verbosity": -1,
        "random_state": get_config().model["random_state"],

        "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 500, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }


# =============================================================
#  Objective
# =============================================================
def _class_weights(y) -> np.ndarray:
    """Inverse-frequency sample weights — counter the FLAT-heavy label.

    Without this the model collapses to "predict FLAT always" because
    the raw class ratio is roughly 85/8/7 for (FLAT, DOWN, UP).
    """
    y = np.asarray(y)
    counts = np.bincount(y, minlength=3).astype(float)
    inv = 1.0 / np.maximum(counts, 1.0)
    inv /= inv.sum()  # normalise so weights average to ~1
    return inv[y] * len(counts)


def _objective(trial: optuna.Trial, splits: Splits) -> float:
    """Average validation log-loss across the walk-forward folds."""
    cfg = get_config()
    params = _suggest_params(trial)
    folds = walk_forward_folds(splits.X_train, splits.y_train, cfg.model["cv_splits"])
    losses = []
    for tr_idx, va_idx in folds:
        tr_w = _class_weights(splits.y_train.iloc[tr_idx].values)
        train_set = lgb.Dataset(
            splits.X_train.iloc[tr_idx],
            label=splits.y_train.iloc[tr_idx],
            weight=tr_w,
        )
        valid_set = lgb.Dataset(
            splits.X_train.iloc[va_idx],
            label=splits.y_train.iloc[va_idx],
            reference=train_set,
        )
        model = lgb.train(
            params,
            train_set,
            num_boost_round=2000,
            valid_sets=[valid_set],
            callbacks=[
                lgb.early_stopping(cfg.model["early_stopping_rounds"], verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        preds = model.predict(splits.X_train.iloc[va_idx])
        losses.append(log_loss(splits.y_train.iloc[va_idx], preds, labels=[0, 1, 2]))
    return float(np.mean(losses))


# =============================================================
#  Public entry point
# =============================================================
def tune_and_train(
    splits: Splits | None = None,
    n_trials: int | None = None,
) -> Tuple[lgb.Booster, Dict]:
    """Run an Optuna study, then refit on train+val with the best params.

    Returns the trained Booster and a metadata dict containing the
    chosen hyperparameters plus validation metrics.
    """
    cfg = get_config()
    splits = splits or make_splits()
    n_trials = n_trials or cfg.model["optuna_trials"]

    log.info("tune_and_train: starting Optuna study (n_trials=%d)", n_trials)
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda t: _objective(t, splits),
        n_trials=n_trials,
        timeout=cfg.model["optuna_timeout_sec"],
        show_progress_bar=False,
    )
    log.info("tune_and_train: best logloss=%.5f", study.best_value)
    log.info("tune_and_train: best params=%s", study.best_params)

    final_params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": cfg.model["metric"],
        "boosting_type": "gbdt",
        "verbosity": -1,
        "random_state": cfg.model["random_state"],
        **study.best_params,
    }

    # Refit on train + val for the final model; we use val only as
    # the early-stopping monitor.  Class weights applied for the
    # same reason as during CV.
    train_set = lgb.Dataset(
        splits.X_train, label=splits.y_train,
        weight=_class_weights(splits.y_train.values),
    )
    valid_set = lgb.Dataset(
        splits.X_val, label=splits.y_val,
        weight=_class_weights(splits.y_val.values),
        reference=train_set,
    )
    final_model = lgb.train(
        final_params,
        train_set,
        num_boost_round=5000,
        valid_sets=[valid_set],
        callbacks=[
            lgb.early_stopping(cfg.model["early_stopping_rounds"], verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )

    # Validation metrics for the registry.
    val_pred = final_model.predict(splits.X_val)
    val_metrics = {
        "val_logloss": float(log_loss(splits.y_val, val_pred, labels=[0, 1, 2])),
        "val_f1_macro": float(
            f1_score(splits.y_val, val_pred.argmax(axis=1), average="macro")
        ),
        "best_iteration": int(final_model.best_iteration or final_model.current_iteration()),
    }
    log.info("tune_and_train: validation metrics=%s", val_metrics)

    metadata = {
        "params": final_params,
        "study_best_value": float(study.best_value),
        "val_metrics": val_metrics,
        "feature_cols": splits.feature_cols,
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }
    return final_model, metadata


def save_model(model: lgb.Booster, metadata: Dict, name: str = "lgbm_model.pkl") -> Path:
    """Persist Booster + metadata next to it as JSON.

    The two-file layout keeps metadata cheap to inspect (no pickle
    load required) while the Booster lives in joblib's format.
    """
    cfg = get_config()
    out_dir: Path = cfg.paths["models_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / name
    meta_path = model_path.with_suffix(".json")

    joblib.dump(model, model_path)
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    log.info("save_model: wrote %s (%.1f KB)", model_path, model_path.stat().st_size / 1024)
    return model_path
