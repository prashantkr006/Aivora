"""Binary UP / DOWN model pair.

The 3-class softmax collapses to "predict FLAT" whenever the FLAT
mass outweighs UP+DOWN — even after inverse-frequency weighting
its calibration is poor and hardly any row emits P(UP) or P(DOWN)
above 0.5.

The remedy: train two independent binary classifiers.

* ``UP-vs-rest``   — target = 1 if forward return is positive
  enough, else 0.  Model outputs a single "conviction to be long"
  probability.
* ``DOWN-vs-rest`` — mirror image for shorts (buying puts).

We combine them into a 3-class-shaped probability vector to keep
the backtester happy::

    p_up   = UP-model's positive probability
    p_down = DOWN-model's positive probability
    p_flat = max(0, 1 - p_up - p_down)   # clipped and renormalised

This is a hack — the two models are trained independently, so
their probabilities aren't a proper joint distribution — but the
backtester only ever compares p_up / p_down against a threshold
and picks the argmax, so the flat term never matters unless it
wins, which is exactly the behaviour we want.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..utils.config import get_config
from ..utils.logger import get_logger
from .dataset import Splits

log = get_logger(__name__)


def _binary_weights(y: np.ndarray) -> np.ndarray:
    """Positive-class weight = neg/pos ratio.  Handles the imbalance
    that would otherwise push the sigmoid toward 0.5 for everyone."""
    y = np.asarray(y).astype(int)
    n_pos = max(int(y.sum()), 1)
    n_neg = max(int(len(y) - n_pos), 1)
    w = np.ones_like(y, dtype=float)
    w[y == 1] = n_neg / n_pos
    return w


def _default_binary_params() -> Dict:
    cfg = get_config()
    return {
        "objective": "binary",
        "metric": "binary_logloss",
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
        "is_unbalance": True,
    }


def _train_one_side(
    splits: Splits,
    positive_class: int,
    params: Optional[Dict] = None,
    num_boost_round: int = 2000,
) -> lgb.Booster:
    """Train a single UP-vs-rest or DOWN-vs-rest binary classifier."""
    cfg = get_config()
    params = {**_default_binary_params(), **(params or {})}

    y_tr = (splits.y_train.values == positive_class).astype(int)
    y_va = (splits.y_val.values == positive_class).astype(int)
    dtrain = lgb.Dataset(splits.X_train, label=y_tr, weight=_binary_weights(y_tr))
    dval = lgb.Dataset(splits.X_val, label=y_va, weight=_binary_weights(y_va), reference=dtrain)

    return lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(cfg.model["early_stopping_rounds"], verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )


def train_binary_pair(
    splits: Splits,
    params: Optional[Dict] = None,
) -> Tuple[lgb.Booster, lgb.Booster]:
    """Train UP and DOWN binary boosters and return the pair."""
    log.info("binary: training UP-vs-rest")
    up = _train_one_side(splits, positive_class=2, params=params)
    log.info("binary: training DOWN-vs-rest")
    down = _train_one_side(splits, positive_class=1, params=params)
    return up, down


def predict_3class_from_binary(
    up_model: lgb.Booster,
    down_model: lgb.Booster,
    X: pd.DataFrame,
) -> np.ndarray:
    """Return a (n, 3) matrix with columns ``[FLAT, DOWN, UP]``.

    See module docstring for the caveat about the flat column —
    it's a placeholder, not a real joint probability.
    """
    p_up = np.asarray(up_model.predict(X))
    p_dn = np.asarray(down_model.predict(X))
    # Clip and renormalise so downstream sum-checks don't blow up.
    p_flat = np.clip(1.0 - p_up - p_dn, 0.0, 1.0)
    total = p_flat + p_up + p_dn
    total[total == 0] = 1.0
    return np.column_stack([p_flat / total, p_dn / total, p_up / total])
