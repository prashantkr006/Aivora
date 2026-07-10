"""Live model inference — one row → one (p_up, p_down) prediction.

Loads the frozen binary UP/DOWN pair created by
``scripts/freeze_model.py`` and applies it to the latest row of
the training Parquet.  Train/serve parity is guaranteed because
we run the *same* feature engineering that produced the training
data — no bespoke live-feature pipeline.

Kept pure — no side effects, no DB writes.  The scheduler owns
persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from ..ml import binary as bin_mod
from ..pipeline.feature_engineering import feature_columns
from ..utils.config import get_config
from ..utils.logger import get_logger

log = get_logger(__name__)


CURRENT_UP = "current_up.pkl"
CURRENT_DOWN = "current_down.pkl"


@dataclass
class InferenceResult:
    symbol: str
    row_time: pd.Timestamp
    p_up: float
    p_down: float
    p_flat: float
    minutes_since_open: float
    vol_regime_pct: float
    spot_close: float
    ce_ltp: Optional[float]
    pe_ltp: Optional[float]


class LiveInference:
    """Cache the frozen boosters so we don't re-load them on every tick."""

    def __init__(self, models_dir: Optional[Path] = None):
        cfg = get_config()
        self.models_dir = Path(models_dir) if models_dir else cfg.paths["models_dir"]
        self._up = None
        self._down = None
        self._loaded_mtime: Tuple[float, float] = (0.0, 0.0)

    # -------------------- model IO --------------------
    def _model_paths(self) -> Tuple[Path, Path]:
        return self.models_dir / CURRENT_UP, self.models_dir / CURRENT_DOWN

    def _needs_reload(self) -> bool:
        up, dn = self._model_paths()
        if not up.exists() or not dn.exists():
            return True
        mtime = (up.stat().st_mtime, dn.stat().st_mtime)
        return mtime != self._loaded_mtime

    def load_if_stale(self) -> None:
        if not self._needs_reload() and self._up is not None:
            return
        up_path, dn_path = self._model_paths()
        if not up_path.exists() or not dn_path.exists():
            raise FileNotFoundError(
                f"Frozen models not found. Run "
                f"`python -m scripts.freeze_model` first.\n"
                f"  expected: {up_path}\n            {dn_path}"
            )
        self._up = joblib.load(up_path)
        self._down = joblib.load(dn_path)
        self._loaded_mtime = (up_path.stat().st_mtime, dn_path.stat().st_mtime)
        log.info("LiveInference: loaded %s + %s", up_path.name, dn_path.name)

    # -------------------- prediction --------------------
    def latest_prediction(self, symbol: str,
                          parquet_path: Optional[Path] = None) -> Optional[InferenceResult]:
        """Return the (p_up, p_down, p_flat) for the LATEST row of ``symbol``.

        None means: no valid feature row is yet available (e.g. the
        first-of-day candle before any lookback windows have warmed).
        """
        cfg = get_config()
        p_path = Path(parquet_path) if parquet_path else cfg.paths["parquet_path"]
        if not p_path.exists():
            raise FileNotFoundError(f"Training Parquet missing at {p_path}")
        df = pd.read_parquet(p_path)
        return self.latest_prediction_from_df(df, symbol)

    def latest_prediction_from_df(
        self,
        feat_df: pd.DataFrame,
        symbol: str,
    ) -> Optional[InferenceResult]:
        """Predict from an already feature-engineered dataframe.

        Same guts as :meth:`latest_prediction` but skips the parquet
        disk round-trip — the live trading engine calls this after
        engineering features on a slim in-memory window. Identical
        output for identical inputs.
        """
        self.load_if_stale()
        sym_rows = feat_df[feat_df["symbol"] == symbol]
        if sym_rows.empty:
            log.warning("No rows for %s in in-memory feature dataframe", symbol)
            return None
        last = sym_rows.sort_values("datetime").iloc[-1]

        # Guard: the very first fresh row after a warmup boundary may
        # still carry NaN in some feature columns.  Feature_columns()
        # already filters all-NaN columns, but rare NaNs on this row
        # are still possible.  LightGBM handles NaN natively but we
        # log a warning to make monitoring easier.
        feat_cols = feature_columns(feat_df)
        X = pd.DataFrame([last[feat_cols].astype("float32").values], columns=feat_cols)
        if X.isna().any().any():
            log.debug("NaN present in live features for %s @ %s", symbol, last["datetime"])

        probs = bin_mod.predict_3class_from_binary(self._up, self._down, X)[0]
        p_flat, p_dn, p_up = float(probs[0]), float(probs[1]), float(probs[2])
        return InferenceResult(
            symbol=symbol,
            row_time=pd.Timestamp(last["datetime"]),
            p_up=p_up,
            p_down=p_dn,
            p_flat=p_flat,
            minutes_since_open=float(last.get("minutes_since_open", np.nan)),
            vol_regime_pct=float(last.get("vol_regime_pct", np.nan)),
            spot_close=float(last["spot_close"]),
            ce_ltp=None if pd.isna(last.get("ce_ltp")) else float(last["ce_ltp"]),
            pe_ltp=None if pd.isna(last.get("pe_ltp")) else float(last["pe_ltp"]),
        )

    # -------------------- entry-signal gate --------------------
    def signal_side(
        self,
        result: InferenceResult,
        settings: Dict[str, Any],
    ) -> Optional[str]:
        """Apply variant #18 gates — return "CE" / "PE" / None.

        Mirrors backtester.run_backtest so paper P&L matches the
        strategy the walk-forward loop validated.
        """
        min_msoo = float(settings.get("min_minutes_since_open", 30))
        max_msoo = float(settings.get("max_minutes_since_open", 300))
        thr_up = float(settings.get("prob_threshold_up", 0.55))
        thr_dn = float(settings.get("prob_threshold_down", 0.60))
        vr_min = settings.get("vol_regime_min")
        vr_max = settings.get("vol_regime_max")

        if not (min_msoo <= result.minutes_since_open <= max_msoo):
            return None
        if vr_min is not None and result.vol_regime_pct < float(vr_min):
            return None
        if vr_max is not None and result.vol_regime_pct > float(vr_max):
            return None

        if result.p_up >= thr_up and result.p_up >= result.p_down:
            return "CE"
        if result.p_down >= thr_dn and result.p_down > result.p_up:
            return "PE"
        return None
