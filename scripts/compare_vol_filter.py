"""Compare P&L with volatility filter ON vs OFF for a given date range."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from aivora.backtest.backtester import run_backtest
from aivora.ml import binary as bin_mod
from aivora.ml.dataset import Splits
from aivora.pipeline.feature_engineering import feature_columns
from aivora.utils.config import get_config
import joblib
import numpy as np

cfg = get_config()
parquet = cfg.paths["parquet_path"]
df = pd.read_parquet(parquet)
df["datetime"] = pd.to_datetime(df["datetime"])
feat_cols = feature_columns(df)

up_model = joblib.load(cfg.paths["models_dir"] / "current_up.pkl")
down_model = joblib.load(cfg.paths["models_dir"] / "current_down.pkl")

DATES = [
    "2026-06-08","2026-06-09","2026-06-10","2026-06-11","2026-06-12",
    "2026-06-15","2026-06-16","2026-06-17","2026-06-18","2026-06-19",
    "2026-06-22","2026-06-23","2026-06-24","2026-06-25","2026-06-26",
    "2026-06-29","2026-06-30","2026-07-01","2026-07-02","2026-07-03",
    "2026-07-06","2026-07-07","2026-07-08","2026-07-09","2026-07-10"
]

SETTINGS_ON = {
    "prob_threshold_up": 0.55, "prob_threshold_down": 0.60,
    "take_profit_pct": 0.60, "stop_loss_pct": 0.30,
    "min_minutes_since_open": 30, "max_minutes_since_open": 300,
    "vol_regime_min": 0.15, "vol_regime_max": 0.90,
    "max_trades_per_day": 10,
}

SETTINGS_OFF = {**SETTINGS_ON, "vol_regime_min": 0.0, "vol_regime_max": 999.0}

def pnl_for_dates(settings):
    total = 0.0
    for d in DATES:
        day = df[df["datetime"].dt.date == pd.to_datetime(d).date()].copy()
        day = day[day["label"].notna()]
        if day.empty:
            continue
        X = day[feat_cols].astype(np.float32).reset_index(drop=True)
        probs = bin_mod.predict_3class_from_binary(up_model, down_model, X)
        meta = day[["datetime","symbol","spot_close","fwd_return","ce_ltp","pe_ltp","minutes_since_open","vol_regime_pct"]].reset_index(drop=True)
        splits = Splits(X_train=pd.DataFrame(), y_train=pd.Series(dtype=int),
                        X_val=pd.DataFrame(), y_val=pd.Series(dtype=int),
                        X_test=X, y_test=pd.Series(dtype=int),
                        feature_cols=feat_cols, meta_test=meta)
        result = run_backtest(probs, splits, overrides=settings, name=f"compare_{d}")
        total += result["summary"]["total_pnl"]
    return total

print("Vol filter ON  total P&L:", round(pnl_for_dates(SETTINGS_ON), 2))
print("Vol filter OFF total P&L:", round(pnl_for_dates(SETTINGS_OFF), 2))