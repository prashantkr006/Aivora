"""Feature-importance + SHAP per ablation config.

Reads the current parquet (whichever feature set is enabled in
``feature_engineering.py`` at rebuild time) and trains ONE
representative walk-forward fold using the same train/val split
``walk_forward_limits.py`` uses. Extracts:

    * LightGBM gain-based feature importance (per UP and DOWN booster)
    * mean(|SHAP|) per feature on the test-month rows (via LGBM's
      built-in ``pred_contrib=True`` — exact, no external library)

Family-level totals are summed too so we can attribute the incremental
edge to EMA vs ADX/Regime.

Usage:
    python -m scripts.importance_and_shap --tag baseline --fold 2025-06
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aivora.ml import binary as bin_mod  # noqa: E402
from aivora.ml.dataset import Splits  # noqa: E402
from aivora.pipeline.feature_engineering import feature_columns  # noqa: E402
from aivora.utils.config import get_config  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "ablation"

# Feature-family assignment. Everything not in EMA / ADX_REGIME sets is
# "baseline" (the original 74). Kept explicit so a rename of any new
# column is caught immediately.
EMA_COLS = {
    "ema_20", "ema_50", "ema_100", "ema_200",
    "ema_20_slope", "ema_50_slope",
    "distance_from_ema20_pct", "distance_from_ema200_pct",
    "ema20_above_ema50", "ema50_above_ema100", "ema100_above_ema200",
    "ema_alignment_score",
}
ADX_REGIME_COLS = {
    "adx_14", "di_plus_14", "di_minus_14", "adx_slope",
    "is_trending", "is_ranging",
}


def _family_of(col: str) -> str:
    if col in EMA_COLS:
        return "EMA"
    if col in ADX_REGIME_COLS:
        return "ADX/Regime"
    return "baseline"


def _build_splits(df: pd.DataFrame, feat_cols: List[str],
                  test_month: pd.Period) -> Splits:
    """Same 11 train + 1 val + 1 test split as walk_forward_limits."""
    train_start = (test_month - 12).to_timestamp()
    val_start = (test_month - 1).to_timestamp()
    test_start = test_month.to_timestamp()
    test_end = (test_month + 1).to_timestamp()

    ts = pd.to_datetime(df["datetime"])
    tr_mask = (ts >= train_start) & (ts < val_start)
    va_mask = (ts >= val_start) & (ts < test_start)
    te_mask = (ts >= test_start) & (ts < test_end)

    df_tr = df.loc[tr_mask].loc[lambda d: d["label"].notna()]
    df_va = df.loc[va_mask].loc[lambda d: d["label"].notna()]
    df_te = df.loc[te_mask].reset_index(drop=True)

    return Splits(
        X_train=df_tr[feat_cols].astype(np.float32),
        y_train=df_tr["label"].astype(int),
        X_val=df_va[feat_cols].astype(np.float32),
        y_val=df_va["label"].astype(int),
        X_test=df_te[feat_cols].astype(np.float32),
        y_test=pd.Series(dtype=int),
        feature_cols=feat_cols,
        meta_test=df_te[["datetime", "symbol", "spot_close"]].reset_index(drop=True),
    )


def _importance_from_booster(booster, feat_cols: List[str]) -> pd.DataFrame:
    imp = booster.feature_importance(importance_type="gain")
    return pd.DataFrame({"feature": feat_cols, "gain": imp})


def _shap_mean_abs(booster, X: pd.DataFrame) -> pd.Series:
    # LightGBM's built-in SHAP: last column is bias, drop it.
    contrib = booster.predict(X, pred_contrib=True)
    contrib = contrib[:, :-1]
    return pd.Series(np.abs(contrib).mean(axis=0), index=X.columns)


def analyze(parquet_path: Path, test_month: pd.Period) -> Dict:
    df = pd.read_parquet(parquet_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    feat_cols = feature_columns(df)

    splits = _build_splits(df, feat_cols, test_month)
    up_model, down_model = bin_mod.train_binary_pair(splits)

    up_imp = _importance_from_booster(up_model, feat_cols)
    dn_imp = _importance_from_booster(down_model, feat_cols)

    up_shap = _shap_mean_abs(up_model, splits.X_test)
    dn_shap = _shap_mean_abs(down_model, splits.X_test)

    combined = pd.DataFrame({
        "feature": feat_cols,
        "family": [_family_of(c) for c in feat_cols],
        "gain_up": up_imp["gain"].values,
        "gain_down": dn_imp["gain"].values,
        "shap_up_abs_mean": [up_shap[c] for c in feat_cols],
        "shap_down_abs_mean": [dn_shap[c] for c in feat_cols],
    })
    combined["gain_total"] = combined["gain_up"] + combined["gain_down"]
    combined["shap_total"] = combined["shap_up_abs_mean"] + combined["shap_down_abs_mean"]

    family = combined.groupby("family").agg(
        n_features=("feature", "count"),
        gain_total=("gain_total", "sum"),
        shap_total=("shap_total", "sum"),
    ).reset_index()
    family["gain_share_pct"] = family["gain_total"] / family["gain_total"].sum() * 100
    family["shap_share_pct"] = family["shap_total"] / family["shap_total"].sum() * 100
    family["gain_per_feature"] = family["gain_total"] / family["n_features"]

    return {
        "test_month": str(test_month),
        "n_features_model_visible": len(feat_cols),
        "n_train_rows": int(len(splits.X_train)),
        "n_val_rows": int(len(splits.X_val)),
        "n_test_rows": int(len(splits.X_test)),
        "features_df": combined.sort_values("gain_total", ascending=False),
        "family_df": family.sort_values("gain_total", ascending=False),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True,
                    help="Label for this config, e.g. baseline / ema_only / adx_only / full")
    ap.add_argument("--fold", default="2025-06",
                    help="Walk-forward test month (YYYY-MM) to use as representative")
    args = ap.parse_args()

    cfg = get_config()
    parquet = cfg.paths["parquet_path"]
    if not parquet.exists():
        print(f"Parquet missing at {parquet}")
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = analyze(parquet, pd.Period(args.fold, freq="M"))

    features_csv = OUT_DIR / f"importance_{args.tag}.csv"
    family_csv = OUT_DIR / f"importance_family_{args.tag}.csv"
    result["features_df"].to_csv(features_csv, index=False)
    result["family_df"].to_csv(family_csv, index=False)

    meta = {
        "tag": args.tag,
        "test_month": result["test_month"],
        "n_features_model_visible": result["n_features_model_visible"],
        "n_train_rows": result["n_train_rows"],
        "n_val_rows": result["n_val_rows"],
        "n_test_rows": result["n_test_rows"],
    }
    (OUT_DIR / f"importance_{args.tag}.meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"\n[{args.tag}]  fold={args.fold}  model_visible={result['n_features_model_visible']}")
    print(f"  train_rows={result['n_train_rows']:,}  "
          f"val_rows={result['n_val_rows']:,}  "
          f"test_rows={result['n_test_rows']:,}")
    print("\n  Family importance (LightGBM gain, summed across UP+DOWN boosters):")
    for _, r in result["family_df"].iterrows():
        print(f"    {r['family']:<12s}  n={int(r['n_features']):>3d}  "
              f"gain_total={r['gain_total']:>12,.0f}  "
              f"({r['gain_share_pct']:>5.1f}% of model)  "
              f"gain/feat={r['gain_per_feature']:>10,.0f}")
    print(f"\nWritten:")
    print(f"  {features_csv}")
    print(f"  {family_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
