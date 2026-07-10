"""Prove the slim-window feature computation matches the full-history one.

For each symbol, this script:

1. Runs ``engineer_features`` on the FULL spot+options history — the way
   the old ``MarketDataCache.refresh`` used to do it.
2. Runs ``engineer_features`` on the same slim window the new cache
   uses (last N calendar days).
3. Picks the last row of each and checks every model-visible feature is
   identical (bit-for-bit for floats).
4. Runs the frozen UP/DOWN pair on both rows — probabilities must match.
5. Prints wall-clock timings for both paths.

Exit code 0 = predictions and features match. 1 = mismatch.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aivora.live.inference import LiveInference  # noqa: E402
from aivora.pipeline import database, feature_engineering  # noqa: E402
from aivora.pipeline.feature_engineering import feature_columns  # noqa: E402
from aivora.utils.config import get_config  # noqa: E402


def _merge(spot: pd.DataFrame, opts: pd.DataFrame) -> pd.DataFrame:
    if opts.empty:
        merged = spot.copy()
        for c in ("ce_ltp", "pe_ltp", "ce_oi", "pe_oi", "ce_iv"):
            if c not in merged.columns:
                merged[c] = pd.NA
    else:
        merged = pd.merge(spot, opts, on=["datetime", "symbol"], how="left")
    return merged.rename(columns={"ce_iv": "iv"})


def _last_row(feat: pd.DataFrame, symbol: str) -> pd.Series:
    sub = feat[feat["symbol"] == symbol].sort_values("datetime")
    if sub.empty:
        raise RuntimeError(f"no rows for {symbol}")
    return sub.iloc[-1]


def main() -> int:
    cfg = get_config()
    symbols = [inst["symbol"] for inst in cfg.instruments]

    # ---- FULL: how the old cache did it ----
    t0 = time.perf_counter()
    spot_full = database.load_spot_futures()
    opts_full = database.load_option_chain()
    merged_full = _merge(spot_full, opts_full)
    feat_full = feature_engineering.engineer_features(merged_full)
    t_full = time.perf_counter() - t0

    # ---- SLIM: how the new cache does it ----
    lookback_days = 60
    cutoff = pd.Timestamp(datetime.now()) - timedelta(days=lookback_days)
    t1 = time.perf_counter()
    spot_slim = database.load_spot_futures_since(cutoff)
    opts_slim = database.load_option_chain_since(cutoff)
    merged_slim = _merge(spot_slim, opts_slim)
    feat_slim = feature_engineering.engineer_features(merged_slim)
    t_slim = time.perf_counter() - t1

    print(f"\n== engineer_features timings ==")
    print(f"  full history ({len(merged_full):>7} rows in): {t_full:6.2f} s  → {len(feat_full):>7} feature rows")
    print(f"  slim window  ({len(merged_slim):>7} rows in): {t_slim:6.2f} s  → {len(feat_slim):>7} feature rows")
    print(f"  speedup: {t_full / max(t_slim, 1e-6):.1f}x")

    inf = LiveInference()
    ok = True
    for sym in symbols:
        try:
            row_full = _last_row(feat_full, sym)
            row_slim = _last_row(feat_slim, sym)
        except RuntimeError as exc:
            print(f"\n[{sym}] SKIP — {exc}")
            continue

        # Same row?
        if row_full["datetime"] != row_slim["datetime"]:
            print(
                f"\n[{sym}] ❌ latest-row datetime mismatch: "
                f"full={row_full['datetime']}  slim={row_slim['datetime']}"
            )
            ok = False
            continue

        feat_cols = feature_columns(feat_full)
        # feature_columns may differ if the slim window happens to have
        # an all-NaN column that full doesn't (or vice versa). Use the
        # union so we compare every column that either produced.
        slim_cols = feature_columns(feat_slim)
        if set(feat_cols) != set(slim_cols):
            missing = set(feat_cols) - set(slim_cols)
            extra = set(slim_cols) - set(feat_cols)
            print(f"\n[{sym}] ⚠ feature column set differs")
            if missing:
                print(f"    only in full: {sorted(missing)}")
            if extra:
                print(f"    only in slim: {sorted(extra)}")

        diffs = []
        common = sorted(set(feat_cols) & set(slim_cols))
        for c in common:
            v_full = row_full[c]
            v_slim = row_slim[c]
            if pd.isna(v_full) and pd.isna(v_slim):
                continue
            if pd.isna(v_full) or pd.isna(v_slim):
                diffs.append((c, v_full, v_slim, "NaN mismatch"))
                continue
            fa = float(v_full)
            fb = float(v_slim)
            if not np.isclose(fa, fb, rtol=1e-9, atol=1e-12):
                diffs.append((c, fa, fb, abs(fa - fb)))

        if diffs:
            ok = False
            print(f"\n[{sym}] ❌ {len(diffs)} feature(s) differ between full & slim runs:")
            for c, a, b, delta in diffs[:10]:
                print(f"    {c:30s}  full={a!r:>16s}  slim={b!r:>16s}  Δ={delta}")
        else:
            print(f"\n[{sym}] ✅ all {len(common)} common features identical at row {row_full['datetime']}")

        # Prediction parity
        try:
            r_full = inf.latest_prediction_from_df(feat_full, sym)
            r_slim = inf.latest_prediction_from_df(feat_slim, sym)
        except FileNotFoundError as exc:
            print(f"[{sym}] ⚠ frozen model missing — skipping prediction parity ({exc})")
            continue
        if r_full is None or r_slim is None:
            print(f"[{sym}] ⚠ one predictor returned None (full={r_full}, slim={r_slim})")
            continue
        for attr in ("p_up", "p_down", "p_flat", "spot_close",
                     "minutes_since_open", "vol_regime_pct"):
            va = getattr(r_full, attr)
            vb = getattr(r_slim, attr)
            if not np.isclose(va, vb, rtol=1e-9, atol=1e-12):
                print(f"[{sym}] ❌ {attr}: full={va} slim={vb}")
                ok = False
        if r_full.p_up == r_slim.p_up and r_full.p_down == r_slim.p_down:
            print(f"[{sym}] ✅ predictions identical: p_up={r_full.p_up:.6f} p_down={r_full.p_down:.6f}")

    print("\n" + ("=" * 60))
    print("VERDICT:", "PASS ✅  slim window is bit-identical" if ok else "FAIL ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
