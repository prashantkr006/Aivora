"""Smoke tests — they don't need network, Kite, or real data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aivora.pipeline import data_cleaning, feature_engineering


def _synthetic(n: int = 600) -> pd.DataFrame:
    """A reproducible synthetic 5-min OHLC + option series."""
    rng = np.random.default_rng(seed=42)
    ts = pd.date_range("2024-01-02 09:15", periods=n, freq="5min")
    # Filter to inside-session timestamps for both symbols.
    rows = []
    for sym, base in [("NIFTY", 22000.0), ("BANKNIFTY", 48000.0)]:
        prices = base + np.cumsum(rng.normal(0, base * 0.0005, size=n))
        df = pd.DataFrame({
            "datetime": ts,
            "symbol": sym,
            "spot_open": prices,
            "spot_high": prices * 1.0005,
            "spot_low": prices * 0.9995,
            "spot_close": prices,
            "fut_open": prices,
            "fut_high": prices,
            "fut_low": prices,
            "fut_close": prices,
            "ce_ltp": rng.uniform(80, 200, size=n),
            "pe_ltp": rng.uniform(80, 200, size=n),
            "ce_oi": rng.integers(50_000, 500_000, size=n).astype(float),
            "pe_oi": rng.integers(50_000, 500_000, size=n).astype(float),
            "iv": rng.uniform(10, 25, size=n),
            "volume": rng.integers(10_000, 100_000, size=n).astype(float),
        })
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def test_clean_keeps_session_only():
    df = _synthetic()
    out = data_cleaning.clean(df)
    times = out["datetime"].dt.time
    assert (times >= pd.Timestamp("09:15").time()).all()
    assert (times <= pd.Timestamp("15:30").time()).all()


def test_feature_engineering_produces_label():
    df = _synthetic()
    cleaned = data_cleaning.clean(df)
    feats = feature_engineering.engineer_features(cleaned)
    assert {"label", "fwd_return", "rsi", "pcr", "macd"} <= set(feats.columns)
    assert set(feats["label"].unique()).issubset({0, 1, 2})
    cols = feature_engineering.feature_columns(feats)
    # Sanity: feature columns must be all numeric and exclude raw prices.
    assert "spot_close" not in cols
    assert "datetime" not in cols
    assert "label" not in cols
    assert feats[cols].dtypes.apply(lambda d: np.issubdtype(d, np.number)).all()


def test_no_lookahead_in_features():
    """Features at row t must not be affected by data after t."""
    df = _synthetic(300)
    cleaned = data_cleaning.clean(df)
    full = feature_engineering.engineer_features(cleaned).sort_values(
        ["symbol", "datetime"]
    ).reset_index(drop=True)

    # Truncate after a fixed midpoint and re-run; the rows present
    # in both must match on every feature column.
    midpoint = cleaned["datetime"].sort_values().iloc[len(cleaned) // 2]
    truncated = cleaned[cleaned["datetime"] <= midpoint]
    truncated_feats = feature_engineering.engineer_features(truncated).sort_values(
        ["symbol", "datetime"]
    ).reset_index(drop=True)

    cols = feature_engineering.feature_columns(full)
    merged = full.merge(
        truncated_feats[["symbol", "datetime"] + cols],
        on=["symbol", "datetime"],
        suffixes=("_full", "_trunc"),
    )
    for col in cols:
        diff = (merged[f"{col}_full"] - merged[f"{col}_trunc"]).abs()
        assert diff.fillna(0).max() < 1e-6, f"Lookahead detected in {col}"
