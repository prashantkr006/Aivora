"""Feature + label engineering.

Two rules guide every feature:

1. **No look-ahead.** Every column at row ``t`` may only use data
   from row ``t`` or earlier.  Rolling windows therefore use
   ``min_periods`` to produce NaN until a full window is available.
2. **Per-symbol grouping.** All operations are scoped by
   ``symbol`` so a Nifty candle never bleeds into Bank Nifty's
   moving average.

Beyond the classical technicals, this module adds an
options-aware feature bundle that survives even when live OI/IV
are unavailable (e.g. early history rows before options snapshots
were being captured):

* :func:`_synthetic_iv` — implied vol backed out of ATM CE+PE
  premium via a straddle approximation.
* Option premium momentum (5m / 15m).
* Put-call premium ratio (a substitute for OI-based PCR).
* Rolling spot-return volatility (10 / 30 periods).
* Average True Range (used for dynamic labelling).
* Time-to-expiry features derived from the weekly expiry weekday.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.calendar import nse_holidays
from ..utils.config import get_config
from ..utils.logger import get_logger

log = get_logger(__name__)

# Realised risk-free rate (approx) used in the synthetic-IV
# closed-form.  A small change here only shifts synth_iv by a
# constant, so the exact value doesn't matter for a tree model.
_RF = 0.065


# =============================================================
#  Classical technicals
# =============================================================
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI (matches trading-platform display)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return pd.DataFrame({"macd": macd, "macd_signal": sig, "macd_hist": hist})


def _bollinger_width(series: pd.Series, period: int = 20, n_std: float = 2.0) -> pd.Series:
    mid = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std()
    return (2 * n_std * std) / mid


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Average True Range.

    We keep it in absolute price units — dividing by close gives an
    ATR%-style variant used by the dynamic labelling helper.
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


# =============================================================
#  Options-aware features that survive missing OI/IV
# =============================================================
def _synthetic_iv(
    ce_ltp: pd.Series,
    pe_ltp: pd.Series,
    spot: pd.Series,
    days_to_expiry: pd.Series,
) -> pd.Series:
    """Back out an ATM straddle IV using the Brenner–Subrahmanyam
    closed-form approximation.

    For an ATM European option this reduces to:

        IV ≈ (straddle_price / spot) * √(2π / T)

    where T is time to expiry in years.  Not exact, but the
    monotone relationship to market-observed IV is what the tree
    model actually uses.  Rows lacking CE / PE prices produce NaN.
    """
    straddle = (ce_ltp.astype(float) + pe_ltp.astype(float))
    T = (days_to_expiry.astype(float).clip(lower=0.5)) / 365.0
    with np.errstate(divide="ignore", invalid="ignore"):
        iv = (straddle / spot) * np.sqrt(2 * np.pi / T)
    return iv


def _days_to_next_weekly_expiry(ts: pd.Series, weekday: int) -> pd.Series:
    """Number of calendar days until the next (holiday-adjusted) weekly expiry.

    ``weekday=3`` is Thursday.  If Thursday is an NSE holiday we
    shift the expiry back to the previous trading day.
    """
    holidays = nse_holidays()
    ts_date = ts.dt.date
    # Compute the next weekly expiry once per unique date, then map back.
    unique = pd.Series(ts_date.unique()).sort_values()

    def _next_expiry(d):
        # First find the coming target-weekday.
        delta = (weekday - d.weekday()) % 7
        target = d + timedelta(days=delta)
        # If the target itself is a holiday / weekend, roll back to
        # the previous trading day.
        for _ in range(5):
            if target.weekday() < 5 and target not in holidays:
                break
            target -= timedelta(days=1)
        return (target - d).days

    mapping = {d: _next_expiry(d) for d in unique}
    return ts_date.map(mapping).astype(float)


# =============================================================
#  Per-symbol feature factory
# =============================================================
def _vwap_intraday(df: pd.DataFrame) -> pd.Series:
    """Session-anchored VWAP.

    Resets at each new trading day so the day's VWAP doesn't carry
    yesterday's baseline.  Uses volume when available, else falls
    back to a simple typical-price rolling mean.
    """
    typical = (df["spot_high"] + df["spot_low"] + df["spot_close"]) / 3.0
    if "volume" in df.columns and df["volume"].notna().any():
        vol = df["volume"].astype(float).fillna(0.0)
        day = df["datetime"].dt.date
        num = (typical * vol).groupby(day).cumsum()
        den = vol.groupby(day).cumsum().replace(0, np.nan)
        return num / den
    # No volume — degrade gracefully.
    return typical.rolling(20, min_periods=1).mean()


def _direction_streak(close: pd.Series, window: int = 5) -> pd.Series:
    """Signed count of consecutive same-direction candles in the last ``window``.

    +N = N up-candles in a row leading up to now, similarly for
    -N.  A tree model latches onto this as a cheap momentum proxy.
    """
    step = np.sign(close.diff().fillna(0)).astype(int)
    signs = step.rolling(window, min_periods=1).apply(
        lambda s: s[-1] * int(np.all(s == s[-1])) * int((s != 0).sum()),
        raw=True,
    )
    return signs


def _engineer_one_symbol(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df.sort_values("datetime").copy()
    close = out["spot_close"].astype(float)
    high = out["spot_high"].astype(float)
    low = out["spot_low"].astype(float)

    # ---- Returns over multiple lags ----
    for lag in params["return_lags_minutes"]:
        n = max(1, lag // 5)
        out[f"ret_{lag}m"] = close.pct_change(n)

    # ---- Range / volatility proxies ----
    out["range"] = high - low
    out["range_pct"] = out["range"] / close

    # Rolling volatility of 5-min returns (10 & 30 lookbacks).
    ret1 = close.pct_change(1)
    out["ret_vol_10"] = ret1.rolling(10, min_periods=10).std()
    out["ret_vol_30"] = ret1.rolling(30, min_periods=30).std()

    # ATR — absolute + normalised.
    out["atr_14"] = _atr(high, low, close, period=14)
    out["atr_14_pct"] = out["atr_14"] / close

    # ---- Volume features (optional source) ----
    if "volume" in out.columns:
        vol = out["volume"].astype(float)
        out["vol_z20"] = (vol - vol.rolling(20, min_periods=20).mean()) / vol.rolling(
            20, min_periods=20
        ).std()

    # ---- Technical indicators ----
    out["rsi"] = _rsi(close, period=params["rsi_period"])
    macd_df = _macd(close)
    out[macd_df.columns] = macd_df
    out["bb_width"] = _bollinger_width(close, period=params["bb_period"], n_std=params["bb_std"])
    out["ma_short"] = close.rolling(params["ma_short"], min_periods=params["ma_short"]).mean()
    out["ma_long"] = close.rolling(params["ma_long"], min_periods=params["ma_long"]).mean()
    out["ma_cross"] = (out["ma_short"] - out["ma_long"]) / close

    # ---- OI-based (may be NaN — resilient to that) ----
    if "pe_oi" in out.columns and "ce_oi" in out.columns:
        out["pcr"] = out["pe_oi"] / out["ce_oi"].replace(0, np.nan)
        out["ce_oi_change"] = out["ce_oi"].diff()
        out["pe_oi_change"] = out["pe_oi"].diff()
        out["oi_buildup_ratio"] = out["pe_oi_change"] / out["ce_oi_change"].replace(0, np.nan)

    # ---- Premium-based (available even when OI isn't) ----
    if "ce_ltp" in out.columns and "pe_ltp" in out.columns:
        ce = out["ce_ltp"].astype(float)
        pe = out["pe_ltp"].astype(float)
        out["premium_spread"] = ce + pe
        out["premium_skew"] = (ce - pe) / out["premium_spread"].replace(0, np.nan)
        # Put-call *premium* ratio — a resilient substitute for OI PCR.
        out["put_call_premium_ratio"] = pe / ce.replace(0, np.nan)
        # Premium momentum (rate of change) over 5- and 15-minute windows.
        out["ce_prem_mom_5m"] = ce.pct_change(1)
        out["pe_prem_mom_5m"] = pe.pct_change(1)
        out["ce_prem_mom_15m"] = ce.pct_change(3)
        out["pe_prem_mom_15m"] = pe.pct_change(3)

    # ---- IV: use provided if present, else synthesise from straddle ----
    ts = out["datetime"]
    expiry_weekday = int(
        get_config().raw.get("historical", {}).get("expiry_weekday", 3)
    )
    dte = _days_to_next_weekly_expiry(ts, expiry_weekday)
    out["days_to_expiry"] = dte
    out["is_expiry_day"] = (dte <= 0.5).astype(int)
    out["is_expiry_week"] = (dte <= 4).astype(int)

    if "iv" in out.columns and out["iv"].notna().any():
        out["iv_change"] = out["iv"].diff()
    if {"ce_ltp", "pe_ltp"}.issubset(out.columns):
        out["synth_iv"] = _synthetic_iv(
            out["ce_ltp"], out["pe_ltp"], close, dte
        )
        out["synth_iv_change"] = out["synth_iv"].diff()

    # ---- Time / session features ----
    out["hour"] = ts.dt.hour
    out["minute"] = ts.dt.minute
    out["day_of_week"] = ts.dt.dayofweek
    hh_mm = ts.dt.hour * 60 + ts.dt.minute
    out["minutes_since_open"] = (hh_mm - (9 * 60 + 15)).clip(lower=0)
    out["is_first_15min"] = (out["minutes_since_open"] < 15).astype(int)
    out["is_first_30min"] = (out["minutes_since_open"] < 30).astype(int)
    out["is_last_30min"] = (hh_mm >= (15 * 60)).astype(int)
    out["is_last_60min"] = (hh_mm >= (14 * 60 + 30)).astype(int)

    # ---- VWAP-relative price ----
    vwap = _vwap_intraday(out)
    out["vwap"] = vwap
    out["vwap_dist_pct"] = (close - vwap) / close

    # ---- Opening-range breakout (first 15 min of the day) ----
    day = out["datetime"].dt.date
    first15 = out["minutes_since_open"] < 15
    or_high = high.where(first15).groupby(day).cummax().groupby(day).ffill()
    or_low = low.where(first15).groupby(day).cummin().groupby(day).ffill()
    out["or_high"] = or_high
    out["or_low"] = or_low
    out["or_break_up"] = ((close > or_high) & (~first15)).astype(int)
    out["or_break_down"] = ((close < or_low) & (~first15)).astype(int)
    out["or_dist_up_pct"] = (close - or_high) / close
    out["or_dist_down_pct"] = (or_low - close) / close

    # ---- Direction streak (last 5 candles same-signed) ----
    out["dir_streak_5"] = _direction_streak(close, window=5)

    # ---- Volatility regime ----
    # Percentile of ATR% inside a 20-day (~1500 candles) window —
    # long enough to capture a regime shift but short enough not to
    # burn 90 trading days of warm-up.
    atr_pct = out["atr_14_pct"]
    out["vol_regime_pct"] = atr_pct.rolling(
        window=20 * 75,
        min_periods=200,
    ).rank(pct=True)

    # =============================================================
    #  Experiment 1 additions — EMAs, ADX, Regime flags
    # =============================================================
    # NOTE: appended at the very end so downstream column order for
    # every existing feature is unchanged. Every column below is
    # side-effect free — no look-ahead, all per-symbol.
    #
    # ABLATION TOGGLES — two independent switches:
    #
    #   _EXP1_EMA_ENABLED  → +12 columns (4 EMAs, 2 slopes, 2 dists,
    #                        3 alignment flags, 1 alignment score)
    #   _EXP1_ADX_ENABLED  → +6 columns (adx_14, di_plus_14, di_minus_14,
    #                        adx_slope, is_trending, is_ranging)
    #
    # Family totals (added on top of the 74-column baseline):
    #   Baseline               →  74
    #   EMA only               →  86
    #   ADX/Regime only        →  80
    #   EMA + ADX/Regime (full)→  92
    _EXP1_EMA_ENABLED = True
    _EXP1_ADX_ENABLED = True

    if _EXP1_EMA_ENABLED:
        # ---- Exponential moving averages ----
        for span in (20, 50, 100, 200):
            out[f"ema_{span}"] = close.ewm(
                span=span, adjust=False, min_periods=span
            ).mean()

        # 5-period % slope (scale-invariant, so 24k Nifty vs 55k BN both fit).
        for span in (20, 50):
            prev = out[f"ema_{span}"].shift(5)
            out[f"ema_{span}_slope"] = (out[f"ema_{span}"] - prev) / prev

        # Signed distance from key EMAs (positive = price above EMA).
        for span in (20, 200):
            ema = out[f"ema_{span}"]
            out[f"distance_from_ema{span}_pct"] = (close - ema) / ema

        # Alignment flags — 1 when the faster EMA is above the slower one.
        out["ema20_above_ema50"] = (out["ema_20"] > out["ema_50"]).astype(int)
        out["ema50_above_ema100"] = (out["ema_50"] > out["ema_100"]).astype(int)
        out["ema100_above_ema200"] = (out["ema_100"] > out["ema_200"]).astype(int)
        out["ema_alignment_score"] = (
            out["ema20_above_ema50"]
            + out["ema50_above_ema100"]
            + out["ema100_above_ema200"]
        )

    if _EXP1_ADX_ENABLED:
        # ---- ADX (14-period Wilder) ----
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=out.index,
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=out.index,
        )

        # Wilder's smoothing = ewm with alpha = 1/period.
        period = 14
        atr_wilder = tr.ewm(
            alpha=1 / period, adjust=False, min_periods=period
        ).mean()
        plus_dm_smooth = plus_dm.ewm(
            alpha=1 / period, adjust=False, min_periods=period
        ).mean()
        minus_dm_smooth = minus_dm.ewm(
            alpha=1 / period, adjust=False, min_periods=period
        ).mean()

        atr_safe = atr_wilder.replace(0, np.nan)
        out["di_plus_14"] = 100 * plus_dm_smooth / atr_safe
        out["di_minus_14"] = 100 * minus_dm_smooth / atr_safe
        di_sum = (out["di_plus_14"] + out["di_minus_14"]).replace(0, np.nan)
        dx = 100 * (out["di_plus_14"] - out["di_minus_14"]).abs() / di_sum
        out["adx_14"] = dx.ewm(
            alpha=1 / period, adjust=False, min_periods=period
        ).mean()

        adx_prev = out["adx_14"].shift(5)
        out["adx_slope"] = (out["adx_14"] - adx_prev) / adx_prev.replace(0, np.nan)

        # ---- Simple market-regime flags ----
        # If EMA family is also enabled, use its alignment score as an
        # additional trend confirmation; otherwise fall back to a pure
        # ADX-based trend flag so this family remains self-contained.
        if _EXP1_EMA_ENABLED:
            out["is_trending"] = (
                (out["adx_14"] > 25) & (out["ema_alignment_score"] >= 2)
            ).astype(int)
        else:
            out["is_trending"] = (out["adx_14"] > 25).astype(int)
        out["is_ranging"] = (out["adx_14"] < 20).astype(int)

    return out


# =============================================================
#  Labels — static or ATR-scaled dynamic
# =============================================================
def add_label(
    df: pd.DataFrame,
    horizon: int,
    up_thr: float,
    down_thr: float,
    mode: str = "static",
    atr_k: float = 0.5,
) -> pd.DataFrame:
    """Add ``fwd_return`` + 3-class ``label`` columns.

    ``mode="static"``  — use ``up_thr`` / ``down_thr`` directly.
    ``mode="dynamic_atr"`` — use ±``atr_k * atr_14_pct`` as the
    per-row threshold, which lets the model see the same "big
    move" signal in both quiet and volatile regimes.
    """
    out = df.sort_values(["symbol", "datetime"]).copy()
    parts = []
    for _, grp in out.groupby("symbol"):
        close = grp["spot_close"].astype(float)
        future = close.shift(-horizon)
        fwd_ret = (future - close) / close

        if mode == "dynamic_atr" and "atr_14_pct" in grp.columns:
            thr = atr_k * grp["atr_14_pct"].astype(float)
            up_mask = fwd_ret > thr
            down_mask = fwd_ret < -thr
        else:
            up_mask = fwd_ret > up_thr
            down_mask = fwd_ret < down_thr

        label = pd.Series(0, index=grp.index, dtype="Int16")
        label[up_mask] = 2
        label[down_mask] = 1
        label[future.isna()] = pd.NA
        g = grp.copy()
        g["fwd_return"] = fwd_ret
        g["label"] = label
        parts.append(g)
    return pd.concat(parts).sort_values(["symbol", "datetime"]).reset_index(drop=True)


# =============================================================
#  Public entry point
# =============================================================
def engineer_features(
    df: pd.DataFrame,
    label_overrides: Optional[Dict] = None,
) -> pd.DataFrame:
    """Run the full feature + label pipeline.

    ``label_overrides`` can carry ``horizon_candles``, ``up_threshold``,
    ``down_threshold``, ``mode`` (``static`` / ``dynamic_atr``) and
    ``atr_k`` — used by the iteration orchestrator to A/B different
    label definitions without editing ``config.yaml``.
    """
    cfg = get_config()
    params = cfg.features
    labels = {**cfg.labels, **(label_overrides or {})}

    parts = []
    for sym, grp in df.groupby("symbol"):
        log.info("engineer_features: %s rows=%d", sym, len(grp))
        parts.append(_engineer_one_symbol(grp, params))
    feat = pd.concat(parts, ignore_index=True)

    feat = add_label(
        feat,
        horizon=int(labels["horizon_candles"]),
        up_thr=float(labels["up_threshold"]),
        down_thr=float(labels["down_threshold"]),
        mode=str(labels.get("mode", "static")),
        atr_k=float(labels.get("atr_k", 0.5)),
    )

    before = len(feat)
    if params["drop_na"]:
        # LightGBM handles NaN natively, so we only drop rows that
        # are missing *essential* spot-derived features — the ones
        # without which no meaningful signal exists.  Options-derived
        # NaNs (older rows without CE/PE) pass through and the tree
        # learns "options absent" as its own branch.
        #
        # CRITICAL: ``label`` is deliberately EXCLUDED from essentials.
        # Rows near the end of the timeline have NaN labels because
        # their forward return is still unknown (the horizon extends
        # into the future).  Those rows are USELESS for training but
        # ESSENTIAL for live inference — they represent the most
        # recent candles, which is what the scheduler wants to
        # predict on.  Training paths filter NaN labels themselves
        # (see ``dataset.make_splits``, ``scripts/freeze_model``,
        # ``scripts/backtest_today``).
        essentials = [c for c in [
            "rsi", "macd", "bb_width", "atr_14", "ret_5m",
            "spot_close",
        ] if c in feat.columns]
        feat = feat.dropna(subset=essentials).reset_index(drop=True)
    dropped = before - len(feat)
    if dropped:
        log.info(
            "engineer_features: dropped %d rows containing NaN (%.2f%%)",
            dropped, 100 * dropped / max(before, 1),
        )

    log.info(
        "engineer_features: final shape=%s, label distribution=%s",
        feat.shape,
        feat["label"].value_counts().to_dict(),
    )
    return feat


def feature_columns(df: pd.DataFrame) -> List[str]:
    """Model-visible columns: everything except identifiers and raw inputs."""
    exclude = {
        "datetime", "symbol", "label", "fwd_return",
        "spot_open", "spot_high", "spot_low", "spot_close",
        "fut_open", "fut_high", "fut_low", "fut_close",
        "ce_ltp", "pe_ltp", "ce_oi", "pe_oi", "iv", "volume",
        "is_filled", "strike", "expiry",
    }
    # Drop columns that are all-NaN in this dataset (older rows may
    # miss options → derived cols end up empty per iteration).
    return [
        c for c in df.columns
        if c not in exclude and not df[c].isna().all()
    ]
