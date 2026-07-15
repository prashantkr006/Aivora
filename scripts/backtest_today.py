"""Backtest a specific date (defaults to today) against variant #18.

Reuses the frozen binary UP/DOWN model at ``models/current_up.pkl`` +
``models/current_down.pkl`` — the same pair the live scheduler uses —
and the same probability/regime/session gates.  So the output tells
you exactly what the live system *would* have done on the target date.

Two important caveats about backtesting today specifically:

    * The training parquet drops rows whose forward return is unknown
      (i.e. the last ~12 candles of any given day, because the
      60-minute horizon extends past 15:30).  During market hours the
      backtest therefore covers only rows up to ~60 minutes before
      the current time.  That's fine for variant #18, whose session
      gate stops entries at 14:15 anyway.

    * If the tick that ran during the day filtered out a signal via
      the regime gate, this script will show the same "no fire" —
      it isn't a second chance at trades the live path missed.  It's
      a faithful replay.

Usage::

    python -m scripts.backtest_today
    python -m scripts.backtest_today --date 2026-07-07
    python -m scripts.backtest_today --date 2026-07-06 --symbol NIFTY
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aivora.backtest.backtester import run_backtest  # noqa: E402
from aivora.ml import binary as bin_mod  # noqa: E402
from aivora.ml.dataset import Splits  # noqa: E402
from aivora.pipeline.feature_engineering import feature_columns  # noqa: E402
from aivora.utils.config import get_config  # noqa: E402
from aivora.utils.logger import get_logger  # noqa: E402

log = get_logger("scripts.backtest_today")


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


VARIANT_18 = {
    "prob_threshold_up": 0.55,
    "prob_threshold_down": 0.55,
    "min_minutes_since_open": 0,
    "max_minutes_since_open": 300,
    "take_profit_pct": 0.60,
    "stop_loss_pct": 0.30,
    "min_minutes_since_open": 30,
    "max_minutes_since_open": 300,
    "vol_regime_min": 0.15,
    "vol_regime_max": 0.90,
    "max_trades_per_day": 3,
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay variant #18 on a specific date")
    ap.add_argument("--date", type=_parse_date, default=date.today(),
                    help="Date to replay (YYYY-MM-DD). Default = today.")
    ap.add_argument("--symbol", choices=["NIFTY", "BANKNIFTY"], default=None,
                    help="Restrict to one symbol (default = both)")
    ap.add_argument("--override", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="Override a variant #18 knob, e.g. --override prob_threshold_up=0.50")
    args = ap.parse_args()

    cfg = get_config()
    target = args.date
    log.info("Replay target: %s  symbol=%s", target, args.symbol or "both")

    # ---- 1. load parquet ----
    parquet = cfg.paths["parquet_path"]
    if not parquet.exists():
        log.error("Training parquet missing at %s — run the pipeline first.", parquet)
        return 2
    df = pd.read_parquet(parquet)
    df["datetime"] = pd.to_datetime(df["datetime"])
    log.info("parquet rows: %d  span=%s → %s",
             len(df), df["datetime"].min(), df["datetime"].max())

    # ---- 2. filter to target date ----
    day = df[df["datetime"].dt.date == target].copy()
    # Backtest needs forward returns to score exits — filter rows
    # whose label is NaN (last ~30-60 min of the target date have
    # no valid forward yet).
    day = day[day["label"].notna()]
    if args.symbol:
        day = day[day["symbol"] == args.symbol]
    if day.empty:
        log.error(
            "No rows in parquet for date=%s symbol=%s.  Either the tick "
            "hasn't run yet today, or this date's rows have future-label "
            "NaNs (last ~60 min are dropped during feature engineering).",
            target, args.symbol or "any",
        )
        return 3
    log.info("rows for target date: %d", len(day))

    # ---- 3. load the frozen binary pair ----
    up_path = cfg.paths["models_dir"] / "current_up.pkl"
    dn_path = cfg.paths["models_dir"] / "current_down.pkl"
    if not up_path.exists() or not dn_path.exists():
        log.error(
            "Frozen model files missing at %s / %s.  Run "
            "`python -m scripts.freeze_model` first.",
            up_path, dn_path,
        )
        return 4
    up_model = joblib.load(up_path)
    down_model = joblib.load(dn_path)
    log.info("loaded frozen UP + DOWN models")

    # ---- 4. run inference ----
    feat_cols = feature_columns(df)      # use full df so the column list is stable
    X = day[feat_cols].astype(np.float32).reset_index(drop=True)
    probs = bin_mod.predict_3class_from_binary(up_model, down_model, X)

    # ---- 5. build a Splits shim (backtester only reads meta_test) ----
    meta_cols = [
        "datetime", "symbol", "spot_close", "fwd_return",
        "ce_ltp", "pe_ltp", "minutes_since_open", "vol_regime_pct",
    ]
    meta = pd.DataFrame(index=day.index)
    for c in meta_cols:
        meta[c] = day[c] if c in day.columns else np.nan
    meta = meta.reset_index(drop=True)

    splits = Splits(
        X_train=pd.DataFrame(), y_train=pd.Series(dtype=int),
        X_val=pd.DataFrame(), y_val=pd.Series(dtype=int),
        X_test=X, y_test=pd.Series(dtype=int),
        feature_cols=feat_cols, meta_test=meta,
    )

    # ---- 6. apply optional overrides ----
    overrides = dict(VARIANT_18)
    for kv in args.override:
        k, _, v = kv.partition("=")
        try:
            overrides[k.strip()] = float(v)
        except ValueError:
            overrides[k.strip()] = v
    log.info("overrides in effect: %s", overrides)

    name = f"today_{target.strftime('%Y%m%d')}"
    if args.symbol:
        name += f"_{args.symbol}"

    # ---- 7. run backtest ----
    result = run_backtest(probs, splits, overrides=overrides, name=name)
    summary = result["summary"]
    trades = result["trades"]

    # ---- 8. printable summary ----
    print()
    print("=" * 64)
    print(f"Backtest replay — {target}   symbol={args.symbol or 'both'}")
    print("=" * 64)
    print(f"  Rows scored          : {len(day)}")
    print(f"  Trades fired         : {summary['n_trades']}")
    if summary["n_trades"] > 0:
        print(f"  Total P&L (₹)        : {summary['total_pnl']:+.2f}")
        print(f"  Win rate             : {summary['win_rate']:.0%}")
        print(f"  Avg win  / avg loss  : {summary['avg_win']:+.2f} / {summary['avg_loss']:+.2f}")
        print(f"  Reward:risk          : {summary['reward_to_risk']:.2f}")
        print(f"  Max drawdown         : {summary['max_drawdown']:+.2%}")
        print()
        print("Trades:")
        show_cols = [c for c in
                     ["datetime", "symbol", "side", "entry_spot", "exit_spot",
                      "entry_premium", "exit_premium", "lots", "exit_reason",
                      "gross_pnl", "costs", "pnl"] if c in trades.columns]
        print(trades[show_cols].to_string(index=False))
    else:
        print("  (No trades fired — gates blocked every candle.)")
        # Show why: worst-blocked reasons across the day.
        meta["p_up"] = probs[:, 2]
        meta["p_down"] = probs[:, 1]
        # A row would fire if:
        #   msoo ∈ [30, 300], vr ∈ [0.15, 0.90], and (p_up ≥ 0.55 or p_down ≥ 0.60)
        in_window = meta["minutes_since_open"].between(30, 300)
        in_regime = meta["vol_regime_pct"].between(0.15, 0.90)
        up_hit = meta["p_up"] >= overrides["prob_threshold_up"]
        dn_hit = meta["p_down"] >= overrides["prob_threshold_down"]
        print(f"  Rows in session window     : {int(in_window.sum())}")
        print(f"  Rows in vol-regime band    : {int(in_regime.sum())}")
        print(f"  Rows with p_up  ≥ threshold: {int(up_hit.sum())}")
        print(f"  Rows with p_dn  ≥ threshold: {int(dn_hit.sum())}")
        print(f"  Peak p_up in day (any row) : {float(meta['p_up'].max()):.3f}")
        print(f"  Peak p_dn in day (any row) : {float(meta['p_down'].max()):.3f}")
    print()
    print(f"Trades CSV:  {result['trades_path']}")
    print(f"Equity plot: {result['equity_curve_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
