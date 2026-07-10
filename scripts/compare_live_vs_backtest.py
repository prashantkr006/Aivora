"""Compare live paper trades (user 27) vs backtest output for a set of dates.

For each date:
  1. Load user 27's live trades from the webapp DB.
  2. Run the backtester on the same date with the same live settings
     (Vol Filter OFF, max_trades_per_day=10, variant #18 gates).
  3. Map each live entry_time to its corresponding backtest candle
     (the last 5-min bar closed before the live tick).
  4. Match by (candle_time, symbol, side).
  5. Re-price the matched trades with the REAL live entry/exit
     premiums (from Kite quotes captured at trade time) and report
     the "real-premium-adjusted" backtest P&L alongside the synthetic
     P&L that the backtester produces on its own.

Note on premium sourcing:
    options_chain has NO rows for 2026-07-08..10 (Dhan only backfills
    options up to ~2026-07-02). Real premium data at those dates
    exists only on the live trade records themselves. So the backtest
    itself uses the synthetic 1%/1.2% premium; we ALSO layer the
    live-recorded premium onto the matched trades so the P&L comparison
    is honest.
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aivora.backtest.backtester import run_backtest  # noqa: E402
from aivora.live.inference import CURRENT_DOWN, CURRENT_UP  # noqa: E402
from aivora.ml import binary as bin_mod  # noqa: E402
from aivora.ml.dataset import Splits  # noqa: E402
from aivora.pipeline.feature_engineering import feature_columns  # noqa: E402
from aivora.utils.config import get_config  # noqa: E402
from aivora.webapp import db as webapp_db  # noqa: E402
from aivora.webapp import portfolios  # noqa: E402
import joblib  # noqa: E402


DATES = [date(2026, 7, 8), date(2026, 7, 9), date(2026, 7, 10)]
USER_ID = 27
OUT_REPORT = Path(__file__).resolve().parents[1] / "logs" / "live_vs_backtest_report.txt"

# Live-system settings (mirrors user 27's portfolio.settings)
LIVE_SETTINGS = {
    "prob_threshold_up": 0.55,
    "prob_threshold_down": 0.60,
    "take_profit_pct": 0.60,
    "stop_loss_pct": 0.30,
    "min_minutes_since_open": 30,
    "max_minutes_since_open": 300,
    "vol_regime_min": 0.0,
    "vol_regime_max": 999.0,
    "max_trades_per_day": 10,
}


def _load_live_trades() -> pd.DataFrame:
    """Read live paper trades for USER_ID."""
    webapp_db.init_db()
    port = portfolios.UserPortfolio(USER_ID, "paper")
    state = port.load()
    if not state["trades"]:
        return pd.DataFrame()
    df = pd.DataFrame(state["trades"])
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
    df["entry_date"] = df["entry_time"].dt.date
    return df


def _live_time_to_backtest_candle(live_time: pd.Timestamp) -> pd.Timestamp:
    """Live tick at HH:MM:SS uses the last-closed 5-min candle.

    Candle at HH:MM covers [HH:MM, HH:MM+5min); it closes at HH:MM+5min.
    So at live time T, the last-closed candle = floor(T / 5min) - 5min
    (T rounded down to the current 5-min mark, then step back one bar).
    Empirical verification: user's 11:05:20 tick was made off the 11:00
    candle's close (which happened at 11:05:00).
    """
    floor_5 = live_time.floor("5min")
    return floor_5 - pd.Timedelta(minutes=5)


def _run_backtest_for_date(df: pd.DataFrame, target: date) -> pd.DataFrame:
    """Freeze-model-based backtest of variant #18 gates on ``target``."""
    cfg = get_config()
    models_dir = cfg.paths["models_dir"]
    up_model = joblib.load(models_dir / CURRENT_UP)
    down_model = joblib.load(models_dir / CURRENT_DOWN)

    # Filter to target date
    day_mask = df["datetime"].dt.date == target
    day = df.loc[day_mask].sort_values(["symbol", "datetime"]).reset_index(drop=True)
    if day.empty:
        return pd.DataFrame()

    feat_cols = feature_columns(df)
    X = day[feat_cols].astype(np.float32)
    probs = bin_mod.predict_3class_from_binary(up_model, down_model, X)

    keep_cols = ["datetime", "symbol", "spot_close", "fwd_return",
                 "ce_ltp", "pe_ltp", "minutes_since_open", "vol_regime_pct"]
    meta = pd.DataFrame(index=day.index)
    for c in keep_cols:
        meta[c] = day[c] if c in day.columns else np.nan
    meta = meta.reset_index(drop=True)

    splits = Splits(
        X_train=pd.DataFrame(), y_train=pd.Series(dtype=int),
        X_val=pd.DataFrame(), y_val=pd.Series(dtype=int),
        X_test=X, y_test=pd.Series(dtype=int),
        feature_cols=feat_cols, meta_test=meta,
    )
    result = run_backtest(probs, splits, overrides=dict(LIVE_SETTINGS),
                          name=f"live_vs_bt_{target}")
    return result["trades"] if result.get("trades") is not None else pd.DataFrame()


def _match_and_reprice(
    live: pd.DataFrame, bt: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict]:
    """Match live vs backtest by (candle, symbol, side)."""
    if live.empty and bt.empty:
        return pd.DataFrame(), {"n_matched": 0, "n_live_only": 0, "n_bt_only": 0}

    live_map: Dict[Tuple, pd.Series] = {}
    for _, r in live.iterrows():
        candle = _live_time_to_backtest_candle(r["entry_time"])
        key = (candle, r["symbol"], r["side"])
        live_map[key] = r

    bt_map: Dict[Tuple, pd.Series] = {}
    if not bt.empty:
        bt["datetime"] = pd.to_datetime(bt["datetime"])
        for _, r in bt.iterrows():
            key = (pd.Timestamp(r["datetime"]), r["symbol"], r["side"])
            bt_map[key] = r

    all_keys = sorted(set(live_map.keys()) | set(bt_map.keys()))
    rows = []
    for k in all_keys:
        L = live_map.get(k)
        B = bt_map.get(k)
        # Initialise every column so the resulting DataFrame always has
        # the same schema, even when there are zero live↔bt matches.
        row: Dict = {
            "candle": k[0], "symbol": k[1], "side": k[2],
            "in_live": L is not None, "in_bt": B is not None,
            "live_entry_prem": np.nan, "live_exit_prem": np.nan, "live_pnl": np.nan,
            "bt_entry_prem": np.nan, "bt_exit_prem": np.nan, "bt_pnl_synth": np.nan,
            "bt_lots": np.nan, "bt_lot_size": np.nan, "bt_pnl_realprem": np.nan,
        }
        if L is not None:
            row["live_entry_prem"] = float(L["entry_premium"])
            row["live_exit_prem"] = float(L["exit_premium"] or L.get("current_premium") or 0.0)
            live_pnl = L.get("realized_pnl")
            if pd.isna(live_pnl):
                live_pnl = L.get("unrealized_pnl", 0.0)
            row["live_pnl"] = float(live_pnl or 0.0)
        if B is not None:
            row["bt_entry_prem"] = float(B["entry_premium"])
            row["bt_exit_prem"] = float(B["exit_premium"])
            row["bt_pnl_synth"] = float(B["pnl"])
            row["bt_lots"] = int(B.get("lots", 1))
            row["bt_lot_size"] = int(B.get("lot_size", 1))
            if L is not None:
                # Re-price the same backtest signal with the live premiums.
                real_gross = (row["live_exit_prem"] - row["live_entry_prem"]) \
                             * row["bt_lots"] * row["bt_lot_size"]
                real_costs = float(B.get("costs", 0.0))
                row["bt_pnl_realprem"] = real_gross - real_costs
        rows.append(row)

    matched_df = pd.DataFrame(rows)
    if matched_df.empty:
        matched_df = pd.DataFrame(columns=[
            "candle", "symbol", "side", "in_live", "in_bt",
            "live_entry_prem", "live_exit_prem", "live_pnl",
            "bt_entry_prem", "bt_exit_prem", "bt_pnl_synth",
            "bt_lots", "bt_lot_size", "bt_pnl_realprem",
        ])
    stats = {
        "n_matched": int((matched_df["in_live"] & matched_df["in_bt"]).sum())
                     if not matched_df.empty else 0,
        "n_live_only": int((matched_df["in_live"] & ~matched_df["in_bt"]).sum())
                     if not matched_df.empty else 0,
        "n_bt_only": int((~matched_df["in_live"] & matched_df["in_bt"]).sum())
                     if not matched_df.empty else 0,
    }
    return matched_df, stats


def main() -> int:
    print("Loading parquet …")
    cfg = get_config()
    parquet = cfg.paths["parquet_path"]
    df = pd.read_parquet(parquet)
    df["datetime"] = pd.to_datetime(df["datetime"])
    print(f"  {len(df):,} rows, columns={df.shape[1]}")

    print("Loading live trades …")
    live_all = _load_live_trades()
    print(f"  {len(live_all)} live trades in user {USER_ID} paper portfolio")

    lines = []
    push = lines.append
    hr = "=" * 84

    push(hr)
    push("AiVora — Live paper trading vs backtest replay")
    push(f"Generated : {datetime.now().isoformat(timespec='seconds')}")
    push(f"User      : {USER_ID} (paper mode)")
    push(f"Settings  : Vol OFF (min=0, max=999), max_trades=10, no cooldown, "
         "variant #18 gates")
    push(f"Model     : {(cfg.paths['models_dir'] / CURRENT_UP).name} + "
         f"{(cfg.paths['models_dir'] / CURRENT_DOWN).name}")
    push(hr)
    push("")
    push(f"{'Date':<12s} {'Live':>6s} {'BT':>6s} {'Match':>6s} "
         f"{'Live P&L':>12s} {'BT synth':>12s} {'BT realprem':>14s} {'Signal match%':>14s}")
    push("-" * 84)

    per_date_details: List[Tuple[date, pd.DataFrame, Dict, dict]] = []
    for target in DATES:
        live = live_all[live_all["entry_date"] == target].copy()
        bt = _run_backtest_for_date(df, target)

        matched, stats = _match_and_reprice(live, bt)

        # Aggregate P&Ls
        live_pnl = float(live["realized_pnl"].fillna(live["unrealized_pnl"]).astype(float).sum()) \
                    if not live.empty else 0.0
        bt_pnl_synth = float(bt["pnl"].astype(float).sum()) if not bt.empty else 0.0
        bt_pnl_realprem = (
            float(matched.loc[matched["in_live"] & matched["in_bt"], "bt_pnl_realprem"].sum())
            + float(matched.loc[~matched["in_live"] & matched["in_bt"], "bt_pnl_synth"].sum())
        )
        # Signal match % = matched / (live ∪ bt total)
        total_signals = stats["n_matched"] + stats["n_live_only"] + stats["n_bt_only"]
        signal_match_pct = 100 * stats["n_matched"] / max(total_signals, 1)

        push(f"{str(target):<12s} {len(live):>6d} {len(bt):>6d} "
             f"{stats['n_matched']:>6d} {live_pnl:>+12,.2f} {bt_pnl_synth:>+12,.2f} "
             f"{bt_pnl_realprem:>+14,.2f} {signal_match_pct:>13.1f}%")

        per_date_details.append((target, matched, stats, {
            "live_n": len(live), "bt_n": len(bt),
            "live_pnl": live_pnl, "bt_pnl_synth": bt_pnl_synth,
            "bt_pnl_realprem": bt_pnl_realprem,
        }))
    push("")

    # Per-date breakdown
    for target, matched, stats, _ in per_date_details:
        push(hr)
        push(f"Detail — {target}")
        push("-" * 84)
        push(f"  Matched trades : {stats['n_matched']}")
        push(f"  Live-only      : {stats['n_live_only']}  (live fired, backtest didn't)")
        push(f"  BT-only        : {stats['n_bt_only']}  (backtest fired, live didn't)")
        if matched.empty:
            push("  (no signals either side)")
            continue
        push("")
        push(f"  {'Candle':<19s} {'Sym':<10s} {'Side':<4s} "
             f"{'In?':<7s} {'Live prem':>10s} {'BT prem':>10s} "
             f"{'Live P&L':>10s} {'BT synth':>10s} {'BT realprem':>12s}")
        for _, r in matched.iterrows():
            in_str = f"{'L' if r['in_live'] else '-'}{'B' if r['in_bt'] else '-'}"
            liveprem = f"{r['live_entry_prem']:.2f}" if r.get("in_live") else "—"
            btprem = f"{r['bt_entry_prem']:.2f}" if r.get("in_bt") else "—"
            livepnl = f"{r['live_pnl']:+.2f}" if r.get("in_live") else "—"
            btsynth = f"{r['bt_pnl_synth']:+.2f}" if r.get("in_bt") else "—"
            btreal = f"{r['bt_pnl_realprem']:+.2f}" if r.get("in_live") and r.get("in_bt") else "—"
            candle_s = r["candle"].strftime("%Y-%m-%d %H:%M")
            push(f"  {candle_s:<19s} {r['symbol']:<10s} {r['side']:<4s} "
                 f"{in_str:<7s} {liveprem:>10s} {btprem:>10s} "
                 f"{livepnl:>10s} {btsynth:>10s} {btreal:>12s}")
        push("")

    push(hr)
    push("Notes")
    push("-" * 84)
    push("• Backtester uses synthetic entry premium (~1% of spot) when the parquet")
    push("  row's ce_ltp/pe_ltp is NaN. options_chain has no rows past 2026-07-02,")
    push("  so all three target dates fall back to synthetic premiums for the")
    push("  raw backtest output (\"BT synth\" column).")
    push("• The \"BT realprem\" column re-prices matched trades using the live")
    push("  system's actual Kite quotes for entry/exit — that is the honest")
    push("  apples-to-apples P&L comparison for the same signal.")
    push("• Signal match % = matched / (live ∪ backtest). 100% means the model is")
    push("  behaving identically live and in backtest; anything less needs to be")
    push("  attributed to (a) tick timing (a live 10:03:32 tick predicts on the")
    push("  09:55 candle rather than 10:00), (b) settings drift, or (c) daily")
    push("  trade-limit cutoffs firing differently.")
    push(hr)

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nReport written to {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
