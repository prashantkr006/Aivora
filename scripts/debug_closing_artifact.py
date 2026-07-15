"""Show ONE trade from the 'closing' ablation where backtester used
NEXT DAY's opening spot as the "exit" price for a trade opened near
today's close. Demonstrates the artifact concretely.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from aivora.backtest.backtester import _exit_premium, run_backtest
from aivora.live.inference import CURRENT_DOWN, CURRENT_UP
from aivora.ml import binary as bin_mod
from aivora.ml.dataset import Splits
from aivora.pipeline.feature_engineering import feature_columns
from aivora.utils.config import get_config
import joblib


def main() -> int:
    cfg = get_config()
    df_all = pd.read_parquet(cfg.paths["parquet_path"])
    df_all["datetime"] = pd.to_datetime(df_all["datetime"])

    # Pick a specific date to demonstrate the artifact — use 2026-03-17
    # which is a random Monday, and show what happens if we trade at 15:20
    # NIFTY with horizon=6.
    target_date = pd.Timestamp("2026-03-17").date()
    day_end = pd.Timestamp("2026-03-17 15:20:00")

    # Meta_test is sorted by (symbol, datetime). Simulate the backtester's
    # exact indexing: find the row for NIFTY at 15:20 and walk 6 candles
    # forward using .iloc[i+1..i+6].
    df_nifty = df_all[df_all["symbol"] == "NIFTY"].sort_values("datetime").reset_index(drop=True)
    idx = df_nifty.index[df_nifty["datetime"] == day_end]
    if idx.empty:
        print(f"No NIFTY bar at {day_end}")
        return 1
    i = int(idx[0])
    entry_bar = df_nifty.iloc[i]
    spot_entry = float(entry_bar["spot_close"])

    print(f"=== Trade entry ===")
    print(f"  bar {i}: {entry_bar['datetime']}  spot_close={spot_entry:.2f}")
    print()
    print(f"=== Backtester walks the next 6 candles (horizon=6, 5min each) ===")
    print(f"  {'step':>4}  {'index':>6}  {'datetime':>19}  {'day':>10}  {'spot_close':>10}  {'synth_exit_premium':>18}")

    # Simulate a CE trade for concreteness
    entry_premium = 100.0  # placeholder; real would be from ce_ltp
    side = "CE"
    horizon = 6
    bt = {"option_delta": 0.5, "expiry_days_assumption": 7}

    for step in range(1, horizon + 1):
        j = i + step
        r = df_nifty.iloc[j]
        r_date = r["datetime"].date()
        marker = "  <-- SAME DAY" if r_date == target_date else "  <-- NEXT DAY (artifact!)"
        candidate = _exit_premium(entry_premium, spot_entry, float(r["spot_close"]),
                                  side, step * 5, bt)
        print(f"  {step:>4}  {j:>6}  {str(r['datetime']):>19}  {str(r_date):>10}  {r['spot_close']:>10.2f}  {candidate:>18.4f}{marker}")

    # Show the huge spot jump
    exit_bar = df_nifty.iloc[i + horizon]
    print()
    print(f"=== Result ===")
    print(f"  Entry:  {entry_bar['datetime']} @ spot {spot_entry:.2f}")
    print(f"  'Exit': {exit_bar['datetime']} @ spot {float(exit_bar['spot_close']):.2f}")
    spot_move = float(exit_bar['spot_close']) - spot_entry
    print(f"  Spot move: {spot_move:+.2f}  ({spot_move / spot_entry * 100:+.2f}%)")
    print(f"  Elapsed CANDLES = 6 (=30 min per backtester's assumption)")
    real_minutes = (exit_bar['datetime'] - entry_bar['datetime']).total_seconds() / 60
    print(f"  Elapsed REAL wall time = {real_minutes:.0f} min  <-- market closed most of this!")
    print()
    print(f"  Backtester computes exit_premium as if 30 min passed at delta=0.5:")
    exit_prem = _exit_premium(entry_premium, spot_entry, float(exit_bar['spot_close']),
                              side, 30, bt)
    print(f"    exit_premium = {exit_prem:.2f}  (vs entry {entry_premium})")
    print(f"    'P&L' per unit = {exit_prem - entry_premium:+.2f}  <-- FAKE (overnight gap)")
    print()
    print("In REAL live trading:")
    print("  * Zerodha auto-squares off open F&O at ~15:20-15:25 IST with actual last-price slippage")
    print("  * You NEVER capture the overnight gap on a same-day trade")
    print("  * If you did hold, you'd be exposed to overnight gap RISK, not gap-as-profit")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
