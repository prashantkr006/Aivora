"""Compare the re-entry cooldown ON vs OFF over the last N trading days.

Runs the same backtest (variant-#18 gates, trailing SL enabled) twice
per day — once with the cooldown active (defaults ``prob_delta=0.05``,
``price_pct=0.001``) and once with it disabled (both thresholds set
to ``0.0``, which makes the "close" check ``abs(delta) < 0`` always
False, so the cooldown never fires).  Every other input is identical
between the two runs, so any difference is attributable purely to
the cooldown.

Outputs:

* ``reports/cooldown_comparison_30days.csv`` — one row per day with
  P&L / trade-count / win-count for both scenarios.
* ``logs/cooldown_analysis.txt`` — aggregate metrics, per-day
  differences, and a recommendation.

Usage::

    python -m scripts.compare_cooldown
    python -m scripts.compare_cooldown --days 60
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

log = get_logger("scripts.compare_cooldown")

# Same knobs the freeze_model + live tick use.
VARIANT_18: Dict[str, float] = {
    "prob_threshold_up": 0.55,
    "prob_threshold_down": 0.60,
    "take_profit_pct": 0.60,
    "stop_loss_pct": 0.30,
    "min_minutes_since_open": 30,
    "max_minutes_since_open": 300,
    "vol_regime_min": 0.15,
    "vol_regime_max": 0.90,
    "max_trades_per_day": 3,
    # Trailing SL is a property of the backtester itself now — no
    # override needed here.  Cooldown thresholds are the only lever.
}


def _run_one_day(
    df_all: pd.DataFrame,
    target: date,
    cooldown_on: bool,
    up_model, down_model, feat_cols: List[str],
) -> Optional[pd.DataFrame]:
    """Return the trades DataFrame for one day, or None if no data."""
    day = df_all[df_all["datetime"].dt.date == target].copy()
    day = day[day["label"].notna()]
    if day.empty:
        return None

    X = day[feat_cols].astype(np.float32).reset_index(drop=True)
    probs = bin_mod.predict_3class_from_binary(up_model, down_model, X)

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

    overrides = dict(VARIANT_18)
    if not cooldown_on:
        # Both < 0 → always False → cooldown never fires.
        overrides["cooldown_prob_delta"] = 0.0
        overrides["cooldown_price_pct"] = 0.0

    tag = "on" if cooldown_on else "off"
    name = f"cmp_{target.strftime('%Y%m%d')}_{tag}"
    result = run_backtest(probs, splits, overrides=overrides, name=name)
    return result["trades"]


def _day_metrics(trades: pd.DataFrame) -> Dict[str, float]:
    if trades is None or trades.empty:
        return {"trades": 0, "wins": 0, "pnl": 0.0}
    pnl = trades["pnl"].astype(float)
    return {
        "trades": int(len(trades)),
        "wins": int((pnl > 0).sum()),
        "pnl": float(pnl.sum()),
    }


def _pick_trading_days(df: pd.DataFrame, n_wanted: int) -> List[date]:
    """Pick the ``n_wanted`` most recent trading days that have any
    labeled rows in the parquet, in ascending order for the report."""
    labelled = df[df["label"].notna()]
    unique = sorted(labelled["datetime"].dt.date.unique(), reverse=True)
    if len(unique) < n_wanted:
        log.warning("Only %d labelled trading days available; asked for %d",
                    len(unique), n_wanted)
    picked = unique[:n_wanted]
    return sorted(picked)


def _aggregate_metrics(rows: List[Dict], key_pnl: str, key_trades: str,
                       key_wins: str, capital: float) -> Dict[str, float]:
    daily = pd.Series([r[key_pnl] for r in rows], dtype=float)
    trades = int(sum(r[key_trades] for r in rows))
    wins = int(sum(r[key_wins] for r in rows))
    total_pnl = float(daily.sum())
    avg_daily = float(daily.mean()) if len(daily) else 0.0
    # Sharpe on daily returns, annualised (~252 trading days).
    daily_ret = daily / capital
    sharpe = (
        float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
        if daily_ret.std() and not np.isnan(daily_ret.std()) else 0.0
    )
    # Max drawdown on cumulative equity curve.
    cum = daily.cumsum() + capital
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    win_rate = (wins / trades) if trades else 0.0
    profitable_days = int((daily > 0).sum())
    profitable_pct = profitable_days / len(daily) if len(daily) else 0.0
    return {
        "total_pnl": total_pnl,
        "avg_daily_pnl": avg_daily,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "total_trades": trades,
        "wins": wins,
        "win_rate": win_rate,
        "avg_pnl_per_trade": (total_pnl / trades) if trades else 0.0,
        "profitable_days": profitable_days,
        "profitable_days_pct": profitable_pct,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Cooldown ON vs OFF comparison")
    ap.add_argument("--days", type=int, default=30,
                    help="Number of most-recent labelled trading days to compare.")
    args = ap.parse_args()

    cfg = get_config()
    capital = float(cfg.project["base_capital"])

    parquet = cfg.paths["parquet_path"]
    if not parquet.exists():
        log.error("Training parquet missing at %s", parquet)
        return 2

    log.info("Loading parquet + frozen models …")
    df_all = pd.read_parquet(parquet)
    df_all["datetime"] = pd.to_datetime(df_all["datetime"])
    up_path = cfg.paths["models_dir"] / "current_up.pkl"
    dn_path = cfg.paths["models_dir"] / "current_down.pkl"
    if not up_path.exists() or not dn_path.exists():
        log.error("Frozen model files missing (%s / %s). Run scripts.freeze_model.",
                  up_path, dn_path)
        return 3
    up_model = joblib.load(up_path)
    down_model = joblib.load(dn_path)
    feat_cols = feature_columns(df_all)

    days = _pick_trading_days(df_all, args.days)
    log.info("Comparing %d trading days: %s → %s",
             len(days), days[0], days[-1])

    # Quiet the backtester's per-run INFO chatter — 60 runs otherwise
    # bury the actual summary in a wall of noise.
    logging.getLogger("aivora.backtest.backtester").setLevel(logging.WARNING)

    rows: List[Dict] = []
    for d in days:
        t_off = _run_one_day(df_all, d, False, up_model, down_model, feat_cols)
        t_on = _run_one_day(df_all, d, True, up_model, down_model, feat_cols)
        moff = _day_metrics(t_off)
        mon = _day_metrics(t_on)
        blocked = max(0, moff["trades"] - mon["trades"])
        blocked_pnl = 0.0
        if t_off is not None and t_on is not None and not t_off.empty:
            # "Blocked" = trades in OFF but not in ON (matched on
            # (symbol, entry-time)).  Cooldown *delays* rather than
            # strictly blocks, so many "blocked" rows have a
            # near-duplicate at a later time in ON — that's fine;
            # we're measuring the trades whose entry moment was
            # exclusively released by the OFF scenario.
            if t_on.empty:
                blocked_only = t_off
            else:
                key_off = set(zip(t_off["symbol"], t_off["datetime"].astype(str)))
                key_on = set(zip(t_on["symbol"], t_on["datetime"].astype(str)))
                blocked_keys = key_off - key_on
                blocked_only = t_off[
                    t_off.apply(lambda r: (r["symbol"], str(r["datetime"])) in blocked_keys, axis=1)
                ]
            blocked_pnl = float(blocked_only["pnl"].sum()) if not blocked_only.empty else 0.0
            blocked_wins = int((blocked_only["pnl"] > 0).sum()) if not blocked_only.empty else 0
            blocked_losses = int((blocked_only["pnl"] < 0).sum()) if not blocked_only.empty else 0
        else:
            blocked_wins = blocked_losses = 0
        rows.append({
            "date": d.isoformat(),
            "pnl_off": moff["pnl"],
            "pnl_on": mon["pnl"],
            "pnl_diff": mon["pnl"] - moff["pnl"],
            "trades_off": moff["trades"],
            "trades_on": mon["trades"],
            "wins_off": moff["wins"],
            "wins_on": mon["wins"],
            "blocked_trades": blocked,
            "blocked_pnl": blocked_pnl,
            "blocked_wins": blocked_wins,
            "blocked_losses": blocked_losses,
        })
        log.info("%s  OFF ₹%+9.2f (%d tr) | ON ₹%+9.2f (%d tr) | Δ ₹%+9.2f",
                 d, moff["pnl"], moff["trades"],
                 mon["pnl"], mon["trades"], mon["pnl"] - moff["pnl"])

    if not rows:
        log.error("No days produced trades in either scenario.  Cannot compare.")
        return 4

    df_out = pd.DataFrame(rows)
    csv_path = cfg.paths["reports_dir"] / "cooldown_comparison_30days.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(csv_path, index=False)
    log.info("Wrote per-day comparison → %s", csv_path)

    agg_off = _aggregate_metrics(rows, "pnl_off", "trades_off", "wins_off", capital)
    agg_on = _aggregate_metrics(rows, "pnl_on", "trades_on", "wins_on", capital)

    # ---- Per-day post-mortem ----
    blocked_summary = {
        "days_with_blocks": int((df_out["blocked_trades"] > 0).sum()),
        "total_blocked_trades": int(df_out["blocked_trades"].sum()),
        "blocked_wins": int(df_out["blocked_wins"].sum()),
        "blocked_losses": int(df_out["blocked_losses"].sum()),
        "blocked_pnl_total": float(df_out["blocked_pnl"].sum()),
    }

    # ---- Recommendation logic ----
    # Prefer higher Sharpe + lower drawdown; break ties on total P&L.
    def _pick_recommendation() -> str:
        pnl_gap = agg_on["total_pnl"] - agg_off["total_pnl"]
        sharpe_gap = agg_on["sharpe"] - agg_off["sharpe"]
        dd_gap = agg_on["max_drawdown"] - agg_off["max_drawdown"]

        # Persistent regression: cooldown loses money on both risk +
        # return dimensions.
        if pnl_gap < 0 and sharpe_gap < 0 and dd_gap < 0:
            return ("REMOVE cooldown — total P&L, Sharpe, AND drawdown all "
                    "worsen with cooldown ON. The rule blocks more winning "
                    "trades than it saves losing ones over this window.")
        # Clean win.
        if pnl_gap > 0 and sharpe_gap >= 0 and dd_gap >= 0:
            return ("KEEP cooldown — total P&L, Sharpe, and drawdown all "
                    "improve. Ship as-is with the current 0.05 / 0.1 % thresholds.")
        # Mixed — better risk-adjusted returns despite lower total P&L.
        if sharpe_gap > 0 and dd_gap > 0 and pnl_gap < 0:
            return ("KEEP cooldown — total P&L is lower but risk-adjusted "
                    "returns (Sharpe) and max drawdown both improve. That "
                    "trade-off is the right one for a leverage strategy.")
        # Mixed the other way — P&L wins but risk is worse.
        if pnl_gap > 0 and (sharpe_gap < 0 or dd_gap < 0):
            return ("KEEP cooldown with looser thresholds — total P&L helps "
                    "but risk metrics slip. Try prob_delta=0.03, price_pct=0.002 "
                    "to catch fewer edge cases while still guarding against "
                    "duplicated entries.")
        return ("TUNE cooldown — mixed signals. Try prob_delta=0.03 and "
                "price_pct=0.002 first, then re-run this comparison.")

    recommendation = _pick_recommendation()

    # ---- Write analysis report ----
    logs_dir = cfg.paths["logs_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)
    report_path = logs_dir / "cooldown_analysis.txt"

    def _row(label: str, off, on, unit: str = "") -> str:
        if isinstance(off, float) and isinstance(on, float):
            diff = on - off
            return f"  {label:<24s}  {off:>12,.2f}{unit:<3s}  {on:>12,.2f}{unit:<3s}  {diff:>+12,.2f}{unit}"
        return f"  {label:<24s}  {off:>12}  {on:>12}  {on - off:>+12}"

    text = [
        "=" * 72,
        f"AiVora — cooldown ON vs OFF   ({len(rows)} trading days, "
        f"{days[0]} → {days[-1]})",
        f"Generated : {datetime.now().isoformat(timespec='seconds')}",
        "=" * 72,
        "",
        "Aggregate metrics",
        "-" * 72,
        f"  {'Metric':<24s}  {'Cooldown OFF':>12s}     {'Cooldown ON':>12s}         Δ (ON − OFF)",
        _row("Total P&L (₹)",           agg_off['total_pnl'], agg_on['total_pnl']),
        _row("Avg daily P&L (₹)",       agg_off['avg_daily_pnl'], agg_on['avg_daily_pnl']),
        _row("Sharpe (annualised)",     agg_off['sharpe'], agg_on['sharpe']),
        _row("Max drawdown",            agg_off['max_drawdown'], agg_on['max_drawdown']),
        _row("Total trades",            agg_off['total_trades'], agg_on['total_trades']),
        _row("Win rate",                agg_off['win_rate'], agg_on['win_rate']),
        _row("Avg P&L per trade (₹)",   agg_off['avg_pnl_per_trade'], agg_on['avg_pnl_per_trade']),
        _row("Profitable days",         agg_off['profitable_days'], agg_on['profitable_days']),
        _row("Profitable days %",       agg_off['profitable_days_pct'], agg_on['profitable_days_pct']),
        "",
        "Blocked-trade analysis (trades present in OFF but not in ON)",
        "-" * 72,
        f"  Days on which cooldown blocked ≥1 entry : {blocked_summary['days_with_blocks']}",
        f"  Total trades blocked                    : {blocked_summary['total_blocked_trades']}",
        f"  Blocked trades that WOULD have won      : {blocked_summary['blocked_wins']}",
        f"  Blocked trades that WOULD have lost     : {blocked_summary['blocked_losses']}",
        f"  Net P&L of blocked trades (₹)           : {blocked_summary['blocked_pnl_total']:+,.2f}",
        "  (Positive → cooldown blocked net-profitable entries)",
        "  (Negative → cooldown correctly filtered net-losing entries)",
        "",
        "Recommendation",
        "-" * 72,
        f"  {recommendation}",
        "",
        "Raw per-day rows in reports/cooldown_comparison_30days.csv",
    ]
    report_path.write_text("\n".join(text), encoding="utf-8")
    log.info("Wrote analysis → %s", report_path)

    # Print recommendation on stdout so the caller sees it immediately.
    print("\n" + "=" * 72)
    print(recommendation)
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
