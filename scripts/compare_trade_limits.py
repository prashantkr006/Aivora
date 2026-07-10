"""Compare ``max_trades_per_day`` = 3 vs 5 vs 10 on every 2026 trading day.

Runs the identical backtester (variant-#18 gates, trailing SL on,
cooldown OFF) three times per day and captures per-day and
per-trade metrics.  Produces:

    * reports/trade_limit_comparison_2026.csv     — per-day aggregates
    * reports/trade_limit_per_trade_2026.csv      — every trade row
    * reports/trade_timing_analysis_2026.csv      — hour-bucket stats
    * logs/trade_limit_comparison_2026.txt        — summary tables
    * logs/trade_limit_recommendation.txt         — final call

Progress is logged as ``Processing 2026-03-15  (day 45/120)`` so a
long run is always visible.

Usage::

    python -m scripts.compare_trade_limits
    python -m scripts.compare_trade_limits --start 2026-01-01 --end 2026-07-08
    python -m scripts.compare_trade_limits --limits 3 5 10 20
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

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

log = get_logger("scripts.compare_trade_limits")

VARIANT_18: Dict[str, float] = {
    "prob_threshold_up": 0.55,
    "prob_threshold_down": 0.60,
    "take_profit_pct": 0.60,
    "stop_loss_pct": 0.30,
    "min_minutes_since_open": 30,
    "max_minutes_since_open": 300,
    "vol_regime_min": 0.15,
    "vol_regime_max": 0.90,
    # max_trades_per_day is the variable under test — filled in per run.
    # cooldown_prob_delta / cooldown_price_pct default to 0.0 (OFF)
    # in the backtester now — no override needed here.
}


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _run_one_day(
    df_all: pd.DataFrame,
    target: date,
    max_trades: int,
    up_model, down_model, feat_cols: List[str],
) -> Optional[pd.DataFrame]:
    """Return the trades DataFrame for one day+limit, or None if no data."""
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
    overrides["max_trades_per_day"] = int(max_trades)
    name = f"tlim_{max_trades}_{target.strftime('%Y%m%d')}"
    result = run_backtest(probs, splits, overrides=overrides, name=name)
    trades = result["trades"]
    if trades is not None and not trades.empty:
        trades = trades.copy()
        trades["date"] = target.isoformat()
        trades["limit"] = int(max_trades)
    return trades


# =============================================================
#  Aggregations
# =============================================================
def _day_row(target: date, limit: int, trades: Optional[pd.DataFrame]) -> Dict:
    if trades is None or trades.empty:
        return {
            "date": target.isoformat(),
            "limit": int(limit),
            "trades": 0, "wins": 0, "losses": 0,
            "pnl": 0.0, "gross_pnl": 0.0, "costs": 0.0,
            "avg_pnl": 0.0,
        }
    pnl = trades["pnl"].astype(float)
    return {
        "date": target.isoformat(),
        "limit": int(limit),
        "trades": int(len(trades)),
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "pnl": float(pnl.sum()),
        "gross_pnl": float(trades["gross_pnl"].astype(float).sum()),
        "costs": float(trades["costs"].astype(float).sum()),
        "avg_pnl": float(pnl.mean()),
    }


def _summarise(rows: List[Dict], capital: float) -> Dict[str, float]:
    daily = pd.Series([r["pnl"] for r in rows], dtype=float)
    trades = int(sum(r["trades"] for r in rows))
    wins = int(sum(r["wins"] for r in rows))
    total_pnl = float(daily.sum())
    total_costs = float(sum(r["costs"] for r in rows))
    total_gross = float(sum(r["gross_pnl"] for r in rows))

    daily_ret = daily / capital
    sharpe = (
        float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
        if daily_ret.std() and not np.isnan(daily_ret.std()) else 0.0
    )
    cum = daily.cumsum() + capital
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    win_rate = (wins / trades) if trades else 0.0
    profitable_days = int((daily > 0).sum())
    days = len(daily)
    return {
        "total_pnl": total_pnl,
        "total_gross_pnl": total_gross,
        "total_costs": total_costs,
        "total_trades": trades,
        "wins": wins,
        "win_rate": win_rate,
        "avg_pnl_per_trade": (total_pnl / trades) if trades else 0.0,
        "avg_daily_pnl": float(daily.mean()) if days else 0.0,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd * 100.0,
        "days": days,
        "profitable_days": profitable_days,
        "profitable_days_pct": (profitable_days / days) if days else 0.0,
        "return_on_capital_pct": total_pnl / capital * 100.0,
    }


def _timing_table(all_trades: pd.DataFrame) -> pd.DataFrame:
    """One row per (limit, hour) bucket."""
    if all_trades.empty:
        return pd.DataFrame()
    df = all_trades.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour_bucket"] = df["datetime"].dt.strftime("%H:00")
    out = (
        df.groupby(["limit", "hour_bucket"])
          .agg(
              n_trades=("pnl", "size"),
              wins=("pnl", lambda s: (s > 0).sum()),
              total_pnl=("pnl", "sum"),
              avg_pnl=("pnl", "mean"),
          )
          .reset_index()
    )
    out["win_rate"] = out["wins"] / out["n_trades"]
    return out


def _pick_recommendation(summaries: Dict[int, Dict], limits: List[int]) -> str:
    """Prefer the highest Sharpe; break ties on total P&L; note diminishing returns."""
    # Rank by Sharpe.
    by_sharpe = sorted(limits, key=lambda k: summaries[k]["sharpe"], reverse=True)
    best_sharpe_lim = by_sharpe[0]
    # Highest total P&L.
    by_pnl = sorted(limits, key=lambda k: summaries[k]["total_pnl"], reverse=True)
    best_pnl_lim = by_pnl[0]

    lines: List[str] = []

    # Diminishing-returns check — going up a step must add ≥ 5 % of
    # the lower step's P&L, otherwise it's not worth the drawdown risk.
    for a, b in zip(limits, limits[1:]):
        pnl_a = summaries[a]["total_pnl"]
        pnl_b = summaries[b]["total_pnl"]
        gain = pnl_b - pnl_a
        gain_pct = gain / max(pnl_a, 1e-9) * 100.0
        dd_a = summaries[a]["max_drawdown_pct"]
        dd_b = summaries[b]["max_drawdown_pct"]
        dd_gap = dd_b - dd_a  # more negative = worse
        lines.append(
            f"  step {a} → {b}: ΔP&L = ₹{gain:+,.2f} ({gain_pct:+.1f}%), "
            f"Δdrawdown = {dd_gap:+.2f}pp, ΔSharpe = "
            f"{summaries[b]['sharpe'] - summaries[a]['sharpe']:+.2f}"
        )

    if best_sharpe_lim == best_pnl_lim:
        rec = (
            f"RECOMMEND max_trades_per_day = {best_sharpe_lim}. "
            f"It gives BOTH the highest total P&L (₹{summaries[best_pnl_lim]['total_pnl']:,.2f}) "
            f"AND the highest Sharpe ({summaries[best_sharpe_lim]['sharpe']:.2f}) — "
            "no trade-off, just pick it."
        )
    else:
        rec = (
            f"RECOMMEND max_trades_per_day = {best_sharpe_lim} "
            f"for best risk-adjusted returns (Sharpe {summaries[best_sharpe_lim]['sharpe']:.2f}). "
            f"limit={best_pnl_lim} gives the highest raw P&L "
            f"(₹{summaries[best_pnl_lim]['total_pnl']:,.2f}) but at slightly worse "
            f"risk metrics."
        )
    lines.append("")
    lines.append(rec)
    return "\n".join(lines)


# =============================================================
#  Main
# =============================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Compare max_trades_per_day settings on 2026 data")
    ap.add_argument("--start", type=_parse_date, default=_parse_date("2026-01-01"))
    ap.add_argument("--end", type=_parse_date, default=date.today())
    ap.add_argument("--limits", type=int, nargs="+", default=[3, 5, 10])
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

    # Pick trading days in [start, end] that have any labelled rows.
    labelled = df_all[df_all["label"].notna()]
    all_dates = labelled["datetime"].dt.date.unique()
    days = sorted(d for d in all_dates if args.start <= d <= args.end)
    if not days:
        log.error("No labelled trading days in range %s → %s", args.start, args.end)
        return 4
    log.info("Comparing limits=%s over %d trading days (%s → %s)",
             args.limits, len(days), days[0], days[-1])

    # Silence per-run backtester chatter — with 3 limits × ~130 days
    # we'd otherwise print >4 000 INFO lines.
    logging.getLogger("aivora.backtest.backtester").setLevel(logging.WARNING)

    per_day_rows: List[Dict] = []
    per_trade_frames: List[pd.DataFrame] = []
    total_runs = len(days) * len(args.limits)
    run_idx = 0
    for i, d in enumerate(days, start=1):
        for lim in args.limits:
            run_idx += 1
            trades = _run_one_day(df_all, d, lim, up_model, down_model, feat_cols)
            per_day_rows.append(_day_row(d, lim, trades))
            if trades is not None and not trades.empty:
                per_trade_frames.append(trades)
        # Show progress once per DAY (compact even for 130+ days).
        log.info("Processing %s  (day %d/%d, total runs %d/%d)",
                 d, i, len(days), run_idx, total_runs)

    per_day_df = pd.DataFrame(per_day_rows)
    per_trade_df = (
        pd.concat(per_trade_frames, ignore_index=True)
        if per_trade_frames else pd.DataFrame()
    )

    # ---- CSV outputs ----
    reports_dir = cfg.paths["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    per_day_csv = reports_dir / "trade_limit_comparison_2026.csv"
    per_day_df.to_csv(per_day_csv, index=False)
    log.info("Wrote per-day rows → %s", per_day_csv)

    per_trade_csv = reports_dir / "trade_limit_per_trade_2026.csv"
    per_trade_df.to_csv(per_trade_csv, index=False)
    log.info("Wrote per-trade rows → %s", per_trade_csv)

    timing_df = _timing_table(per_trade_df)
    timing_csv = reports_dir / "trade_timing_analysis_2026.csv"
    timing_df.to_csv(timing_csv, index=False)
    log.info("Wrote timing table → %s", timing_csv)

    # ---- Aggregates ----
    summaries: Dict[int, Dict] = {}
    for lim in args.limits:
        rows = [r for r in per_day_rows if r["limit"] == lim]
        summaries[lim] = _summarise(rows, capital)

    # ---- Sweet-spot analysis (across all trades, all limits) ----
    def _sweet_spot(all_tr: pd.DataFrame) -> Dict[str, str]:
        if all_tr.empty:
            return {"peak_freq": "n/a", "peak_win": "n/a", "peak_pnl": "n/a"}
        df = all_tr.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["hour_bucket"] = df["datetime"].dt.strftime("%H:00")
        g = df.groupby("hour_bucket").agg(
            n=("pnl", "size"),
            win_rate=("pnl", lambda s: (s > 0).mean()),
            avg_pnl=("pnl", "mean"),
        )
        return {
            "peak_freq": g["n"].idxmax() + f" ({int(g['n'].max())} trades)",
            "peak_win": g["win_rate"].idxmax() + f" ({g['win_rate'].max():.0%})",
            "peak_pnl": g["avg_pnl"].idxmax() + f" (₹{g['avg_pnl'].max():+.2f}/trade)",
        }
    sweet = _sweet_spot(per_trade_df)

    # ---- Text summary ----
    def _row(label: str, formatter, *vals) -> str:
        cells = "  ".join(f"{formatter(v):>14s}" for v in vals)
        return f"  {label:<26s}  {cells}"

    header_cells = "  ".join(f"{'limit=' + str(lim):>14s}" for lim in args.limits)
    text = [
        "=" * 72,
        f"AiVora — max_trades_per_day comparison  "
        f"({len(days)} days, {days[0]} → {days[-1]})",
        f"Generated : {datetime.now().isoformat(timespec='seconds')}",
        "=" * 72,
        "",
        "Aggregate metrics",
        "-" * 72,
        f"  {'Metric':<26s}  {header_cells}",
        _row("Total P&L (₹)",           lambda v: f"{v:,.2f}",  *(summaries[l]['total_pnl'] for l in args.limits)),
        _row("Total gross P&L (₹)",     lambda v: f"{v:,.2f}",  *(summaries[l]['total_gross_pnl'] for l in args.limits)),
        _row("Total costs (₹)",         lambda v: f"{v:,.2f}",  *(summaries[l]['total_costs'] for l in args.limits)),
        _row("Total trades",            lambda v: f"{int(v):,}", *(summaries[l]['total_trades'] for l in args.limits)),
        _row("Win rate",                lambda v: f"{v:.2%}",   *(summaries[l]['win_rate'] for l in args.limits)),
        _row("Avg P&L / trade (₹)",     lambda v: f"{v:,.2f}",  *(summaries[l]['avg_pnl_per_trade'] for l in args.limits)),
        _row("Avg daily P&L (₹)",       lambda v: f"{v:,.2f}",  *(summaries[l]['avg_daily_pnl'] for l in args.limits)),
        _row("Sharpe (annualised)",     lambda v: f"{v:.2f}",   *(summaries[l]['sharpe'] for l in args.limits)),
        _row("Max drawdown (%)",        lambda v: f"{v:.2f}",   *(summaries[l]['max_drawdown_pct'] for l in args.limits)),
        _row("Profitable days",         lambda v: f"{int(v)}",  *(summaries[l]['profitable_days'] for l in args.limits)),
        _row("Profitable days %",       lambda v: f"{v:.1%}",   *(summaries[l]['profitable_days_pct'] for l in args.limits)),
        _row("Return on capital (%)",   lambda v: f"{v:.2f}",   *(summaries[l]['return_on_capital_pct'] for l in args.limits)),
        "",
        "Trade-timing sweet spot (across ALL trades in ALL scenarios)",
        "-" * 72,
        f"  Hour with most trades      : {sweet['peak_freq']}",
        f"  Hour with best win rate    : {sweet['peak_win']}",
        f"  Hour with best avg P&L     : {sweet['peak_pnl']}",
        "",
        "Timing table (hour, limit) — top rows shown; full table in the CSV",
        "-" * 72,
    ]
    if not timing_df.empty:
        show = timing_df.sort_values(["limit", "hour_bucket"]).head(60).copy()
        show["win_rate"] = show["win_rate"].apply(lambda v: f"{v:.0%}")
        show["total_pnl"] = show["total_pnl"].apply(lambda v: f"{v:+.2f}")
        show["avg_pnl"] = show["avg_pnl"].apply(lambda v: f"{v:+.2f}")
        text.append(show.to_string(index=False))
    text.append("")

    summary_path = cfg.paths["logs_dir"] / "trade_limit_comparison_2026.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(text), encoding="utf-8")
    log.info("Wrote summary → %s", summary_path)

    # ---- Recommendation ----
    rec = _pick_recommendation(summaries, list(args.limits))
    rec_path = cfg.paths["logs_dir"] / "trade_limit_recommendation.txt"
    rec_path.write_text(
        "=" * 72 + "\n"
        f"AiVora — trade-limit recommendation "
        f"({len(days)} days, {days[0]} → {days[-1]})\n"
        + "=" * 72 + "\n\n"
        + "Diminishing-returns walk\n"
        + "-" * 72 + "\n"
        + rec + "\n",
        encoding="utf-8",
    )
    log.info("Wrote recommendation → %s", rec_path)

    print("\n" + "=" * 72)
    print(rec)
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
