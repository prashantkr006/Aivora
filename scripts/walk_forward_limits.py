"""Honest walk-forward comparison of ``max_trades_per_day`` = 3 / 5 / 10.

The earlier 2026 comparison in ``compare_trade_limits.py`` was
trained on the full 13-month dataset that already contained every
test month → data leakage → inflated numbers.  This script fixes
that by retraining the binary UP + DOWN model pair once per test
month, using ONLY data prior to that month's first day.

Fold layout for each test month M::

    ├───── 11 training months ────┼── 1 val ──┼── 1 test ──┤
    (M-12 .. M-2)                  (M-1)        (M)

Total look-back = 12 months, of which the last month is used as
the internal early-stopping validation set — standard time-series
practice.  No row from month M is ever seen during training.

Outputs:

    * reports/walk_forward_2026_comparison.csv    — per (fold, limit)
    * logs/walk_forward_2026_summary.txt          — aggregate table
    * logs/walk_forward_2026_recommendation.txt   — final call

Usage::

    python -m scripts.walk_forward_limits
    python -m scripts.walk_forward_limits --limits 3 5 10 20
    python -m scripts.walk_forward_limits --test-months 2026-01 2026-02 2026-03
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aivora.backtest.backtester import run_backtest  # noqa: E402
from aivora.ml import binary as bin_mod  # noqa: E402
from aivora.ml.dataset import Splits  # noqa: E402
from aivora.pipeline.feature_engineering import feature_columns  # noqa: E402
from aivora.utils.config import get_config  # noqa: E402
from aivora.utils.logger import get_logger  # noqa: E402

log = get_logger("scripts.walk_forward_limits")

VARIANT_18: Dict[str, float] = {
    "prob_threshold_up": 0.55,
    "prob_threshold_down": 0.60,
    "take_profit_pct": 0.60,
    "stop_loss_pct": 0.30,
    "min_minutes_since_open": 30,
    "max_minutes_since_open": 300,
    "vol_regime_min": 0.15,
    "vol_regime_max": 0.90,
    # cooldown defaults to 0.0 (OFF) in the backtester itself.
    # max_trades_per_day is the variable under test.
}


# =============================================================
#  Fold + training
# =============================================================
def _month_bounds(month: pd.Period) -> tuple:
    """Return (first_day_of_month, first_day_of_next_month)."""
    start = month.to_timestamp()
    end = (month + 1).to_timestamp()
    return start, end


def _build_splits(df: pd.DataFrame, feat_cols: List[str],
                  train_range: tuple, val_range: tuple, test_range: tuple) -> Splits:
    ts = pd.to_datetime(df["datetime"])
    tr_mask = (ts >= train_range[0]) & (ts < train_range[1])
    va_mask = (ts >= val_range[0]) & (ts < val_range[1])
    te_mask = (ts >= test_range[0]) & (ts < test_range[1])

    df_tr = df.loc[tr_mask]
    df_va = df.loc[va_mask]
    df_te = df.loc[te_mask].reset_index(drop=True)

    # Training / val need labels; test does NOT (backtester walks
    # forward per row using spot_close, not label).
    df_tr = df_tr[df_tr["label"].notna()]
    df_va = df_va[df_va["label"].notna()]

    if len(df_tr) < 500 or len(df_va) < 20:
        raise ValueError(
            f"Insufficient rows: train={len(df_tr)}, val={len(df_va)}"
        )

    keep_cols = ["datetime", "symbol", "spot_close", "fwd_return",
                 "ce_ltp", "pe_ltp", "minutes_since_open", "vol_regime_pct"]
    meta_test = pd.DataFrame(index=df_te.index)
    for c in keep_cols:
        meta_test[c] = df_te[c] if c in df_te.columns else np.nan

    return Splits(
        X_train=df_tr[feat_cols].astype(np.float32),
        y_train=df_tr["label"].astype(int),
        X_val=df_va[feat_cols].astype(np.float32),
        y_val=df_va["label"].astype(int),
        X_test=df_te[feat_cols].astype(np.float32),
        y_test=pd.Series(dtype=int),
        feature_cols=feat_cols,
        meta_test=meta_test.reset_index(drop=True),
    )


def _run_one_fold(
    df: pd.DataFrame, feat_cols: List[str], test_month: pd.Period,
    limits: List[int],
) -> Optional[List[Dict]]:
    """Retrain the binary pair for one test month, run backtest per limit.

    Returns one row per limit with pnl / trades / wins / costs.
    """
    # 12-month look-back: 11 train + 1 val.  The last month of the
    # look-back is validation; training data ends 1 month BEFORE
    # the test month.
    train_start = (test_month - 12).to_timestamp()
    val_start = (test_month - 1).to_timestamp()
    test_start, test_end = _month_bounds(test_month)

    try:
        splits = _build_splits(
            df, feat_cols,
            train_range=(train_start, val_start),
            val_range=(val_start, test_start),
            test_range=(test_start, test_end),
        )
    except ValueError as exc:
        log.warning("Skipping %s: %s", test_month, exc)
        return None

    if splits.X_test.empty:
        log.warning("Skipping %s: no test rows", test_month)
        return None

    # Fit UP + DOWN binary pair on train + early-stop on val.
    log.info(
        "%s — train %d rows (%s → %s), val %d rows (%s → %s), test %d rows",
        test_month,
        len(splits.X_train), train_start.date(), val_start.date(),
        len(splits.X_val), val_start.date(), test_start.date(),
        len(splits.X_test),
    )
    up_model, down_model = bin_mod.train_binary_pair(splits)

    # One prediction pass — reused across every limit setting.
    probs = bin_mod.predict_3class_from_binary(up_model, down_model, splits.X_test)

    rows: List[Dict] = []
    for lim in limits:
        overrides = dict(VARIANT_18)
        overrides["max_trades_per_day"] = int(lim)
        name = f"wf_{test_month}_{lim}"
        result = run_backtest(probs, splits, overrides=overrides, name=name)
        trades = result["trades"]
        if trades is None or trades.empty:
            rows.append({
                "test_month": str(test_month),
                "limit": int(lim),
                "trades": 0, "wins": 0, "losses": 0,
                "pnl": 0.0, "gross_pnl": 0.0, "costs": 0.0,
            })
            continue
        pnl = trades["pnl"].astype(float)
        rows.append({
            "test_month": str(test_month),
            "limit": int(lim),
            "trades": int(len(trades)),
            "wins": int((pnl > 0).sum()),
            "losses": int((pnl < 0).sum()),
            "pnl": float(pnl.sum()),
            "gross_pnl": float(trades["gross_pnl"].astype(float).sum()),
            "costs": float(trades["costs"].astype(float).sum()),
        })
    return rows


# =============================================================
#  Aggregation
# =============================================================
def _summarise(fold_rows: List[Dict], capital: float) -> Dict[str, float]:
    if not fold_rows:
        return {
            "total_pnl": 0.0, "total_gross_pnl": 0.0, "total_costs": 0.0,
            "total_trades": 0, "wins": 0, "win_rate": 0.0,
            "avg_pnl_per_trade": 0.0, "sharpe": 0.0, "max_drawdown_pct": 0.0,
            "months": 0, "profitable_months": 0,
            "avg_monthly_return_pct": 0.0, "months_positive_pct": 0.0,
        }
    monthly = pd.Series([r["pnl"] for r in fold_rows], dtype=float)
    trades = int(sum(r["trades"] for r in fold_rows))
    wins = int(sum(r["wins"] for r in fold_rows))
    total_pnl = float(monthly.sum())
    total_costs = float(sum(r["costs"] for r in fold_rows))
    total_gross = float(sum(r["gross_pnl"] for r in fold_rows))

    # Annualised Sharpe from monthly returns: (mean/std) × sqrt(12).
    monthly_ret = monthly / capital
    sharpe = (
        float(monthly_ret.mean() / monthly_ret.std() * np.sqrt(12))
        if monthly_ret.std() and not np.isnan(monthly_ret.std()) else 0.0
    )
    cum = monthly.cumsum() + capital
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    win_rate = (wins / trades) if trades else 0.0
    profitable_months = int((monthly > 0).sum())
    n_months = len(monthly)
    return {
        "total_pnl": total_pnl,
        "total_gross_pnl": total_gross,
        "total_costs": total_costs,
        "total_trades": trades,
        "wins": wins,
        "win_rate": win_rate,
        "avg_pnl_per_trade": (total_pnl / trades) if trades else 0.0,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd * 100.0,
        "months": n_months,
        "profitable_months": profitable_months,
        "avg_monthly_return_pct": float(monthly_ret.mean()) * 100.0,
        "months_positive_pct": (profitable_months / n_months) if n_months else 0.0,
    }


def _pick_recommendation(agg: Dict[int, Dict], limits: List[int]) -> str:
    by_sharpe = sorted(limits, key=lambda k: agg[k]["sharpe"], reverse=True)
    by_pnl = sorted(limits, key=lambda k: agg[k]["total_pnl"], reverse=True)
    best_sharpe = by_sharpe[0]
    best_pnl = by_pnl[0]

    steps: List[str] = []
    for a, b in zip(limits, limits[1:]):
        pnl_gap = agg[b]["total_pnl"] - agg[a]["total_pnl"]
        pnl_gain_pct = pnl_gap / max(abs(agg[a]["total_pnl"]), 1e-9) * 100.0
        dd_gap = agg[b]["max_drawdown_pct"] - agg[a]["max_drawdown_pct"]
        sharpe_gap = agg[b]["sharpe"] - agg[a]["sharpe"]
        steps.append(
            f"  step {a} → {b}: ΔP&L = ₹{pnl_gap:+,.2f} ({pnl_gain_pct:+.1f}%), "
            f"Δdrawdown = {dd_gap:+.2f}pp, ΔSharpe = {sharpe_gap:+.2f}"
        )

    if best_sharpe == best_pnl:
        verdict = (
            f"RECOMMEND max_trades_per_day = {best_sharpe} — highest total P&L "
            f"(₹{agg[best_pnl]['total_pnl']:,.2f}) AND highest Sharpe "
            f"({agg[best_sharpe]['sharpe']:.2f}) in walk-forward out-of-sample."
        )
    else:
        verdict = (
            f"RECOMMEND max_trades_per_day = {best_sharpe} for best risk-adjusted "
            f"returns (Sharpe {agg[best_sharpe]['sharpe']:.2f}). limit={best_pnl} "
            f"gives the highest raw P&L (₹{agg[best_pnl]['total_pnl']:,.2f}) but "
            f"at worse Sharpe/drawdown."
        )
    return "\n".join(steps) + "\n\n" + verdict


# =============================================================
#  Main
# =============================================================
def _default_test_months() -> List[pd.Period]:
    months = pd.period_range("2022-01", "2026-07", freq="M")
    return list(months)


def main() -> int:
    ap = argparse.ArgumentParser(description="Walk-forward limit comparison")
    ap.add_argument("--limits", type=int, nargs="+", default=[3, 5, 10])
    ap.add_argument(
        "--test-months", nargs="+",
        default=[str(p) for p in _default_test_months()],
        help="Test months as YYYY-MM (default = 2026-01 through 2026-07)",
    )
    args = ap.parse_args()

    test_months = [pd.Period(m, freq="M") for m in args.test_months]

    cfg = get_config()
    capital = float(cfg.project["base_capital"])

    parquet = cfg.paths["parquet_path"]
    if not parquet.exists():
        log.error("Training parquet missing at %s", parquet)
        return 2

    log.info("Loading parquet …")
    df_all = pd.read_parquet(parquet)
    df_all["datetime"] = pd.to_datetime(df_all["datetime"])
    feat_cols = feature_columns(df_all)

    # Silence backtester per-run INFO lines — with 7 folds × 3 limits
    # they would drown the summary.
    logging.getLogger("aivora.backtest.backtester").setLevel(logging.WARNING)

    per_fold_rows: List[Dict] = []
    for i, month in enumerate(test_months, start=1):
        log.info("=== Fold %d/%d — test month %s ===", i, len(test_months), month)
        rows = _run_one_fold(df_all, feat_cols, month, args.limits)
        if rows:
            per_fold_rows.extend(rows)

    if not per_fold_rows:
        log.error("No fold produced results.")
        return 3

    df_out = pd.DataFrame(per_fold_rows)
    reports_dir = cfg.paths["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "walk_forward_2026_comparison.csv"
    df_out.to_csv(csv_path, index=False)
    log.info("Wrote per-fold rows → %s", csv_path)

    # Aggregate per limit.
    agg: Dict[int, Dict] = {}
    for lim in args.limits:
        rows = [r for r in per_fold_rows if r["limit"] == lim]
        agg[lim] = _summarise(rows, capital)

    # ---- Text summary ----
    def _row(label: str, formatter, *vals) -> str:
        cells = "  ".join(f"{formatter(v):>14s}" for v in vals)
        return f"  {label:<26s}  {cells}"

    header_cells = "  ".join(f"{'limit=' + str(l):>14s}" for l in args.limits)
    tested = sorted({r["test_month"] for r in per_fold_rows})
    lines = [
        "=" * 72,
        f"AiVora — walk-forward max_trades_per_day comparison  "
        f"({len(tested)} test months: {tested[0]} → {tested[-1]})",
        f"Generated : {datetime.now().isoformat(timespec='seconds')}",
        "Method    : retrain binary UP+DOWN pair per test month, "
        "12-month look-back (11 train + 1 val), NO row from test month in training.",
        "=" * 72,
        "",
        "Aggregate out-of-sample metrics",
        "-" * 72,
        f"  {'Metric':<26s}  {header_cells}",
        _row("Total P&L (₹)",           lambda v: f"{v:,.2f}",  *(agg[l]['total_pnl'] for l in args.limits)),
        _row("Total gross P&L (₹)",     lambda v: f"{v:,.2f}",  *(agg[l]['total_gross_pnl'] for l in args.limits)),
        _row("Total costs (₹)",         lambda v: f"{v:,.2f}",  *(agg[l]['total_costs'] for l in args.limits)),
        _row("Total trades",            lambda v: f"{int(v):,}", *(agg[l]['total_trades'] for l in args.limits)),
        _row("Win rate",                lambda v: f"{v:.2%}",   *(agg[l]['win_rate'] for l in args.limits)),
        _row("Avg P&L / trade (₹)",     lambda v: f"{v:,.2f}",  *(agg[l]['avg_pnl_per_trade'] for l in args.limits)),
        _row("Sharpe (annualised)",     lambda v: f"{v:.2f}",   *(agg[l]['sharpe'] for l in args.limits)),
        _row("Max drawdown (%)",        lambda v: f"{v:.2f}",   *(agg[l]['max_drawdown_pct'] for l in args.limits)),
        _row("Profitable months",       lambda v: f"{int(v)}",  *(agg[l]['profitable_months'] for l in args.limits)),
        _row("Profitable months %",     lambda v: f"{v:.1%}",   *(agg[l]['months_positive_pct'] for l in args.limits)),
        _row("Avg monthly return (%)",  lambda v: f"{v:.2f}",   *(agg[l]['avg_monthly_return_pct'] for l in args.limits)),
        "",
        "Per-fold P&L (rupees)",
        "-" * 72,
    ]
    fold_matrix = df_out.pivot_table(
        index="test_month", columns="limit", values="pnl", aggfunc="sum",
    ).round(2)
    lines.append(fold_matrix.to_string())
    lines.append("")

    logs_dir = cfg.paths["logs_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = logs_dir / "walk_forward_2026_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote summary → %s", summary_path)

    rec = _pick_recommendation(agg, list(args.limits))
    rec_path = logs_dir / "walk_forward_2026_recommendation.txt"
    rec_path.write_text(
        "=" * 72 + "\n"
        f"AiVora — HONEST walk-forward recommendation "
        f"({len(tested)} test months, {tested[0]} → {tested[-1]})\n"
        + "=" * 72 + "\n"
        + "Method: for each test month M, retrain binary UP + DOWN pair on\n"
        + "        months [M-12, M-1) with month M-1 as early-stopping val.\n"
        + "        Test month M itself is NEVER in the training set.\n\n"
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
