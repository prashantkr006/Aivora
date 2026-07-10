"""Compare two walk-forward CSVs produced by ``walk_forward_limits.py``.

Reads:
    reports/wf_74feat_volOFF.csv   (baseline, 74 features)
    reports/wf_92feat_volOFF.csv   (experiment 1, 92 features = +EMA/ADX/Regime)

Writes:
    logs/feature_ablation_report.txt

Also prints the same report to stdout.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aivora.utils.config import get_config  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
CSV_74 = ROOT / "reports" / "wf_74feat_volOFF.csv"
CSV_92 = ROOT / "reports" / "wf_92feat_volOFF.csv"
REPORT = ROOT / "logs" / "feature_ablation_report.txt"


def _totals(df: pd.DataFrame) -> dict:
    return {
        "n_folds": len(df),
        "total_trades": int(df["trades"].sum()),
        "total_pnl": float(df["pnl"].sum()),
        "total_gross_pnl": float(df["gross_pnl"].sum()),
        "total_costs": float(df["costs"].sum()),
        "total_wins": int(df["wins"].sum()),
        "profitable_months": int((df["pnl"] > 0).sum()),
    }


def _monthly_metrics(df: pd.DataFrame, capital: float) -> dict:
    monthly = df["pnl"].astype(float)
    monthly_ret = monthly / capital
    sharpe = (
        float(monthly_ret.mean() / monthly_ret.std() * np.sqrt(12))
        if monthly_ret.std() and not np.isnan(monthly_ret.std()) else 0.0
    )
    cum = monthly.cumsum() + capital
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd_pct = float(drawdown.min()) * 100.0 if not drawdown.empty else 0.0
    return {"sharpe": sharpe, "max_dd_pct": max_dd_pct}


def main() -> int:
    if not CSV_74.exists() or not CSV_92.exists():
        raise FileNotFoundError(f"Need both {CSV_74} and {CSV_92}")
    df74 = pd.read_csv(CSV_74)
    df92 = pd.read_csv(CSV_92)

    cfg = get_config()
    capital = float(cfg.project["base_capital"])

    t74 = _totals(df74)
    t92 = _totals(df92)
    m74 = _monthly_metrics(df74, capital)
    m92 = _monthly_metrics(df92, capital)

    # Overlap: identical pnl per (test_month, limit)
    merged = df74.merge(
        df92, on=["test_month", "limit"], suffixes=("_74", "_92"),
    )
    same_pnl = int((np.isclose(merged["pnl_74"], merged["pnl_92"], atol=1e-6)).sum())
    same_trades = int((merged["trades_74"] == merged["trades_92"]).sum())
    same_all = int(
        ((merged["trades_74"] == merged["trades_92"])
         & np.isclose(merged["pnl_74"], merged["pnl_92"], atol=1e-6)).sum()
    )
    n_months = len(merged)

    # Month-by-month diff (top 10 biggest by |Δpnl|)
    merged["delta_pnl"] = merged["pnl_92"] - merged["pnl_74"]
    merged["delta_trades"] = merged["trades_92"] - merged["trades_74"]
    top = merged.reindex(merged["delta_pnl"].abs().sort_values(ascending=False).index).head(10)

    lines = []
    push = lines.append
    hr = "=" * 74

    push(hr)
    push("AiVora — Feature Ablation Report (Experiment 1: EMA/ADX/Regime)")
    push(f"Generated : {datetime.now().isoformat(timespec='seconds')}")
    push(f"Baseline  : 74 features  →  {CSV_74.name}")
    push(f"Enriched  : 92 features  →  {CSV_92.name}  (+18: EMA×4, slope×2, dist×2,")
    push("            align_flag×3, align_score×1, adx_14, di_plus/minus_14, adx_slope,")
    push("            is_trending, is_ranging)")
    push(f"Setup     : max_trades_per_day=10, Vol Filter OFF (min=0, max=999), no cooldown")
    push(f"Test months in each run: {t74['n_folds']} (74-feat) / {t92['n_folds']} (92-feat)")
    push(hr)
    push("")

    push("Aggregate comparison")
    push("-" * 74)
    push(f"  {'Metric':<28s}  {'74 feat':>16s}  {'92 feat':>16s}  {'Δ (92−74)':>12s}")

    def _row(label, v74, v92, fmt):
        return f"  {label:<28s}  {fmt.format(v74):>16s}  {fmt.format(v92):>16s}  {fmt.format(v92 - v74):>12s}"

    push(_row("Total P&L (₹)",        t74["total_pnl"],       t92["total_pnl"],       "{:,.2f}"))
    push(_row("Total gross P&L (₹)",  t74["total_gross_pnl"], t92["total_gross_pnl"], "{:,.2f}"))
    push(_row("Total costs (₹)",      t74["total_costs"],     t92["total_costs"],     "{:,.2f}"))
    push(_row("Total trades",         t74["total_trades"],    t92["total_trades"],    "{:,}"))
    push(_row("Winning trades",       t74["total_wins"],      t92["total_wins"],      "{:,}"))
    wr74 = t74["total_wins"] / max(t74["total_trades"], 1) * 100
    wr92 = t92["total_wins"] / max(t92["total_trades"], 1) * 100
    push(_row("Win rate (%)",         wr74,                   wr92,                   "{:.2f}"))
    push(_row("Sharpe (annualised)",  m74["sharpe"],          m92["sharpe"],          "{:.2f}"))
    push(_row("Max drawdown (%)",     m74["max_dd_pct"],      m92["max_dd_pct"],      "{:.2f}"))
    push(_row("Profitable months",    t74["profitable_months"], t92["profitable_months"], "{:d}"))
    push("")

    push("Overlap between the two runs")
    push("-" * 74)
    push(f"  Months present in both runs        : {n_months}")
    push(f"  Months with IDENTICAL P&L          : {same_pnl}  ({100*same_pnl/max(n_months,1):.1f}%)")
    push(f"  Months with IDENTICAL trade count  : {same_trades}  ({100*same_trades/max(n_months,1):.1f}%)")
    push(f"  Months with BOTH identical         : {same_all}  ({100*same_all/max(n_months,1):.1f}%)")
    push("")

    push("Top 10 months by absolute P&L difference")
    push("-" * 74)
    push(f"  {'Month':<10s} {'Trades 74':>10s} {'Trades 92':>10s} {'P&L 74':>14s} {'P&L 92':>14s} {'Δ P&L':>12s}")
    for _, r in top.iterrows():
        push(f"  {r['test_month']:<10s} "
             f"{int(r['trades_74']):>10d} {int(r['trades_92']):>10d} "
             f"{r['pnl_74']:>14,.2f} {r['pnl_92']:>14,.2f} {r['delta_pnl']:>+12,.2f}")
    push("")

    # Verdict
    push(hr)
    push("Verdict")
    push("-" * 74)
    delta_pnl_total = t92["total_pnl"] - t74["total_pnl"]
    delta_sharpe = m92["sharpe"] - m74["sharpe"]
    delta_dd = m92["max_dd_pct"] - m74["max_dd_pct"]
    pnl_gain_pct = delta_pnl_total / max(abs(t74["total_pnl"]), 1e-9) * 100.0
    push(f"  ΔTotal P&L   : ₹{delta_pnl_total:+,.2f}  ({pnl_gain_pct:+.1f}%)")
    push(f"  ΔSharpe      : {delta_sharpe:+.2f}")
    push(f"  ΔMax DD      : {delta_dd:+.2f} pp")
    if delta_pnl_total > 0 and delta_sharpe > 0:
        push("  → EMA/ADX enrichment IMPROVES both P&L and risk-adjusted returns.")
    elif delta_pnl_total > 0 and delta_sharpe <= 0:
        push("  → EMA/ADX enrichment lifts P&L but not risk-adjusted returns.")
    elif delta_pnl_total <= 0 and delta_sharpe > 0:
        push("  → EMA/ADX enrichment improves Sharpe despite lower total P&L.")
    else:
        push("  → EMA/ADX enrichment does NOT help — keep the 74-feature baseline.")
    push(hr)

    report = "\n".join(lines)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport written to {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
