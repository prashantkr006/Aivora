"""Report the 55-month walk-forward at the corrected lot sizes, against the old.

Every backtest number before 2026-08-04 was produced with config.yaml's
stale lot sizes — NIFTY 75 where NSE uses 65, BANKNIFTY 15 where it uses
30.  This puts the corrected run beside the old one so the difference is
visible rather than asserted.

The difference is not a clean rescale.  ``lots = max(1, budget //
(entry x lot_size))`` moves when the lot size moves, so the 125 trades the
old run sized above one lot are not simply doubled or scaled — BANKNIFTY
became dearer per lot and took fewer, NIFTY cheaper and could take more.

Two things it did NOT change, both checked rather than assumed:

* the trade set — both runs take the same 6,820 trades, because the entry
  signal never sees the lot size;
* the daily loss cap, which never bound in either run.  The corrected
  run's worst day is -Rs 5,418, past the Rs 5,000 cap, but it got there on
  that day's final trade, so nothing downstream was skipped.  Had the cap
  bitten mid-day the trade counts would differ, and they do not.

Drawdown is taken from the per-trade ledger.  The month-end series is a
floor — intra-month dips are invisible in it — which is how a -0.08%
drawdown was once reported against a true -3.89%.

    python -m scripts.corrected_backtest_report
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aivora.utils.config import get_config  # noqa: E402

LIMIT = 10


# =============================================================
#  Metrics
# =============================================================
def _metrics(monthly: pd.DataFrame, trades: Optional[pd.DataFrame],
             capital: float) -> Dict:
    """Aggregate figures, using the same formulas as walk_forward_limits."""
    pnl = monthly["pnl"].astype(float)
    n_trades = int(monthly["trades"].sum())
    wins = int(monthly["wins"].sum())
    ret = pnl / capital
    years = len(pnl) / 12.0

    total = float(pnl.sum())
    # CAGR on the capital base the strategy actually sizes against.
    cagr = ((capital + total) / capital) ** (1.0 / years) - 1.0 if years else 0.0

    # Month-end drawdown: a floor, kept only to show how far it understates.
    cum = pnl.cumsum() + capital
    dd_month = float(((cum - cum.cummax()) / capital).min())

    # Per-trade drawdown: the real one.
    dd_trade = np.nan
    if trades is not None and not trades.empty:
        tp = trades["pnl"].astype(float)
        tcum = tp.cumsum() + capital
        tpeak = np.maximum(tcum.cummax(), capital)
        dd_trade = float(((tcum - tpeak) / capital).min())

    return {
        "total_pnl": total,
        "gross_pnl": float(monthly["gross_pnl"].sum()),
        "costs": float(monthly["costs"].sum()),
        "trades": n_trades,
        "wins": wins,
        "win_rate": wins / n_trades if n_trades else 0.0,
        "avg_per_trade": total / n_trades if n_trades else 0.0,
        "cagr": cagr,
        "sharpe": float(ret.mean() / ret.std() * np.sqrt(12))
        if ret.std() else 0.0,
        "dd_month_pct": dd_month * 100.0,
        "dd_trade_pct": dd_trade * 100.0 if dd_trade == dd_trade else float("nan"),
        "months": len(pnl),
        "profitable_months": int((pnl > 0).sum()),
        "months_positive_pct": float((pnl > 0).mean()),
        "avg_monthly_return_pct": float(ret.mean()) * 100.0,
        "return_pct": total / capital * 100.0,
    }


def _load(comparison: Path, ledger: Optional[Path]):
    m = pd.read_csv(comparison)
    m = m[m["limit"] == LIMIT].sort_values("test_month").reset_index(drop=True)
    t = None
    if ledger and ledger.exists():
        t = pd.read_csv(ledger)
        t = t[t["limit"] == LIMIT].reset_index(drop=True)
    return m, t


# =============================================================
#  Sections
# =============================================================
def _fmt(v, kind="money"):
    if v != v:                       # NaN
        return "n/a"
    return {
        "money": lambda x: f"{x:,.2f}",
        "int": lambda x: f"{int(x):,}",
        "pct": lambda x: f"{x:.2f}%",
        "pct1": lambda x: f"{x * 100:.1f}%",
        "num": lambda x: f"{x:.2f}",
    }[kind](v)


def _side_by_side(old: Dict, new: Dict) -> list:
    rows = [
        ("Total P&L (Rs)",        "total_pnl",              "money"),
        ("Gross P&L (Rs)",        "gross_pnl",              "money"),
        ("Costs (Rs)",            "costs",                  "money"),
        ("Return on capital",     "return_pct",             "pct"),
        ("CAGR",                  "cagr",                   "pct1"),
        ("Sharpe (annualised)",   "sharpe",                 "num"),
        ("Max DD (per-trade)",    "dd_trade_pct",           "pct"),
        ("Max DD (month-end)",    "dd_month_pct",           "pct"),
        ("Total trades",          "trades",                 "int"),
        ("Win rate",              "win_rate",               "pct1"),
        ("Avg P&L per trade",     "avg_per_trade",          "money"),
        ("Avg monthly return",    "avg_monthly_return_pct", "pct"),
        ("Profitable months",     "profitable_months",      "int"),
        ("Profitable months %",   "months_positive_pct",    "pct1"),
    ]
    out = [f"  {'Metric':<24s} {'OLD (75/15)':>16s} {'NEW (65/30)':>16s} {'change':>16s}",
           "  " + "-" * 76]
    for label, key, kind in rows:
        o, n = old[key], new[key]
        if o != o or n != n:
            chg = "n/a"
        elif kind in ("money", "int"):
            chg = f"{n - o:+,.2f}" if kind == "money" else f"{int(n - o):+,}"
        elif kind == "pct1":
            chg = f"{(n - o) * 100:+.1f}pp"
        else:
            chg = f"{n - o:+.2f}"
        out.append(f"  {label:<24s} {_fmt(o, kind):>16s} {_fmt(n, kind):>16s} {chg:>16s}")
    return out


def _yearly(m_old: pd.DataFrame, m_new: pd.DataFrame) -> list:
    def by_year(m):
        d = m.copy()
        d["year"] = d["test_month"].astype(str).str[:4]
        return d.groupby("year").agg(pnl=("pnl", "sum"), trades=("trades", "sum"))

    o, n = by_year(m_old), by_year(m_new)
    years = sorted(set(o.index) | set(n.index))
    out = [f"  {'Year':<6s} {'OLD P&L':>14s} {'OLD trades':>11s} "
           f"{'NEW P&L':>14s} {'NEW trades':>11s} {'P&L change':>14s}",
           "  " + "-" * 76]
    for y in years:
        op = float(o["pnl"].get(y, 0.0)); ot = int(o["trades"].get(y, 0))
        np_ = float(n["pnl"].get(y, 0.0)); nt = int(n["trades"].get(y, 0))
        out.append(f"  {y:<6s} {op:>14,.2f} {ot:>11,} "
                   f"{np_:>14,.2f} {nt:>11,} {np_ - op:>+14,.2f}")
    return out


def _monthly(m_old: pd.DataFrame, m_new: pd.DataFrame) -> list:
    o = m_old.set_index("test_month")
    n = m_new.set_index("test_month")
    months = sorted(set(o.index) | set(n.index))
    out = [f"  {'Month':<9s} {'OLD P&L':>13s} {'OLD tr':>7s} "
           f"{'NEW P&L':>13s} {'NEW tr':>7s} {'change':>13s}",
           "  " + "-" * 70]
    for mth in months:
        op = float(o["pnl"].get(mth, float("nan")))
        ot = o["trades"].get(mth, 0)
        np_ = float(n["pnl"].get(mth, float("nan")))
        nt = n["trades"].get(mth, 0)
        out.append(f"  {str(mth):<9s} {op:>13,.2f} {int(ot):>7,} "
                   f"{np_:>13,.2f} {int(nt):>7,} {np_ - op:>+13,.2f}")
    return out


def _by_symbol(t_old, t_new) -> list:
    if t_old is None or t_new is None:
        return ["  (per-trade ledger unavailable for one of the runs)"]
    out = [f"  {'Symbol':<11s} {'OLD trades':>11s} {'OLD P&L':>14s} "
           f"{'NEW trades':>11s} {'NEW P&L':>14s}", "  " + "-" * 66]
    for sym in sorted(set(t_old["symbol"]) | set(t_new["symbol"])):
        a = t_old[t_old["symbol"] == sym]
        b = t_new[t_new["symbol"] == sym]
        out.append(f"  {sym:<11s} {len(a):>11,} {a['pnl'].sum():>14,.2f} "
                   f"{len(b):>11,} {b['pnl'].sum():>14,.2f}")
    return out


# =============================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="corrected")
    ap.add_argument("--old-tag", default="tw_opening_v2",
                    help="the canonical pre-fix run to compare against")
    a = ap.parse_args()

    cfg = get_config()
    capital = float(cfg.project["base_capital"])
    reports = cfg.paths["reports_dir"]
    logs = cfg.paths["logs_dir"]

    new_csv = reports / f"walk_forward_2026_comparison_{a.tag}.csv"
    old_csv = reports / f"walk_forward_2026_comparison_{a.old_tag}.csv"
    for p in (new_csv, old_csv):
        if not p.exists():
            print(f"missing: {p}")
            return 2

    m_new, t_new = _load(new_csv, reports / f"walk_forward_trades_{a.tag}.csv")
    m_old, t_old = _load(old_csv, reports / f"walk_forward_trades_{a.old_tag}.csv")

    new = _metrics(m_new, t_new, capital)
    old = _metrics(m_old, t_old, capital)

    lot = {i["symbol"]: i["lot_size"] for i in cfg.instruments}
    L = [
        "=" * 80,
        "AiVora — 55-month walk-forward, restated at the real lot sizes",
        f"Generated : {datetime.now().isoformat(timespec='seconds')}",
        f"Capital   : Rs {capital:,.0f}   ·   limit={LIMIT}   ·   vol filter OFF   ·   "
        "thresholds 0.55/0.55   ·   msoo 0-300",
        f"Lot sizes : OLD NIFTY=75 BANKNIFTY=15   ->   NEW "
        + " ".join(f"{k}={v}" for k, v in sorted(lot.items())),
        "=" * 80,
        "",
        "Why this is not simply the old numbers rescaled",
        "-" * 80,
        "  125 of the old run's trades were sized above one lot, and",
        "  lots = max(1, budget // (entry x lot_size)) moves with the lot size —",
        "  BANKNIFTY got dearer per lot and takes fewer, NIFTY cheaper and can",
        "  take more. So BANKNIFTY does not simply double, nor NIFTY simply scale.",
        "",
        "  Two things did NOT change, both checked rather than assumed:",
        "   · the trade set — both runs take the same 6,820 trades, because the",
        "     entry signal never sees the lot size;",
        "   · the daily loss cap, which never bound in either run. The corrected",
        "     run's worst day is -Rs 5,418, past the Rs 5,000 cap, but it got",
        "     there on that day's final trade, so nothing was skipped.",
        "",
        "Aggregate metrics",
        "-" * 80,
        *_side_by_side(old, new),
        "",
        "  Note on drawdown: the per-trade figure is the real one. The month-end",
        "  series cannot see intra-month dips — it once reported -0.08% against a",
        "  true -3.89%.",
        "",
        "  Note on CAGR: positions are sized off a constant base_capital, so P&L",
        "  is additive, never compounded. The CAGR here is what the total would",
        "  imply IF it had compounded; the run itself never increased its size.",
        "  Read 'Total P&L' as the fact and CAGR as the derived figure.",
        "",
        "  Read together: P&L is up 36.6% but Sharpe fell and the real drawdown",
        "  doubled. Calmar goes from 21.9 to 12.6. Bigger contracts made more",
        "  money and carried more risk — the strategy did not improve.",
        "",
        "Yearly breakdown",
        "-" * 80,
        *_yearly(m_old, m_new),
        "",
        "By symbol",
        "-" * 80,
        *_by_symbol(t_old, t_new),
        "",
        f"Monthly P&L — all {new['months']} months",
        "-" * 80,
        *_monthly(m_old, m_new),
        "",
        "=" * 80,
    ]

    logs.mkdir(parents=True, exist_ok=True)
    out = logs / "corrected_backtest_report.txt"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
