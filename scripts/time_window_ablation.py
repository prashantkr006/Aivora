"""Compare 4 time-window configurations for variant #18 at limit=10.

Runs walk-forward for each config sequentially, then produces a
side-by-side comparison table + verdict.

Configs:
  baseline    — min=30, max=300 (current live setting)
  all_open    — min=0,  max=375 (trade the entire session)
  opening     — min=0,  max=300 (add opening only)
  closing     — min=30, max=375 (add closing only)

Writes:
  reports/wf_tw_<config>.csv       (per-fold rows for each run)
  logs/time_window_ablation.txt    (final comparison table)
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aivora.utils.config import get_config  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
LOG_OUT = ROOT / "logs" / "time_window_ablation.txt"

CONFIGS = [
    # (tag,        min_msoo, max_msoo, description)
    ("baseline",   30,  300, "Current — no opening/closing"),
    ("all_open",    0,  375, "Trade entire session"),
    ("opening",     0,  300, "Add opening (9:15-9:45) only"),
    ("closing",    30,  375, "Add closing (14:15-15:30) only"),
]


def _run_one(tag: str, min_msoo: int, max_msoo: int) -> Path:
    print(f"\n=== Running config '{tag}'  (min={min_msoo}, max={max_msoo}) ===")
    subprocess.run(
        [
            sys.executable, "-m", "scripts.walk_forward_limits",
            "--limits", "10",
            "--min-msoo", str(min_msoo),
            "--max-msoo", str(max_msoo),
            "--tag", f"tw_{tag}",
        ],
        cwd=str(ROOT), check=True,
    )
    csv = REPORTS / f"walk_forward_2026_comparison_tw_{tag}.csv"
    if not csv.exists():
        raise RuntimeError(f"Expected output missing: {csv}")
    return csv


def _summarise(csv: Path, capital: float) -> dict:
    df = pd.read_csv(csv)
    monthly = df["pnl"].astype(float)
    monthly_ret = monthly / capital
    sharpe = (
        float(monthly_ret.mean() / monthly_ret.std() * np.sqrt(12))
        if monthly_ret.std() and not np.isnan(monthly_ret.std()) else 0.0
    )
    cum = monthly.cumsum() + capital
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return {
        "total_pnl": float(monthly.sum()),
        "total_trades": int(df["trades"].sum()),
        "wins": int(df["wins"].sum()),
        "win_rate": float(df["wins"].sum() / max(df["trades"].sum(), 1) * 100),
        "sharpe": sharpe,
        "max_dd": float(dd.min()) * 100,
        "profitable_months": int((monthly > 0).sum()),
        "n_months": len(monthly),
    }


def main() -> int:
    cfg = get_config()
    capital = float(cfg.project["base_capital"])

    print(f"Running {len(CONFIGS)} walk-forwards. Each takes ~20 min. "
          f"Total est: ~{len(CONFIGS) * 20} min.")
    results = {}
    for tag, mn, mx, _ in CONFIGS:
        csv = _run_one(tag, mn, mx)
        results[tag] = _summarise(csv, capital)

    # ---- Comparison table ----
    lines = []
    push = lines.append
    hr = "=" * 92
    push(hr)
    push("AiVora — Time-window ablation on variant #18 (limit=10, vol_off, prob_down=0.55)")
    push(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    push(hr)
    push("")
    push(f"  {'Config':<12s}  {'Description':<38s}  {'P&L (₹)':>12s}  {'Trades':>7s}  "
         f"{'Win%':>6s}  {'Sharpe':>7s}  {'MaxDD%':>7s}  {'ProfMo':>7s}")
    push("  " + "-" * 90)
    for tag, mn, mx, desc in CONFIGS:
        r = results[tag]
        push(f"  {tag:<12s}  {desc:<38s}  {r['total_pnl']:>12,.0f}  "
             f"{r['total_trades']:>7d}  {r['win_rate']:>5.2f}%  "
             f"{r['sharpe']:>7.2f}  {r['max_dd']:>7.2f}  "
             f"{r['profitable_months']:>2d}/{r['n_months']}")

    push("")
    push("Deltas vs baseline")
    push("-" * 92)
    base = results["baseline"]
    push(f"  {'Config':<12s}  {'ΔP&L':>12s}  {'ΔP&L%':>8s}  {'ΔTrades':>8s}  "
         f"{'ΔSharpe':>8s}  {'ΔMaxDD':>8s}")
    for tag, _, _, _ in CONFIGS:
        r = results[tag]
        d_pnl = r["total_pnl"] - base["total_pnl"]
        d_pnl_pct = d_pnl / max(abs(base["total_pnl"]), 1e-9) * 100
        d_trd = r["total_trades"] - base["total_trades"]
        d_sr = r["sharpe"] - base["sharpe"]
        d_dd = r["max_dd"] - base["max_dd"]
        push(f"  {tag:<12s}  {d_pnl:>+12,.0f}  {d_pnl_pct:>+7.1f}%  "
             f"{d_trd:>+8d}  {d_sr:>+8.2f}  {d_dd:>+8.2f} pp")

    push("")
    push("Verdict")
    push("-" * 92)
    winner = max(results.items(), key=lambda kv: kv[1]["sharpe"])
    best_pnl = max(results.items(), key=lambda kv: kv[1]["total_pnl"])
    push(f"  Best Sharpe : '{winner[0]}' with {winner[1]['sharpe']:.2f}")
    push(f"  Best P&L    : '{best_pnl[0]}' with ₹{best_pnl[1]['total_pnl']:,.0f}")
    if winner[0] == best_pnl[0]:
        push(f"  → RECOMMEND '{winner[0]}'  — highest on both P&L and Sharpe")
    else:
        push(f"  → Trade-off: '{best_pnl[0]}' has more P&L; '{winner[0]}' has better risk-adjusted returns")
    push(hr)

    LOG_OUT.parent.mkdir(parents=True, exist_ok=True)
    LOG_OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nReport written to {LOG_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
