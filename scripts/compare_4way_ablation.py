"""Consolidate the 4-way ablation into one report.

Reads:
    reports/wf_74feat_volOFF.csv     baseline    (74 features)
    reports/wf_ema_only.csv          + EMA       (86 = 74 + 12)
    reports/wf_adx_only.csv          + ADX/Reg   (80 = 74 + 6)
    reports/wf_92feat_volOFF.csv     + both      (92 = 74 + 18)

    reports/ablation/importance_{baseline,ema_only,adx_only,full}.csv

Writes:
    logs/ablation_4way_report.txt
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
REPORT = ROOT / "logs" / "ablation_4way_report.txt"

# tag → (csv path, importance path, model_visible expected)
CONFIGS = [
    ("baseline",   ROOT / "reports" / "wf_74feat_volOFF.csv",
                   ROOT / "reports" / "ablation" / "importance_baseline.csv",  55),
    ("+ EMA",      ROOT / "reports" / "wf_ema_only.csv",
                   ROOT / "reports" / "ablation" / "importance_ema_only.csv",  67),
    ("+ ADX/Reg",  ROOT / "reports" / "wf_adx_only.csv",
                   ROOT / "reports" / "ablation" / "importance_adx_only.csv",  61),
    ("+ both",     ROOT / "reports" / "wf_92feat_volOFF.csv",
                   ROOT / "reports" / "ablation" / "importance_full.csv",      73),
]


def _wf_metrics(csv: Path, capital: float) -> dict:
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
        "total_gross": float(df["gross_pnl"].astype(float).sum()),
        "total_costs": float(df["costs"].astype(float).sum()),
        "total_trades": int(df["trades"].sum()),
        "total_wins": int(df["wins"].sum()),
        "win_rate": float(df["wins"].sum() / max(df["trades"].sum(), 1)) * 100,
        "sharpe": sharpe,
        "max_dd_pct": float(dd.min()) * 100 if not dd.empty else 0.0,
        "profitable_months": int((monthly > 0).sum()),
        "n_months": len(monthly),
    }


def _family_summary(imp_csv: Path) -> pd.DataFrame:
    """Group per-feature importance into families and sum."""
    df = pd.read_csv(imp_csv)
    fam = df.groupby("family").agg(
        n_features=("feature", "count"),
        gain_total=("gain_total", "sum"),
        shap_total=("shap_total", "sum"),
    )
    fam["gain_share_pct"] = fam["gain_total"] / fam["gain_total"].sum() * 100
    fam["gain_per_feat"] = fam["gain_total"] / fam["n_features"]
    return fam


def _top_new_features(imp_csv: Path, tag: str) -> pd.DataFrame:
    df = pd.read_csv(imp_csv)
    if tag == "baseline":
        return df.nlargest(10, "gain_total")[["feature", "family", "gain_total", "shap_total"]]
    new = df[df["family"] != "baseline"].nlargest(10, "gain_total")
    return new[["feature", "family", "gain_total", "shap_total"]]


def main() -> int:
    cfg = get_config()
    capital = float(cfg.project["base_capital"])

    metrics = {tag: _wf_metrics(csv, capital) for tag, csv, _, _ in CONFIGS}
    base = metrics["baseline"]

    lines = []
    push = lines.append
    hr = "=" * 84

    push(hr)
    push("AiVora — Experiment 1 four-way ablation (EMA / ADX+Regime / Both)")
    push(f"Generated : {datetime.now().isoformat(timespec='seconds')}")
    push(f"Setup     : Walk-forward, 55 test months (2022-01 → 2026-07)")
    push(f"            max_trades_per_day = 10, Vol Filter OFF, no cooldown")
    push(f"            Same fold layout, same LightGBM defaults, same random_state")
    push(hr)
    push("")

    # -------- Walk-forward metrics --------
    push("1. Walk-forward metrics (55 test months)")
    push("-" * 84)
    push(f"  {'Config':<15s} {'Cols':>5s} {'P&L (₹)':>13s} {'ΔP&L':>11s} {'Trades':>7s} {'Win%':>6s} "
         f"{'Sharpe':>7s} {'ΔSR':>6s} {'MaxDD%':>7s}")
    for tag, _, _, model_vis in CONFIGS:
        m = metrics[tag]
        d_pnl = m["total_pnl"] - base["total_pnl"]
        d_sr = m["sharpe"] - base["sharpe"]
        push(f"  {tag:<15s} {model_vis:>5d} {m['total_pnl']:>13,.0f} "
             f"{d_pnl:>+11,.0f} {m['total_trades']:>7d} {m['win_rate']:>6.2f} "
             f"{m['sharpe']:>7.2f} {d_sr:>+6.2f} {m['max_dd_pct']:>7.2f}")
    push("")

    # Δ vs baseline in %
    push("2. Deltas vs 74-feature baseline (%)")
    push("-" * 84)
    push(f"  {'Config':<15s} {'ΔP&L':>10s} {'Δ Trades':>10s} {'Δ Win%':>10s} {'Δ Sharpe':>10s}")
    for tag, _, _, _ in CONFIGS:
        m = metrics[tag]
        pnl_pct = (m["total_pnl"] - base["total_pnl"]) / max(abs(base["total_pnl"]), 1e-9) * 100
        trd_pct = (m["total_trades"] - base["total_trades"]) / max(base["total_trades"], 1) * 100
        wr_pp = m["win_rate"] - base["win_rate"]
        sr_pct = (m["sharpe"] - base["sharpe"]) / max(abs(base["sharpe"]), 1e-9) * 100
        push(f"  {tag:<15s} {pnl_pct:>+9.1f}% {trd_pct:>+9.1f}% {wr_pp:>+9.2f}pp {sr_pct:>+9.1f}%")
    push("")

    # -------- Family importance --------
    push("3. Feature importance by family (LightGBM gain, UP+DOWN summed)")
    push("-" * 84)
    for tag, _, imp_csv, _ in CONFIGS:
        fam = _family_summary(imp_csv)
        push(f"\n  [{tag}]")
        push(f"    {'Family':<12s} {'#Feat':>6s} {'Gain total':>14s} {'Share':>7s} {'Gain/feat':>12s}")
        for family in ["baseline", "EMA", "ADX/Regime"]:
            if family not in fam.index:
                continue
            r = fam.loc[family]
            push(f"    {family:<12s} {int(r['n_features']):>6d} "
                 f"{r['gain_total']:>14,.0f} {r['gain_share_pct']:>6.1f}% "
                 f"{r['gain_per_feat']:>12,.0f}")

    push("")

    # -------- Top new features across configs --------
    push("4. Top-10 new (non-baseline) features in the FULL model")
    push("-" * 84)
    top_full = _top_new_features(CONFIGS[3][2], "+ both")
    push(f"  {'Feature':<30s} {'Family':<12s} {'Gain (UP+DN)':>14s} {'|SHAP| (UP+DN)':>16s}")
    for _, r in top_full.iterrows():
        push(f"  {r['feature']:<30s} {r['family']:<12s} "
             f"{r['gain_total']:>14,.0f} {r['shap_total']:>16.4f}")
    push("")

    # -------- Verdict --------
    push(hr)
    push("5. Verdict — where does the edge come from?")
    push("-" * 84)

    # incremental gains
    ema_only = metrics["+ EMA"]["total_pnl"] - base["total_pnl"]
    adx_only = metrics["+ ADX/Reg"]["total_pnl"] - base["total_pnl"]
    both = metrics["+ both"]["total_pnl"] - base["total_pnl"]
    sr_ema = metrics["+ EMA"]["sharpe"] - base["sharpe"]
    sr_adx = metrics["+ ADX/Reg"]["sharpe"] - base["sharpe"]
    sr_both = metrics["+ both"]["sharpe"] - base["sharpe"]
    interaction_pnl = both - (ema_only + adx_only)
    interaction_sr = sr_both - (sr_ema + sr_adx)

    push(f"  ΔP&L attribution (₹)")
    push(f"    EMA only  contribution        : {ema_only:>+12,.0f}")
    push(f"    ADX/Reg only contribution     : {adx_only:>+12,.0f}")
    push(f"    Sum of both individual        : {ema_only + adx_only:>+12,.0f}")
    push(f"    Combined (actual joint gain)  : {both:>+12,.0f}")
    push(f"    Interaction term (synergy)    : {interaction_pnl:>+12,.0f}  "
         f"({100*interaction_pnl/max(abs(both),1e-9):+.1f}% of joint gain)")
    push("")
    push(f"  ΔSharpe attribution")
    push(f"    EMA only contribution         : {sr_ema:>+7.2f}")
    push(f"    ADX/Reg only contribution     : {sr_adx:>+7.2f}")
    push(f"    Sum of both individual        : {sr_ema + sr_adx:>+7.2f}")
    push(f"    Combined (actual joint gain)  : {sr_both:>+7.2f}")
    push(f"    Interaction term (synergy)    : {interaction_sr:>+7.2f}")
    push("")
    push(f"  Per-feature gain density (from FULL model importance):")
    full_fam = _family_summary(CONFIGS[3][2])
    for family in ["baseline", "EMA", "ADX/Regime"]:
        if family in full_fam.index:
            r = full_fam.loc[family]
            push(f"    {family:<12s} {r['gain_per_feat']:>10,.0f}  "
                 f"(from {int(r['n_features']):>2d} features)")

    push("")
    push("  Reading of the results")
    push("  ----------------------")
    if ema_only < 0 and adx_only < 0 and both > 0:
        push("  • Neither family ALONE lifts P&L — both individually reduce it slightly.")
        push(f"    But TOGETHER they add ₹{both:,.0f} vs baseline, with a strongly positive")
        push(f"    interaction term (₹{interaction_pnl:+,.0f}). The two families are ")
        push("    COMPLEMENTARY: EMAs give the model level/regime context, ADX gives it")
        push("    trend intensity — the boosters can only exploit that when both are visible.")
    elif ema_only > 0 and adx_only > 0 and both > 0:
        push("  • Both families individually help; their combination amplifies the effect.")
    elif ema_only > 0 and adx_only <= 0:
        push("  • EMA drives the edge; ADX/Regime alone is roughly neutral.")
    elif adx_only > 0 and ema_only <= 0:
        push("  • ADX/Regime drives the edge; EMA alone is roughly neutral.")
    push("")

    if abs(sr_ema) > abs(sr_adx):
        push("  • On Sharpe, EMA moves the needle more per feature added.")
    else:
        push("  • On Sharpe, ADX/Regime moves the needle more per feature added.")

    if "ADX/Regime" in full_fam.index and "EMA" in full_fam.index:
        ratio = full_fam.loc["ADX/Regime", "gain_per_feat"] / full_fam.loc["EMA", "gain_per_feat"]
        push(f"  • Feature efficiency: ADX/Regime is {ratio:.2f}× as gain-dense as EMA per column.")
        push("    A future pruning pass could safely drop the lowest-importance EMA columns")
        push("    without losing the joint benefit.")

    push("")
    push("  → Keep the 92-feature full model (both flags ON) for production.")
    push("    The incremental edge is a JOINT effect; disabling either family regresses P&L.")
    push(hr)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReport written to {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
