"""Options-buying backtest engine.

The model emits a 3-class probability vector per 5-minute candle.
When the top-1 class is UP or DOWN and its probability clears the
threshold, we buy 1+ lots of the ATM CE (UP) or PE (DOWN) and
exit either after ``horizon_candles`` candles (default 30 min) or
sooner if a TP/SL threshold is hit — whichever comes first.

Option P&L is approximated with a delta + linear-theta model
because Dhan's intraday endpoint doesn't ship a live option price
path.  Costs are the full Indian-retail F&O schedule from
:mod:`.costs`.

The engine is a pure function: given a ``probs`` array and a
``Splits`` object, it returns a metrics dict.  All knobs are
overridable via the ``overrides`` argument so the iteration
orchestrator can vary threshold / TP-SL / cost profile without
editing ``config.yaml``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..ml.dataset import Splits  # noqa: E402
from ..utils.config import get_config  # noqa: E402
from ..utils.logger import get_logger  # noqa: E402
from .costs import compute_round_trip  # noqa: E402

log = get_logger(__name__)


@dataclass
class Trade:
    datetime: pd.Timestamp
    symbol: str
    side: str
    entry_spot: float
    exit_spot: float
    entry_premium: float
    exit_premium: float
    lots: int
    gross_pnl: float
    costs: float
    pnl: float
    pnl_pct: float
    exit_reason: str = "horizon"


# =============================================================
#  Helpers
# =============================================================
def _instrument_meta(symbol: str) -> Dict:
    for inst in get_config().instruments:
        if inst["symbol"] == symbol:
            return inst
    raise KeyError(f"Unknown symbol: {symbol}")


def _estimate_entry_premium(spot: float, symbol: str) -> float:
    """ATM premium heuristic when no live option chain is available.

    ~1 % of spot for Nifty, ~1.2 % for Bank Nifty — within the
    typical weekly-ATM range.  Replace with real premium data if
    the parquet has it (see ``entry_premium_from_row``).
    """
    if symbol == "BANKNIFTY":
        return spot * 0.012
    return spot * 0.010


def _entry_premium_from_row(row: pd.Series, side: str) -> float:
    """Prefer real ATM premium from the row if available."""
    if side == "CE" and pd.notna(row.get("ce_ltp")):
        return float(row["ce_ltp"])
    if side == "PE" and pd.notna(row.get("pe_ltp")):
        return float(row["pe_ltp"])
    return _estimate_entry_premium(float(row["spot_close"]), row["symbol"])


def _exit_premium(
    entry_premium: float,
    spot_entry: float,
    spot_exit: float,
    side: str,
    elapsed_minutes: int,
    cfg: Dict,
) -> float:
    delta = float(cfg["option_delta"])
    expiry_minutes = int(cfg["expiry_days_assumption"]) * 6 * 60
    theta_per_minute = entry_premium / max(expiry_minutes, 1)

    direction = +1 if side == "CE" else -1
    spot_move = (spot_exit - spot_entry) * direction
    intrinsic = max(0.0, spot_move) * delta

    decayed_premium = entry_premium - theta_per_minute * elapsed_minutes
    return max(0.0, decayed_premium + intrinsic)


# =============================================================
#  Backtest core
# =============================================================
def run_backtest(
    probs: np.ndarray,
    splits: Splits,
    overrides: Optional[Dict[str, Any]] = None,
    name: str = "backtest",
) -> Dict:
    """Simulate the trading strategy on the test fold.

    ``overrides`` accepted keys (all optional):
        probability_threshold, max_trades_per_day,
        daily_loss_limit_pct, risk_per_trade_pct,
        take_profit_pct, stop_loss_pct,
        symbols_allow_list,
        min_minutes_since_open, max_minutes_since_open,
        prob_threshold_up, prob_threshold_down,     # asymmetric long/short
        vol_regime_min, vol_regime_max,             # 0..1 gate on vol_regime_pct
    """
    cfg = get_config()
    bt = {**cfg.backtest, **(overrides or {})}
    costs_cfg = {**bt.get("costs", {}), **(overrides or {}).get("costs", {})}
    capital = float(cfg.project["base_capital"])
    horizon = int(cfg.labels["horizon_candles"])

    meta = splits.meta_test.copy().reset_index(drop=True)
    if len(meta) != len(probs):
        raise ValueError("probs and meta_test length mismatch")
    meta["p_flat"] = probs[:, 0]
    meta["p_down"] = probs[:, 1]
    meta["p_up"] = probs[:, 2]
    meta["date"] = meta["datetime"].dt.date

    # Optional CE/PE prices for realistic entry premium.
    if "ce_ltp" not in meta.columns:
        meta["ce_ltp"] = np.nan
    if "pe_ltp" not in meta.columns:
        meta["pe_ltp"] = np.nan

    allow = set(overrides.get("symbols_allow_list", [])) if overrides else set()
    min_msoo = float(bt.get("min_minutes_since_open", 0))
    max_msoo = float(bt.get("max_minutes_since_open", 375))
    tp_pct = bt.get("take_profit_pct")
    sl_pct = bt.get("stop_loss_pct")
    # Asymmetric per-side thresholds — falls back to the symmetric one.
    thr_up = float(bt.get("prob_threshold_up", bt["probability_threshold"]))
    thr_dn = float(bt.get("prob_threshold_down", bt["probability_threshold"]))
    vr_min = bt.get("vol_regime_min")
    vr_max = bt.get("vol_regime_max")
    # Margin filter: require the two binary models to disagree by at
    # least this much before firing (useful when using
    # ``predict_3class_from_binary`` where p_up and p_down are
    # independent probabilities that can both be high).
    margin = float(bt.get("prob_margin_min", 0.0))

    trades: List[Trade] = []
    daily_pnl: Dict[Any, float] = {}
    trade_open_until: Dict[str, int] = {}
    # Per-symbol last-exit metadata for the re-entry cooldown check.
    # Reset every trading day so yesterday's tail can't block today.
    last_exit: Dict[Any, Dict[str, Dict]] = {}
    # Cooldown knobs — defaults set to 0 (OFF) based on the 30-day
    # comparison (logs/cooldown_analysis.txt).  Any consumer can
    # opt in by passing ``cooldown_prob_delta=0.05`` and
    # ``cooldown_price_pct=0.001`` (the original candidate values).
    cd_prob_delta = float(bt.get("cooldown_prob_delta", 0.0))
    cd_price_pct = float(bt.get("cooldown_price_pct", 0.0))

    # Local import keeps the pipeline surface small.
    from ..live import trailing_sl as tsl

    for i, row in meta.iterrows():
        sym = row["symbol"]
        day = row["date"]

        if allow and sym not in allow:
            continue

        # Daily loss brake.
        if daily_pnl.get(day, 0.0) <= -bt["daily_loss_limit_pct"] * capital:
            continue

        # Max trades per day.
        if sum(1 for t in trades if t.datetime.date() == day) >= bt["max_trades_per_day"]:
            continue

        # No overlapping trades on the same symbol.
        if i < trade_open_until.get(sym, 0):
            continue

        # Time-of-day filter.
        msoo = row.get("minutes_since_open", None)
        if msoo is None:
            hh_mm = row["datetime"].hour * 60 + row["datetime"].minute
            msoo = max(0, hh_mm - (9 * 60 + 15))
        if not (min_msoo <= msoo <= max_msoo):
            continue

        # Volatility-regime gate.
        if vr_min is not None or vr_max is not None:
            vr = row.get("vol_regime_pct", np.nan)
            if pd.isna(vr):
                continue
            if vr_min is not None and vr < float(vr_min):
                continue
            if vr_max is not None and vr > float(vr_max):
                continue

        # Entry signal (asymmetric UP/DOWN thresholds + margin gate).
        diff = row["p_up"] - row["p_down"]
        if row["p_up"] >= thr_up and diff >= margin:
            side = "CE"
            entry_prob = float(row["p_up"])
        elif row["p_down"] >= thr_dn and -diff >= margin:
            side = "PE"
            entry_prob = float(row["p_down"])
        else:
            continue

        # -------- Re-entry cooldown --------
        # Only skip if the previous exit was ``horizon`` (natural
        # timeout) AND conviction AND price both look unchanged.
        # TP / SL / trailing exits never trigger cooldown — those are
        # decisive outcomes worth reacting to on the next signal.
        prev = last_exit.get(day, {}).get(sym)
        if prev is not None and prev["reason"] == "horizon":
            prob_close = abs(entry_prob - prev["prob"]) < cd_prob_delta
            price_close = (
                abs(float(row["spot_close"]) - prev["spot"]) / max(prev["spot"], 1e-9)
                < cd_price_pct
            )
            if prob_close and price_close:
                continue

        # Locate exit — trailing SL / TP / fixed SL / horizon.
        entry_premium = _entry_premium_from_row(row, side)
        spot_entry = float(row["spot_close"])
        exit_i = i + horizon
        exit_reason = "horizon"
        exit_premium = None
        spot_exit = None
        peak_premium = entry_premium
        trailing_sl = None   # activates on +10 % peak

        # Walk the intervening candles — replicates the live tick's
        # per-step trailing update + exit check.
        for step in range(1, horizon + 1):
            j = i + step
            if j >= len(meta):
                exit_i = j - 1
                break
            r = meta.iloc[j]
            if r["symbol"] != sym:
                exit_i = j - 1
                break
            candidate_premium = _exit_premium(
                entry_premium, spot_entry, float(r["spot_close"]),
                side, step * 5, bt,
            )
            # Update peak + trailing SL BEFORE the exit checks, so a
            # spike that would have unlocked a milestone gets counted
            # even if the same bar reverses.
            peak_premium = max(peak_premium, candidate_premium)
            trailing_sl = tsl.update_trailing_sl(entry_premium, peak_premium, trailing_sl)

            move = (candidate_premium - entry_premium) / max(entry_premium, 1e-9)
            if tp_pct is not None and move >= float(tp_pct):
                exit_premium = candidate_premium
                spot_exit = float(r["spot_close"])
                exit_i = j
                exit_reason = "take_profit"
                break
            if tsl.would_stop_here(candidate_premium, trailing_sl):
                exit_premium = candidate_premium
                spot_exit = float(r["spot_close"])
                exit_i = j
                exit_reason = "trailing_stop"
                break
            if trailing_sl is None and sl_pct is not None and move <= -float(sl_pct):
                # Fixed SL only applies before trailing SL activates.
                exit_premium = candidate_premium
                spot_exit = float(r["spot_close"])
                exit_i = j
                exit_reason = "stop_loss"
                break

        if exit_premium is None:
            if exit_i >= len(meta):
                continue
            exit_row = meta.iloc[exit_i]
            if exit_row["symbol"] != sym:
                continue
            spot_exit = float(exit_row["spot_close"])
            exit_premium = _exit_premium(
                entry_premium, spot_entry, spot_exit, side, horizon * 5, bt
            )

        # Position sizing.
        inst = _instrument_meta(sym)
        lot_size = int(inst["lot_size"])
        max_loss_per_lot = entry_premium * lot_size
        risk_budget = bt["risk_per_trade_pct"] * capital
        lots = max(1, int(risk_budget // max(max_loss_per_lot, 1)))

        # Costs and P&L.
        gross_pnl = (exit_premium - entry_premium) * lots * lot_size
        rt_costs = compute_round_trip(
            entry_premium=entry_premium,
            exit_premium=exit_premium,
            lots=lots,
            lot_size=lot_size,
            cfg=costs_cfg,
        )
        pnl = gross_pnl - rt_costs.total
        daily_pnl[day] = daily_pnl.get(day, 0.0) + pnl

        trades.append(Trade(
            datetime=row["datetime"],
            symbol=sym,
            side=side,
            entry_spot=spot_entry,
            exit_spot=spot_exit,
            entry_premium=entry_premium,
            exit_premium=exit_premium,
            lots=lots,
            gross_pnl=float(gross_pnl),
            costs=float(rt_costs.total),
            pnl=float(pnl),
            pnl_pct=float(pnl / capital),
            exit_reason=exit_reason,
        ))
        trade_open_until[sym] = exit_i
        # Stash last-exit metadata for the cooldown check on subsequent
        # signals for the same symbol today.
        last_exit.setdefault(day, {})[sym] = {
            "reason": exit_reason,
            "prob": entry_prob,
            "spot": spot_entry,
        }

    trades_df = pd.DataFrame([asdict(t) for t in trades])
    summary = _summarise(trades_df, capital)
    log.info("[%s] backtest: %s", name, {k: round(v, 4) for k, v in summary.items()})

    plots_dir = cfg.paths["reports_dir"] / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_equity(trades_df, capital, plots_dir / f"equity_{name}.png")

    trades_path = cfg.paths["reports_dir"] / f"trades_{name}.csv"
    trades_df.to_csv(trades_path, index=False)
    log.info("[%s] wrote %d trades → %s", name, len(trades_df), trades_path)

    return {
        "summary": summary,
        "trades": trades_df,
        "equity_curve_path": str(plots_dir / f"equity_{name}.png"),
        "trades_path": str(trades_path),
    }


# =============================================================
#  Stats + plotting
# =============================================================
def _summarise(trades: pd.DataFrame, capital: float) -> Dict[str, float]:
    """Trade-level and monthly aggregates for goal evaluation."""
    if trades.empty:
        return {
            "n_trades": 0, "total_pnl": 0.0, "win_rate": 0.0,
            "sharpe": 0.0, "max_drawdown": 0.0, "return_pct": 0.0,
            "avg_monthly_return_pct": 0.0, "monthly_std_pct": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0, "reward_to_risk": 0.0,
            "n_months": 0, "months_positive_pct": 0.0,
        }
    pnl = trades["pnl"].astype(float)
    cum = pnl.cumsum() + capital
    peak = cum.cummax()
    drawdown = (cum - peak) / peak

    # Daily P&L → annualised Sharpe.
    daily = trades.assign(day=trades["datetime"].dt.date).groupby("day")["pnl"].sum()
    daily_ret = daily / capital
    sharpe = (
        float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
        if daily_ret.std() and not np.isnan(daily_ret.std())
        else 0.0
    )

    # Monthly aggregates for the "3–5% monthly" target.
    monthly = (
        trades.assign(month=pd.to_datetime(trades["datetime"]).dt.to_period("M"))
        .groupby("month")["pnl"].sum()
    )
    monthly_ret = monthly / capital
    n_months = int(len(monthly_ret))
    avg_monthly = float(monthly_ret.mean() if n_months else 0.0)
    std_monthly = float(monthly_ret.std() if n_months > 1 else 0.0)
    months_pos = float((monthly_ret > 0).mean() if n_months else 0.0)

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    rr = float(-avg_win / avg_loss) if avg_loss < 0 else 0.0

    return {
        "n_trades": int(len(trades)),
        "total_pnl": float(pnl.sum()),
        "win_rate": float((pnl > 0).mean()),
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "return_pct": float(pnl.sum() / capital),
        "avg_monthly_return_pct": avg_monthly,
        "monthly_std_pct": std_monthly,
        "n_months": n_months,
        "months_positive_pct": months_pos,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "reward_to_risk": rr,
    }


def _plot_equity(trades: pd.DataFrame, capital: float, out: Path) -> None:
    if trades.empty:
        log.warning("No trades to plot for %s", out.name)
        return
    eq = trades["pnl"].cumsum() + capital
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pd.to_datetime(trades["datetime"]).values, eq.values, color="steelblue")
    ax.axhline(capital, linestyle="--", color="grey", linewidth=1)
    ax.set_title(f"Cumulative equity — {out.stem}")
    ax.set_ylabel("Equity (INR)")
    ax.set_xlabel("Time")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
