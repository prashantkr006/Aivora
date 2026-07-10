"""Paper-trade executor.

Simulates option-buy fills using the *same* delta+theta model
that :mod:`aivora.backtest.backtester` uses.  Any P&L number the
paper mode reports is directly comparable to the backtest, which
means the walk-forward numbers you saw in `logs/final_report.txt`
translate 1:1 to what paper mode will accrue if the market
behaves like the last 4 years.

Costs use the same Indian F&O schedule as the backtest via
:mod:`aivora.backtest.costs`.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from ..backtest.costs import compute_round_trip
from ..utils.calendar import is_trading_day
from ..utils.config import get_config
from ..utils.logger import get_logger
from .portfolio import Portfolio, Trade, make_trade_id

log = get_logger(__name__)


def _instrument_for(symbol: str):
    for inst in get_config().instruments:
        if inst["symbol"] == symbol:
            return inst
    raise KeyError(symbol)


def _estimate_entry_premium(spot: float, symbol: str) -> float:
    """Match the backtester's simplified ATM premium heuristic."""
    return spot * (0.012 if symbol == "BANKNIFTY" else 0.010)


def theoretical_exit_premium(
    entry_premium: float,
    spot_entry: float,
    spot_now: float,
    side: str,
    elapsed_minutes: float,
    settings: dict,
    delta: float = 0.5,
    expiry_days_assumption: int = 7,
) -> float:
    """Delta + linear theta - same formula the backtester uses."""
    expiry_minutes = expiry_days_assumption * 6 * 60
    theta_per_min = entry_premium / max(expiry_minutes, 1)
    direction = +1 if side == "CE" else -1
    intrinsic = max(0.0, (spot_now - spot_entry) * direction) * delta
    decayed = entry_premium - theta_per_min * elapsed_minutes
    return max(0.0, decayed + intrinsic)


def _horizon_close(entry_time: datetime, horizon_candles: int) -> datetime:
    """Absolute time by which the horizon rule forces an exit."""
    return entry_time + timedelta(minutes=5 * horizon_candles)


# =============================================================
#  Public API
# =============================================================
def open_paper_trade(
    portfolio: Portfolio,
    symbol: str,
    side: str,
    spot: float,
    entry_time: datetime,
    live_ce_ltp: Optional[float] = None,
    live_pe_ltp: Optional[float] = None,
    entry_prob: Optional[float] = None,
) -> Trade:
    """Record a paper entry.

    If a real live LTP is available (from Kite quote() during
    market hours), use it as the fill price; otherwise fall back
    to the backtest's ATM premium heuristic.
    """
    if not is_trading_day(entry_time.date()):
        raise RuntimeError(f"{entry_time.date()} is not a trading day")

    inst = _instrument_for(symbol)
    lot_size = int(inst["lot_size"])
    strike_step = int(inst["strike_step"])
    strike = round(spot / strike_step) * strike_step

    if side == "CE" and live_ce_ltp:
        entry_premium = float(live_ce_ltp)
    elif side == "PE" and live_pe_ltp:
        entry_premium = float(live_pe_ltp)
    else:
        entry_premium = _estimate_entry_premium(spot, symbol)

    settings = portfolio.load()["settings"]
    # Position size — same rule as backtest_summary_pct.
    capital = float(portfolio.load()["current_capital"])
    risk_budget = float(settings["risk_per_trade_pct"]) * capital
    per_lot_risk = entry_premium * lot_size
    lots = max(1, int(risk_budget // max(per_lot_risk, 1.0)))

    horizon_candles = int(settings.get("horizon_candles", 12))
    trade = Trade(
        trade_id=make_trade_id(),
        entry_time=entry_time.isoformat(timespec="seconds"),
        symbol=symbol,
        side=side,
        strike=float(strike),
        lots=lots,
        lot_size=lot_size,
        entry_premium=entry_premium,
        entry_spot=float(spot),
        current_premium=entry_premium,
        unrealized_pnl=0.0,
        horizon_close_time=_horizon_close(entry_time, horizon_candles).isoformat(timespec="seconds"),
        entry_prob=float(entry_prob) if entry_prob is not None else None,
        peak_premium=entry_premium,
        trailing_sl_price=None,
    )
    portfolio.open_trade(trade)
    log.info(
        "PAPER open %s %s strike=%.0f lots=%d premium=%.2f -> tp=%.0f%% sl=%.0f%%",
        symbol, side, strike, lots, entry_premium,
        100 * float(settings["take_profit_pct"]),
        100 * float(settings["stop_loss_pct"]),
    )
    return trade


def close_paper_trade(
    portfolio: Portfolio,
    trade_dict: dict,
    exit_time: datetime,
    exit_premium: float,
    exit_reason: str,
    exit_spot: Optional[float] = None,
) -> None:
    """Compute costs, book P&L, persist."""
    settings = portfolio.load()["settings"]
    costs = compute_round_trip(
        entry_premium=float(trade_dict["entry_premium"]),
        exit_premium=float(exit_premium),
        lots=int(trade_dict["lots"]),
        lot_size=int(trade_dict["lot_size"]),
        cfg=settings,
    )
    gross_pnl = (float(exit_premium) - float(trade_dict["entry_premium"])) * \
                int(trade_dict["lots"]) * int(trade_dict["lot_size"])
    portfolio.close_trade(
        trade_id=trade_dict["trade_id"],
        exit_time=exit_time,
        exit_premium=exit_premium,
        exit_reason=exit_reason,
        gross_pnl=gross_pnl,
        costs=costs.total,
        exit_spot=exit_spot,
    )
