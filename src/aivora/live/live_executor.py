"""LIVE executor — places real Kite orders.

Every function here is a thin shell that:

1. Consults the master switch + safety module.
2. Places a LIMIT order at (approximate) mid-price via KiteClient.
3. Polls order status for up to ``fill_timeout_sec``.
4. Records the fill (or the rejection) in the ``Portfolio`` file.

If ANY safety check fails, or if the order isn't filled inside
the timeout, we do NOT retry silently — we log LOUDLY and hand
control back to the scheduler.  Silent partial fills are the
single most common way retail auto-trading blows up.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Optional

from ..utils.calendar import is_trading_day
from ..utils.config import get_config
from ..utils.logger import get_logger
from .kite_client import KiteClient
from .portfolio import Portfolio, Trade, make_trade_id
from .safety import assert_can_trade_live

log = get_logger(__name__)


def _round_to_tick(price: float, tick: float = 0.05) -> float:
    """Kite requires option prices at 0.05 ticks."""
    return round(round(price / tick) * tick, 2)


def _wait_for_fill(kite: KiteClient, order_id: str, timeout_sec: int = 20) -> Optional[dict]:
    """Poll order_status until COMPLETE / REJECTED / CANCELLED."""
    deadline = time.time() + timeout_sec
    last = None
    while time.time() < deadline:
        try:
            last = kite.order_status(order_id)
        except Exception as exc:
            log.warning("order_status(%s) failed: %s", order_id, exc)
            time.sleep(1.0)
            continue
        status = (last or {}).get("status")
        if status in ("COMPLETE", "REJECTED", "CANCELLED"):
            return last
        time.sleep(1.0)
    return last


def open_live_trade(
    portfolio: Portfolio,
    kite: KiteClient,
    symbol: str,
    side: str,
    spot: float,
    entry_time: datetime,
    live_ce_ltp: float,
    live_pe_ltp: float,
    fill_timeout_sec: int = 20,
) -> Optional[Trade]:
    """Place a real LIMIT BUY at ~mid, wait for fill, record it.

    Returns the Trade on success, None on rejection or timeout.
    """
    assert_can_trade_live(portfolio)
    if not is_trading_day(entry_time.date()):
        log.warning("Not a trading day — refusing live order")
        return None

    inst = None
    for i in get_config().instruments:
        if i["symbol"] == symbol:
            inst = i
            break
    assert inst is not None
    strike_step = int(inst["strike_step"])
    strike = round(spot / strike_step) * strike_step

    # Resolve tradingsymbol via Kite instruments dump.
    info = kite.atm_option_symbols(symbol, spot)
    tradingsymbol = info["CE"] if side == "CE" else info["PE"]
    lot_size = int(info["lot_size"])
    ltp = float(live_ce_ltp if side == "CE" else live_pe_ltp)
    limit_price = _round_to_tick(ltp * 1.001)   # tiny cross to help fill

    settings = portfolio.load()["settings"]
    capital = float(portfolio.load()["current_capital"])
    risk_budget = float(settings["risk_per_trade_pct"]) * capital
    lots = max(1, int(risk_budget // max(ltp * lot_size, 1.0)))
    qty = lots * lot_size

    log.warning(
        "LIVE placing BUY %s qty=%d px=%.2f (ltp=%.2f)",
        tradingsymbol, qty, limit_price, ltp,
    )
    try:
        order_id = kite.place_limit_buy(tradingsymbol, qty, limit_price)
    except Exception as exc:
        portfolio.append_log(f"LIVE order placement failed: {exc}", "error")
        return None

    final = _wait_for_fill(kite, order_id, timeout_sec=fill_timeout_sec)
    status = (final or {}).get("status")
    if status != "COMPLETE":
        portfolio.append_log(
            f"LIVE order {order_id} not filled: status={status}", "error"
        )
        return None

    avg_price = float(final.get("average_price") or limit_price)
    horizon = int(settings.get("horizon_candles", 12))
    trade = Trade(
        trade_id=make_trade_id(),
        entry_time=entry_time.isoformat(timespec="seconds"),
        symbol=symbol,
        side=side,
        strike=float(strike),
        lots=lots,
        lot_size=lot_size,
        entry_premium=avg_price,
        entry_spot=float(spot),
        current_premium=avg_price,
        unrealized_pnl=0.0,
        entry_order_id=order_id,
        horizon_close_time=(entry_time + timedelta(minutes=5 * horizon)).isoformat(timespec="seconds"),
    )
    portfolio.open_trade(trade)
    return trade


def close_live_trade(
    portfolio: Portfolio,
    kite: KiteClient,
    trade_dict: dict,
    exit_time: datetime,
    exit_reason: str,
    fill_timeout_sec: int = 20,
) -> None:
    """Send a LIMIT SELL close and reconcile the fill into the trade."""
    assert_can_trade_live(portfolio)
    lots = int(trade_dict["lots"])
    lot_size = int(trade_dict["lot_size"])
    qty = lots * lot_size

    # We need the tradingsymbol we entered with; re-derive from
    # the current chain (strike/side stable within intraday window).
    info = kite.atm_option_symbols(
        trade_dict["symbol"], float(trade_dict["entry_spot"] or 0.0),
    )
    ts = info["CE"] if trade_dict["side"] == "CE" else info["PE"]
    # Pull latest quote to size the LIMIT price fairly.
    q = kite.atm_option_quote(trade_dict["symbol"], float(trade_dict["entry_spot"] or 0.0))
    ltp = float(q["ce_ltp"] if trade_dict["side"] == "CE" else q["pe_ltp"])
    limit_price = _round_to_tick(ltp * 0.999)   # cross slightly the other way

    log.warning("LIVE placing SELL %s qty=%d px=%.2f", ts, qty, limit_price)
    try:
        order_id = kite.place_limit_sell(ts, qty, limit_price)
    except Exception as exc:
        portfolio.append_log(f"LIVE close failed to place: {exc}", "error")
        return

    final = _wait_for_fill(kite, order_id, timeout_sec=fill_timeout_sec)
    status = (final or {}).get("status")
    if status != "COMPLETE":
        portfolio.append_log(
            f"LIVE close order {order_id} status={status} — MANUAL REVIEW NEEDED",
            "error",
        )
        return

    avg = float(final.get("average_price") or limit_price)
    from ..backtest.costs import compute_round_trip
    settings = portfolio.load()["settings"]
    costs = compute_round_trip(
        entry_premium=float(trade_dict["entry_premium"]),
        exit_premium=avg,
        lots=lots, lot_size=lot_size,
        cfg=settings,
    )
    gross = (avg - float(trade_dict["entry_premium"])) * lots * lot_size
    portfolio.close_trade(
        trade_id=trade_dict["trade_id"],
        exit_time=exit_time,
        exit_premium=avg,
        exit_reason=exit_reason,
        gross_pnl=gross,
        costs=costs.total,
        exit_order_id=order_id,
    )
