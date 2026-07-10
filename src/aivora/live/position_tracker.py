"""Position tracker — every 5-min tick, decide who exits.

For each open trade in the portfolio:

1. Compute the current option premium (paper: theoretical model /
   live: fresh Kite quote).
2. Update ``current_premium`` and ``unrealized_pnl`` on the trade.
3. Check exit conditions in this order:
     a. Take-profit  (premium up by ≥ take_profit_pct)
     b. Stop-loss    (premium down by ≥ stop_loss_pct)
     c. Horizon      (now >= horizon_close_time)
   First hit wins; execute via the paper or live executor.

The tracker never opens trades — that's the scheduler's job.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from ..utils.logger import get_logger
from . import trailing_sl as tsl
from .kite_client import KiteClient
from .paper_executor import close_paper_trade, theoretical_exit_premium
from .portfolio import Portfolio

log = get_logger(__name__)


def _minutes(a: datetime, b: datetime) -> float:
    return (a - b).total_seconds() / 60.0


def _live_premium(kite: KiteClient, symbol: str, spot_now: float, side: str) -> float:
    q = kite.atm_option_quote(symbol, spot_now)
    return float(q["ce_ltp"] if side == "CE" else q["pe_ltp"])


def _decide_exit(
    trade: dict,
    now: datetime,
    current_premium: float,
    settings: Dict,
) -> Optional[str]:
    """Return the exit reason to fire, or None to hold.

    Priority order (first hit wins):
      1. take_profit — premium up ≥ take_profit_pct.
      2. trailing_stop — premium fallen to the trailing SL (only
         active once the peak has hit +10 %).
      3. stop_loss — fixed −stop_loss_pct from entry, active while
         trailing SL hasn't kicked in yet.
      4. horizon — hold time expired.
    """
    entry_prem = float(trade["entry_premium"])
    if entry_prem <= 0:
        return None
    move = (current_premium - entry_prem) / entry_prem
    tp = float(settings.get("take_profit_pct") or 0.0)
    sl = float(settings.get("stop_loss_pct") or 0.0)
    if tp > 0 and move >= tp:
        return "take_profit"

    trailing = trade.get("trailing_sl_price")
    if tsl.would_stop_here(current_premium, trailing):
        return "trailing_stop"

    # Fixed SL still applies while trailing SL is dormant (peak
    # never reached +10 %).  Once trailing SL activates it supersedes
    # the fixed one (since trailing is always at or above entry).
    if trailing is None and sl > 0 and move <= -sl:
        return "stop_loss"

    horizon_close = trade.get("horizon_close_time")
    if horizon_close and now >= datetime.fromisoformat(horizon_close):
        return "horizon"
    return None


def _step_trailing_sl(trade: dict, current_premium: float,
                      portfolio: Portfolio) -> Optional[float]:
    """Refresh peak_premium and trailing_sl_price on a trade.

    Returns the new trailing SL price (or None if not yet active).
    The trade dict is mutated in place; caller pushes the updated
    values through ``portfolio.update_open_marks`` alongside the
    current-mark refresh.
    """
    entry = float(trade["entry_premium"])
    prev_peak = float(trade.get("peak_premium") or entry)
    new_peak = max(prev_peak, float(current_premium))
    prev_trail = trade.get("trailing_sl_price")
    new_trail = tsl.update_trailing_sl(entry, new_peak, prev_trail)

    trade["peak_premium"] = new_peak
    trade["trailing_sl_price"] = new_trail

    # Log only when the trailing SL steps up to a new level.
    if new_trail is not None and (prev_trail is None or new_trail > prev_trail):
        pct = (new_trail - entry) / entry
        try:
            portfolio.append_log(
                f"🛡️ Trailing SL updated to ₹{new_trail:.2f} ({pct:+.0%}) "
                f"— {trade['symbol']} {trade['side']}",
                "info",
            )
        except AttributeError:
            # File-based Portfolio uses append_log; UserPortfolio shim
            # bridges to log_event.  Both are handled elsewhere; be
            # defensive here in case neither is available.
            log.info("Trailing SL updated to %.2f (%.0f%%) — %s %s",
                     new_trail, pct * 100, trade["symbol"], trade["side"])
    return new_trail


def tick(
    portfolio: Portfolio,
    now: datetime,
    spot_prices: Dict[str, float],
    kite: Optional[KiteClient] = None,
) -> None:
    """Run one 5-min tick over every open trade.

    ``spot_prices``: latest spot close per symbol, so the paper
    engine has a fresh underlying to price against.
    """
    state = portfolio.load()
    open_trades = [t for t in state["trades"] if not t.get("exit_time")]
    if not open_trades:
        return

    settings = state["settings"]
    live = state["mode"] == "live"
    updates: Dict[str, Dict[str, float]] = {}
    to_close = []

    for t in open_trades:
        sym = t["symbol"]
        spot = float(spot_prices.get(sym) or t.get("entry_spot") or 0.0)
        # Current premium: paper uses the theoretical model, live
        # pulls a fresh quote (rate-limited by KiteClient).
        if live and kite is not None:
            try:
                current = _live_premium(kite, sym, spot, t["side"])
            except Exception as exc:
                log.warning("live quote failed for %s (%s): %s", sym, t["side"], exc)
                current = float(t.get("current_premium") or t["entry_premium"])
        else:
            elapsed = _minutes(now, datetime.fromisoformat(t["entry_time"]))
            current = theoretical_exit_premium(
                entry_premium=float(t["entry_premium"]),
                spot_entry=float(t.get("entry_spot") or spot),
                spot_now=spot,
                side=t["side"],
                elapsed_minutes=max(0.0, elapsed),
                settings=settings,
            )
        # Refresh peak + trailing SL before the exit decision so a
        # sudden spike is captured on the same tick that caused it.
        _step_trailing_sl(t, current, portfolio)

        unrealized = (current - float(t["entry_premium"])) * int(t["lots"]) * int(t["lot_size"])
        marks = {
            "current_premium": float(current),
            "unrealized_pnl": float(unrealized),
            "peak_premium": float(t["peak_premium"]),
        }
        if t.get("trailing_sl_price") is not None:
            marks["trailing_sl_price"] = float(t["trailing_sl_price"])
        updates[t["trade_id"]] = marks

        reason = _decide_exit(t, now, current, settings)
        if reason is not None:
            to_close.append((t, current, spot, reason))

    if updates:
        portfolio.update_open_marks(updates)

    for trade_dict, exit_prem, spot_now, reason in to_close:
        log.info("EXIT trigger: id=%s reason=%s exit_premium=%.2f",
                 trade_dict["trade_id"][:8], reason, exit_prem)
        if live and kite is not None:
            from .live_executor import close_live_trade
            close_live_trade(portfolio, kite, trade_dict, now, reason)
        else:
            close_paper_trade(portfolio, trade_dict, now, exit_prem, reason, exit_spot=spot_now)


def emergency_square_off(
    portfolio: Portfolio,
    now: datetime,
    spot_prices: Dict[str, float],
    kite: Optional[KiteClient] = None,
) -> int:
    """Force-close every open position immediately.  Returns count."""
    state = portfolio.load()
    open_trades = [t for t in state["trades"] if not t.get("exit_time")]
    if not open_trades:
        return 0
    settings = state["settings"]
    live = state["mode"] == "live"
    n = 0
    for t in open_trades:
        sym = t["symbol"]
        spot = float(spot_prices.get(sym) or t.get("entry_spot") or 0.0)
        if live and kite is not None:
            try:
                current = _live_premium(kite, sym, spot, t["side"])
            except Exception:
                current = float(t.get("current_premium") or t["entry_premium"])
            from .live_executor import close_live_trade
            close_live_trade(portfolio, kite, t, now, "emergency")
        else:
            elapsed = _minutes(now, datetime.fromisoformat(t["entry_time"]))
            current = theoretical_exit_premium(
                entry_premium=float(t["entry_premium"]),
                spot_entry=float(t.get("entry_spot") or spot),
                spot_now=spot,
                side=t["side"],
                elapsed_minutes=max(0.0, elapsed),
                settings=settings,
            )
            close_paper_trade(portfolio, t, now, current, "emergency", exit_spot=spot)
        n += 1
    portfolio.append_log(f"Emergency square-off closed {n} position(s)", "warn")
    return n
