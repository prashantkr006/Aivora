"""Bring the book's capital in line with the money actually in the account.

AiVora's capital was derived entirely from its own trades:
``current = initial + realised``.  Cash added or withdrawn in Kite was
invisible to it, so the moment money moved outside AiVora the two numbers
parted company — and stayed parted, because nothing ever looked.

Same shape of bug as positions exited by hand at the broker: the book was
a record of what AiVora did, never checked against what the account holds.

When this may run
-----------------
Only before the first trade of the day, with nothing open.  This is not
caution for its own sake — with a position open the two numbers are not
comparable at all:

* the option premium has already left the account;
* margin is blocked against the position;
* and F&O profit from yesterday does not reach the balance until it
  settles, so a sync taken too early books a withdrawal that never
  happened and reverses it the next morning.

Pre-market, flat, after the day's token refresh is exactly the moment all
three are quiet — which is why that is where this is wired in.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from ..utils.logger import get_logger

log = get_logger(__name__)

# Below this, treat it as rounding rather than a real cash movement.
MIN_DELTA = 1.0


class CashSyncSkipped(Exception):
    """Not a failure — a reason the sync must not run right now."""


def _guard(portfolio, now: datetime) -> None:
    state = portfolio.load()

    if state["mode"] != "live":
        # Paper capital is imaginary on purpose. Tracking a real account
        # would make paper results depend on money that was never at risk.
        raise CashSyncSkipped("portfolio is not in live mode")

    open_trades = [t for t in state["trades"] if not t.get("exit_time")]
    if open_trades:
        raise CashSyncSkipped(
            f"{len(open_trades)} position(s) open — premium is spent and "
            "margin is blocked, so the broker's cash is not comparable"
        )

    last = state.get("cash_synced_at")
    if last and str(last)[:10] == now.date().isoformat():
        raise CashSyncSkipped("already synced today")


def sync_capital_from_broker(
    portfolio, kite, now: Optional[datetime] = None,
) -> Optional[float]:
    """Set the book's capital to the account's cash.  Returns the adjustment.

    ``None`` means nothing was applied — either a guard said not now, or the
    difference was too small to be a real movement.  Never raises for an
    unreachable broker: failing to sync is a stale number, while trading
    stops altogether if this is allowed to propagate.
    """
    now = now or datetime.now()

    try:
        _guard(portfolio, now)
    except CashSyncSkipped as why:
        log.info("cash sync skipped: %s", why)
        return None

    try:
        cash = float(kite.available_funds())
    except Exception as exc:  # noqa: BLE001
        portfolio.append_log(
            f"Could not read your broker balance ({exc}) — capital not synced",
            "warn",
        )
        return None

    if cash <= 0:
        # Zerodha returns 0 in some fields on the day a deposit lands.
        # Zeroing the book on that is far worse than leaving it stale.
        portfolio.append_log(
            "Broker reported a balance of 0 — refusing to sync capital to it",
            "warn",
        )
        return None

    before = float(portfolio.load()["current_capital"])
    delta = cash - before
    if abs(delta) < MIN_DELTA:
        portfolio.set_broker_cash(cash, now)      # records the timestamp
        log.info("cash sync: already in line (Rs %.2f)", cash)
        return 0.0

    portfolio.set_broker_cash(cash, now)
    direction = "added to" if delta > 0 else "withdrawn from"
    portfolio.append_log(
        f"💰 Capital synced with Kite: ₹{before:,.2f} → ₹{cash:,.2f} "
        f"(₹{abs(delta):,.2f} {direction} your account outside AiVora)",
        "warn",
    )
    log.warning("cash sync: %.2f -> %.2f (delta %+.2f)", before, cash, delta)
    return delta
