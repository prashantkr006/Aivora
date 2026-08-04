"""Reconcile the book against the broker's own positions.

AiVora's book is a record of what AiVora did.  The account is the truth.
Nothing kept the two in step, so anything that happened to a position
outside AiVora — you exiting by hand in Kite, the broker squaring off at
close, a margin call — left the book claiming a holding that no longer
existed.

That phantom is not harmless.  While it sits there:

* the position tracker keeps marking it and can fire a "stop loss" on a
  position nobody owns;
* the symbol counts as occupied, so no new entry is taken in it;
* realised P&L and the equity curve are wrong;
* and the emergency square-off sends a sell order for it, which the
  broker rejects — that is how the panic button ended up failing.

So on every live tick, before anything else looks at the book, ask Kite
what it actually holds and close what it does not.

Deliberately conservative: this only ever *closes* book entries the
broker reports flat.  It never opens a trade AiVora did not take (a
manual position in the same account stays the user's business), and when
the broker's answer is ambiguous it logs loudly and changes nothing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from ..utils.logger import get_logger

log = get_logger(__name__)

# Written into exit_reason so these are distinguishable from AiVora's own
# exits when reviewing history.
EXIT_REASON = "closed_at_broker"


def _index(raw: dict) -> Dict[str, dict]:
    """Flatten Kite's ``{"net": [...], "day": [...]}`` into EXCH:SYMBOL → row.

    ``day`` is applied second so it wins: a contract bought and sold today
    appears there with quantity 0 and the ``sell_price`` we need to book
    the exit at, which is exactly the case we care about.
    """
    out: Dict[str, dict] = {}
    for bucket in ("net", "day"):
        for row in raw.get(bucket) or []:
            exch = row.get("exchange")
            sym = row.get("tradingsymbol")
            if not sym:
                continue
            out[f"{exch}:{sym}" if exch else str(sym)] = row
    return out


def _exit_price(row: dict, trade: dict) -> float:
    """What the broker actually got out at.

    ``sell_price`` is the average across today's sells of that contract —
    the real number when the exit happened outside AiVora.  Fall back to
    the last traded price, then to our own last mark, so a missing field
    never books an exit at zero.
    """
    for candidate in (row.get("sell_price"), row.get("last_price")):
        px = float(candidate or 0.0)
        if px > 0:
            return px
    return float(trade.get("current_premium") or trade.get("entry_premium") or 0.0)


def _close(portfolio, trade: dict, now: datetime, exit_px: float) -> None:
    from ..backtest.costs import compute_round_trip, live_cost_cfg

    lots = int(trade["lots"])
    lot_size = int(trade["lot_size"])
    entry = float(trade["entry_premium"])
    gross = (exit_px - entry) * lots * lot_size
    costs = compute_round_trip(
        entry_premium=entry,
        exit_premium=exit_px,
        lots=lots, lot_size=lot_size,
        cfg=live_cost_cfg(),
    )
    portfolio.close_trade(
        trade_id=trade["trade_id"],
        exit_time=now,
        exit_premium=exit_px,
        exit_reason=EXIT_REASON,
        gross_pnl=gross,
        costs=costs.total,
    )
    portfolio.append_log(
        f"↔️ {trade.get('symbol')} {trade.get('side')} was closed at your broker "
        f"(not by AiVora) — booked at ₹{exit_px:.2f}, "
        f"P&L ₹{gross - costs.total:+,.0f}",
        "warn",
    )


def reconcile_live_book(portfolio, kite, now: Optional[datetime] = None) -> int:
    """Close book entries the broker no longer holds.  Returns how many.

    Safe to call on every tick: one ``positions()`` request, and a no-op
    when the book is already flat or already agrees with the account.
    """
    now = now or datetime.now()
    state = portfolio.load()
    if state["mode"] != "live":
        return 0

    open_trades = [t for t in state["trades"] if not t.get("exit_time")]
    if not open_trades:
        return 0

    try:
        raw = kite.positions()
    except Exception as exc:  # noqa: BLE001
        # Never guess when the broker is unreachable — an empty answer and
        # a failed call look identical, and one of them would close the
        # whole book.
        portfolio.append_log(
            f"Could not read broker positions ({exc}) — book not reconciled",
            "warn",
        )
        return 0

    book = _index(raw)

    # Group by contract: the broker nets everything on one tradingsymbol
    # into a single row, so per-trade comparison is not possible.
    by_symbol: Dict[str, List[dict]] = {}
    for t in open_trades:
        ts = t.get("tradingsymbol")
        if not ts:
            # Opened before the contract was recorded.  Re-deriving it risks
            # matching the wrong strike and closing a live position that is
            # genuinely open, so leave it alone and say so.
            portfolio.append_log(
                f"{t.get('symbol')} {t.get('side')} has no contract recorded — "
                "cannot check it against your broker",
                "warn",
            )
            continue
        by_symbol.setdefault(ts, []).append(t)

    closed = 0
    for ts, trades in by_symbol.items():
        row = book.get(ts)
        our_qty = sum(int(t["lots"]) * int(t["lot_size"]) for t in trades)

        if row is None:
            # A contract traded today should always appear in day positions.
            # Absent means something we do not understand — say so, touch
            # nothing.
            portfolio.append_log(
                f"{ts} is open in AiVora but your broker does not list it — "
                "check your positions manually",
                "warn",
            )
            continue

        broker_qty = int(row.get("quantity") or 0)
        if broker_qty >= our_qty:
            continue                      # account agrees, or holds more

        if broker_qty > 0:
            # Partially exited elsewhere.  Splitting one book trade across a
            # partial fill is guesswork; flag it and let the user decide.
            portfolio.append_log(
                f"{ts}: your broker holds {broker_qty} but AiVora expects "
                f"{our_qty} — partially exited elsewhere, leaving it open",
                "warn",
            )
            continue

        exit_px = _exit_price(row, trades[0])
        for t in trades:
            _close(portfolio, t, now, exit_px)
            closed += 1

    if closed:
        log.warning("reconciled %d trade(s) closed outside AiVora", closed)
    return closed
