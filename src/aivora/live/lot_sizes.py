"""Contract size per lot — asked of the exchange, not read from a file.

NSE revises index-option lot sizes, and config.yaml carried NIFTY 75 and
BANKNIFTY 15 long after they had become 65 and 30.  Nothing ever checked,
so the stale numbers did quiet damage:

* every backtested BANKNIFTY P&L was computed on half the real contract
  size, so the strategy's results were half of what those trades would
  actually have made or lost;
* and, worse, one BANKNIFTY lot looked like 21% of a 51,000 account when
  it is really 43%.  The backtest never saw a position that large, so its
  drawdowns were optimistic about a concentration that genuinely exists.

So live and paper ask Kite on every entry.  The value is cached for the
trading day — the instruments dump is already cached by KiteClient, and a
lot size does not change intraday.

config.yaml stays as a fallback for the two callers that have no broker to
ask: the backtest, and offline tests.  When both are available and they
disagree, that is logged as a warning, so the config can never silently
drift out of date again.
"""

from __future__ import annotations

import threading
from datetime import date
from typing import Dict, Optional, Tuple

from ..utils.config import get_config
from ..utils.logger import get_logger

log = get_logger(__name__)

_lock = threading.RLock()
_cache: Dict[Tuple[str, date], int] = {}
# Symbols already warned about today, so a mismatch is reported once a day
# rather than on every entry.
_warned: set = set()


def configured_lot_size(symbol: str) -> int:
    """The fallback value from config.yaml."""
    for inst in get_config().instruments:
        if inst["symbol"] == symbol:
            return int(inst["lot_size"])
    raise KeyError(f"{symbol} is not in config.yaml instruments")


def lot_size_for(symbol: str, kite=None, today: Optional[date] = None) -> int:
    """Units in one lot of ``symbol`` right now.

    Asks ``kite`` when one is given; falls back to config.yaml when there
    is no broker, or when the broker cannot answer.  Never raises on the
    broker path — a lot size that is stale by one revision is bad, but
    failing to trade at all because the instruments dump timed out is
    worse, and the fallback is at least a value someone checked.
    """
    day = today or date.today()
    key = (symbol, day)

    with _lock:
        if key in _cache:
            return _cache[key]

    fallback = configured_lot_size(symbol)
    if kite is None:
        return fallback

    try:
        live = int(kite.lot_size(symbol))
    except Exception as exc:  # noqa: BLE001
        log.warning("lot size for %s unavailable from Kite (%s) — "
                    "using config value %d", symbol, exc, fallback)
        return fallback

    if live <= 0:
        log.warning("Kite returned lot size %r for %s — using config value %d",
                    live, symbol, fallback)
        return fallback

    with _lock:
        _cache[key] = live
        if live != fallback and symbol not in _warned:
            _warned.add(symbol)
            log.warning(
                "lot size drift: config.yaml says %s=%d, the exchange says %d. "
                "Trading uses %d; update config.yaml so the backtest agrees.",
                symbol, fallback, live, live,
            )
    return live


def reset_cache() -> None:
    """Drop the day's cached values (tests, and a long-running worker that
    crosses midnight)."""
    with _lock:
        _cache.clear()
        _warned.clear()
