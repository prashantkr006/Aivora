"""Zerodha Kite Connect wrapper — sole broker for data and orders.

Responsibilities:

* Pull the latest 5-minute spot candles for NIFTY 50 and NIFTY BANK.
* Fetch live ATM CE/PE quotes and snapshot them per tick.
* Historical spot + option contract candles (with the Historical Data
  add-on subscription).
* Read available funds.
* Place / monitor / cancel orders.

Construction does not perform any network I/O; the underlying
KiteConnect instance is lazy so unit tests + import-time UI
renders stay offline.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..utils.config import KiteCredentials, get_config
from ..utils.logger import get_logger

log = get_logger(__name__)


# =============================================================
#  Rate limiter
# =============================================================
@dataclass
class _MinGap:
    gap_sec: float
    _last: float = 0.0

    def wait(self) -> None:
        delta = _time.time() - self._last
        if delta < self.gap_sec:
            _time.sleep(self.gap_sec - delta)
        self._last = _time.time()


# =============================================================
#  Client
# =============================================================
class KiteClient:
    """Thin, retrying wrapper over kiteconnect.KiteConnect."""

    # Cached NSE/NFO tokens for the two index spot underlyings.
    _SPOT_TRADINGSYMBOL = {
        "NIFTY":     ("NSE", "NIFTY 50"),
        "BANKNIFTY": ("NSE", "NIFTY BANK"),
    }

    def __init__(self, creds: Optional[KiteCredentials] = None):
        cfg = get_config()
        self.cfg = cfg
        self.creds = creds or cfg.kite_credentials()
        self._rate = _MinGap(gap_sec=1.0 / max(1, cfg.zerodha["rate_limit_per_sec"]))
        self._sdk = None
        self._instruments_cache: Dict[str, pd.DataFrame] = {}
        self._spot_token_cache: Dict[str, int] = {}

    # ---- Lazy SDK ----
    def _client(self):
        if self._sdk is not None:
            return self._sdk
        if not self.creds.api_key or not self.creds.access_token:
            raise RuntimeError(
                "Kite credentials missing. Set KITE_API_KEY and "
                "KITE_ACCESS_TOKEN in .env before using KiteClient."
            )
        from kiteconnect import KiteConnect  # local import keeps tests offline

        kite = KiteConnect(api_key=self.creds.api_key)
        kite.set_access_token(self.creds.access_token)
        self._sdk = kite
        log.info("Initialised Kite Connect SDK for api_key=%s...", self.creds.api_key[:6])
        return kite

    # ---- Retry ----
    @retry(
        reraise=True,
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
    )
    def _call(self, fn, *args, **kwargs):
        self._rate.wait()
        return fn(*args, **kwargs)

    # ---- Instruments dump (cached per exchange, per day) ----
    def instruments(self, exchange: str) -> pd.DataFrame:
        if exchange in self._instruments_cache:
            return self._instruments_cache[exchange]
        raw = self._call(self._client().instruments, exchange)
        df = pd.DataFrame(raw)
        self._instruments_cache[exchange] = df
        log.info("Kite instruments(%s): %d rows", exchange, len(df))
        return df

    def _spot_token(self, symbol: str) -> int:
        if symbol in self._spot_token_cache:
            return self._spot_token_cache[symbol]
        exch, sym = self._SPOT_TRADINGSYMBOL[symbol]
        df = self.instruments(exch)
        rows = df[df["tradingsymbol"].str.upper() == sym.upper()]
        if rows.empty:
            raise RuntimeError(f"Kite: no instrument found for {exch}:{sym}")
        tok = int(rows.iloc[0]["instrument_token"])
        self._spot_token_cache[symbol] = tok
        return tok

    # =========================================================
    #  Spot candles
    # =========================================================
    def spot_candles(
        self,
        symbol: str,
        from_dt: datetime,
        to_dt: datetime,
        interval: str = "5minute",
    ) -> pd.DataFrame:
        """Return an OHLCV dataframe for the spot index between two datetimes.

        Columns match what ``data_ingestion.SPOT_SCHEMA`` expects:
        ``datetime, symbol, spot_open/high/low/close, volume`` +
        NaN futures columns.
        """
        token = self._spot_token(symbol)
        # Kite caps historical calls at ~60 days for 5-minute data.
        # We only ever ask for a few days so no chunking needed.
        raw = self._call(
            self._client().historical_data, token, from_dt, to_dt, interval,
        )
        if not raw:
            return pd.DataFrame(columns=[
                "datetime", "symbol",
                "spot_open", "spot_high", "spot_low", "spot_close",
                "fut_open", "fut_high", "fut_low", "fut_close", "volume",
            ])
        df = pd.DataFrame(raw)
        df = df.rename(columns={
            "date": "datetime",
            "open": "spot_open", "high": "spot_high",
            "low": "spot_low", "close": "spot_close",
        })
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
        df["symbol"] = symbol
        for c in ("fut_open", "fut_high", "fut_low", "fut_close"):
            df[c] = pd.NA
        cols = ["datetime", "symbol",
                "spot_open", "spot_high", "spot_low", "spot_close",
                "fut_open", "fut_high", "fut_low", "fut_close", "volume"]
        return df[cols]

    def fetch_recent_spot(self, symbol: str, days_back: int = 2) -> pd.DataFrame:
        """Convenience: last ``days_back`` trading days of 5-min candles."""
        now = datetime.now()
        start = now - timedelta(days=days_back + 4)  # weekend buffer
        return self.spot_candles(symbol, start.replace(hour=9, minute=15), now)

    # =========================================================
    #  Live quotes & ATM options
    # =========================================================
    def atm_option_symbols(self, symbol: str, spot: float) -> Dict[str, str]:
        """Resolve the tradingsymbols for ATM CE + PE at the nearest expiry.

        Returns ``{"CE": "NFO:NIFTY25JULxxxxxCE", "PE": "..."}``.
        """
        step = 50 if symbol == "NIFTY" else 100
        atm = round(spot / step) * step
        nfo = self.instruments("NFO")
        nfo = nfo[nfo["name"] == symbol].copy()
        nfo["expiry"] = pd.to_datetime(nfo["expiry"]).dt.date
        today = date.today()
        upcoming = nfo[nfo["expiry"] >= today]
        if upcoming.empty:
            raise RuntimeError(f"No upcoming Kite F&O expiries for {symbol}")
        exp = upcoming["expiry"].min()
        window = upcoming[upcoming["expiry"] == exp]
        ce = window[(window["strike"] == atm) & (window["instrument_type"] == "CE")]
        pe = window[(window["strike"] == atm) & (window["instrument_type"] == "PE")]
        if ce.empty or pe.empty:
            raise RuntimeError(f"ATM {atm} not found for {symbol} exp {exp}")
        return {
            "CE": f"NFO:{ce.iloc[0]['tradingsymbol']}",
            "PE": f"NFO:{pe.iloc[0]['tradingsymbol']}",
            "expiry": str(exp),
            "atm_strike": float(atm),
            "lot_size": int(ce.iloc[0]["lot_size"]),
        }

    def lot_size(self, symbol: str) -> int:
        """Units in one lot of ``symbol``, at the nearest expiry.

        The exchange's own answer, from the instruments dump this client
        already caches.  Nearest expiry specifically: when NSE revises a
        lot size, the new value appears on newly listed expiries first, and
        the nearest one is what an entry today would actually trade.
        """
        nfo = self.instruments("NFO")
        rows = nfo[nfo["name"] == symbol].copy()
        if rows.empty:
            raise RuntimeError(f"Kite: no NFO contracts found for {symbol}")
        rows["expiry"] = pd.to_datetime(rows["expiry"]).dt.date
        upcoming = rows[rows["expiry"] >= date.today()]
        if upcoming.empty:
            raise RuntimeError(f"No upcoming Kite F&O expiries for {symbol}")
        nearest = upcoming[upcoming["expiry"] == upcoming["expiry"].min()]
        return int(nearest.iloc[0]["lot_size"])

    def atm_option_quote(self, symbol: str, spot: float) -> Dict[str, Any]:
        """LTP / OI / IV(if available) for the ATM CE + PE."""
        info = self.atm_option_symbols(symbol, spot)
        q = self._call(self._client().quote, [info["CE"], info["PE"]])
        ce_q = q[info["CE"]]
        pe_q = q[info["PE"]]
        return {
            "expiry": info["expiry"],
            "atm_strike": info["atm_strike"],
            "lot_size": info["lot_size"],
            "ce_tradingsymbol": info["CE"],
            "pe_tradingsymbol": info["PE"],
            "ce_ltp": float(ce_q.get("last_price") or 0.0),
            "pe_ltp": float(pe_q.get("last_price") or 0.0),
            "ce_oi": float(ce_q.get("oi") or 0.0),
            "pe_oi": float(pe_q.get("oi") or 0.0),
        }

    def ltp_for(self, tradingsymbol: str) -> float:
        """Last traded price for one exact contract.

        Marking a position must ask about the contract we actually hold.
        Deriving it from the current spot instead returns whichever strike
        happens to be at-the-money now, which is a different option as soon
        as the market moves.
        """
        q = self._call(self._client().quote, [tradingsymbol])
        row = q.get(tradingsymbol) or {}
        return float(row.get("last_price") or 0.0)

    def positions(self) -> Dict[str, Any]:
        """Broker's own view of open positions.

        The authoritative record of what we hold and what it is worth —
        quantity, average price, last price and P&L straight from Kite.
        Used to reconcile our book against the account rather than
        inferring it.
        """
        return self._call(self._client().positions)

    # =========================================================
    #  Funds
    # =========================================================
    def available_funds(self) -> float:
        """Best-guess "tradable cash right now" from Kite's margin block.

        Kite exposes a few overlapping fields under ``available``:

            live_balance   — realtime cash + collateral, minus utilised;
                             what Zerodha's web dashboard shows as
                             "Margin available".  This is the field
                             that matches the number the user sees.
            cash           — cash brought forward from prior-day, does
                             NOT reflect same-day fund additions right
                             away.  We saw 0 here even when the user
                             had funded the account and Zerodha's UI
                             showed 50k Margin available.
            opening_balance— 0 on the day the deposit lands.

        We prefer ``live_balance``, then ``cash``, then the top-level
        ``net`` (kept as a last-ditch fallback since ``net`` includes
        unrealised P&L on open positions).
        """
        margins = self._call(self._client().margins, segment="equity")
        try:
            avail = margins.get("available") or {}
        except AttributeError:
            avail = {}
        for key in ("live_balance", "cash"):
            try:
                v = float(avail.get(key) or 0.0)
                if v > 0:
                    return v
            except (TypeError, ValueError):
                continue
        try:
            v = float(margins.get("net") or 0.0)
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass
        return 0.0

    # =========================================================
    #  Orders — LIVE ONLY
    # =========================================================
    def place_limit_buy(
        self,
        tradingsymbol: str,      # e.g. "NFO:NIFTY25JULxxxxxCE"
        quantity: int,           # already in shares (lots × lot_size)
        limit_price: float,
        tag: str = "aivora",
    ) -> str:
        """Place a LIMIT BUY order.  Returns Kite's ``order_id``."""
        exch, sym = tradingsymbol.split(":")
        kite = self._client()
        order_id = self._call(
            kite.place_order,
            variety=kite.VARIETY_REGULAR,
            exchange=exch,
            tradingsymbol=sym,
            transaction_type=kite.TRANSACTION_TYPE_BUY,
            quantity=int(quantity),
            product=kite.PRODUCT_MIS,
            order_type=kite.ORDER_TYPE_LIMIT,
            price=float(limit_price),
            validity=kite.VALIDITY_DAY,
            tag=tag[:20],
        )
        log.info("Kite LIMIT BUY placed: %s qty=%d px=%.2f id=%s",
                 tradingsymbol, quantity, limit_price, order_id)
        return str(order_id)

    def place_limit_sell(
        self,
        tradingsymbol: str,
        quantity: int,
        limit_price: float,
        tag: str = "aivora",
    ) -> str:
        exch, sym = tradingsymbol.split(":")
        kite = self._client()
        order_id = self._call(
            kite.place_order,
            variety=kite.VARIETY_REGULAR,
            exchange=exch,
            tradingsymbol=sym,
            transaction_type=kite.TRANSACTION_TYPE_SELL,
            quantity=int(quantity),
            product=kite.PRODUCT_MIS,
            order_type=kite.ORDER_TYPE_LIMIT,
            price=float(limit_price),
            validity=kite.VALIDITY_DAY,
            tag=tag[:20],
        )
        log.info("Kite LIMIT SELL placed: %s qty=%d px=%.2f id=%s",
                 tradingsymbol, quantity, limit_price, order_id)
        return str(order_id)

    def order_status(self, order_id: str) -> Dict[str, Any]:
        history = self._call(self._client().order_history, order_id)
        return history[-1] if history else {}

    def cancel_order(self, order_id: str) -> None:
        kite = self._client()
        self._call(kite.cancel_order, kite.VARIETY_REGULAR, order_id)
        log.info("Kite CANCEL order_id=%s", order_id)
