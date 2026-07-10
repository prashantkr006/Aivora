"""DhanHQ adapter (v2 SDK).

Thin wrapper over the official ``dhanhq`` Python SDK, exposing just
what the pipeline needs:

* :py:meth:`DhanClient.spot_intraday` — N-minute OHLCV history for the
  underlying index. Wraps ``dhan.intraday_minute_data``.

* :py:meth:`DhanClient.expired_option_intraday` — N-minute OHLC + OI/IV
  for an ATM (or ATM±N) option contract from an already-expired weekly
  series. Wraps ``dhan.expired_options_data`` (the ``/charts/rollingoption``
  endpoint). This is what makes multi-month option backfill possible —
  Dhan does the ATM/strike and contract resolution server-side, so no
  scrip-master lookup is needed.

* :py:meth:`DhanClient.load_full_historical_data` — top-level loader
  that walks every weekly expiry between two dates, pulls spot +
  CE/PE option history for each, and stitches it into the canonical
  wide schema the rest of the pipeline expects.

* :py:meth:`DhanClient.option_chain` / :py:meth:`expiry_list` /
  :py:meth:`atm_option_snapshot` — live option-chain snapshot, used by
  the daily-update path (unrelated to the expired-options endpoint).

Requirements & known quirks (reverse-engineered against the live API —
Dhan's docs and SDK examples disagree with each other and with actual
behaviour in a few places)
----------------------------------------------------------------------
* Requires a **Data APIs subscription** (₹499+tax/month, separate from
  basic account access) — without it every data call fails with
  ``DH-902``.
* ``instrument_type`` for :py:meth:`expired_option_intraday` must be
  ``"OPTIDX"``, *not* ``"INDEX"`` — the official SDK README example
  uses "INDEX" and silently returns empty data for every query.
* ``expiry_code=0`` is rejected by the server ("expiryCode is
  required") despite docs listing 0 as a valid value ("current/near
  expiry"). Only 1 ("next") and 2 ("far") reliably work — we try both,
  bracketing the request window tightly around the target expiry so
  "next/far relative to that window" resolves to the contract we want.
* The rolling-option endpoint rate-limits much harder than the general
  data API — expect ``DH-904`` after a handful of rapid calls. We
  throttle aggressively and retry with backoff on that specific error.

Token lifetime
--------------
The access token generated from web.dhan.co lasts 24 hours. The
client doesn't auto-refresh — when it expires, the next call raises;
:func:`scripts/refresh_dhan_token.py` documents the manual flow.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from dhanhq import DhanContext, dhanhq

from ..utils.calendar import nse_holidays
from ..utils.config import DhanCredentials, get_config
from ..utils.logger import get_logger

log = get_logger(__name__)


# =============================================================
#  Rate limiter — simple min-gap between calls
# =============================================================
@dataclass
class _MinGap:
    """Ensure consecutive calls are separated by at least ``gap_sec``."""

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
class DhanClient:
    """Adapter over ``dhanhq.dhanhq`` (v2 SDK, ``DhanContext``-based auth)."""

    def __init__(self, creds: Optional[DhanCredentials] = None):
        cfg = get_config()
        self.cfg = cfg
        self.creds = creds or cfg.dhan_credentials()

        rate = cfg.dhan["data_rate_per_sec"]
        self._data_gap = _MinGap(gap_sec=1.0 / rate)
        self._oc_gap = _MinGap(gap_sec=float(cfg.dhan["option_chain_min_gap_sec"]))
        self._expired_gap = _MinGap(gap_sec=float(cfg.dhan["expired_options_min_gap_sec"]))
        self._sdk = None  # lazy

    # ---- SDK instantiation ----
    def _client(self):
        if self._sdk is not None:
            return self._sdk
        if not self.creds.client_id or not self.creds.access_token:
            raise RuntimeError(
                "DhanHQ credentials missing. Set DHAN_CLIENT_ID and "
                "DHAN_ACCESS_TOKEN in .env before calling data endpoints."
            )
        context = DhanContext(self.creds.client_id, self.creds.access_token)
        self._sdk = dhanhq(context)
        log.info("Initialised DhanHQ SDK for client_id=%s", self.creds.client_id)
        return self._sdk

    # ---- retry / rate-limit wrapper ----
    def _call(self, fn, *args, _gap: Optional[_MinGap] = None, _max_retries: int = 5, **kwargs):
        """Apply the requested rate-limit, then invoke ``fn``.

        Retries with backoff specifically on Dhan's ``DH-904`` rate-limit
        error, since the rolling-option endpoint throttles harder than
        the documented general data-API rate.
        """
        gap = _gap or self._data_gap
        last_remarks = None
        for attempt in range(1, _max_retries + 1):
            gap.wait()
            resp = fn(*args, **kwargs)
            if isinstance(resp, dict) and resp.get("status") == "success":
                return resp
            remarks = (resp or {}).get("remarks")
            last_remarks = remarks
            err_code = remarks.get("error_code") if isinstance(remarks, dict) else None
            if err_code == "DH-904" and attempt < _max_retries:
                backoff = min(30.0, 2.0 ** attempt)
                log.warning(
                    "Dhan rate-limited (DH-904) — retrying in %.0fs (attempt %d/%d)",
                    backoff, attempt, _max_retries,
                )
                _time.sleep(backoff)
                continue
            raise RuntimeError(f"DhanHQ error: {remarks}")
        raise RuntimeError(f"DhanHQ error after {_max_retries} retries: {last_remarks}")

    # =========================================================
    #  Spot intraday OHLCV
    # =========================================================
    def spot_intraday(
        self,
        security_id: str,
        exchange_segment: str,
        instrument_type: str,
        from_dt: datetime,
        to_dt: datetime,
        interval_minutes: int = 5,
    ) -> pd.DataFrame:
        """Fetch N-minute spot/index OHLCV, chunked to stay under Dhan's
        ~90-day-per-call cap (we use 80 days to leave headroom)."""
        chunk = timedelta(days=80)
        cursor = from_dt
        frames: List[pd.DataFrame] = []

        while cursor < to_dt:
            chunk_end = min(cursor + chunk, to_dt)
            resp = self._call(
                self._client().intraday_minute_data,
                security_id=security_id,
                exchange_segment=exchange_segment,
                instrument_type=instrument_type,
                from_date=cursor.strftime("%Y-%m-%d"),
                to_date=chunk_end.strftime("%Y-%m-%d"),
                interval=interval_minutes,
            )
            frames.append(_normalise_candles(resp))
            cursor = chunk_end + timedelta(days=1)

        if not frames:
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
        out = pd.concat(frames, ignore_index=True)
        return out.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    # =========================================================
    #  Expired options (rolling-option endpoint)
    # =========================================================
    def expired_option_intraday(
        self,
        security_id: str,
        from_dt: date,
        to_dt: date,
        option_type: str,          # "CALL" / "PUT"
        expiry_flag: str = "WEEK",
        strike: str = "ATM",
        interval_minutes: int = 5,
    ) -> pd.DataFrame:
        """Fetch N-minute OHLC + OI/IV for an expired ATM option.

        Tries ``expiry_code=1`` then ``2`` (0 is rejected by the
        server) — bracketing ``from_dt``/``to_dt`` tightly around the
        target week is what makes the right expiry resolve.
        """
        for expiry_code in (1, 2):
            resp = self._call(
                self._client().expired_options_data,
                security_id=security_id,
                exchange_segment="NSE_FNO",
                instrument_type="OPTIDX",
                expiry_flag=expiry_flag,
                expiry_code=expiry_code,
                strike=strike,
                drv_option_type=option_type,
                required_data=["open", "high", "low", "close", "volume", "oi", "iv", "spot", "strike"],
                interval=interval_minutes,
                from_date=from_dt.isoformat(),
                to_date=to_dt.isoformat(),
                _gap=self._expired_gap,
            )
            df = _normalise_expired_option(resp)
            if not df.empty:
                return df
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "oi", "iv", "spot", "strike"])

    # =========================================================
    #  Live option chain (used by the daily-update path only)
    # =========================================================
    def expiry_list(self, under_security_id: str, under_segment: str) -> List[str]:
        resp = self._call(
            self._client().expiry_list,
            under_security_id=under_security_id,
            under_exchange_segment=under_segment,
        )
        data = (resp or {}).get("data", {})
        if isinstance(data, dict):
            return list(data.get("expiry_list", []))
        return list(data)

    def option_chain(self, under_security_id: str, under_segment: str, expiry: str) -> pd.DataFrame:
        resp = self._call(
            self._client().option_chain,
            under_security_id=under_security_id,
            under_exchange_segment=under_segment,
            expiry=expiry,
            _gap=self._oc_gap,
        )
        return _flatten_option_chain(resp, expiry)

    def atm_option_snapshot(
        self,
        under_security_id: str,
        under_segment: str,
        strike_step: int,
        expiry: Optional[str] = None,
    ) -> Dict[str, Any]:
        if expiry is None:
            expiries = self.expiry_list(under_security_id, under_segment)
            today = date.today().isoformat()
            future = sorted(e for e in expiries if e >= today)
            if not future:
                raise RuntimeError(f"No upcoming expiries for {under_security_id}/{under_segment}")
            expiry = future[0]

        chain = self.option_chain(under_security_id, under_segment, expiry)
        if chain.empty:
            raise RuntimeError(f"Empty option chain for {under_security_id} {expiry}")

        spot = float(chain["underlying_ltp"].iloc[0])
        atm = round(spot / strike_step) * strike_step
        atm_rows = chain[chain["strike"] == atm]
        if atm_rows.empty:
            atm = float(chain["strike"].iloc[(chain["strike"] - spot).abs().argmin()])
            atm_rows = chain[chain["strike"] == atm]
        ce = atm_rows[atm_rows["type"] == "CE"].head(1)
        pe = atm_rows[atm_rows["type"] == "PE"].head(1)

        def _val(df: pd.DataFrame, col: str) -> Any:
            return None if df.empty else df[col].iloc[0]

        return {
            "atm_strike": float(atm),
            "expiry": expiry,
            "underlying_ltp": spot,
            "ce_ltp": _val(ce, "ltp"),
            "pe_ltp": _val(pe, "ltp"),
            "ce_oi": _val(ce, "oi"),
            "pe_oi": _val(pe, "oi"),
            "ce_iv": _val(ce, "iv"),
            "pe_iv": _val(pe, "iv"),
        }

    # =========================================================
    #  Historical (multi-month) loader
    # =========================================================
    def load_full_historical_data(
        self,
        start_date: date,
        end_date: date,
        symbols: Optional[Iterable[str]] = None,
        cache_path: Optional[Path] = None,
        interval_minutes: Optional[int] = None,
    ) -> pd.DataFrame:
        """Walk every weekly expiry between ``start_date`` and
        ``end_date``, pulling spot + ATM CE/PE history for each, and
        stitch everything into the canonical spot+option schema.
        """
        cfg = get_config()
        interval_minutes = interval_minutes or int(cfg.market.get("candle_interval_minutes", 5))
        cache_path = cache_path or (cfg.paths["data_raw_dir"] / "dhan_historical.parquet")
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        symbols = list(symbols) if symbols else [i["symbol"] for i in cfg.instruments]
        expiries_all = _weekly_expiries(start_date, end_date)
        log.info(
            "load_full_historical_data: %d expiries x %d symbols, %s -> %s",
            len(expiries_all), len(symbols), start_date, end_date,
        )

        combined_per_symbol: List[pd.DataFrame] = []
        for symbol in symbols:
            inst = _instrument_for_symbol(symbol)
            log.info("[1/2] Spot history for %s", symbol)
            spot = self.spot_intraday(
                security_id=str(inst["dhan_security_id"]),
                exchange_segment=inst["dhan_segment"],
                instrument_type=inst["dhan_instrument_type"],
                from_dt=datetime.combine(start_date, datetime.min.time()),
                to_dt=datetime.combine(end_date, datetime.max.time()),
                interval_minutes=interval_minutes,
            )
            if spot.empty:
                log.error("No spot history for %s — skipping symbol", symbol)
                continue
            log.info("  spot rows fetched: %d", len(spot))

            log.info("[2/2] Option history for %s (per weekly expiry)", symbol)
            per_expiry_frames: List[pd.DataFrame] = []
            for i, expiry in enumerate(expiries_all, start=1):
                if expiry < start_date or expiry > end_date:
                    continue
                window_start = expiry - timedelta(days=4)   # Monday of expiry week
                log.info("  expiry %d/%d — %s", i, len(expiries_all), expiry)

                sides: Dict[str, pd.DataFrame] = {}
                for side, opt_type in (("CE", "CALL"), ("PE", "PUT")):
                    try:
                        df = self.expired_option_intraday(
                            security_id=str(inst["dhan_security_id"]),
                            from_dt=window_start,
                            to_dt=expiry,
                            option_type=opt_type,
                            interval_minutes=interval_minutes,
                        )
                    except Exception as exc:
                        log.warning("    %s %s %s — fetch failed (%s)", symbol, expiry, side, exc)
                        continue
                    if not df.empty:
                        sides[side] = df

                if not sides:
                    continue
                merged_sides = _merge_ce_pe(sides, symbol, expiry)
                per_expiry_frames.append(merged_sides)

            if per_expiry_frames:
                opts = pd.concat(per_expiry_frames, ignore_index=True)
                opts = opts.drop_duplicates(subset=["datetime", "symbol"], keep="last")
            else:
                opts = pd.DataFrame(
                    columns=["datetime", "symbol", "ce_ltp", "pe_ltp", "ce_oi", "pe_oi", "ce_iv", "pe_iv"]
                )

            spot_renamed = spot.rename(columns={
                "open": "spot_open", "high": "spot_high",
                "low": "spot_low", "close": "spot_close",
            })
            spot_renamed["symbol"] = symbol
            merged = pd.merge(spot_renamed, opts, on=["datetime", "symbol"], how="left")
            if "ce_iv" in merged.columns:
                merged["iv"] = merged["ce_iv"]
            combined_per_symbol.append(merged)
            log.info("  %s combined rows: %d", symbol, len(merged))

        if not combined_per_symbol:
            raise RuntimeError(
                "Historical load returned no data — check dates, credentials, and Data API subscription."
            )

        out = pd.concat(combined_per_symbol, ignore_index=True)
        out = out.sort_values(["symbol", "datetime"]).reset_index(drop=True)
        out.to_parquet(cache_path, index=False)
        log.info("load_full_historical_data: cached %d rows to %s", len(out), cache_path)
        return out


# =============================================================
#  Response normalisers
# =============================================================
def _normalise_candles(resp: Dict[str, Any]) -> pd.DataFrame:
    """Turn a Dhan intraday-candle response into a clean dataframe."""
    cols = ["datetime", "open", "high", "low", "close", "volume"]
    data = (resp or {}).get("data") or {}
    ts = data.get("timestamp") or []
    if not ts:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame({
        "datetime": pd.to_datetime(ts, unit="s"),
        "open":   data.get("open", []),
        "high":   data.get("high", []),
        "low":    data.get("low", []),
        "close":  data.get("close", []),
        "volume": data.get("volume", []),
    })
    # Dhan timestamps are genuine UTC epoch seconds (confirmed empirically:
    # epoch for 09:15 IST market open decodes to 03:45 under a naive UTC
    # interpretation) — add the IST offset to get real wall-clock time.
    df["datetime"] = df["datetime"] + pd.Timedelta(hours=5, minutes=30)
    return df[cols]


def _normalise_expired_option(resp: Dict[str, Any]) -> pd.DataFrame:
    """Turn an expired_options_data response into a clean dataframe.

    Response shape observed live: ``{"data": {"data": {"ce": {...}}}}``
    (double-nested) — we defensively unwrap either one or two levels.
    """
    d = (resp or {}).get("data") or {}
    inner = d.get("data") if "data" in d and "ce" not in d else d
    side = inner.get("ce") or inner.get("pe") or {}
    ts = side.get("timestamp") or []
    cols = ["datetime", "open", "high", "low", "close", "volume", "oi", "iv", "spot", "strike"]
    if not ts:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame({
        # Same genuine-UTC-epoch convention as the intraday endpoint —
        # add the IST offset to get real wall-clock time.
        "datetime": pd.to_datetime(ts, unit="s") + pd.Timedelta(hours=5, minutes=30),
        "open":  side.get("open", []),
        "high":  side.get("high", []),
        "low":   side.get("low", []),
        "close": side.get("close", []),
        "volume": side.get("volume", []),
        "oi":    side.get("oi", []),
        "iv":    side.get("iv", []),
        "spot":  side.get("spot", []),
        "strike": side.get("strike", []),
    })
    return df[cols]


def _merge_ce_pe(sides: Dict[str, pd.DataFrame], symbol: str, expiry: date) -> pd.DataFrame:
    """Combine a CE and/or PE dataframe into the wide per-timestamp schema."""
    parts = []
    for side, df in sides.items():
        renamed = df.rename(columns={
            "close": f"{side.lower()}_ltp",
            "oi": f"{side.lower()}_oi",
            "iv": f"{side.lower()}_iv",
            "strike": f"{side.lower()}_strike",
        })[["datetime", f"{side.lower()}_ltp", f"{side.lower()}_oi",
            f"{side.lower()}_iv", f"{side.lower()}_strike"]]
        parts.append(renamed.set_index("datetime"))
    wide = pd.concat(parts, axis=1, join="outer").reset_index()
    wide["symbol"] = symbol
    return wide


def _flatten_option_chain(resp: Dict[str, Any], expiry: str) -> pd.DataFrame:
    """Turn the live option-chain nested ``oc`` dict into a long-format dataframe."""
    data = (resp or {}).get("data") or {}
    underlying_ltp = float(data.get("last_price") or 0.0)
    oc = data.get("oc") or {}
    rows: List[Dict[str, Any]] = []
    for strike_str, sides in oc.items():
        try:
            strike = float(strike_str)
        except (TypeError, ValueError):
            continue
        for side_key, side_label in (("ce", "CE"), ("pe", "PE")):
            payload = sides.get(side_key) or {}
            if not payload:
                continue
            greeks = payload.get("greeks") or {}
            rows.append({
                "expiry": expiry,
                "strike": strike,
                "type": side_label,
                "ltp": payload.get("last_price"),
                "oi": payload.get("oi"),
                "iv": payload.get("implied_volatility"),
                "volume": payload.get("volume"),
                "delta": greeks.get("delta"),
                "gamma": greeks.get("gamma"),
                "theta": greeks.get("theta"),
                "vega": greeks.get("vega"),
                "underlying_ltp": underlying_ltp,
            })
    if not rows:
        return pd.DataFrame(columns=[
            "expiry", "strike", "type", "ltp", "oi", "iv", "volume",
            "delta", "gamma", "theta", "vega", "underlying_ltp",
        ])
    return pd.DataFrame(rows).sort_values(["strike", "type"]).reset_index(drop=True)


# =============================================================
#  Weekly expiry enumeration
# =============================================================
def _weekly_expiries(start: date, end: date, weekday: int = 3) -> List[date]:
    """Yield every weekly expiry date between ``start`` and ``end``.

    ``weekday=3`` is Thursday (Mon=0). If Thursday is an NSE holiday
    the expiry shifts to the previous trading day — matching NSE's
    actual practice.
    """
    holidays = nse_holidays()
    days_ahead = (weekday - start.weekday()) % 7
    cur = start + timedelta(days=days_ahead)
    out: List[date] = []
    while cur <= end:
        adj = cur
        for _ in range(5):
            if adj.weekday() < 5 and adj not in holidays:
                break
            adj -= timedelta(days=1)
        out.append(adj)
        cur += timedelta(days=7)
    return out


def _instrument_for_symbol(symbol: str) -> Dict:
    for inst in get_config().instruments:
        if inst["symbol"] == symbol:
            return inst
    raise KeyError(f"Unknown symbol: {symbol}")
