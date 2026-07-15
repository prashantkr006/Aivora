"""Multi-user trading engine.

Reuses every proven piece from the single-user scheduler:

    * ``KiteClient`` for spot candles + option quotes.
    * ``data_cleaning`` + ``feature_engineering`` for the parquet.
    * ``LiveInference`` for the frozen UP/DOWN model.
    * ``paper_executor`` + ``live_executor`` + ``position_tracker`` —
      these are already portfolio-agnostic: they call ``.load()``,
      ``.open_trade()``, ``.update_open_marks()``, ``.close_trade()``
      — all of which ``UserPortfolio`` implements with the same
      signatures as the file-based ``Portfolio``.

Design decisions:

    * **Market data is shared, trades are per-user.**  The
      ``spot_futures`` and ``options_chain`` SQLite tables and the
      training parquet are common to every user — the market data
      itself doesn't depend on who's watching.  A ``MarketDataCache``
      makes sure a 5-minute window fetches exactly once, even if
      three users all have master-switch ON.

    * **One user's crash never touches another user's job.**  Every
      exception inside ``run_user_tick`` is caught, logged to that
      user's event log, and swallowed.  The APScheduler ``max_instances=1``
      / ``coalesce=True`` settings on the per-user job handle
      overlap.

    * **Data-fetch uses the calling user's Kite connection.**  If
      their token is expired, cache refresh is skipped (falling back
      to the last cached snapshot) and the tick continues with what
      it has.  No user's stale token blocks other users.
"""

from __future__ import annotations

import threading
import time as _time
from datetime import datetime, time as dt_time, timedelta
from typing import Dict, Optional

import pandas as pd

from ..live import position_tracker as _tracker
from ..live.inference import InferenceResult, LiveInference
from ..live.kite_client import KiteClient
from ..live.paper_executor import open_paper_trade
from ..pipeline import data_cleaning, database, feature_engineering
from ..utils.calendar import is_trading_day
from ..utils.config import KiteCredentials, get_config
from ..utils.logger import get_logger
from . import brokers
from .portfolios import UserPortfolio

log = get_logger(__name__)


# =============================================================
#  Shim: bridge Portfolio.append_log ↔ UserPortfolio.log_event
# =============================================================
class _UserPortfolioShim:
    """Wrap a ``UserPortfolio`` so downstream code that calls
    ``.append_log()`` (the file-based Portfolio interface) transparently
    hits ``.log_event()`` instead.  Everything else passes through.
    """

    def __init__(self, up: UserPortfolio):
        self._up = up

    def __getattr__(self, name):
        return getattr(self._up, name)

    def append_log(self, msg: str, level: str = "info") -> None:
        self._up.log_event(msg, level)


# =============================================================
#  Shared market-data cache
# =============================================================
class MarketDataCache:
    """One fetch per 5-minute window, shared across users.

    Performance note
    ----------------
    The tick used to rebuild the entire feature parquet (~184k rows,
    90–120 s) on every 5-minute call. It now:

    1. Fetches new candles from Kite (unchanged).
    2. Upserts into the SQLite spot/options tables (unchanged).
    3. Loads a **slim window** — only the last
       :attr:`_LOOKBACK_DAYS` calendar days per symbol — from the DB.
    4. Runs the same :func:`engineer_features` on that slim window.
    5. Feeds the resulting dataframe directly to
       :meth:`LiveInference.latest_prediction_from_df` — no parquet
       round-trip.

    Why this is safe: the longest rolling window used by any feature
    is ``vol_regime_pct``'s 1500-candle (~20 trading days) percentile
    rank. As long as the slim window contains at least 1500 candles
    preceding the row being predicted, the rank at that row is
    **bit-identical** to what the full-history run would produce —
    ``pandas.rolling`` only looks at the last N values, so extending
    the window backwards past those N values changes nothing.

    Result: same 74 features, same predictions, ~30× faster.
    """

    _lock = threading.RLock()
    _last_fetch: Optional[datetime] = None
    _predictions: Dict[str, InferenceResult] = {}
    _spot_prices: Dict[str, float] = {}
    _last_error: Optional[str] = None
    _window_sec = 240   # slightly under 5 min so we refresh a beat early
    _inference = LiveInference()

    # Calendar-day window loaded from SQLite for the slim rebuild.
    # 60 days ≈ 43 trading days ≈ 3200 candles per symbol — comfortably
    # above the 1500-candle floor imposed by vol_regime_pct's rolling
    # rank, with plenty of headroom for weekends and holidays.
    _LOOKBACK_DAYS = 60

    @classmethod
    def snapshot(cls) -> Dict[str, InferenceResult]:
        with cls._lock:
            return dict(cls._predictions)

    @classmethod
    def spot_prices(cls) -> Dict[str, float]:
        with cls._lock:
            return dict(cls._spot_prices)

    @classmethod
    def last_error(cls) -> Optional[str]:
        with cls._lock:
            return cls._last_error

    @classmethod
    def _needs_refresh(cls, now: datetime) -> bool:
        if cls._last_fetch is None:
            return True
        return (now - cls._last_fetch).total_seconds() >= cls._window_sec

    @classmethod
    def refresh_if_stale(cls, kite: KiteClient, now: datetime) -> bool:
        """Fetch → clean → upsert → engineer features (slim window) → inference.

        Returns True if a refresh happened this call. Only the first
        caller inside the 5-min window actually does the work; the
        rest see the cached snapshot.
        """
        with cls._lock:
            if not cls._needs_refresh(now):
                return False
            t0 = _time.perf_counter()
            try:
                cfg = get_config()
                # ---- 1. new spot candles from Kite (~1-2s) ----
                for inst in cfg.instruments:
                    sym = inst["symbol"]
                    df = kite.fetch_recent_spot(sym, days_back=2)
                    if df.empty:
                        continue
                    cleaned = data_cleaning.clean(df)
                    database.upsert_spot_futures(cleaned)
                t_fetch = _time.perf_counter() - t0

                # ---- 1b. ATM option snapshot per symbol (Kite) ----
                # Keeps options_chain warm without Dhan. Failures here
                # never block the tick — options-derived features are only
                # ~10% of model importance and LightGBM handles NaN.
                t_before_opts = _time.perf_counter()
                try:
                    cls._snapshot_options(kite, now, cfg.instruments)
                except Exception as exc:  # noqa: BLE001
                    log.warning("option snapshot failed (skipping): %s", exc)
                t_opts = _time.perf_counter() - t_before_opts

                # ---- 2. slim-window load + feature engineering ----
                # A 60-day window is enough for every rolling feature
                # to produce identical values at the latest row (see
                # class-level docstring for the correctness proof).
                cutoff = pd.Timestamp(now) - timedelta(days=cls._LOOKBACK_DAYS)
                spot = database.load_spot_futures_since(cutoff)
                opts = database.load_option_chain_since(cutoff)
                if spot.empty:
                    raise RuntimeError(
                        "spot_futures window empty — has the historical load "
                        "been run? (`python -m scripts.run_historical_load`)"
                    )
                if opts.empty:
                    merged = spot
                    for c in ("ce_ltp", "pe_ltp", "ce_oi", "pe_oi", "ce_iv"):
                        merged[c] = pd.NA
                else:
                    merged = pd.merge(spot, opts, on=["datetime", "symbol"], how="left")
                merged = merged.rename(columns={"ce_iv": "iv"})
                feats = feature_engineering.engineer_features(merged)
                t_feats = _time.perf_counter() - t0 - t_fetch

                # ---- 3. inference per symbol (in-memory, no parquet) ----
                preds: Dict[str, InferenceResult] = {}
                spots: Dict[str, float] = {}
                for inst in cfg.instruments:
                    sym = inst["symbol"]
                    try:
                        r = cls._inference.latest_prediction_from_df(feats, sym)
                    except FileNotFoundError as exc:
                        cls._last_error = f"frozen model missing: {exc}"
                        raise
                    except Exception as exc:  # noqa: BLE001
                        log.warning("inference %s failed: %s", sym, exc)
                        continue
                    if r is None:
                        continue
                    preds[sym] = r
                    spots[sym] = r.spot_close
                t_total = _time.perf_counter() - t0

                cls._predictions = preds
                cls._spot_prices = spots
                cls._last_fetch = now
                cls._last_error = None
                log.info(
                    "MarketDataCache refreshed: %d predictions "
                    "(spot=%.2fs, opts=%.2fs, features=%.2fs, total=%.2fs; "
                    "slim window=%d rows)",
                    len(preds), t_fetch, t_opts, t_feats, t_total, len(feats),
                )
                return True
            except Exception as exc:  # noqa: BLE001
                cls._last_error = str(exc)
                log.exception("MarketDataCache refresh failed")
                raise

    @classmethod
    def _snapshot_options(cls, kite: KiteClient, now: datetime, instruments) -> None:
        """Snapshot the ATM CE + PE for every configured symbol and
        upsert into ``options_chain``. Called once per 5-min tick.

        This is what Dhan's ``run_daily_update`` used to do — now
        happens through Kite so no Dhan dependency in production.
        Snapshot timestamp is bucketed to the current 5-min mark so
        it aligns with the spot candle for the join.
        """
        # Bucket to the last 5-min boundary (matches spot candle labels).
        bucket = now.replace(second=0, microsecond=0)
        bucket = bucket.replace(minute=(bucket.minute // 5) * 5)

        rows = []
        for inst in instruments:
            sym = inst["symbol"]
            spot = cls._spot_prices.get(sym)
            if spot is None:
                # First tick of the process — fall back to querying Kite
                # spot LTP so we still get an option snapshot.
                try:
                    spot_df = kite.fetch_recent_spot(sym, days_back=1)
                    if not spot_df.empty:
                        spot = float(spot_df.iloc[-1]["spot_close"])
                except Exception:  # noqa: BLE001
                    continue
            if not spot:
                continue

            try:
                q = kite.atm_option_quote(sym, spot)
            except Exception as exc:  # noqa: BLE001
                log.warning("option quote %s failed: %s", sym, exc)
                continue

            strike = q["atm_strike"]
            for side, ltp_key, oi_key in (
                ("CE", "ce_ltp", "ce_oi"),
                ("PE", "pe_ltp", "pe_oi"),
            ):
                rows.append({
                    "datetime": bucket,
                    "symbol": sym,
                    "strike": strike,
                    "type": side,
                    "ltp": float(q.get(ltp_key) or 0.0),
                    "oi": float(q.get(oi_key) or 0.0),
                    "iv": None,  # Kite's quote endpoint doesn't include IV
                })

        if rows:
            df = pd.DataFrame(rows)
            database.upsert_option_chain(df)

    # ---- test helper ----
    @classmethod
    def _reset(cls) -> None:
        with cls._lock:
            cls._last_fetch = None
            cls._predictions = {}
            cls._spot_prices = {}
            cls._last_error = None


# =============================================================
#  Public: per-user tick
# =============================================================
def _build_kite_from_broker(zer) -> KiteClient:
    creds = KiteCredentials(
        api_key=zer.api_key or "",
        api_secret=zer.api_secret or "",
        access_token=zer.access_token or "",
        user_id=zer.client_id or "",
    )
    return KiteClient(creds=creds)


def _open_live_trade_shim(*args, **kwargs):
    """Import live_executor lazily so the module works even when
    kiteconnect isn't installed (tests / paper-only deployments)."""
    from ..live.live_executor import open_live_trade
    return open_live_trade(*args, **kwargs)


def run_user_tick(user_id: int, mode: str, now: Optional[datetime] = None) -> Dict:
    """One full 5-minute tick for one user.

    Wraps everything in try/except so a single user's failure —
    expired token, feature NaN, Kite outage — is contained to
    their event log and never breaks another user's job.
    """
    now = now or datetime.now()
    report: Dict = {"user_id": user_id, "mode": mode,
                    "tick_time": now.isoformat(timespec="seconds")}

    # Load portfolio FIRST so every early-exit path can still write a
    # heartbeat entry — silence looks identical to "the scheduler is
    # dead" from the user's side.
    try:
        portfolio = _UserPortfolioShim(UserPortfolio(user_id, mode))
    except Exception as exc:  # noqa: BLE001
        log.exception("portfolio load failed for user_id=%s", user_id)
        return {**report, "error": f"portfolio: {exc}"}

    # Session guards — quietly return; the heartbeat is only useful
    # during market hours, and off-session ticks would otherwise spam
    # the event log with irrelevant entries.
    hhmm = now.strftime("%H:%M")
    if not is_trading_day(now.date()):
        return {**report, "skipped": "non-trading-day"}
    if not (dt_time(9, 15) <= now.time() <= dt_time(15, 30)):
        return {**report, "skipped": "off-session"}

    # Load broker creds — encrypted at rest, decrypted here in memory only.
    zer = brokers.get(user_id, "ZERODHA")
    if not zer or not zer.access_token or not zer.api_key:
        portfolio.append_log(
            f"⚠️ {hhmm} — Kite disconnected. Reconnect via Profile page.",
            "warn",
        )
        return {**report, "skipped": "no-kite-token"}

    kite = _build_kite_from_broker(zer)

    # Refresh the shared market snapshot; if it fails we still tick
    # (with the LAST snapshot) so trades in flight can still exit.
    try:
        MarketDataCache.refresh_if_stale(kite, now)
        portfolio.set_last_data_update(now)
    except Exception as exc:  # noqa: BLE001
        portfolio.append_log(
            f"Market data refresh failed ({exc}); using cached snapshot.",
            "warn",
        )

    predictions = MarketDataCache.snapshot()
    spot_prices = MarketDataCache.spot_prices()
    report["predictions"] = list(predictions.keys())

    # Position tracker: update marks + close any hitting TP/SL/timeout.
    try:
        _tracker.tick(
            portfolio, now, spot_prices,
            kite=kite if mode == "live" else None,
        )
    except Exception as exc:  # noqa: BLE001
        portfolio.append_log(f"tracker error: {exc}", "error")

    # Entry logic — only fire if master switch is on.
    state = portfolio.load()
    open_ct = sum(1 for t in state["trades"] if not t.get("exit_time"))
    if not state["master_switch"]:
        portfolio.append_log(
            f"⏸️ {hhmm} — paused (master switch OFF). "
            f"{len(predictions)} predictions computed, {open_ct} open trade(s) monitored.",
            "info",
        )
        return {**report, "entry_skipped": "master_switch_off"}

    settings = state["settings"]
    today = now.date().isoformat()
    trades_today = sum(
        1 for t in state["trades"] if str(t.get("entry_time", ""))[:10] == today
    )
    if trades_today >= int(settings.get("max_trades_per_day", 3)):
        portfolio.append_log(
            f"🔒 {hhmm} — daily trade limit reached ({trades_today}/day). "
            "No new entries; existing trades still monitored.",
            "info",
        )
        return {**report, "entry_skipped": "max_trades_per_day"}

    open_syms = {t["symbol"] for t in state["trades"] if not t.get("exit_time")}
    actions = []
    considered = []  # human-readable per-symbol verdicts

    thr_up = float(settings.get("prob_threshold_up", 0.55))
    thr_dn = float(settings.get("prob_threshold_down", 0.55))
    vr_min = float(settings.get("vol_regime_min") or 0.0)
    vr_max = float(settings.get("vol_regime_max") or 1.0)
    # Cooldown defaults to OFF — the 30-day comparison (see
    # ``logs/cooldown_analysis.txt``) showed it costs ~₹1,725 of P&L
    # and 1.1 Sharpe units without a compensating drawdown reduction.
    # Left as portfolio settings so an individual user can turn it
    # back on (e.g. cooldown_prob_delta=0.05) if they want to
    # experiment.
    cd_prob_delta = float(settings.get("cooldown_prob_delta", 0.0))
    cd_price_pct = float(settings.get("cooldown_price_pct", 0.0))

    # Build "last exit today" map per symbol so we can enforce the
    # re-entry cooldown (skip only if horizon exit + unchanged
    # conviction + unchanged spot).
    today_str = now.date().isoformat()
    last_exit_by_sym: Dict[str, Dict] = {}
    for t in state["trades"]:
        et = str(t.get("exit_time", ""))
        if not et or et[:10] != today_str:
            continue
        prev = last_exit_by_sym.get(t["symbol"])
        if prev is None or et > prev.get("exit_time", ""):
            last_exit_by_sym[t["symbol"]] = {
                "exit_time": et,
                "reason": t.get("exit_reason"),
                "prob": t.get("entry_prob"),
                "spot": t.get("entry_spot"),
            }

    for sym, result in predictions.items():
        pu, pd, vr = result.p_up, result.p_down, result.vol_regime_pct
        if sym in open_syms:
            considered.append(f"{sym} 🔁 already holding a trade")
            continue
        side = MarketDataCache._inference.signal_side(result, settings)
        if side is None:
            # Explain WHY it didn't fire — the model's peak vs threshold.
            if not (vr_min <= vr <= vr_max):
                why = f"volatility {vr:.2f} outside [{vr_min:.2f}, {vr_max:.2f}]"
            elif pu >= pd and pu < thr_up:
                gap = thr_up - pu
                why = f"UP conviction {pu:.2f} — needs {thr_up:.2f} (short by {gap:.2f})"
            elif pd > pu and pd < thr_dn:
                gap = thr_dn - pd
                why = f"DOWN conviction {pd:.2f} — needs {thr_dn:.2f} (short by {gap:.2f})"
            else:
                why = f"gates blocked (p_up={pu:.2f} p_down={pd:.2f} vr={vr:.2f})"
            considered.append(f"{sym} 🚫 {why}")
            continue

        # Cooldown — only if last today's exit for this symbol was a
        # horizon timeout AND nothing meaningful changed since then.
        prev_exit = last_exit_by_sym.get(sym)
        if prev_exit and prev_exit.get("reason") == "horizon":
            prev_prob = prev_exit.get("prob")
            prev_spot = prev_exit.get("spot") or 0.0
            entry_prob_now = pu if side == "CE" else pd
            prob_close = (
                prev_prob is not None
                and abs(entry_prob_now - float(prev_prob)) < cd_prob_delta
            )
            price_close = (
                prev_spot
                and abs(result.spot_close - float(prev_spot)) / float(prev_spot)
                < cd_price_pct
            )
            if prob_close and price_close:
                gap_p = abs(entry_prob_now - float(prev_prob))
                gap_px = abs(result.spot_close - float(prev_spot)) / float(prev_spot)
                considered.append(
                    f"{sym} ⏳ cooldown (prob unchanged @ {entry_prob_now:.2f}, "
                    f"price {gap_px:+.2%})"
                )
                portfolio.append_log(
                    f"Skipping {sym} re-entry — cooldown (prob {gap_p:+.2f}, "
                    f"price {gap_px:+.2%})",
                    "info",
                )
                continue

        # Try to fetch a live quote to make the paper fill realistic.
        live_ce = live_pe = None
        try:
            q = kite.atm_option_quote(sym, result.spot_close)
            live_ce = q["ce_ltp"]
            live_pe = q["pe_ltp"]
        except Exception as exc:  # noqa: BLE001
            log.warning("live quote for %s failed: %s", sym, exc)

        try:
            if mode == "live":
                trade = _open_live_trade_shim(
                    portfolio, kite, sym, side, result.spot_close, now,
                    live_ce or 0.0, live_pe or 0.0,
                )
                actions.append({"entered_live": bool(trade), "symbol": sym, "side": side})
            else:
                entry_prob_now = pu if side == "CE" else pd
                open_paper_trade(
                    portfolio, sym, side, result.spot_close, now,
                    live_ce_ltp=live_ce, live_pe_ltp=live_pe,
                    entry_prob=entry_prob_now,
                )
                actions.append({"entered_paper": True, "symbol": sym, "side": side})
        except Exception as exc:  # noqa: BLE001
            portfolio.append_log(f"entry error {sym} {side}: {exc}", "error")

    # Single, human-readable summary of the tick.  One entry per tick
    # (not two like before) — keeps the event log signal-dense.
    if actions:
        entered_bits = []
        for a in actions:
            sym = a.get("symbol", "?")
            side = a.get("side", "?")
            side_txt = "CALL" if side == "CE" else "PUT"
            entered_bits.append(f"{sym} {side_txt}")
        headline = "🚀 " + ", ".join(entered_bits) + " — trade opened"
    else:
        headline = f"✅ {hhmm} — checked, nothing to trade"
    portfolio.append_log(
        headline + "  |  " + "  •  ".join(considered),
        "info",
    )

    return {**report, "actions": actions}
