"""Every-5-minute tick loop.

Wraps APScheduler's ``BackgroundScheduler`` so the Streamlit app
can start it once (via ``ensure_started()``) and forget about it.
The scheduler:

1. Fetches the latest 5-min spot candles from KiteClient.
2. Upserts them into ``spot_futures``.
3. Rebuilds the training Parquet (the feature engineer is
   idempotent - cheap enough at 5-min cadence).
4. Loads the frozen UP/DOWN pair, produces the latest prediction
   per symbol.
5. Runs the position tracker (updates unrealized P&L, exits).
6. If master switch is ON and the signal fires, opens a new
   trade via the paper or live executor.
7. Writes ``last_data_update`` / ``last_signal_time`` on the
   portfolio so the UI has fresh timestamps.

Failures at any step are logged, appended to the portfolio's
banner log, and the tick continues (the next one might succeed).
"""

from __future__ import annotations

import threading
from datetime import datetime, time as dt_time
from typing import Dict, Optional

import pandas as pd

from ..pipeline import data_cleaning, database, feature_engineering
from ..utils.calendar import is_trading_day
from ..utils.config import get_config
from ..utils.logger import get_logger
from .inference import LiveInference
from .kite_client import KiteClient
from .paper_executor import open_paper_trade
from .portfolio import Portfolio
from .position_tracker import tick as position_tick

log = get_logger(__name__)

_STATE_LOCK = threading.RLock()
_SCHEDULER = None       # single BackgroundScheduler shared across imports
_LAST_TICK: Optional[datetime] = None


# =============================================================
#  Tick body — runnable standalone for tests
# =============================================================
def run_tick(portfolio: Portfolio, now: Optional[datetime] = None) -> Dict:
    now = now or datetime.now()
    if not is_trading_day(now.date()):
        return {"skipped": "non-trading-day"}
    if not (dt_time(9, 15) <= now.time() <= dt_time(15, 30)):
        return {"skipped": "off-session"}

    cfg = get_config()
    report: Dict = {"tick_time": now.isoformat(timespec="seconds"), "actions": []}

    # ---- 1. pull spot candles ----
    try:
        kite = KiteClient()
        rows_per_sym = {}
        for inst in cfg.instruments:
            sym = inst["symbol"]
            df = kite.fetch_recent_spot(sym, days_back=2)
            if df.empty:
                continue
            cleaned = data_cleaning.clean(df)
            n = database.upsert_spot_futures(cleaned)
            rows_per_sym[sym] = n
        portfolio.set_last_data_update(now)
        report["upserted"] = rows_per_sym
    except Exception as exc:
        log.exception("scheduler: spot fetch failed")
        portfolio.append_log(f"scheduler: spot fetch failed - {exc}", "error")
        return {**report, "error": f"spot fetch: {exc}"}

    # ---- 2. rebuild parquet ----
    try:
        spot = database.load_spot_futures()
        opts = database.load_option_chain()
        merged = spot if opts.empty else pd.merge(spot, opts, on=["datetime", "symbol"], how="left")
        if opts.empty:
            for c in ("ce_ltp", "pe_ltp", "ce_oi", "pe_oi", "ce_iv"):
                merged[c] = pd.NA
        merged = merged.rename(columns={"ce_iv": "iv"})
        feats = feature_engineering.engineer_features(merged)
        feats.to_parquet(cfg.paths["parquet_path"], index=False)
    except Exception as exc:
        log.exception("scheduler: feature rebuild failed")
        portfolio.append_log(f"scheduler: feature rebuild failed - {exc}", "error")
        return {**report, "error": f"features: {exc}"}

    # ---- 3-4. inference + spot map for tracker ----
    inf = LiveInference()
    spot_prices: Dict[str, float] = {}
    predictions: Dict[str, Dict] = {}
    for inst in cfg.instruments:
        sym = inst["symbol"]
        try:
            result = inf.latest_prediction(sym)
            if result is None:
                continue
            spot_prices[sym] = result.spot_close
            predictions[sym] = {
                "p_up": result.p_up, "p_down": result.p_down,
                "p_flat": result.p_flat,
                "minutes_since_open": result.minutes_since_open,
                "vol_regime_pct": result.vol_regime_pct,
                "spot_close": result.spot_close,
                "row_time": str(result.row_time),
            }
        except FileNotFoundError as exc:
            portfolio.append_log(
                "Frozen model missing - skipping inference. "
                "Run `python -m scripts.freeze_model`.", "error"
            )
            return {**report, "error": f"model: {exc}"}
        except Exception as exc:
            log.warning("inference for %s failed: %s", sym, exc)
    report["predictions"] = predictions

    # ---- 5. tracker updates open trades and closes any that hit exit rules ----
    try:
        position_tick(portfolio, now, spot_prices, kite=kite if portfolio.load()["mode"] == "live" else None)
    except Exception as exc:
        log.exception("scheduler: position_tick failed")
        portfolio.append_log(f"scheduler: position tracker error - {exc}", "error")

    # ---- 6. entry ----
    state = portfolio.load()
    if not state["master_switch"]:
        report["entry_skipped"] = "master_switch_off"
        return report

    settings = state["settings"]
    trades_today = sum(
        1 for t in state["trades"]
        if str(t.get("entry_time", ""))[:10] == now.date().isoformat()
    )
    if trades_today >= int(settings.get("max_trades_per_day", 3)):
        report["entry_skipped"] = "max_trades_per_day"
        return report

    # Do not open a fresh entry on top of an existing open trade
    # for the same symbol - matches backtest behaviour.
    open_syms = {
        t["symbol"] for t in state["trades"] if not t.get("exit_time")
    }
    for sym in list(predictions):
        if sym in open_syms:
            continue
        result = inf.latest_prediction(sym)
        if result is None:
            continue
        side = inf.signal_side(result, settings)
        if side is None:
            continue
        # Optional: fetch a live option quote for a realistic paper fill.
        try:
            q = kite.atm_option_quote(sym, result.spot_close)
            live_ce, live_pe = q["ce_ltp"], q["pe_ltp"]
        except Exception as exc:
            log.warning("quote fetch for %s failed: %s", sym, exc)
            live_ce = live_pe = None
        # Paper vs live entry.
        if state["mode"] == "live":
            from .live_executor import open_live_trade
            try:
                trade = open_live_trade(
                    portfolio, kite, sym, side, result.spot_close,
                    now, live_ce or 0.0, live_pe or 0.0,
                )
                report["actions"].append({"entered_live": bool(trade), "symbol": sym, "side": side})
            except Exception as exc:
                portfolio.append_log(f"LIVE entry error {sym} {side}: {exc}", "error")
        else:
            trade = open_paper_trade(
                portfolio, sym, side, result.spot_close, now,
                live_ce_ltp=live_ce, live_pe_ltp=live_pe,
            )
            report["actions"].append({"entered_paper": True, "symbol": sym, "side": side})
    state = portfolio.load()
    state["last_signal_time"] = now.isoformat(timespec="seconds")
    portfolio.save(state)
    return report


# =============================================================
#  BackgroundScheduler wrapper
# =============================================================
def ensure_started(portfolio: Portfolio, interval_seconds: int = 300) -> None:
    """Start APScheduler if it isn't already.  Safe to call every render."""
    global _SCHEDULER
    with _STATE_LOCK:
        if _SCHEDULER is not None and _SCHEDULER.running:
            return
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
        except ImportError as exc:
            log.error("apscheduler not installed: %s", exc)
            raise

        _SCHEDULER = BackgroundScheduler(daemon=True, timezone="Asia/Kolkata")

        def _job():
            global _LAST_TICK
            try:
                run_tick(portfolio)
                _LAST_TICK = datetime.now()
            except Exception as exc:  # protect the scheduler thread
                log.exception("scheduler: tick crashed - %s", exc)

        _SCHEDULER.add_job(_job, "interval", seconds=int(interval_seconds), id="aivora-tick")
        _SCHEDULER.start()
        log.warning("Scheduler started (every %ds)", interval_seconds)


def stop() -> None:
    global _SCHEDULER
    with _STATE_LOCK:
        if _SCHEDULER is not None and _SCHEDULER.running:
            _SCHEDULER.shutdown(wait=False)
            _SCHEDULER = None
            log.warning("Scheduler stopped")


def last_tick() -> Optional[datetime]:
    return _LAST_TICK
