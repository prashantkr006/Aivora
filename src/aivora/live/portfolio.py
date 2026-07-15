"""Portfolio state — the single source of truth.

Everything the UI displays (capital, P&L, trade table, equity
curve, drawdown, win rate) is derived from this file — nothing
lives in an unpersisted variable.  The scheduler writes, the UI
reads, an atomic swap keeps them from colliding.

The state is a plain JSON blob:

    {
      "mode": "paper" | "live",
      "initial_capital": 100000.0,
      "current_capital": 100540.20,
      "peak_capital": 101200.0,
      "master_switch": true,
      "settings": {...},                 # strategy knobs the UI edits
      "last_data_update": "ISO-8601 or null",
      "last_signal_time": "ISO-8601 or null",
      "trades": [Trade, ...],
      "log": [{"ts", "level", "msg"}]    # last N banner-worthy events
    }

Math invariant we enforce on every save:

    current_capital == initial_capital
                       + sum(t.realized_pnl for t in closed_trades)

Unrealized P&L is reported separately so the invariant stays
exactly checkable.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional

from ..utils.config import get_config
from ..utils.logger import get_logger

log = get_logger(__name__)

_LOCK = RLock()


# =============================================================
#  Trade
# =============================================================
@dataclass
class Trade:
    trade_id: str
    entry_time: str                      # ISO-8601
    symbol: str                          # "NIFTY" / "BANKNIFTY"
    side: str                            # "CE" / "PE"
    strike: float
    lots: int
    lot_size: int
    entry_premium: float
    exit_premium: Optional[float] = None
    exit_time: Optional[str] = None
    exit_reason: Optional[str] = None    # take_profit | stop_loss | horizon | emergency | manual
    entry_spot: Optional[float] = None
    exit_spot: Optional[float] = None
    # Live-order metadata (paper leaves these None).
    entry_order_id: Optional[str] = None
    exit_order_id: Optional[str] = None
    # For live-inference tick monitoring.
    horizon_close_time: Optional[str] = None   # when to force-exit if TP/SL don't hit
    # Populated on close.
    gross_pnl: Optional[float] = None
    costs: Optional[float] = None
    realized_pnl: Optional[float] = None
    # Kept fresh by the position tracker while open.
    current_premium: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    # ---- Trailing SL + cooldown state ----
    entry_prob: Optional[float] = None       # p_up / p_down that triggered entry
    trailing_sl_price: Optional[float] = None  # dynamic stop; monotonically rises
    peak_premium: Optional[float] = None       # highest premium seen since entry

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    @property
    def status(self) -> str:
        return "OPEN" if self.is_open else "CLOSED"


# =============================================================
#  Default settings — mirror the winning backtest variant #18
# =============================================================
def default_settings() -> Dict[str, Any]:
    return {
        "prob_threshold_up": 0.55,
        "prob_threshold_down": 0.55,  # was 0.60 — 30-day sensitivity test showed
                                      # ₹21k→₹60k P&L jump (44→134 trades) at 0.55.
                                      # Symmetric with UP now; both gate at 55 %.
        "take_profit_pct": 0.60,
        "stop_loss_pct": 0.30,
        "min_minutes_since_open": 0,       # 09:15 — opening window opened after
                                           # 55-mo WF confirmed +19% P&L
                                           # (₹13.5L → ₹16.1L) with better Sharpe.
        "max_minutes_since_open": 300,     # 14:15 — CLOSING kept blocked;
                                           # WF numbers past 14:15 are backtest
                                           # artifacts (synthetic post-market
                                           # exits), 3-mo dominate 55% of P&L.
        "vol_regime_min": 0.15,
        "vol_regime_max": 0.90,
        "max_trades_per_day": 10,
        "horizon_candles": 12,             # 60-minute forward horizon
        "risk_per_trade_pct": 0.02,
        "brokerage_flat_per_order": 20.0,
        "brokerage_pct_cap": 0.0003,
        "stt_pct_sell": 0.001,
        "sebi_pct": 0.00001,
        "exchange_txn_pct": 0.0003503,
        "stamp_duty_pct_buy": 0.00003,
        "gst_pct": 0.18,
        "slippage_pct": 0.001,
        # ---- Re-entry cooldown ----
        # Both set to 0.0 means the cooldown never blocks.  Rationale:
        # the 30-day comparison (see logs/cooldown_analysis.txt) showed
        # cooldown ON removes ~₹1,725 of P&L and 1.1 Sharpe units over
        # the sample.  Left tunable so an individual user can opt in.
        "cooldown_prob_delta": 0.0,
        "cooldown_price_pct": 0.0,
    }


# =============================================================
#  Portfolio
# =============================================================
class Portfolio:
    """Read-modify-write facade over a JSON file.

    Two portfolios coexist: ``paper_portfolio.json`` and
    ``live_portfolio.json``.  Switching mode swaps which file the
    UI + scheduler point at — the other keeps its state intact.
    """

    def __init__(self, mode: str = "paper", path: Optional[Path] = None):
        assert mode in ("paper", "live")
        self.mode = mode
        self.path = Path(path) if path else self._default_path(mode)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._init_blank_file()

    # -------------------- load / save --------------------
    @staticmethod
    def _default_path(mode: str) -> Path:
        cfg = get_config()
        return cfg.paths["db_dir"].parent / f"{mode}_portfolio.json"

    def _init_blank_file(self) -> None:
        cap = float(get_config().project["base_capital"])
        state = {
            "mode": self.mode,
            "initial_capital": cap,
            "current_capital": cap,
            "peak_capital": cap,
            "master_switch": False,
            "settings": default_settings(),
            "last_data_update": None,
            "last_signal_time": None,
            "trades": [],
            "log": [],
        }
        self._atomic_write(state)
        log.info("Initialised blank portfolio at %s", self.path)

    def load(self) -> Dict[str, Any]:
        with _LOCK:
            return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, state: Dict[str, Any]) -> None:
        with _LOCK:
            self._recompute_invariants(state)
            self._verify_invariants(state)
            self._atomic_write(state)

    def _atomic_write(self, state: Dict[str, Any]) -> None:
        """Write to a temp file in the same dir, then os.replace() —
        atomic on both POSIX and Windows so readers never see a
        half-written file."""
        tmp_dir = self.path.parent
        fd, tmp_path = tempfile.mkstemp(prefix=".portfolio.", suffix=".tmp", dir=tmp_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2, default=str)
            os.replace(tmp_path, self.path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # -------------------- math invariants --------------------
    @staticmethod
    def _recompute_invariants(state: Dict[str, Any]) -> None:
        """Re-derive ``current_capital`` and ``peak_capital`` from trades."""
        closed = [t for t in state["trades"] if t.get("exit_time")]
        realized = sum(float(t.get("realized_pnl") or 0.0) for t in closed)
        state["current_capital"] = float(state["initial_capital"]) + realized
        state["peak_capital"] = max(
            float(state.get("peak_capital") or state["initial_capital"]),
            state["current_capital"],
        )

    @staticmethod
    def _verify_invariants(state: Dict[str, Any]) -> None:
        closed = [t for t in state["trades"] if t.get("exit_time")]
        realized = sum(float(t.get("realized_pnl") or 0.0) for t in closed)
        expected = float(state["initial_capital"]) + realized
        actual = float(state["current_capital"])
        if abs(actual - expected) > 1e-6:
            raise RuntimeError(
                f"Portfolio invariant violated: current_capital={actual} "
                f"expected={expected} (initial={state['initial_capital']} "
                f"+ realized={realized})"
            )

    # -------------------- convenience API --------------------
    def append_log(self, msg: str, level: str = "info", max_entries: int = 200) -> None:
        state = self.load()
        state["log"].append({
            "ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "level": level,
            "msg": msg,
        })
        state["log"] = state["log"][-max_entries:]
        self.save(state)

    def set_master_switch(self, on: bool) -> None:
        state = self.load()
        state["master_switch"] = bool(on)
        self.save(state)
        self.append_log(f"Master switch {'ON' if on else 'OFF'}", "warn")

    def set_initial_capital(self, capital: float) -> None:
        """Editable only when no trades exist yet, or via explicit reset."""
        state = self.load()
        if state["trades"]:
            raise RuntimeError(
                "Cannot change initial capital after trading has started. "
                "Reset the portfolio via reset()."
            )
        delta = float(capital) - float(state["initial_capital"])
        state["initial_capital"] = float(capital)
        state["current_capital"] = float(capital)
        state["peak_capital"] = float(capital)
        self.save(state)
        self.append_log(f"Initial capital set to {capital:.2f} (delta {delta:+.2f})")

    def update_settings(self, patch: Dict[str, Any]) -> None:
        state = self.load()
        state["settings"].update(patch)
        self.save(state)
        self.append_log(f"Settings updated: {list(patch)}")

    def set_last_data_update(self, ts: datetime) -> None:
        state = self.load()
        state["last_data_update"] = ts.isoformat(timespec="seconds")
        self.save(state)

    # -------------------- trade lifecycle --------------------
    def open_trade(self, trade: Trade) -> None:
        state = self.load()
        state["trades"].append(asdict(trade))
        self.save(state)
        self.append_log(
            f"OPEN {trade.symbol} {trade.side} strike={trade.strike:.0f} "
            f"lots={trade.lots} @ ₹{trade.entry_premium:.2f}"
        )

    def update_open_marks(self, updates: Dict[str, Dict[str, float]]) -> None:
        """Update current_premium + unrealized_pnl for open trades.

        ``updates`` is ``{trade_id: {"current_premium": px, "unrealized_pnl": pnl}}``.
        """
        state = self.load()
        for t in state["trades"]:
            if t["trade_id"] in updates and not t.get("exit_time"):
                t.update(updates[t["trade_id"]])
        self.save(state)

    def close_trade(
        self,
        trade_id: str,
        exit_time: datetime,
        exit_premium: float,
        exit_reason: str,
        gross_pnl: float,
        costs: float,
        exit_spot: Optional[float] = None,
        exit_order_id: Optional[str] = None,
    ) -> None:
        state = self.load()
        for t in state["trades"]:
            if t["trade_id"] != trade_id:
                continue
            t["exit_time"] = exit_time.isoformat(timespec="seconds")
            t["exit_premium"] = float(exit_premium)
            t["exit_reason"] = exit_reason
            t["exit_spot"] = float(exit_spot) if exit_spot is not None else None
            t["gross_pnl"] = float(gross_pnl)
            t["costs"] = float(costs)
            t["realized_pnl"] = float(gross_pnl - costs)
            t["unrealized_pnl"] = 0.0
            t["current_premium"] = float(exit_premium)
            if exit_order_id:
                t["exit_order_id"] = exit_order_id
            break
        self.save(state)
        self.append_log(
            f"CLOSE {trade_id[:8]} reason={exit_reason} "
            f"P&L={gross_pnl - costs:+.2f}"
        )

    # -------------------- reporting helpers --------------------
    def summary(self) -> Dict[str, Any]:
        """Compute all UI-facing metrics from a fresh load.

        Kept read-only so the UI can call it as often as it wants
        without racing the scheduler.
        """
        state = self.load()
        trades = state["trades"]
        closed = [t for t in trades if t.get("exit_time")]
        open_ = [t for t in trades if not t.get("exit_time")]
        realized = sum(float(t.get("realized_pnl") or 0.0) for t in closed)
        unrealized = sum(float(t.get("unrealized_pnl") or 0.0) for t in open_)

        # Today's numbers, in local time.
        today = datetime.now().date().isoformat()
        today_closed = [
            t for t in closed if str(t.get("exit_time", ""))[:10] == today
        ]
        today_open = [
            t for t in open_ if str(t.get("entry_time", ""))[:10] == today
        ]
        today_realized = sum(float(t.get("realized_pnl") or 0.0) for t in today_closed)
        today_unrealized = sum(float(t.get("unrealized_pnl") or 0.0) for t in today_open)
        today_pnl = today_realized + today_unrealized

        wins = [t for t in closed if float(t.get("realized_pnl") or 0.0) > 0]
        today_wins = [t for t in today_closed if float(t.get("realized_pnl") or 0.0) > 0]

        peak = float(state["peak_capital"])
        cur = float(state["current_capital"]) + unrealized
        drawdown = (peak - cur) / peak if peak > 0 else 0.0

        return {
            "mode": state["mode"],
            "master_switch": bool(state["master_switch"]),
            "initial_capital": float(state["initial_capital"]),
            "current_capital": float(state["current_capital"]),
            "current_capital_incl_unrealized": cur,
            "peak_capital": peak,
            "drawdown_pct": float(drawdown),
            "realized_pnl_total": float(realized),
            "unrealized_pnl_total": float(unrealized),
            "today_pnl": float(today_pnl),
            "today_realized_pnl": float(today_realized),
            "today_unrealized_pnl": float(today_unrealized),
            "trades_today": int(len(today_closed) + len(today_open)),
            "closed_trades_today": int(len(today_closed)),
            "win_rate_today": (len(today_wins) / len(today_closed)) if today_closed else 0.0,
            "win_rate_overall": (len(wins) / len(closed)) if closed else 0.0,
            "n_open_trades": int(len(open_)),
            "n_closed_trades": int(len(closed)),
            "last_data_update": state.get("last_data_update"),
            "last_signal_time": state.get("last_signal_time"),
            "settings": state["settings"],
        }

    # -------------------- destructive helpers --------------------
    def reset(self, capital: Optional[float] = None) -> None:
        """Blow away trades + logs, keep the file. Requires explicit call."""
        cap = float(capital) if capital is not None else float(get_config().project["base_capital"])
        state = {
            "mode": self.mode,
            "initial_capital": cap,
            "current_capital": cap,
            "peak_capital": cap,
            "master_switch": False,
            "settings": default_settings(),
            "last_data_update": None,
            "last_signal_time": None,
            "trades": [],
            "log": [],
        }
        self._atomic_write(state)
        log.warning("%s portfolio reset to %.2f", self.mode, cap)


# =============================================================
#  Public factory helpers
# =============================================================
def make_trade_id() -> str:
    return uuid.uuid4().hex


def get_portfolio(mode: str = "paper") -> Portfolio:
    """Convenience for callers that don't need to hold a reference."""
    return Portfolio(mode=mode)
