"""Per-user portfolio + trades — DB-backed replacement for
``aivora.live.portfolio.Portfolio``.

Same math invariant — ``current_capital == initial + Σ realized_pnl``
— enforced on every save.  All queries are scoped by ``user_id``
so data isolation is impossible to bypass in application code.

The single-user file-based Portfolio still exists and still works
for the classic dashboard.  This module is only used by the
multi-user webapp.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..live.portfolio import Trade, default_settings
from ..utils.logger import get_logger
from . import db as db_mod

log = get_logger(__name__)


# =============================================================
#  Portfolio (DB-backed)
# =============================================================
class UserPortfolio:
    """One (user_id, mode) portfolio row + related trades/events."""

    def __init__(self, user_id: int, mode: str):
        assert mode in ("paper", "live")
        self.user_id = int(user_id)
        self.mode = mode
        self._ensure_row()

    # ---- lifecycle ----
    def _ensure_row(self) -> None:
        with db_mod.connect() as conn:
            row = conn.execute(
                "SELECT id FROM user_portfolios WHERE user_id = ? AND mode = ?",
                (self.user_id, self.mode),
            ).fetchone()
            if row is not None:
                return
            conn.execute(
                """
                INSERT INTO user_portfolios
                    (user_id, mode, initial_capital, current_capital,
                     peak_capital, master_switch, settings_json)
                VALUES (?, ?, ?, ?, ?, 0, ?)
                """,
                (self.user_id, self.mode,
                 100_000.0, 100_000.0, 100_000.0,
                 json.dumps(default_settings())),
            )

    def reset(self, initial_capital: float = 100_000.0) -> None:
        cap = float(initial_capital)
        with db_mod.connect() as conn:
            conn.execute(
                "DELETE FROM user_trades WHERE user_id = ? AND mode = ?",
                (self.user_id, self.mode),
            )
            conn.execute(
                "DELETE FROM user_events WHERE user_id = ?",
                (self.user_id,),
            )
            conn.execute(
                """
                UPDATE user_portfolios
                SET initial_capital = ?, current_capital = ?,
                    peak_capital = ?, master_switch = 0,
                    settings_json = ?, last_data_update = NULL,
                    last_signal_time = NULL
                WHERE user_id = ? AND mode = ?
                """,
                (cap, cap, cap, json.dumps(default_settings()),
                 self.user_id, self.mode),
            )
        self.log_event(f"Portfolio reset to {cap:.2f}", "warn")

    # ---- read ----
    def load(self) -> Dict[str, Any]:
        with db_mod.connect() as conn:
            row = conn.execute(
                "SELECT * FROM user_portfolios WHERE user_id = ? AND mode = ?",
                (self.user_id, self.mode),
            ).fetchone()
            trades = self._load_trades(conn)
            events = self._load_events(conn, limit=200)
        return {
            "mode": row["mode"],
            "initial_capital": float(row["initial_capital"]),
            "current_capital": float(row["current_capital"]),
            "peak_capital": float(row["peak_capital"]),
            "master_switch": bool(row["master_switch"]),
            "settings": json.loads(row["settings_json"] or "{}") or default_settings(),
            "last_data_update": row["last_data_update"],
            "last_signal_time": row["last_signal_time"],
            "trades": trades,
            "log": events,
        }

    def _load_trades(self, conn) -> List[Dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT * FROM user_trades
            WHERE user_id = ? AND mode = ?
            ORDER BY entry_time
            """,
            (self.user_id, self.mode),
        ).fetchall()
        return [dict(r) for r in rows]

    def _load_events(self, conn, limit: int = 200) -> List[Dict[str, str]]:
        rows = conn.execute(
            """
            SELECT ts, level, msg FROM user_events
            WHERE user_id = ?
            ORDER BY id DESC LIMIT ?
            """,
            (self.user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- settings ----
    def update_settings(self, patch: Dict[str, Any]) -> None:
        state = self.load()
        state["settings"].update(patch)
        with db_mod.connect() as conn:
            conn.execute(
                """
                UPDATE user_portfolios SET settings_json = ?
                WHERE user_id = ? AND mode = ?
                """,
                (json.dumps(state["settings"]), self.user_id, self.mode),
            )
        self.log_event(f"Settings updated: {list(patch)}")

    def set_master_switch(self, on: bool) -> None:
        with db_mod.connect() as conn:
            conn.execute(
                """
                UPDATE user_portfolios SET master_switch = ?
                WHERE user_id = ? AND mode = ?
                """,
                (int(on), self.user_id, self.mode),
            )
        self.log_event(f"Master switch {'ON' if on else 'OFF'}", "warn")

    def set_initial_capital(self, capital: float) -> None:
        with db_mod.connect() as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM user_trades WHERE user_id = ? AND mode = ?",
                (self.user_id, self.mode),
            ).fetchone()[0]
        if int(n) > 0:
            raise RuntimeError(
                "Cannot change initial capital after trading has started. "
                "Reset the portfolio first."
            )
        cap = float(capital)
        with db_mod.connect() as conn:
            conn.execute(
                """
                UPDATE user_portfolios SET
                    initial_capital = ?, current_capital = ?, peak_capital = ?
                WHERE user_id = ? AND mode = ?
                """,
                (cap, cap, cap, self.user_id, self.mode),
            )
        self.log_event(f"Initial capital set to {cap:.2f}")

    def set_last_data_update(self, ts: datetime) -> None:
        with db_mod.connect() as conn:
            conn.execute(
                """
                UPDATE user_portfolios SET last_data_update = ?
                WHERE user_id = ? AND mode = ?
                """,
                (ts.isoformat(timespec="seconds"), self.user_id, self.mode),
            )

    # ---- trade lifecycle ----
    def open_trade(self, trade: Trade) -> None:
        d = asdict(trade)
        cols = [
            "user_id", "mode", "trade_id", "entry_time", "symbol", "side",
            "strike", "lots", "lot_size", "entry_premium",
            "current_premium", "entry_spot", "entry_order_id",
            "horizon_close_time",
        ]
        vals = [
            self.user_id, self.mode, d["trade_id"], d["entry_time"],
            d["symbol"], d["side"], d["strike"], d["lots"], d["lot_size"],
            d["entry_premium"], d["current_premium"], d["entry_spot"],
            d.get("entry_order_id"), d.get("horizon_close_time"),
        ]
        placeholders = ", ".join(["?"] * len(cols))
        with db_mod.connect() as conn:
            conn.execute(
                f"INSERT INTO user_trades ({', '.join(cols)}) VALUES ({placeholders})",
                vals,
            )
        self.log_event(
            f"OPEN {trade.symbol} {trade.side} strike={trade.strike:.0f} "
            f"lots={trade.lots} @ ₹{trade.entry_premium:.2f}"
        )

    def update_open_marks(self, updates: Dict[str, Dict[str, float]]) -> None:
        """Update current_premium + unrealized_pnl for open trades."""
        if not updates:
            return
        with db_mod.connect() as conn:
            for tid, patch in updates.items():
                conn.execute(
                    """
                    UPDATE user_trades
                    SET current_premium = ?, unrealized_pnl = ?
                    WHERE user_id = ? AND mode = ? AND trade_id = ?
                      AND exit_time IS NULL
                    """,
                    (
                        float(patch["current_premium"]),
                        float(patch["unrealized_pnl"]),
                        self.user_id, self.mode, tid,
                    ),
                )

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
        realized = float(gross_pnl - costs)
        with db_mod.connect() as conn:
            conn.execute(
                """
                UPDATE user_trades
                SET exit_time = ?, exit_premium = ?, exit_reason = ?,
                    exit_spot = ?, exit_order_id = ?,
                    gross_pnl = ?, costs = ?, realized_pnl = ?,
                    unrealized_pnl = 0.0, current_premium = ?
                WHERE user_id = ? AND mode = ? AND trade_id = ?
                """,
                (
                    exit_time.isoformat(timespec="seconds"),
                    float(exit_premium), exit_reason,
                    float(exit_spot) if exit_spot is not None else None,
                    exit_order_id,
                    float(gross_pnl), float(costs), realized,
                    float(exit_premium),
                    self.user_id, self.mode, trade_id,
                ),
            )
            self._recompute_capital(conn)
        self.log_event(f"CLOSE {trade_id[:8]} reason={exit_reason} P&L={realized:+.2f}")

    def _recompute_capital(self, conn) -> None:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(realized_pnl), 0.0) AS r,
                   (SELECT initial_capital FROM user_portfolios
                    WHERE user_id = ? AND mode = ?) AS init
            FROM user_trades
            WHERE user_id = ? AND mode = ? AND exit_time IS NOT NULL
            """,
            (self.user_id, self.mode, self.user_id, self.mode),
        ).fetchone()
        realized = float(row["r"])
        initial = float(row["init"])
        current = initial + realized
        conn.execute(
            """
            UPDATE user_portfolios
            SET current_capital = ?, peak_capital = MAX(peak_capital, ?)
            WHERE user_id = ? AND mode = ?
            """,
            (current, current, self.user_id, self.mode),
        )
        # Invariant check — same firewall as the file-based portfolio.
        row = conn.execute(
            """
            SELECT current_capital, initial_capital
            FROM user_portfolios WHERE user_id = ? AND mode = ?
            """,
            (self.user_id, self.mode),
        ).fetchone()
        if abs(float(row["current_capital"]) - (float(row["initial_capital"]) + realized)) > 1e-6:
            raise RuntimeError(
                "UserPortfolio invariant violated. "
                f"current={row['current_capital']} "
                f"initial={row['initial_capital']} "
                f"realized={realized}"
            )

    # ---- logging ----
    def log_event(self, msg: str, level: str = "info",
                  retain_last: int = 500) -> None:
        """Append an event and auto-prune the user's oldest ones.

        We keep the newest ``retain_last`` entries per user — enough
        to review the last ~2 trading days — and drop the rest on
        each write.  This stops the ``user_events`` table growing
        unboundedly over weeks of use.
        """
        with db_mod.connect() as conn:
            conn.execute(
                "INSERT INTO user_events (user_id, ts, level, msg) VALUES (?, ?, ?, ?)",
                (self.user_id,
                 datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                 level, msg),
            )
            # Prune anything older than the retain_last window.
            conn.execute(
                """
                DELETE FROM user_events
                WHERE user_id = ?
                  AND id NOT IN (
                      SELECT id FROM user_events
                      WHERE user_id = ?
                      ORDER BY id DESC LIMIT ?
                  )
                """,
                (self.user_id, self.user_id, int(retain_last)),
            )

    def prune_events(self, retain_last: int = 500) -> int:
        """One-shot cleanup helper — returns rows deleted."""
        with db_mod.connect() as conn:
            cur = conn.execute(
                """
                DELETE FROM user_events
                WHERE user_id = ?
                  AND id NOT IN (
                      SELECT id FROM user_events
                      WHERE user_id = ?
                      ORDER BY id DESC LIMIT ?
                  )
                """,
                (self.user_id, self.user_id, int(retain_last)),
            )
            return int(cur.rowcount or 0)

    # ---- read-only summary for the UI ----
    def summary(self) -> Dict[str, Any]:
        state = self.load()
        trades = state["trades"]
        closed = [t for t in trades if t.get("exit_time")]
        open_ = [t for t in trades if not t.get("exit_time")]
        realized = sum(float(t.get("realized_pnl") or 0.0) for t in closed)
        unrealized = sum(float(t.get("unrealized_pnl") or 0.0) for t in open_)

        today = datetime.now().date().isoformat()
        today_closed = [t for t in closed if str(t.get("exit_time", ""))[:10] == today]
        today_open = [t for t in open_ if str(t.get("entry_time", ""))[:10] == today]
        today_realized = sum(float(t.get("realized_pnl") or 0.0) for t in today_closed)
        today_unrealized = sum(float(t.get("unrealized_pnl") or 0.0) for t in today_open)

        wins = [t for t in closed if float(t.get("realized_pnl") or 0.0) > 0]
        today_wins = [t for t in today_closed if float(t.get("realized_pnl") or 0.0) > 0]

        peak = float(state["peak_capital"])
        cur = float(state["current_capital"]) + unrealized
        drawdown = (peak - cur) / peak if peak > 0 else 0.0

        return {
            "user_id": self.user_id,
            "mode": state["mode"],
            "master_switch": bool(state["master_switch"]),
            "initial_capital": float(state["initial_capital"]),
            "current_capital": float(state["current_capital"]),
            "current_capital_incl_unrealized": cur,
            "peak_capital": peak,
            "drawdown_pct": float(drawdown),
            "realized_pnl_total": float(realized),
            "unrealized_pnl_total": float(unrealized),
            "today_pnl": float(today_realized + today_unrealized),
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


def make_trade_id() -> str:
    return uuid4().hex
