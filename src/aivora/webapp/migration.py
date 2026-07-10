"""One-shot migration from the legacy single-user
``data/paper_portfolio.json`` into a target user's paper portfolio.

Idempotent: an already-migrated trade (matched by ``trade_id``) is
skipped, so re-running the import doesn't create duplicates."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from ..utils.config import get_config
from ..utils.logger import get_logger
from . import db as db_mod
from .portfolios import UserPortfolio

log = get_logger(__name__)


def legacy_portfolio_path() -> Path:
    return get_config().paths["db_dir"].parent / "paper_portfolio.json"


def preview() -> Optional[Dict]:
    """Return the legacy portfolio's summary without importing it."""
    p = legacy_portfolio_path()
    if not p.exists():
        return None
    try:
        state = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("legacy portfolio unreadable: %s", exc)
        return None
    return {
        "initial_capital": state.get("initial_capital"),
        "current_capital": state.get("current_capital"),
        "n_trades": len(state.get("trades", [])),
        "n_closed": sum(1 for t in state.get("trades", []) if t.get("exit_time")),
        "path": str(p),
    }


def import_into(user_id: int, mode: str = "paper") -> Dict[str, int]:
    """Copy legacy trades into ``user_id``'s target-mode portfolio."""
    p = legacy_portfolio_path()
    if not p.exists():
        return {"imported": 0, "skipped": 0, "reason": "no legacy file"}
    state = json.loads(p.read_text(encoding="utf-8"))
    trades = state.get("trades", [])

    target = UserPortfolio(user_id, mode)
    tgt_state = target.load()

    # Preserve invariants — set initial capital first if the user hasn't traded yet.
    if not tgt_state["trades"] and state.get("initial_capital"):
        try:
            target.set_initial_capital(float(state["initial_capital"]))
        except Exception as exc:
            log.warning("could not set initial capital during migration: %s", exc)

    existing = {t["trade_id"] for t in tgt_state["trades"]}
    imported = 0
    skipped = 0
    with db_mod.connect() as conn:
        for t in trades:
            tid = t.get("trade_id")
            if not tid or tid in existing:
                skipped += 1
                continue
            conn.execute(
                """
                INSERT INTO user_trades (
                    user_id, mode, trade_id, entry_time, exit_time,
                    symbol, side, strike, lots, lot_size,
                    entry_premium, exit_premium, current_premium,
                    entry_spot, exit_spot,
                    entry_order_id, exit_order_id, horizon_close_time,
                    gross_pnl, costs, realized_pnl, unrealized_pnl,
                    exit_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id, mode, tid,
                    t.get("entry_time"), t.get("exit_time"),
                    t.get("symbol"), t.get("side"), t.get("strike"),
                    t.get("lots"), t.get("lot_size"),
                    t.get("entry_premium"), t.get("exit_premium"),
                    t.get("current_premium"),
                    t.get("entry_spot"), t.get("exit_spot"),
                    t.get("entry_order_id"), t.get("exit_order_id"),
                    t.get("horizon_close_time"),
                    t.get("gross_pnl"), t.get("costs"),
                    t.get("realized_pnl"), t.get("unrealized_pnl"),
                    t.get("exit_reason"),
                ),
            )
            imported += 1
        # Re-derive current_capital + peak_capital from all imported closed trades.
        target._recompute_capital(conn)  # noqa: SLF001
    log.info(
        "migration into user_id=%s mode=%s: imported=%d skipped=%d",
        user_id, mode, imported, skipped,
    )
    return {"imported": imported, "skipped": skipped}


def deactivate_legacy_file() -> Path:
    """Rename the legacy file so subsequent runs don't re-offer it."""
    src = legacy_portfolio_path()
    if not src.exists():
        return src
    dst = src.with_suffix(f".json.migrated.{datetime.now().strftime('%Y%m%d%H%M%S')}")
    src.rename(dst)
    log.info("legacy portfolio archived at %s", dst)
    return dst
