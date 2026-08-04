"""Read-only dump of one user's trades for a given day.

Written for repairing the 2026-08-04 live trades, but deliberately generic
and side-effect free: it opens the webapp DB, prints what is there, and
exits.  Nothing is written.

    python -m scripts.inspect_trades --user 27 --mode live --date 2026-08-04
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aivora.webapp import db as webapp_db  # noqa: E402

FIELDS = [
    "id", "trade_id", "entry_time", "exit_time", "symbol", "side", "strike",
    "lots", "lot_size", "entry_premium", "exit_premium", "current_premium",
    "gross_pnl", "costs", "realized_pnl", "unrealized_pnl", "exit_reason",
    "tradingsymbol", "entry_order_id", "exit_order_id",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, required=True)
    ap.add_argument("--mode", default="live")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (entry date)")
    a = ap.parse_args()

    print(f"DB: {webapp_db.default_db_path()}")

    with webapp_db.connect() as c:
        rows = c.execute(
            f"SELECT {', '.join(FIELDS)} FROM user_trades "
            "WHERE user_id=? AND mode=? AND date(entry_time)=? "
            "ORDER BY entry_time",
            (a.user, a.mode, a.date),
        ).fetchall()

        print(f"\n=== {len(rows)} trade(s) on {a.date} "
              f"(user={a.user} mode={a.mode}) ===")
        for r in rows:
            print()
            for f in FIELDS:
                print(f"  {f:<16} {r[f]}")

        pf = c.execute(
            "SELECT initial_capital, current_capital, master_switch "
            "FROM user_portfolios WHERE user_id=? AND mode=?",
            (a.user, a.mode),
        ).fetchone()
        print("\n=== portfolio ===")
        if pf is None:
            print("  (no row)")
        else:
            for k in ("initial_capital", "current_capital", "master_switch"):
                print(f"  {k:<16} {pf[k]}")

        tot = c.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(realized_pnl),0) s "
            "FROM user_trades WHERE user_id=? AND mode=? AND exit_time IS NOT NULL",
            (a.user, a.mode),
        ).fetchone()
        print(f"\n=== all closed trades in this portfolio ===")
        print(f"  count            {tot['n']}")
        print(f"  sum realized_pnl {tot['s']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
