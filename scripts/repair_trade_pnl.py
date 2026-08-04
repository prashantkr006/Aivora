"""Repair closed trades whose exit was recorded by AiVora but happened at the broker.

Trades exited by hand in Kite, before reconciliation existed, were later
force-closed by the emergency square-off at whatever mark AiVora happened
to be carrying.  That mark came from the wrong option contract, so the
recorded exit price — and with it the P&L, and the portfolio capital —
bear no relation to what actually happened in the account.

This rewrites those rows from the real per-trade P&L.

Safety, because this edits financial records:

* dry run unless ``--apply``;
* the database is copied to a timestamped backup before any write;
* every spec must match exactly one trade, or nothing is written;
* by default it refuses to touch a trade unless it is closed and its
  exit_reason is one of ``--expect-reason`` — repairing the wrong row is
  worse than repairing none;
* capital is recomputed by the portfolio's own routine, so the
  initial + realised invariant is enforced rather than reimplemented.

Spec file — a JSON list, one object per trade::

    [
      {"symbol": "NIFTY", "side": "PE", "strike": 24650, "pnl": -107.25},
      {"symbol": "NIFTY", "side": "PE", "strike": 24600, "pnl": -783.25},
      {"symbol": "BANKNIFTY", "side": "CE", "strike": 57700, "pnl": 3139.50,
       "charges": 61.20, "exit_time": "2026-08-04T13:42:00"}
    ]

``pnl`` is the broker's figure for that position.  ``charges`` and
``exit_time`` are optional; without them charges are estimated and the
existing exit_time is kept.

    python -m scripts.repair_trade_pnl --user 27 --mode live \
        --date 2026-08-04 --spec repair.json            # dry run
    python -m scripts.repair_trade_pnl ... --apply      # writes
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aivora.backtest.costs import compute_round_trip, live_cost_cfg  # noqa: E402
from aivora.live.reconcile import EXIT_REASON  # noqa: E402
from aivora.utils.config import get_config  # noqa: E402
from aivora.webapp import db as webapp_db  # noqa: E402
from aivora.webapp.portfolios import UserPortfolio  # noqa: E402


def _cost_cfg(include_slippage: bool) -> dict:
    """Charge model for a trade that really executed — same one the live
    closers use, so a repaired row is priced exactly like a fresh one."""
    cfg = live_cost_cfg()
    if include_slippage:
        cfg["slippage_pct"] = float(
            get_config().raw.get("backtest", {}).get("costs", {})
            .get("slippage_pct", 0.001)
        )
    return cfg


def _find(conn, user: int, mode: str, date: str, spec: dict):
    return conn.execute(
        """
        SELECT * FROM user_trades
         WHERE user_id=? AND mode=? AND date(entry_time)=?
           AND symbol=? AND side=? AND ABS(strike - ?) < 0.5
        """,
        (user, mode, date, spec["symbol"], spec["side"], float(spec["strike"])),
    ).fetchall()


def _plan(row, spec: dict, pnl_is_gross: bool, cost_cfg: dict) -> dict:
    """Work out the new numbers for one trade.  Pure — writes nothing."""
    lots, lot_size = int(row["lots"]), int(row["lot_size"])
    qty = lots * lot_size
    entry = float(row["entry_premium"])
    given = float(spec["pnl"])
    stated = spec.get("charges")

    def _charges_at(exit_px: float) -> float:
        if stated is not None:
            return float(stated)
        return compute_round_trip(entry, exit_px, lots, lot_size, cost_cfg).total

    # The broker's P&L is (exit - entry) x qty, so the exit price it implies
    # is exact — provided our entry matches the account's buy price, which
    # the report below puts in front of you.
    if pnl_is_gross:
        gross = given
    else:
        # A net figure has the charges already taken out, and those charges
        # depend on the exit price, which is what we are solving for.  Walk
        # the fixed point: it settles in two or three passes because charges
        # are a fraction of a percent of turnover.
        gross = given
        for _ in range(50):
            nxt = given + _charges_at(entry + gross / qty)
            if abs(nxt - gross) < 1e-9:
                break
            gross = nxt

    exit_px = entry + gross / qty
    charges = _charges_at(exit_px)

    return {
        "row": row,
        "qty": qty,
        "entry": entry,
        "exit_premium": exit_px,
        "gross_pnl": gross,
        "costs": charges,
        "realized_pnl": gross - charges,
        "exit_time": spec.get("exit_time") or row["exit_time"],
    }


def _report(plans: list) -> None:
    for p in plans:
        r = p["row"]
        print(f"\n  {r['symbol']} {int(r['strike'])} {r['side']}"
              f"   ({r['lots']} lot x {r['lot_size']} = {p['qty']})")
        print(f"    {'':<16} {'BEFORE':>14}   {'AFTER':>14}")
        for label, before, after in (
            ("entry_premium", r["entry_premium"], p["entry"]),
            ("exit_premium", r["exit_premium"], p["exit_premium"]),
            ("gross_pnl", r["gross_pnl"], p["gross_pnl"]),
            ("costs", r["costs"], p["costs"]),
            ("realized_pnl", r["realized_pnl"], p["realized_pnl"]),
        ):
            b = "None" if before is None else f"{float(before):,.2f}"
            print(f"    {label:<16} {b:>14}   {after:>14,.2f}")
        print(f"    {'exit_reason':<16} {str(r['exit_reason']):>14}"
              f"   {EXIT_REASON:>14}")
        print(f"    {'exit_time':<16} {str(r['exit_time']):>14}"
              f"   {str(p['exit_time']):>14}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, required=True)
    ap.add_argument("--mode", default="live")
    ap.add_argument("--date", required=True)
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--pnl", choices=["gross", "net"], default="gross",
                    help="gross = Kite Positions (charges excluded, default); "
                         "net = Console P&L (charges already deducted)")
    ap.add_argument("--include-slippage", action="store_true",
                    help="also charge the backtest slippage model (rarely right)")
    ap.add_argument("--expect-reason", default="emergency",
                    help="comma-separated exit_reason values allowed on the "
                         "rows being repaired; '*' to skip the check")
    ap.add_argument("--apply", action="store_true",
                    help="write the changes (otherwise dry run)")
    ap.add_argument("--db", type=Path, default=None,
                    help="webapp DB to operate on (default: the configured one)")
    a = ap.parse_args()

    specs = json.loads(a.spec.read_text(encoding="utf-8"))
    cost_cfg = _cost_cfg(a.include_slippage)
    allowed = None if a.expect_reason.strip() == "*" else {
        s.strip().lower() for s in a.expect_reason.split(",")
    }

    db_path = a.db or webapp_db.default_db_path()
    print(f"DB    : {db_path}")
    print(f"Target: user={a.user} mode={a.mode} date={a.date}")
    print(f"P&L   : treated as {a.pnl.upper()}"
          f"{'' if a.include_slippage else '  (slippage excluded from charges)'}")

    plans, problems = [], []
    with webapp_db.connect(db_path) as conn:
        for spec in specs:
            label = f"{spec['symbol']} {spec['strike']} {spec['side']}"
            rows = _find(conn, a.user, a.mode, a.date, spec)
            if len(rows) != 1:
                problems.append(f"{label}: matched {len(rows)} trades, expected 1")
                continue
            row = rows[0]
            if row["exit_time"] is None:
                problems.append(f"{label}: still open — refusing to rewrite")
                continue
            if allowed is not None and str(row["exit_reason"] or "").lower() not in allowed:
                problems.append(
                    f"{label}: exit_reason is {row['exit_reason']!r}, "
                    f"expected one of {sorted(allowed)}"
                )
                continue
            plans.append(_plan(row, spec, a.pnl == "gross", cost_cfg))

    if problems:
        print("\n!! refusing to proceed:")
        for p in problems:
            print("   -", p)
        return 1

    print(f"\n=== {len(plans)} trade(s) ===")
    _report(plans)

    tot_gross = sum(p["gross_pnl"] for p in plans)
    tot_net = sum(p["realized_pnl"] for p in plans)
    print(f"\n  total gross (should match your broker) : {tot_gross:+,.2f}")
    print(f"  total after charges                    : {tot_net:+,.2f}")

    if not a.apply:
        print("\nDRY RUN — nothing written.  Re-run with --apply to commit.")
        return 0

    backup = db_path.with_name(
        f"{db_path.stem}.backup-{datetime.now():%Y%m%d-%H%M%S}{db_path.suffix}"
    )
    shutil.copy2(db_path, backup)
    print(f"\nbackup: {backup}")

    pf = UserPortfolio(a.user, a.mode)
    with webapp_db.connect(db_path) as conn:
        for p in plans:
            conn.execute(
                """
                UPDATE user_trades
                   SET exit_premium=?, current_premium=?, gross_pnl=?, costs=?,
                       realized_pnl=?, unrealized_pnl=0.0,
                       exit_reason=?, exit_time=?
                 WHERE id=?
                """,
                (p["exit_premium"], p["exit_premium"], p["gross_pnl"],
                 p["costs"], p["realized_pnl"], EXIT_REASON, p["exit_time"],
                 p["row"]["id"]),
            )
        # The portfolio's own routine, so the initial + realised invariant is
        # enforced here exactly as it is on every normal close.
        pf._recompute_capital(conn)

        # Leave a trace in the user's event log.  Written on this connection
        # rather than via log_event, which would open the configured DB and
        # ignore --db.
        conn.execute(
            "INSERT INTO user_events (user_id, mode, ts, level, msg) "
            "VALUES (?,?,?,?,?)",
            (a.user, a.mode, datetime.now().isoformat(timespec="seconds"),
             "warn",
             f"Repaired {len(plans)} trade(s) against broker records "
             f"(net {tot_net:+,.2f})"),
        )

        cap = conn.execute(
            "SELECT initial_capital, current_capital FROM user_portfolios "
            "WHERE user_id=? AND mode=?", (a.user, a.mode),
        ).fetchone()

    print(f"\ncapital: initial {cap['initial_capital']:,.2f} "
          f"-> current {cap['current_capital']:,.2f}")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
