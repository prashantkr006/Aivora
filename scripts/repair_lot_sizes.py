"""Rewrite past trades that were sized on a stale lot size.

config.yaml said NIFTY 75 and BANKNIFTY 15 long after NSE had moved to 65
and 30.  Live was never affected — it reads the size off the Kite
instruments dump — but paper took the config value, so every paper trade
was sized on the wrong contract and its P&L is the P&L of a position that
could not have been held.

This restates those trades at the real contract size.  Entry and exit
premiums, timestamps and lot counts are left exactly as they were: the
prices are what they are, only the multiplier was wrong.

Costs
-----
Charges scale with turnover, and turnover scales with lot size, so a
BANKNIFTY trade at double the contract size really did cost roughly double
to trade.  Leaving the old figure would understate it.  They are
recomputed by default; ``--keep-costs`` preserves the recorded value if
you would rather not move it.

Safety, because this edits financial records: dry run unless ``--apply``,
a timestamped DB backup before any write, and capital is rebuilt by the
portfolio's own routine so the initial + realised invariant is enforced.

    python -m scripts.repair_lot_sizes --user 27 --mode paper
    python -m scripts.repair_lot_sizes --user 27 --mode paper --apply
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aivora.backtest.costs import compute_round_trip  # noqa: E402
from aivora.live.lot_sizes import configured_lot_size  # noqa: E402
from aivora.utils.config import get_config  # noqa: E402
from aivora.webapp import db as webapp_db  # noqa: E402
from aivora.webapp.portfolios import UserPortfolio  # noqa: E402


def _paper_cost_cfg() -> dict:
    """Paper's charge model — the full schedule, slippage included.

    Unlike a live fill, a paper fill never crossed a real spread, so that
    term has to be modelled here or paper flatters itself.
    """
    return dict(get_config().raw.get("backtest", {}).get("costs", {}) or {})


def _plan(row, correct_lot: int, keep_costs: bool, cost_cfg: dict) -> dict:
    """New numbers for one trade.  Pure — writes nothing."""
    lots = int(row["lots"])
    entry = float(row["entry_premium"])
    exit_px = float(row["exit_premium"]) if row["exit_premium"] is not None \
        else float(row["current_premium"] or entry)

    gross = (exit_px - entry) * lots * correct_lot
    if keep_costs:
        costs = float(row["costs"] or 0.0)
    else:
        costs = compute_round_trip(entry, exit_px, lots, correct_lot,
                                   cost_cfg).total

    closed = row["exit_time"] is not None
    return {
        "row": row,
        "lot_size": correct_lot,
        "gross_pnl": gross,
        "costs": costs,
        "realized_pnl": (gross - costs) if closed else 0.0,
        "unrealized_pnl": 0.0 if closed else (gross - costs),
        "closed": closed,
    }


def _report(plans: list) -> None:
    print(f"\n{'trade':<10} {'symbol':<10} {'lot':>9}  "
          f"{'old P&L':>11} {'new P&L':>11} {'change':>11}")
    print("  " + "-" * 68)
    for p in plans:
        r = p["row"]
        old = float((r["realized_pnl"] if p["closed"] else r["unrealized_pnl"]) or 0.0)
        new = p["realized_pnl"] if p["closed"] else p["unrealized_pnl"]
        tag = "" if p["closed"] else "  (open)"
        print(f"{str(r['trade_id'])[:8]:<10} {r['symbol']:<10} "
              f"{int(r['lot_size']):>3} -> {p['lot_size']:<3}  "
              f"{old:>11,.2f} {new:>11,.2f} {new - old:>+11,.2f}{tag}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, required=True)
    ap.add_argument("--mode", default="paper")
    ap.add_argument("--apply", action="store_true",
                    help="write the changes (otherwise dry run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="explicit no-op; dry run is already the default")
    ap.add_argument("--keep-costs", action="store_true",
                    help="leave recorded costs alone (they were computed on "
                         "the wrong turnover, so they are too low/high)")
    ap.add_argument("--include-open", action="store_true",
                    help="also restate open trades' unrealised P&L")
    ap.add_argument("--lot-size", action="append", default=[], metavar="SYM=N",
                    help="override the correct size for one symbol, e.g. "
                         "--lot-size BANKNIFTY=30. Use this when config.yaml "
                         "may itself be stale; repeatable.")
    ap.add_argument("--db", type=Path, default=None)
    a = ap.parse_args()

    if a.dry_run and a.apply:
        print("--dry-run and --apply contradict each other")
        return 2

    overrides = {}
    for item in a.lot_size:
        sym, _, val = item.partition("=")
        if not val.isdigit() or int(val) <= 0:
            print(f"bad --lot-size {item!r}; expected SYMBOL=N")
            return 2
        overrides[sym.strip().upper()] = int(val)

    db_path = a.db or webapp_db.default_db_path()
    cost_cfg = _paper_cost_cfg()

    print(f"DB    : {db_path}")
    print(f"Target: user={a.user} mode={a.mode}")
    print(f"Costs : {'left as recorded' if a.keep_costs else 'recomputed on the real turnover'}")
    # Print what we are treating as correct. The first run of this script
    # reported "32 already correct" against a container whose config.yaml
    # was itself stale — it compared a wrong number to a wrong number and
    # matched. Showing the source makes that impossible to miss.
    print("Sizes :", ", ".join(
        f"{s}={overrides.get(s, configured_lot_size(s))}"
        f"{' (override)' if s in overrides else ' (config.yaml)'}"
        for s in sorted({i["symbol"] for i in get_config().instruments} | set(overrides))
    ))

    with webapp_db.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM user_trades WHERE user_id=? AND mode=? "
            "ORDER BY entry_time",
            (a.user, a.mode),
        ).fetchall()

    plans, skipped_open, already_right, unknown = [], 0, 0, []
    for r in rows:
        if r["symbol"] in overrides:
            correct = overrides[r["symbol"]]
        else:
            try:
                correct = configured_lot_size(r["symbol"])
            except KeyError:
                unknown.append(r["symbol"])
                continue
        if int(r["lot_size"] or 0) == correct:
            already_right += 1
            continue
        if r["exit_time"] is None and not a.include_open:
            skipped_open += 1
            continue
        plans.append(_plan(r, correct, a.keep_costs, cost_cfg))

    print(f"\n{len(rows)} {a.mode} trade(s) found · {len(plans)} to restate"
          f" · {already_right} already correct"
          + (f" · {skipped_open} open (use --include-open)" if skipped_open else ""))
    if unknown:
        print(f"  !! symbols not in config.yaml, left alone: {sorted(set(unknown))}")
    if not plans:
        print("\nNothing to do.")
        return 0

    _report(plans)

    old_tot = sum(float((p["row"]["realized_pnl"] if p["closed"]
                         else p["row"]["unrealized_pnl"]) or 0.0) for p in plans)
    new_tot = sum(p["realized_pnl"] if p["closed"] else p["unrealized_pnl"]
                  for p in plans)
    print("  " + "-" * 68)
    print(f"{'TOTAL':<10} {'':<10} {'':>9}  "
          f"{old_tot:>11,.2f} {new_tot:>11,.2f} {new_tot - old_tot:>+11,.2f}")

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
                   SET lot_size=?, gross_pnl=?, costs=?,
                       realized_pnl=?, unrealized_pnl=?
                 WHERE id=?
                """,
                (p["lot_size"], p["gross_pnl"], p["costs"],
                 p["realized_pnl"], p["unrealized_pnl"], p["row"]["id"]),
            )
        pf._recompute_capital(conn)
        conn.execute(
            "INSERT INTO user_events (user_id, mode, ts, level, msg) "
            "VALUES (?,?,?,?,?)",
            (a.user, a.mode, datetime.now().isoformat(timespec="seconds"), "warn",
             f"Restated {len(plans)} trade(s) at the correct lot size "
             f"(P&L {new_tot - old_tot:+,.2f})"),
        )
        cap = conn.execute(
            "SELECT initial_capital, current_capital FROM user_portfolios "
            "WHERE user_id=? AND mode=?", (a.user, a.mode),
        ).fetchone()

    print(f"\n{len(plans)} trade(s) updated")
    print(f"capital: initial {cap['initial_capital']:,.2f} "
          f"-> current {cap['current_capital']:,.2f}")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
