"""Put one user's paper and live day side by side, trade for trade.

Paper and live share one signal source — the same MarketDataCache, the same
model, the same gates — so they should enter the same things at the same
times.  Everything after the entry decision is different, and that is where
a divergence in P&L or win rate has to come from:

* **lot size** — live reads it off the Kite instruments dump; paper read it
  from config.yaml until 2026-08-04.  A stale config meant paper traded a
  BANKNIFTY contract half the size of live's, so live's P&L on the same
  price move was double paper's regardless of who had more capital.
* **entry price** — live is the broker's actual fill; paper is the live LTP
  when a quote was available, and a synthetic estimate when it was not.
* **exit price** — live re-quotes the contract it holds.  Paper never
  re-quotes anything: it prices exits with ``theoretical_exit_premium``, a
  model of what the premium ought to be given the spot move and time
  elapsed.  If that model is kind, paper wins trades live would lose.

This prints the numbers rather than reasoning about them.

    python -m scripts.compare_modes --user 27 --date 2026-08-05
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aivora.webapp import db as webapp_db  # noqa: E402

MODES = ("paper", "live")


def _rows(conn, user: int, mode: str, day: str) -> List[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM user_trades WHERE user_id=? AND mode=? "
        "AND date(entry_time)=? ORDER BY entry_time",
        (user, mode, day),
    )]


def _portfolio(conn, user: int, mode: str) -> Optional[dict]:
    r = conn.execute(
        "SELECT * FROM user_portfolios WHERE user_id=? AND mode=?",
        (user, mode),
    ).fetchone()
    return dict(r) if r else None


def _num(v, default=0.0) -> float:
    try:
        return float(v if v is not None else default)
    except (TypeError, ValueError):
        return default


def _summary(trades: List[dict]) -> dict:
    closed = [t for t in trades if t.get("exit_time")]
    wins = [t for t in closed if _num(t.get("realized_pnl")) > 0]
    return {
        "n": len(trades),
        "closed": len(closed),
        "open": len(trades) - len(closed),
        "wins": len(wins),
        "win_rate": len(wins) / len(closed) if closed else 0.0,
        "pnl": sum(_num(t.get("realized_pnl")) for t in closed),
        "gross": sum(_num(t.get("gross_pnl")) for t in closed),
        "costs": sum(_num(t.get("costs")) for t in closed),
    }


def _key(t: dict) -> tuple:
    """Pair a paper trade with its live twin: same minute, symbol, side."""
    return (str(t["entry_time"])[:16], t["symbol"], t["side"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, required=True)
    ap.add_argument("--date", default=datetime.now().date().isoformat())
    ap.add_argument("--db", type=Path, default=None)
    a = ap.parse_args()

    db_path = a.db or webapp_db.default_db_path()
    print(f"DB   : {db_path}")
    print(f"User : {a.user}   Date: {a.date}")

    with webapp_db.connect(db_path) as conn:
        trades = {m: _rows(conn, a.user, m, a.date) for m in MODES}
        pf = {m: _portfolio(conn, a.user, m) for m in MODES}

    # ---- capital ----
    print("\n=== Portfolio ===")
    print(f"  {'':<20} {'PAPER':>16} {'LIVE':>16}")
    for label, col in (("initial capital", "initial_capital"),
                       ("external flows", "external_flows"),
                       ("current capital", "current_capital")):
        vals = []
        for m in MODES:
            vals.append("n/a" if not pf[m] else f"{_num(pf[m].get(col)):,.2f}")
        print(f"  {label:<20} {vals[0]:>16} {vals[1]:>16}")

    # ---- day summary ----
    s = {m: _summary(trades[m]) for m in MODES}
    print("\n=== Today ===")
    print(f"  {'':<20} {'PAPER':>16} {'LIVE':>16}")
    for label, key, fmt in (
        ("trades", "n", "d"), ("closed", "closed", "d"), ("still open", "open", "d"),
        ("wins", "wins", "d"), ("win rate", "win_rate", "pct"),
        ("gross P&L", "gross", "money"), ("costs", "costs", "money"),
        ("net P&L", "pnl", "money"),
    ):
        cells = []
        for m in MODES:
            v = s[m][key]
            cells.append(f"{v:,.2f}" if fmt == "money"
                         else f"{v:.0%}" if fmt == "pct" else f"{v:,}")
        print(f"  {label:<20} {cells[0]:>16} {cells[1]:>16}")

    # ---- lot sizes actually used ----
    print("\n=== Lot size actually recorded on each trade ===")
    used: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for m in MODES:
        for t in trades[m]:
            used[t["symbol"]][m].add(int(_num(t.get("lot_size"))))
    if not used:
        print("  (no trades)")
    for sym in sorted(used):
        p = sorted(used[sym].get("paper", set())) or ["-"]
        l = sorted(used[sym].get("live", set())) or ["-"]
        flag = "  <-- MISMATCH" if p != l and "-" not in (p + l) else ""
        print(f"  {sym:<12} paper={p}  live={l}{flag}")

    # ---- trade-by-trade pairing ----
    print("\n=== Trade by trade (paired on entry minute + symbol + side) ===")
    paired: Dict[tuple, Dict[str, dict]] = defaultdict(dict)
    for m in MODES:
        for t in trades[m]:
            paired[_key(t)][m] = t

    for k in sorted(paired):
        p, l = paired[k].get("paper"), paired[k].get("live")
        when, sym, side = k
        tag = "" if (p and l) else ("  [PAPER ONLY]" if p else "  [LIVE ONLY]")
        print(f"\n  {when}  {sym} {side}{tag}")
        print(f"    {'':<16} {'PAPER':>14} {'LIVE':>14} {'diff':>14}")
        for label, col in (("lot_size", "lot_size"), ("lots", "lots"),
                           ("entry premium", "entry_premium"),
                           ("exit premium", "exit_premium"),
                           ("entry spot", "entry_spot"),
                           ("exit spot", "exit_spot"),
                           ("gross P&L", "gross_pnl"),
                           ("costs", "costs"),
                           ("net P&L", "realized_pnl")):
            pv = _num(p.get(col)) if p else None
            lv = _num(l.get(col)) if l else None
            f = lambda v: "-" if v is None else f"{v:,.2f}"
            d = "-" if (pv is None or lv is None) else f"{lv - pv:+,.2f}"
            print(f"    {label:<16} {f(pv):>14} {f(lv):>14} {d:>14}")
        pr = (p or {}).get("exit_reason") or "open"
        lr = (l or {}).get("exit_reason") or "open"
        star = "  <-- different exit" if (p and l and pr != lr) else ""
        print(f"    {'exit reason':<16} {pr:>14} {lr:>14}{star}")

    print("\n" + "=" * 66)
    print("What to look at first:")
    print("  · lot_size mismatch  -> live and paper traded different contract")
    print("                          sizes, so P&L cannot be compared directly")
    print("  · exit premium diff  -> paper priced the exit with a model, live")
    print("                          re-quoted the real contract")
    print("  · different exit     -> the same trade hit different rules because")
    print("                          it was marked at different prices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
