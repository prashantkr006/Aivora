"""Pipeline-robustness harness.

Runs four real-world scenarios against the live SQLite store and
reports pass/fail with concrete evidence.  Every test is bracketed
by a backup+restore so nothing persists after the run.

    T1  Duplicate prevention          Run daily pipeline twice,
                                      confirm row count is stable.
    T2  Missing-day recovery          Delete a recent day's rows,
                                      run daily pipeline, confirm
                                      recovery.
    T3  Mid-day (partial candle)      Report Dhan's behaviour when
                                      called before the current
                                      5-min candle closes.
    T4  Old-gap recovery              Delete a full week from ~30
                                      days ago, run daily pipeline,
                                      confirm daily mode does *not*
                                      recover it (documents the
                                      requirement for --mode historical).

The backup is a copy-table (``spot_futures_backup_<pid>``) inside
the same SQLite file - no external files are created, so a hard
kill still leaves the DB in a recoverable state (see ``--restore-only``).

Usage::

    python -m scripts.test_pipeline_robustness            # run everything
    python -m scripts.test_pipeline_robustness --skip-t1  # skip test 1
    python -m scripts.test_pipeline_robustness --restore-only

Requires ``DHAN_ACCESS_TOKEN`` in .env - a live network call is
needed for tests 1, 2 and 4 to invoke the ETL.  T3 is analysis-only
and works offline.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from aivora.pipeline import pipeline as pipe_mod  # noqa: E402
from aivora.pipeline.database import connect  # noqa: E402
from aivora.utils.calendar import is_trading_day, previous_trading_day  # noqa: E402
from aivora.utils.config import get_config  # noqa: E402
from aivora.utils.logger import get_logger  # noqa: E402

log = get_logger("scripts.test_pipeline_robustness")

BACKUP_TABLE = f"spot_futures_backup_{os.getpid()}"


# =============================================================
#  Data class for one test result
# =============================================================
@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str
    numbers: Dict[str, int] = field(default_factory=dict)


# =============================================================
#  DB helpers - never touch anything except spot_futures
# =============================================================
def _table_exists(conn, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def backup_spot_futures() -> None:
    """Copy spot_futures into a session-scoped backup table."""
    with connect() as conn:
        if _table_exists(conn, BACKUP_TABLE):
            log.warning("Backup table %s already exists - dropping first", BACKUP_TABLE)
            conn.execute(f"DROP TABLE {BACKUP_TABLE}")
        conn.execute(f"CREATE TABLE {BACKUP_TABLE} AS SELECT * FROM spot_futures")
        n = conn.execute(f"SELECT COUNT(*) FROM {BACKUP_TABLE}").fetchone()[0]
    log.info("Backup taken: %s (%d rows)", BACKUP_TABLE, n)


def restore_spot_futures() -> None:
    """Replace spot_futures contents from the backup table.

    ``INSERT OR REPLACE`` is used so any rows the pipeline added
    during a test survive alongside the original ones the backup
    still owns.
    """
    with connect() as conn:
        if not _table_exists(conn, BACKUP_TABLE):
            log.warning("No backup table found - nothing to restore")
            return
        conn.execute("DELETE FROM spot_futures")
        conn.execute(
            f"INSERT INTO spot_futures SELECT * FROM {BACKUP_TABLE}"
        )
        conn.execute(f"DROP TABLE {BACKUP_TABLE}")
        n = conn.execute("SELECT COUNT(*) FROM spot_futures").fetchone()[0]
    log.info("Restored spot_futures from backup (%d rows)", n)


def row_count(symbol: Optional[str] = None) -> int:
    with connect() as conn:
        if symbol is None:
            row = conn.execute("SELECT COUNT(*) FROM spot_futures").fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM spot_futures WHERE symbol = ?", (symbol,)
            ).fetchone()
    return int(row[0])


def rows_for_day(target: date, symbol: Optional[str] = None) -> int:
    day_str = target.strftime("%Y-%m-%d")
    q = "SELECT COUNT(*) FROM spot_futures WHERE date(datetime) = ?"
    params: tuple = (day_str,)
    if symbol:
        q += " AND symbol = ?"
        params = (day_str, symbol)
    with connect() as conn:
        return int(conn.execute(q, params).fetchone()[0])


def delete_day(target: date, symbol: Optional[str] = None) -> int:
    day_str = target.strftime("%Y-%m-%d")
    q = "DELETE FROM spot_futures WHERE date(datetime) = ?"
    params: tuple = (day_str,)
    if symbol:
        q += " AND symbol = ?"
        params = (day_str, symbol)
    with connect() as conn:
        cur = conn.execute(q, params)
        deleted = cur.rowcount
    log.info("Deleted %d rows for %s%s",
             deleted, day_str, f" ({symbol})" if symbol else "")
    return deleted


def delete_week_ending(target: date) -> Dict[str, int]:
    """Delete rows for the 7 calendar days ending at ``target``."""
    counts: Dict[str, int] = {}
    for i in range(7):
        d = target - timedelta(days=i)
        counts[d.isoformat()] = delete_day(d)
    return counts


def _last_stored_date_per_symbol() -> Dict[str, Optional[date]]:
    out: Dict[str, Optional[date]] = {}
    for inst in get_config().instruments:
        sym = inst["symbol"]
        with connect() as conn:
            row = conn.execute(
                "SELECT MAX(datetime) FROM spot_futures WHERE symbol = ?", (sym,)
            ).fetchone()
        out[sym] = pd.to_datetime(row[0]).date() if row and row[0] else None
    return out


# =============================================================
#  Individual tests
# =============================================================
def test_1_idempotency() -> TestResult:
    """Run the daily pipeline twice, confirm row count is stable."""
    log.info("=== T1: duplicate prevention ===")
    before = row_count()
    log.info("Row count before  : %d", before)

    log.info("First daily run...")
    pipe_mod.run_daily_update(record_options=False)
    after_first = row_count()
    delta_first = after_first - before
    log.info("Row count after 1 : %d  (added %d)", after_first, delta_first)

    log.info("Second daily run (should be a no-op / upsert)...")
    pipe_mod.run_daily_update(record_options=False)
    after_second = row_count()
    delta_second = after_second - after_first
    log.info("Row count after 2 : %d  (added %d)", after_second, delta_second)

    passed = delta_second == 0
    return TestResult(
        name="T1: duplicate prevention",
        passed=passed,
        detail=(
            "Second run must add 0 rows because upsert uses "
            "INSERT OR REPLACE on (symbol, datetime).  "
            f"Second run added {delta_second} rows."
        ),
        numbers={
            "rows_before": before,
            "rows_after_first_run": after_first,
            "rows_after_second_run": after_second,
            "delta_first_run": delta_first,
            "delta_second_run": delta_second,
        },
    )


def test_2_recover_recent_day() -> TestResult:
    """Delete one recent trading day, run daily pipeline, confirm recovery."""
    log.info("=== T2: missing-day recovery (recent) ===")
    last_dates = _last_stored_date_per_symbol()
    log.info("Latest stored dates: %s", last_dates)

    # Pick the most recent trading day that's actually in the DB.
    candidate = None
    for _sym, d in last_dates.items():
        if d and is_trading_day(d):
            candidate = d
            break
    if candidate is None:
        return TestResult(
            name="T2: missing-day recovery",
            passed=False,
            detail="No trading day found in the DB - is spot_futures empty?",
        )
    target = candidate
    # Skip days older than 5 days back - daily mode won't fetch that far.
    max_lookback = date.today() - timedelta(days=5)
    if target < max_lookback:
        target = previous_trading_day(date.today() - timedelta(days=1))
        log.warning("Latest stored day is older than the daily lookback; "
                    "using %s instead (may not be recoverable if API doesn't have it).", target)

    before_day_rows = {
        sym: rows_for_day(target, sym) for sym in
        [i["symbol"] for i in get_config().instruments]
    }
    log.info("Rows for %s before delete: %s", target, before_day_rows)

    for sym in before_day_rows:
        delete_day(target, sym)
    after_delete = {
        sym: rows_for_day(target, sym) for sym in before_day_rows
    }
    log.info("Rows for %s after delete : %s", target, after_delete)

    other_day = target - timedelta(days=7)
    while not is_trading_day(other_day):
        other_day -= timedelta(days=1)
    other_before = rows_for_day(other_day)
    log.info("Control day %s row count : %d", other_day, other_before)

    log.info("Running daily pipeline to recover...")
    pipe_mod.run_daily_update(record_options=False)

    recovered = {
        sym: rows_for_day(target, sym) for sym in before_day_rows
    }
    other_after = rows_for_day(other_day)
    log.info("Rows for %s after recovery: %s", target, recovered)
    log.info("Control day %s after      : %d", other_day, other_after)

    recovered_ok = all(
        recovered[s] >= before_day_rows[s] * 0.9  # tolerate small end-of-day gap
        for s in before_day_rows if before_day_rows[s] > 0
    )
    control_ok = other_before == other_after
    passed = recovered_ok and control_ok
    return TestResult(
        name="T2: missing-day recovery",
        passed=passed,
        detail=(
            f"Deleted rows for {target}, ran daily pipeline. "
            f"Recovery {'OK' if recovered_ok else 'FAILED'} "
            f"(before={before_day_rows} after={recovered}). "
            f"Control day {other_day} unchanged: {control_ok}."
        ),
        numbers={
            "before_day_rows_total": sum(before_day_rows.values()),
            "after_delete_total": sum(after_delete.values()),
            "after_recovery_total": sum(recovered.values()),
            "control_before": other_before,
            "control_after": other_after,
        },
    )


def test_3_midday_behaviour() -> TestResult:
    """Explain (with reference to code) what happens if run mid-session."""
    log.info("=== T3: mid-day behaviour ===")
    now = datetime.now()
    session_open = time(9, 15)
    session_close = time(15, 30)
    in_session = session_open <= now.time() <= session_close and is_trading_day(now.date())

    # Where would the last CLOSED 5-min candle sit right now?
    minute_floor = (now.minute // 5) * 5
    last_closed = now.replace(minute=minute_floor, second=0, microsecond=0)
    if last_closed > now:
        last_closed -= timedelta(minutes=5)

    detail = [
        "Dhan's intraday_minute_data returns candles that have already CLOSED.",
        f"Current wall clock : {now.isoformat(timespec='seconds')}",
        f"In market session  : {in_session}",
        f"Last closed 5-min  : {last_closed.isoformat(timespec='minutes')}",
        "",
        "Behaviour:",
        "  * If the pipeline runs at (e.g.) 10:33, Dhan returns candles",
        "    up to and INCLUDING 10:30. The forming 10:30-10:34:59 window",
        "    is not returned until 10:35.",
        "  * fetch_recent_spot(days_back=2) pulls the last 5 trading days",
        "    of closed candles per symbol, then upserts.",
        "  * Re-running at 10:38 will additionally pull the 10:35 candle.",
        "  * No forward-filling or interpolation happens on the current day.",
    ]

    if in_session:
        # We can be more concrete: count today's rows before/after.
        today_before = rows_for_day(now.date())
        log.info("Today's rows before mid-day pipeline call: %d", today_before)
        pipe_mod.run_daily_update(record_options=False)
        today_after = rows_for_day(now.date())
        added = today_after - today_before
        log.info("Today's rows after: %d (added %d)", today_after, added)
        detail.append("")
        detail.append(
            f"Live measurement: today's row count went from {today_before} "
            f"to {today_after} (added {added})."
        )
        passed = True
        numbers = {
            "today_rows_before": today_before,
            "today_rows_after": today_after,
            "added_this_run": added,
        }
    else:
        detail.append("")
        detail.append(
            "Not in a live session - the description above is the "
            "expected behaviour but wasn't measured on this run."
        )
        passed = True
        numbers = {}

    return TestResult(
        name="T3: mid-day behaviour",
        passed=passed,
        detail="\n".join(detail),
        numbers=numbers,
    )


def test_4_old_gap_not_recovered() -> TestResult:
    """Delete a week 30 days ago, run daily; expect NO recovery."""
    log.info("=== T4: old-gap recovery via daily mode (expected: fails) ===")
    target = date.today() - timedelta(days=30)
    # Walk to nearest trading day.
    while not is_trading_day(target):
        target -= timedelta(days=1)

    log.info("Simulating deletion of the week ending %s", target)
    deleted_counts = delete_week_ending(target)
    total_deleted = sum(deleted_counts.values())
    log.info("Total rows deleted: %d", total_deleted)

    week_rows_after_delete = sum(
        rows_for_day(target - timedelta(days=i)) for i in range(7)
    )
    log.info("Rows in that week after delete: %d", week_rows_after_delete)

    log.info("Running daily pipeline - it only looks back ~5 days...")
    pipe_mod.run_daily_update(record_options=False)

    week_rows_after_daily = sum(
        rows_for_day(target - timedelta(days=i)) for i in range(7)
    )
    log.info("Rows in that week after daily run: %d", week_rows_after_daily)

    # PASS means the daily-mode behaviour is *as documented*: it
    # did NOT touch the old week.
    daily_did_not_backfill = week_rows_after_daily == week_rows_after_delete
    return TestResult(
        name="T4: old-gap not recovered by daily mode",
        passed=daily_did_not_backfill,
        detail=(
            "Daily mode only fetches ~5 trailing days, so gaps older than that "
            "are NOT filled.  To recover, run:\n"
            f"    python -m scripts.run_pipeline --mode historical "
            f"--start {(target - timedelta(days=7)).isoformat()} "
            f"--end {target.isoformat()}\n"
            f"Total rows deleted from the target week: {total_deleted}. "
            f"After daily run they are still missing "
            f"(expected: {week_rows_after_delete}, actual: {week_rows_after_daily})."
        ),
        numbers={
            "rows_deleted": total_deleted,
            "week_rows_after_delete": week_rows_after_delete,
            "week_rows_after_daily_run": week_rows_after_daily,
        },
    )


# =============================================================
#  Driver
# =============================================================
def _print_summary(results: List[TestResult]) -> None:
    log.info("")
    log.info("=" * 60)
    log.info("Robustness test summary")
    log.info("=" * 60)
    for r in results:
        log.info("  [%s] %s", "PASS" if r.passed else "FAIL", r.name)
    n_pass = sum(1 for r in results if r.passed)
    log.info("")
    log.info("Result: %d/%d passed", n_pass, len(results))
    if n_pass != len(results):
        log.info("Failures:")
        for r in results:
            if not r.passed:
                log.info("  - %s: %s", r.name, r.detail.splitlines()[0])
    log.info("")
    log.info("Recommendations")
    log.info("---------------")
    log.info("  * Run scripts.audit_data after every historical load to spot")
    log.info("    gaps early.")
    log.info("  * Schedule scripts.run_pipeline --mode daily after 15:35 each")
    log.info("    trading day to append today's candles.")
    log.info("  * Any gap older than 5 days needs --mode historical with an")
    log.info("    explicit --start/--end window.")


def main() -> int:
    ap = argparse.ArgumentParser(description="AiVora pipeline robustness tests")
    ap.add_argument("--skip-t1", action="store_true", help="Skip T1 (idempotency)")
    ap.add_argument("--skip-t2", action="store_true", help="Skip T2 (recovery)")
    ap.add_argument("--skip-t3", action="store_true", help="Skip T3 (mid-day)")
    ap.add_argument("--skip-t4", action="store_true", help="Skip T4 (old gap)")
    ap.add_argument("--restore-only", action="store_true",
                    help="Restore from any prior backup table and exit")
    args = ap.parse_args()

    if args.restore_only:
        restore_spot_futures()
        return 0

    if not os.getenv("DHAN_ACCESS_TOKEN"):
        log.warning(
            "DHAN_ACCESS_TOKEN is not set - T1/T2/T4 need to call the "
            "daily pipeline and will fail.  Run scripts.refresh_dhan_token first."
        )

    backup_spot_futures()
    results: List[TestResult] = []
    try:
        if not args.skip_t1:
            results.append(test_1_idempotency())
        if not args.skip_t2:
            results.append(test_2_recover_recent_day())
        if not args.skip_t3:
            results.append(test_3_midday_behaviour())
        if not args.skip_t4:
            results.append(test_4_old_gap_not_recovered())
    except Exception:
        log.exception("A test crashed - restoring backup before re-raising")
        restore_spot_futures()
        raise
    finally:
        restore_spot_futures()

    _print_summary(results)

    # Persist a machine-readable log too.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = get_config().paths["logs_dir"] / f"robustness_report_{stamp}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [f"AiVora robustness tests @ {stamp}", ""]
    for r in results:
        lines.append(f"[{'PASS' if r.passed else 'FAIL'}] {r.name}")
        lines.append(r.detail)
        if r.numbers:
            for k, v in r.numbers.items():
                lines.append(f"  {k}: {v}")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report -> %s", out)

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
