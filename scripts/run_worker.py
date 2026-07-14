"""Standalone scheduler worker — runs ticks without the dashboard being open.

Previous architecture: APScheduler lived inside the Streamlit process
and only got armed on dashboard visits. If nobody visited, no ticks
fired. Container restart + no visit = silent trading outage.

This worker is a dedicated container that:

  1. Reads every user_portfolios row with master_switch=1 from the
     webapp DB on startup.
  2. Registers a per-user tick with APScheduler (cron: */5, sec=20,
     Asia/Kolkata).
  3. Reruns the discovery every 60s so a user who flips their master
     switch ON via the dashboard gets picked up automatically.
  4. Runs forever — restart:unless-stopped in compose keeps it alive.

The dashboard remains for UI only; it no longer registers ticks.
"""
from __future__ import annotations

import signal
import sys
import time
from pathlib import Path
from typing import Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from apscheduler.schedulers.background import BackgroundScheduler  # noqa: E402
from apscheduler.triggers.cron import CronTrigger  # noqa: E402

from aivora.utils.logger import get_logger  # noqa: E402
from aivora.webapp import db as webapp_db  # noqa: E402
from aivora.webapp.trading_engine import run_user_tick  # noqa: E402

log = get_logger("scripts.run_worker")

# (user_id, mode) tuples currently registered with the scheduler.
_ACTIVE: Set[Tuple[int, str]] = set()


def _active_portfolios() -> Set[Tuple[int, str]]:
    """Read the webapp DB for all portfolios with master_switch=1."""
    webapp_db.init_db()
    with webapp_db.connect() as c:
        rows = c.execute(
            "SELECT user_id, mode FROM user_portfolios WHERE master_switch = 1"
        ).fetchall()
    return {(int(r[0]), str(r[1])) for r in rows}


def _tick_job(user_id: int, mode: str) -> None:
    """APScheduler job wrapper — never raises so the scheduler keeps running."""
    try:
        run_user_tick(user_id, mode)
    except Exception as exc:  # noqa: BLE001
        log.exception("tick fatal for user_id=%s mode=%s: %s", user_id, mode, exc)


def _sync(scheduler: BackgroundScheduler) -> None:
    """Add/remove APScheduler jobs so they match the active portfolios."""
    desired = _active_portfolios()

    # Add jobs that appeared.
    for user_id, mode in desired - _ACTIVE:
        job_id = f"tick-{user_id}-{mode}"
        scheduler.add_job(
            _tick_job, args=[user_id, mode],
            trigger=CronTrigger(minute="*/5", second=20, timezone="Asia/Kolkata"),
            id=job_id, replace_existing=True,
            max_instances=1, coalesce=True,
        )
        log.info("registered tick: user=%s mode=%s", user_id, mode)

    # Remove jobs that disappeared (user turned master switch OFF).
    for user_id, mode in _ACTIVE - desired:
        job_id = f"tick-{user_id}-{mode}"
        scheduler.remove_job(job_id)
        log.info("unregistered tick: user=%s mode=%s", user_id, mode)

    _ACTIVE.clear()
    _ACTIVE.update(desired)


def main() -> int:
    log.info("=== AiVora worker starting ===")
    scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
    scheduler.start()

    # Graceful shutdown on SIGTERM / SIGINT.
    def _shutdown(*_):
        log.info("worker: shutdown signal received")
        scheduler.shutdown(wait=False)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # Sync every 60s so dashboard-triggered master-switch changes propagate.
    while True:
        try:
            _sync(scheduler)
        except Exception as exc:  # noqa: BLE001
            log.exception("_sync failed: %s", exc)
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
