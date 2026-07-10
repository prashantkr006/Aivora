"""Per-user 5-minute tick manager.

Wraps a single APScheduler ``BackgroundScheduler`` that owns one
job per (user_id, mode) pair whose master switch is ON.  Adding,
removing and inspecting jobs is idempotent so the UI can call
:func:`sync_user` on every render without side effects.

The tick body itself is deliberately thin — it delegates to a
callback function so the scheduler stays test-friendly (we can
inject a mock in the smoke test).
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, Optional

from ..utils.logger import get_logger

log = get_logger(__name__)

_LOCK = threading.RLock()
_SCHEDULER = None                        # single BackgroundScheduler
_JOBS: Dict[str, str] = {}               # (user_id, mode) → job_id
TickFn = Callable[[int, str], None]      # signature: fn(user_id, mode) -> None
_TICK_FN: Optional[TickFn] = None


# =============================================================
#  Public API — used by the UI
# =============================================================
def set_tick_function(fn: TickFn) -> None:
    """Register the per-user tick body.  Called once at import."""
    global _TICK_FN
    with _LOCK:
        _TICK_FN = fn


def ensure_started(interval_seconds: int = 300) -> None:
    """Boot the shared APScheduler if it isn't already running."""
    global _SCHEDULER
    with _LOCK:
        if _SCHEDULER is not None and _SCHEDULER.running:
            return
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
        except ImportError as exc:
            log.error("apscheduler not installed: %s", exc)
            raise
        _SCHEDULER = BackgroundScheduler(daemon=True, timezone="Asia/Kolkata")
        _SCHEDULER.start()
        log.info("scheduler booted (default interval=%ds)", interval_seconds)


def sync_user(user_id: int, mode: str, master_switch: bool,
              interval_seconds: int = 300) -> None:
    """Reconcile the scheduler with the desired state for one user.

    Ticks fire on the **standard market 5-minute boundary + 20 sec**
    — i.e. 09:15:20, 09:20:20, ... 15:30:20 IST — regardless of
    when the user toggled the master switch.  That way every user's
    trade decisions are aligned to the same candle-close times that
    a human trader would look at (00, 05, 10, … minutes past the
    hour).  The 20-second offset gives Kite Connect a moment to
    finalise the just-closed candle before we fetch it.
    """
    key = _key(user_id, mode)
    with _LOCK:
        ensure_started(interval_seconds)
        if master_switch:
            if key in _JOBS:
                return
            job_id = f"aivora-{key}"
            _SCHEDULER.add_job(
                _tick_wrapper,
                "cron",
                minute="*/5",     # 00, 05, 10, ..., 55
                second=20,        # 20-sec buffer after candle close
                id=job_id,
                args=[int(user_id), mode],
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )
            _JOBS[key] = job_id
            log.info("scheduler+ user=%s mode=%s (cron: every 5-min mark + 20s)",
                     user_id, mode)
        else:
            job_id = _JOBS.pop(key, None)
            if job_id:
                try:
                    _SCHEDULER.remove_job(job_id)
                    log.info("scheduler- user=%s mode=%s", user_id, mode)
                except Exception as exc:  # noqa: BLE001
                    log.warning("remove_job(%s) failed: %s", job_id, exc)


def active_users() -> Dict[str, str]:
    with _LOCK:
        return dict(_JOBS)


def shutdown() -> None:
    global _SCHEDULER
    with _LOCK:
        if _SCHEDULER is not None and _SCHEDULER.running:
            _SCHEDULER.shutdown(wait=False)
            _SCHEDULER = None
        _JOBS.clear()


# =============================================================
#  Internals
# =============================================================
def _key(user_id: int, mode: str) -> str:
    return f"{int(user_id)}:{mode}"


def _tick_wrapper(user_id: int, mode: str) -> None:
    """Guard around the injected tick fn so one user's crash never
    kills another user's job."""
    if _TICK_FN is None:
        log.warning("no tick function registered; skipping user=%s", user_id)
        return
    try:
        _TICK_FN(int(user_id), mode)
    except Exception as exc:  # noqa: BLE001
        log.exception("user=%s mode=%s tick raised: %s", user_id, mode, exc)
