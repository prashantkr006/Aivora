"""Monthly retrain wrapper.

Rebuilds the training Parquet from the latest SQLite state and freezes
a fresh binary UP/DOWN model pair on the trailing 13 months of data
(12 train, 1 validate) — the same shape production live-inference
consumes.  Writes status to ``models/retrain_status.json`` so the
admin dashboard can show a live "in progress / done / failed"
indicator without shelling out.

Two intended callers:

  * Cron — 1st Saturday of every month, 20:00 IST.
  * Admin dashboard button — "Retrain now" (admin only, blocked
    during market hours).

A lock file at ``models/retrain.lock`` prevents concurrent runs (cron
overlapping with a manual click).  Failures leave the lock behind so
the operator sees the last-attempt error; the script clears an old
lock on start if its recorded PID is no longer alive.

Usage::

    docker compose exec dashboard python -m scripts.monthly_retrain
    docker compose exec dashboard python -m scripts.monthly_retrain --tag manual
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess  # noqa: F401  (re-exported for callers if they wrap us)
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aivora.pipeline import pipeline  # noqa: E402
from aivora.utils.config import get_config  # noqa: E402
from aivora.utils.logger import get_logger  # noqa: E402

log = get_logger("scripts.monthly_retrain")


def _paths():
    cfg = get_config()
    models_dir = Path(cfg.paths["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    return {
        "lock": models_dir / "retrain.lock",
        "status": models_dir / "retrain_status.json",
        "up": models_dir / "current_up.pkl",
        "down": models_dir / "current_down.pkl",
    }


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _acquire_lock(lock_path: Path) -> bool:
    """Best-effort atomic lock via O_EXCL.  Cleans up stale locks
    whose PID is no longer running."""
    if lock_path.exists():
        try:
            old_pid = int(lock_path.read_text().strip())
        except (OSError, ValueError):
            old_pid = 0
        if old_pid and _pid_alive(old_pid):
            log.warning("retrain: another run (pid=%s) is in progress", old_pid)
            return False
        log.warning("retrain: clearing stale lock (pid=%s not running)", old_pid)
        try:
            lock_path.unlink()
        except OSError:
            pass
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(fd, str(os.getpid()).encode("ascii"))
        os.close(fd)
        return True
    except FileExistsError:
        log.warning("retrain: lock re-appeared during acquisition")
        return False


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except OSError:
        pass


def _write_status(status_path: Path, payload: dict) -> None:
    try:
        status_path.write_text(json.dumps(payload, indent=2, default=str))
    except OSError as exc:
        log.warning("could not write status file: %s", exc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="monthly",
                    help="Free-form label written to the status file "
                         "(e.g. 'monthly', 'manual').")
    ap.add_argument("--train-months", type=int, default=12)
    ap.add_argument("--val-months", type=int, default=1)
    args = ap.parse_args()

    P = _paths()
    if not _acquire_lock(P["lock"]):
        _write_status(P["status"], {
            "state": "skipped",
            "reason": "another retrain is already running",
            "attempted_at": datetime.now().isoformat(timespec="seconds"),
            "tag": args.tag,
        })
        return 2

    started_at = datetime.now()
    _write_status(P["status"], {
        "state": "running",
        "tag": args.tag,
        "started_at": started_at.isoformat(timespec="seconds"),
        "pid": os.getpid(),
    })

    try:
        log.info("retrain[%s]: step 1/2 — rebuild parquet from SQLite", args.tag)
        pipeline.build_training_dataset()

        log.info("retrain[%s]: step 2/2 — freeze binary UP/DOWN model", args.tag)
        # freeze_model does its own imports; call it as a module so any
        # future signature changes stay contained to that file.
        rc = subprocess.run(
            [sys.executable, "-m", "scripts.freeze_model",
             "--train-months", str(args.train_months),
             "--val-months", str(args.val_months)],
            check=False,
        ).returncode
        if rc != 0:
            raise RuntimeError(f"freeze_model exited with code {rc}")

        finished_at = datetime.now()
        _write_status(P["status"], {
            "state": "success",
            "tag": args.tag,
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "duration_sec": (finished_at - started_at).total_seconds(),
            "model_up_mtime": P["up"].stat().st_mtime if P["up"].exists() else None,
            "model_down_mtime": P["down"].stat().st_mtime if P["down"].exists() else None,
        })
        log.info("retrain[%s]: SUCCESS in %.1fs",
                 args.tag, (finished_at - started_at).total_seconds())
        return 0
    except Exception as exc:  # noqa: BLE001
        finished_at = datetime.now()
        _write_status(P["status"], {
            "state": "failed",
            "tag": args.tag,
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "error": str(exc),
        })
        log.exception("retrain[%s]: FAILED", args.tag)
        return 1
    finally:
        _release_lock(P["lock"])


if __name__ == "__main__":
    raise SystemExit(main())
