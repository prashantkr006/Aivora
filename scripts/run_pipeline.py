"""Entry point: run the ETL pipeline (DhanHQ-only).

Usage::

    python -m scripts.run_pipeline --mode historical
    python -m scripts.run_pipeline --mode daily
    python -m scripts.run_pipeline --mode rebuild
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Make ``src`` importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aivora.pipeline import pipeline  # noqa: E402
from aivora.utils.logger import get_logger  # noqa: E402

log = get_logger("scripts.run_pipeline")


def _parse_date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def main() -> int:
    ap = argparse.ArgumentParser(description="AiVora ETL runner (DhanHQ)")
    ap.add_argument(
        "--mode",
        choices=["historical", "daily", "rebuild"],
        default="historical",
        help=(
            "historical = multi-month DhanHQ cold start;  "
            "daily = incremental DhanHQ pull;  "
            "rebuild = re-run feature engineering only"
        ),
    )
    ap.add_argument("--start", type=_parse_date, default=None,
                    help="Override historical start date (YYYY-MM-DD)")
    ap.add_argument("--end", type=_parse_date, default=None,
                    help="Override historical end date (YYYY-MM-DD)")
    args = ap.parse_args()

    try:
        if args.mode == "historical":
            out = pipeline.run_historical_load(args.start, args.end)
        elif args.mode == "daily":
            out = pipeline.run_daily_update()
        else:
            out = pipeline.build_training_dataset()
        log.info("Pipeline finished — output: %s", out)
        return 0
    except Exception:
        log.exception("Pipeline failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
