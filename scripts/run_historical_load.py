"""Dedicated historical-load entrypoint.

Runs the multi-month DhanHQ pull, prints a summary, then triggers
feature engineering so the training Parquet is ready immediately.

Usage::

    python -m scripts.run_historical_load
    python -m scripts.run_historical_load --start 2026-01-05 --end 2026-07-05
    python -m scripts.run_historical_load --symbol NIFTY
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aivora.pipeline import pipeline  # noqa: E402
from aivora.pipeline.dhan_client import DhanClient  # noqa: E402
from aivora.utils.config import get_config  # noqa: E402
from aivora.utils.logger import get_logger  # noqa: E402

log = get_logger("scripts.run_historical_load")


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _print_summary(client_df) -> None:
    """Log a compact summary of the raw combined dataframe."""
    if client_df.empty:
        log.warning("Loader returned no rows.")
        return
    by_symbol = client_df.groupby("symbol").agg(
        rows=("datetime", "size"),
        start=("datetime", "min"),
        end=("datetime", "max"),
    )
    log.info("\n%s", by_symbol.to_string())


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the full DhanHQ historical load + feature build"
    )
    ap.add_argument("--start", type=_parse_date, default=None,
                    help="Start date (YYYY-MM-DD). Default = config.historical.start_date.")
    ap.add_argument("--end", type=_parse_date, default=None,
                    help="End date (YYYY-MM-DD). Default = config.historical.end_date or today.")
    ap.add_argument("--symbol", choices=["NIFTY", "BANKNIFTY"], default=None,
                    help="Limit to a single symbol (default = all configured)")
    ap.add_argument("--skip-features", action="store_true",
                    help="Stop after raw load; skip feature engineering")
    args = ap.parse_args()

    cfg = get_config()
    hist_cfg = cfg.raw.get("historical", {})
    start = args.start or _parse_date(hist_cfg.get("start_date", "2026-01-05"))
    end = args.end or (
        _parse_date(hist_cfg["end_date"]) if hist_cfg.get("end_date") else date.today()
    )
    symbols = [args.symbol] if args.symbol else None

    log.info(
        "Starting full historical load - range=%s -> %s, symbols=%s",
        start, end, symbols or "all",
    )

    try:
        client = DhanClient()
        combined = client.load_full_historical_data(
            start_date=start,
            end_date=end,
            symbols=symbols,
        )
        log.info("Raw combined rows: %d", len(combined))
        _print_summary(combined)

        if args.skip_features:
            log.info("--skip-features set; not running feature engineering.")
            return 0

        # Persist into SQLite (so the daily-update path can keep
        # appending) and produce the training Parquet. Reuse the
        # already-fetched dataframe instead of pulling it again.
        out = pipeline.run_historical_load(start, end, combined=combined)
        log.info("Training Parquet written: %s", out)
        return 0
    except Exception:
        log.exception("Historical load failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
