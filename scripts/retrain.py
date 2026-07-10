"""Periodic retraining job.

Convenience wrapper that:

1. Pulls fresh candles from Dhan (``run_daily_update``).
2. Re-runs the full training pipeline.
3. Registers the new model version.

Drop this into Windows Task Scheduler / cron to retrain weekly or
whenever the daily update brings in significant new data.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aivora.backtest.backtester import run_backtest  # noqa: E402
from aivora.ml import registry, train  # noqa: E402
from aivora.ml.dataset import make_splits  # noqa: E402
from aivora.ml.evaluate import evaluate_model  # noqa: E402
from aivora.pipeline import pipeline  # noqa: E402
from aivora.utils.logger import get_logger  # noqa: E402

log = get_logger("scripts.retrain")


def main() -> int:
    try:
        log.info("Step 1/3 — daily update")
        pipeline.run_daily_update()

        log.info("Step 2/3 — training")
        splits = make_splits()
        model, metadata = train.tune_and_train(splits)
        model_path = train.save_model(model, metadata)

        log.info("Step 3/3 — evaluation + registration")
        test_results = evaluate_model(model, splits)
        bt = run_backtest(test_results["probs"], splits)
        metrics = {
            "accuracy": test_results["accuracy"],
            "directional_accuracy": test_results["directional_accuracy"],
            "up_precision": test_results["up_precision"],
            "down_precision": test_results["down_precision"],
            **bt["summary"],
        }
        version = registry.register(model_path, metadata, metrics)
        log.info("Retrain complete — registered as %s", version)
        return 0
    except Exception:
        log.exception("Retrain failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
