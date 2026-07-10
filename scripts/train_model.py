"""Entry point: train + evaluate + backtest in one shot.

Usage::

    python -m scripts.train_model --trials 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aivora.backtest.backtester import run_backtest  # noqa: E402
from aivora.ml import registry, train  # noqa: E402
from aivora.ml.dataset import make_splits  # noqa: E402
from aivora.ml.evaluate import evaluate_model  # noqa: E402
from aivora.utils.logger import get_logger  # noqa: E402

log = get_logger("scripts.train_model")


def main() -> int:
    ap = argparse.ArgumentParser(description="Train and evaluate the AiVora model")
    ap.add_argument("--trials", type=int, default=None,
                    help="Override Optuna trial count from config.yaml")
    ap.add_argument("--name", type=str, default="lgbm_model.pkl",
                    help="Filename for the saved model")
    args = ap.parse_args()

    try:
        splits = make_splits()
        model, metadata = train.tune_and_train(splits, n_trials=args.trials)
        model_path = train.save_model(model, metadata, name=args.name)

        test_results = evaluate_model(model, splits)
        bt_results = run_backtest(test_results["probs"], splits)

        metrics = {
            "accuracy": test_results["accuracy"],
            "directional_accuracy": test_results["directional_accuracy"],
            "up_precision": test_results["up_precision"],
            "down_precision": test_results["down_precision"],
            **bt_results["summary"],
        }
        version = registry.register(model_path, metadata, metrics)
        log.info("All done — registered as %s", version)
        return 0
    except Exception:
        log.exception("Training failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
