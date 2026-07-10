"""Model evaluation utilities.

Reports go to ``reports/`` (text) and ``reports/plots/`` (PNG).
We deliberately keep matplotlib usage simple: one figure per plot,
no shared global state, ``Agg`` backend so it runs in CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from ..utils.config import get_config  # noqa: E402
from ..utils.logger import get_logger  # noqa: E402
from .dataset import Splits  # noqa: E402

log = get_logger(__name__)

CLASS_NAMES = ["FLAT", "DOWN", "UP"]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def evaluate_model(model, splits: Splits) -> Dict:
    """Compute the standard metrics on the held-out test fold.

    Also writes a confusion-matrix heatmap and a feature-importance
    bar chart so the user can sanity-check the model visually.
    """
    cfg = get_config()
    reports_dir = _ensure_dir(cfg.paths["reports_dir"])
    plots_dir = _ensure_dir(cfg.paths["reports_dir"] / "plots")

    probs = model.predict(splits.X_test)
    preds = probs.argmax(axis=1)
    y_test = splits.y_test.values

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])
    report = classification_report(
        y_test, preds, target_names=CLASS_NAMES, digits=4, zero_division=0
    )

    # Directional accuracy ignores FLAT and reports the hit rate
    # on the trades we'd actually take.
    dir_mask = np.isin(preds, [1, 2])
    if dir_mask.any():
        directional_acc = float((preds[dir_mask] == y_test[dir_mask]).mean())
    else:
        directional_acc = float("nan")

    up_mask = preds == 2
    down_mask = preds == 1
    up_acc = float((y_test[up_mask] == 2).mean()) if up_mask.any() else float("nan")
    down_acc = float((y_test[down_mask] == 1).mean()) if down_mask.any() else float("nan")

    log.info("Test accuracy = %.4f", acc)
    log.info("Directional accuracy = %.4f", directional_acc)
    log.info("UP precision = %.4f | DOWN precision = %.4f", up_acc, down_acc)
    log.info("\n%s", report)

    # ---- Confusion matrix plot ----
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test)")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(plots_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    # ---- Feature importance plot ----
    try:
        importances = model.feature_importance(importance_type="gain")
        order = np.argsort(importances)[-25:]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(
            np.array(splits.feature_cols)[order],
            np.array(importances)[order],
            color="steelblue",
        )
        ax.set_title("Top-25 feature importance (gain)")
        fig.tight_layout()
        fig.savefig(plots_dir / "feature_importance.png", dpi=120)
        plt.close(fig)
    except Exception as exc:
        log.warning("Could not plot feature importance: %s", exc)

    # ---- Text report ----
    report_path = reports_dir / "classification_report.txt"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write(report)
        fh.write(f"\nAccuracy: {acc:.4f}\n")
        fh.write(f"Directional accuracy: {directional_acc:.4f}\n")
        fh.write(f"UP precision: {up_acc:.4f}\n")
        fh.write(f"DOWN precision: {down_acc:.4f}\n")
    log.info("evaluate_model: wrote %s", report_path)

    return {
        "accuracy": float(acc),
        "directional_accuracy": directional_acc,
        "up_precision": up_acc,
        "down_precision": down_acc,
        "confusion_matrix": cm.tolist(),
        "probs": probs,
        "preds": preds,
    }
