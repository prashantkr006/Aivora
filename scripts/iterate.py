"""Iterative improvement loop.

Runs up to N model / strategy variants against the same training
parquet, comparing each against the ``goals:`` section of
``config.yaml``.  Stops early on the first variant that meets all
targets, otherwise reports the closest-scoring variant and its
gaps.

Each iteration produces two files under ``logs/``:

    iteration_XX_changes.txt   — what was changed vs baseline
    iteration_XX_results.txt   — backtest summary + goal report

A final ``logs/final_report.txt`` summarises the whole run.

Nothing here fetches new data — it operates on whatever training
Parquet already exists.  If the Parquet is missing, the script
prints a clear "run the historical load first" message and
exits non-zero.

Usage::

    python -m scripts.iterate                     # run all variants
    python -m scripts.iterate --max-iterations 5  # cap early
    python -m scripts.iterate --list              # just list variants
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aivora.backtest.backtester import run_backtest  # noqa: E402
from aivora.ml import binary as bin_mod  # noqa: E402
from aivora.ml import goals as goals_mod  # noqa: E402
from aivora.ml import registry as reg_mod  # noqa: E402
from aivora.ml import train as train_mod  # noqa: E402
from aivora.ml import walk_forward as wf_mod  # noqa: E402
from aivora.ml.dataset import Splits, make_splits  # noqa: E402
from aivora.ml.evaluate import evaluate_model  # noqa: E402
from aivora.pipeline import feature_engineering, pipeline  # noqa: E402
from aivora.pipeline.database import load_option_chain, load_spot_futures  # noqa: E402
from aivora.utils.config import get_config  # noqa: E402
from aivora.utils.logger import get_logger  # noqa: E402

log = get_logger("scripts.iterate")


# =============================================================
#  Variant definitions
# =============================================================
@dataclass
class Variant:
    name: str
    description: str
    # Overrides on top of the baseline config.  Empty dict = baseline.
    label_overrides: Dict[str, Any] = field(default_factory=dict)
    backtest_overrides: Dict[str, Any] = field(default_factory=dict)
    train_overrides: Dict[str, Any] = field(default_factory=dict)
    # If True, force a full parquet rebuild + retrain (label/feature change).
    rebuild_dataset: bool = False
    # If True, retrain the model even when the dataset is unchanged.
    retrain: bool = True
    # Advanced pipelines the vanilla runner doesn't know about.
    #   "3class"       — the existing softmax + threshold path (default).
    #   "binary"       — pair of UP-vs-rest / DOWN-vs-rest LightGBMs.
    #   "walk_forward" — rolling monthly retrain, 3-class model.
    #   "walk_forward_binary" — walk-forward + binary pair per fold.
    strategy: str = "3class"
    # Kwargs forwarded to walk-forward-based strategies.
    wf_kwargs: Dict[str, Any] = field(default_factory=dict)


VARIANTS: List[Variant] = [
    Variant(
        name="01_baseline",
        description="Config defaults — static ±0.3 % labels, prob>=0.65, both symbols.",
    ),
    Variant(
        name="02_high_threshold",
        description="Only trade when top-1 probability >= 0.50.",
        backtest_overrides={"probability_threshold": 0.50},
        retrain=False,
    ),
    Variant(
        name="03_very_high_threshold",
        description="Only trade when top-1 probability >= 0.55.",
        backtest_overrides={"probability_threshold": 0.55},
        retrain=False,
    ),
    Variant(
        name="04_dynamic_atr_labels",
        description="Dynamic ATR-scaled labels (k=0.5) — adapts to vol regime.",
        label_overrides={"mode": "dynamic_atr", "atr_k": 0.5},
        rebuild_dataset=True,
    ),
    Variant(
        name="05_nifty_only",
        description="Trade NIFTY exclusively; BANKNIFTY filtered out.",
        backtest_overrides={"symbols_allow_list": ["NIFTY"]},
        retrain=False,
    ),
    Variant(
        name="06_tp_sl_50_30",
        description="Add option-premium TP=+50 %, SL=-30 %.",
        backtest_overrides={"take_profit_pct": 0.50, "stop_loss_pct": 0.30},
        retrain=False,
    ),
    Variant(
        name="07_mid_session_only",
        description="Skip first 30 min and last 45 min of the session.",
        backtest_overrides={
            "min_minutes_since_open": 30, "max_minutes_since_open": 330,
        },
        retrain=False,
    ),
    Variant(
        name="08_short_horizon_15m",
        description="15-minute forward horizon instead of 30.",
        label_overrides={"horizon_candles": 3},
        rebuild_dataset=True,
    ),
    Variant(
        name="09_long_horizon_60m",
        description="60-minute forward horizon.",
        label_overrides={"horizon_candles": 12},
        rebuild_dataset=True,
    ),
    Variant(
        name="10_ensemble_conservative",
        description=(
            "High-conviction + TP/SL + mid-session filter — the mix "
            "of the best knobs above."
        ),
        backtest_overrides={
            "probability_threshold": 0.55,
            "take_profit_pct": 0.50,
            "stop_loss_pct": 0.30,
            "min_minutes_since_open": 30,
            "max_minutes_since_open": 330,
        },
        retrain=False,
    ),

    # -------- New variants: walk-forward + binary pair + regime -----
    Variant(
        name="11_walk_forward_baseline",
        description=(
            "Rolling monthly retrain over the full 4-year history — "
            "gives an honest 12+ month sample of monthly returns."
        ),
        strategy="walk_forward",
        wf_kwargs={"train_months": 12, "val_months": 1,
                   "test_months": 1, "step_months": 1},
        backtest_overrides={"probability_threshold": 0.42},
        rebuild_dataset=True,
    ),
    Variant(
        name="12_binary_up_down",
        description=(
            "Two independent binary models (UP-vs-rest / DOWN-vs-rest); "
            "escape the softmax collapse toward FLAT."
        ),
        strategy="binary",
        backtest_overrides={
            "probability_threshold": 0.55,
            "take_profit_pct": 0.50, "stop_loss_pct": 0.30,
            "min_minutes_since_open": 30, "max_minutes_since_open": 330,
        },
    ),
    Variant(
        name="13_wf_binary_regime",
        description=(
            "Walk-forward + binary pair + mid-vol regime gate + "
            "session filter — most conservative combination."
        ),
        strategy="walk_forward_binary",
        wf_kwargs={"train_months": 12, "val_months": 1,
                   "test_months": 1, "step_months": 1},
        backtest_overrides={
            "probability_threshold": 0.55,
            "take_profit_pct": 0.50, "stop_loss_pct": 0.30,
            "min_minutes_since_open": 30, "max_minutes_since_open": 330,
            "vol_regime_min": 0.20, "vol_regime_max": 0.85,
        },
        rebuild_dataset=True,
    ),
    Variant(
        name="14_wf_asymmetric_long_bias",
        description=(
            "Walk-forward with an asymmetric long bias — NIFTY has a "
            "structural upward drift, so let CEs fire at a lower "
            "threshold than PEs."
        ),
        strategy="walk_forward",
        wf_kwargs={"train_months": 12, "val_months": 1,
                   "test_months": 1, "step_months": 1},
        backtest_overrides={
            "prob_threshold_up": 0.38,
            "prob_threshold_down": 0.48,
            "take_profit_pct": 0.50, "stop_loss_pct": 0.30,
            "min_minutes_since_open": 30, "max_minutes_since_open": 330,
        },
        rebuild_dataset=True,
    ),
    Variant(
        name="15_wf_binary_full_kitchen_sink",
        description=(
            "Walk-forward binary + asymmetric long bias + regime gate + "
            "session filter + TP/SL — everything combined."
        ),
        strategy="walk_forward_binary",
        wf_kwargs={"train_months": 12, "val_months": 1,
                   "test_months": 1, "step_months": 1},
        backtest_overrides={
            "prob_threshold_up": 0.55,
            "prob_threshold_down": 0.60,
            "take_profit_pct": 0.60, "stop_loss_pct": 0.30,
            "min_minutes_since_open": 30, "max_minutes_since_open": 330,
            "vol_regime_min": 0.15, "vol_regime_max": 0.90,
            "max_trades_per_day": 4,
        },
        rebuild_dataset=True,
    ),

    # -------- Win-rate tuning ------------------------------------------
    # Variant 15 hits every goal except win rate (43.2 % vs 45 % target).
    # 16-18 tighten the entry to nudge win rate above 45 % while trying
    # to preserve monthly-return and Sharpe.
    Variant(
        name="16_wf_binary_higher_threshold",
        description=(
            "Same as 15 but tighter thresholds (up=0.62, down=0.65) — "
            "fewer trades, higher-conviction only."
        ),
        strategy="walk_forward_binary",
        wf_kwargs={"train_months": 12, "val_months": 1,
                   "test_months": 1, "step_months": 1},
        backtest_overrides={
            "prob_threshold_up": 0.62,
            "prob_threshold_down": 0.65,
            "take_profit_pct": 0.60, "stop_loss_pct": 0.30,
            "min_minutes_since_open": 30, "max_minutes_since_open": 330,
            "vol_regime_min": 0.15, "vol_regime_max": 0.90,
            "max_trades_per_day": 3,
        },
    ),
    Variant(
        name="17_wf_binary_mid_vol_only",
        description=(
            "Restrict to mid-vol regime (0.30-0.75) — avoids both dead "
            "markets and panic sessions."
        ),
        strategy="walk_forward_binary",
        wf_kwargs={"train_months": 12, "val_months": 1,
                   "test_months": 1, "step_months": 1},
        backtest_overrides={
            "prob_threshold_up": 0.58,
            "prob_threshold_down": 0.62,
            "take_profit_pct": 0.60, "stop_loss_pct": 0.30,
            "min_minutes_since_open": 30, "max_minutes_since_open": 330,
            "vol_regime_min": 0.30, "vol_regime_max": 0.75,
            "max_trades_per_day": 4,
        },
    ),
    Variant(
        name="18_wf_binary_longer_horizon",
        description=(
            "60-minute horizon (12 candles) — the direction has more "
            "time to materialise, target higher win rate."
        ),
        strategy="walk_forward_binary",
        wf_kwargs={"train_months": 12, "val_months": 1,
                   "test_months": 1, "step_months": 1},
        label_overrides={"horizon_candles": 12},
        backtest_overrides={
            "prob_threshold_up": 0.55,
            "prob_threshold_down": 0.60,
            "take_profit_pct": 0.60, "stop_loss_pct": 0.30,
            "min_minutes_since_open": 30, "max_minutes_since_open": 300,
            "vol_regime_min": 0.15, "vol_regime_max": 0.90,
            "max_trades_per_day": 3,
        },
        rebuild_dataset=True,
    ),
    Variant(
        name="19_wf_binary_margin_gate",
        description=(
            "Variant 18 + confidence-margin gate: require p_up - p_down "
            ">= 0.15 (or the reverse for PE).  Trades only when the two "
            "binary models strongly disagree."
        ),
        strategy="walk_forward_binary",
        wf_kwargs={"train_months": 12, "val_months": 1,
                   "test_months": 1, "step_months": 1},
        label_overrides={"horizon_candles": 12},
        backtest_overrides={
            "prob_threshold_up": 0.55,
            "prob_threshold_down": 0.60,
            "prob_margin_min": 0.15,
            "take_profit_pct": 0.60, "stop_loss_pct": 0.30,
            "min_minutes_since_open": 30, "max_minutes_since_open": 300,
            "vol_regime_min": 0.15, "vol_regime_max": 0.90,
            "max_trades_per_day": 3,
        },
    ),
]


# =============================================================
#  Helpers
# =============================================================
def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _dataset_from_db(label_overrides: Optional[Dict]) -> pd.DataFrame:
    """Re-run feature engineering against the DB with modified labels.

    Used by variants that change ``label_overrides`` — we can't
    reuse the on-disk parquet because its labels are stale.
    """
    spot = load_spot_futures()
    opts = load_option_chain()
    if not opts.empty:
        merged = pd.merge(spot, opts, on=["datetime", "symbol"], how="left")
    else:
        merged = spot
        for c in ("ce_ltp", "pe_ltp", "ce_oi", "pe_oi", "ce_iv"):
            merged[c] = pd.NA
    merged = merged.rename(columns={"ce_iv": "iv"})
    return feature_engineering.engineer_features(merged, label_overrides=label_overrides)


def _summarise_signature(v: Variant) -> str:
    """Small string summarising what changes vs baseline — for the changes file."""
    parts = []
    if v.label_overrides:
        parts.append(f"labels={v.label_overrides}")
    if v.train_overrides:
        parts.append(f"train={v.train_overrides}")
    if v.backtest_overrides:
        parts.append(f"backtest={v.backtest_overrides}")
    if not parts:
        return "(no overrides — pure baseline)"
    return " | ".join(parts)


# =============================================================
#  One iteration
# =============================================================
def _run_walk_forward(variant: Variant, df, use_binary: bool):
    """Shared code path for walk-forward and walk-forward-binary strategies."""
    if not use_binary:
        return wf_mod.walk_forward_backtest(
            df,
            backtest_overrides=variant.backtest_overrides,
            name=variant.name,
            **variant.wf_kwargs,
        )
    # Binary + walk-forward: refit two binary models per fold.
    # We can't reuse walk_forward_predict because it trains a
    # single 3-class booster; write a bespoke loop that trains
    # the binary pair per fold and stitches the probabilities.
    from aivora.ml.walk_forward import make_folds as _mf
    from aivora.pipeline.feature_engineering import feature_columns as _fc
    ts = pd.to_datetime(df["datetime"])
    feat_cols = _fc(df)
    folds = _mf(df, **variant.wf_kwargs)
    all_probs = []
    all_meta = []
    for i, fold in enumerate(folds, start=1):
        log.info("[%s] wf-binary fold %d/%d %s → %s",
                 variant.name, i, len(folds),
                 fold.val_end.date(), fold.test_end.date())
        train_mask = (ts >= fold.train_start) & (ts < fold.train_end)
        val_mask = (ts >= fold.train_end) & (ts < fold.val_end)
        test_mask = (ts >= fold.val_end) & (ts < fold.test_end)
        n_tr, n_va, n_te = int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum())
        if n_tr < 500 or n_va < 20 or n_te < 20:
            log.warning("  skip (train=%d val=%d test=%d)", n_tr, n_va, n_te)
            continue
        splits_fold = Splits(
            X_train=df.loc[train_mask, feat_cols].astype(np.float32),
            y_train=df.loc[train_mask, "label"].astype(int),
            X_val=df.loc[val_mask, feat_cols].astype(np.float32),
            y_val=df.loc[val_mask, "label"].astype(int),
            X_test=df.loc[test_mask, feat_cols].astype(np.float32),
            y_test=df.loc[test_mask, "label"].astype(int),
            feature_cols=feat_cols,
            meta_test=pd.DataFrame(),  # not used here
        )
        up_m, dn_m = bin_mod.train_binary_pair(splits_fold)
        probs = bin_mod.predict_3class_from_binary(up_m, dn_m, splits_fold.X_test)
        meta = df.loc[test_mask].reset_index(drop=True)
        keep = ["datetime", "symbol", "spot_close", "fwd_return",
                "ce_ltp", "pe_ltp", "minutes_since_open", "vol_regime_pct"]
        meta = meta[[c for c in keep if c in meta.columns]].copy()
        all_probs.append(probs)
        all_meta.append(meta)
    if not all_probs:
        raise RuntimeError("walk_forward_binary produced no folds.")
    probs = np.vstack(all_probs)
    meta = pd.concat(all_meta, ignore_index=True)
    splits_shim = Splits(
        X_train=pd.DataFrame(), y_train=pd.Series(dtype=int),
        X_val=pd.DataFrame(), y_val=pd.Series(dtype=int),
        X_test=pd.DataFrame(), y_test=pd.Series(dtype=int),
        feature_cols=[], meta_test=meta,
    )
    return run_backtest(probs, splits_shim,
                        overrides=variant.backtest_overrides,
                        name=variant.name)


def run_one_iteration(
    variant: Variant,
    cache: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute a single variant; return its metrics + goal report.

    ``cache`` holds carry-over state between iterations so we can
    skip the expensive steps when a variant only tweaks backtest
    knobs (no rebuild, no retrain).
    """
    cfg = get_config()
    logs_dir = cfg.paths["logs_dir"]

    log.info("=" * 70)
    log.info("Iteration %s — %s (strategy=%s)",
             variant.name, variant.description, variant.strategy)
    log.info("Overrides: %s", _summarise_signature(variant))

    # ---- dataset ----
    if variant.rebuild_dataset or cache.get("df") is None:
        log.info("[%s] rebuilding feature parquet from DB (labels override)", variant.name)
        df = _dataset_from_db(variant.label_overrides or None)
        cache["df"] = df
    else:
        df = cache["df"]

    # ---- walk-forward path bypasses the standard train/backtest ----
    if variant.strategy in ("walk_forward", "walk_forward_binary"):
        bt = _run_walk_forward(
            variant, df, use_binary=(variant.strategy == "walk_forward_binary"),
        )
        summary = bt["summary"]
        report = goals_mod.evaluate(summary)
        model_path = "walk_forward (per-fold, not persisted)"

        # Persist logs + return.
        results_path = logs_dir / f"iteration_{variant.name}_results.txt"
        _write(logs_dir / f"iteration_{variant.name}_changes.txt", "\n".join([
            f"Iteration: {variant.name}",
            f"Description: {variant.description}",
            f"Strategy: {variant.strategy}",
            f"Signature: {_summarise_signature(variant)}",
        ]))
        _write(results_path, "\n".join([
            f"=== {variant.name} ===",
            f"Timestamp: {datetime.now().isoformat()}",
            f"Strategy: {variant.strategy}",
            f"WF kwargs: {variant.wf_kwargs}",
            "",
            "-- Backtest summary --",
        ] + [f"  {k:22s}: {v}" for k, v in summary.items()] + [
            "", *report.as_lines(),
        ]))
        return {
            "variant": variant.name,
            "summary": summary,
            "report": report,
            "model_path": model_path,
            "metrics": {**summary},
        }

    # ---- splits ----
    if variant.rebuild_dataset or cache.get("splits") is None:
        splits = make_splits(df)
        cache["splits"] = splits
    else:
        splits = cache["splits"]

    # ---- model ----
    if variant.strategy == "binary":
        # Always retrain — the binary pair is cheap and doesn't
        # share state with the 3-class cache.
        log.info("[%s] training binary UP/DOWN pair", variant.name)
        up_m, dn_m = bin_mod.train_binary_pair(splits)
        probs = bin_mod.predict_3class_from_binary(up_m, dn_m, splits.X_test)
        model_path = cfg.paths["models_dir"] / f"model_{variant.name}.pkl"
        joblib.dump({"up": up_m, "down": dn_m}, model_path)
        meta = {"strategy": "binary"}
        test_results_metrics = {"accuracy": float("nan"),
                                "directional_accuracy": float("nan"),
                                "up_precision": float("nan"),
                                "down_precision": float("nan")}
    else:
        if variant.retrain or cache.get("model") is None:
            n_trials = int(variant.train_overrides.get(
                "n_trials", cfg.model["optuna_trials"]
            ))
            log.info("[%s] training model (n_trials=%d)", variant.name, n_trials)
            model, meta = train_mod.tune_and_train(splits, n_trials=n_trials)
            model_path = train_mod.save_model(
                model, meta, name=f"model_{variant.name}.pkl"
            )
            cache["model"] = model
            cache["model_meta"] = meta
            cache["model_path"] = model_path
        else:
            model = cache["model"]
            meta = cache["model_meta"]
            model_path = cache["model_path"]
            log.info("[%s] reusing cached model %s", variant.name, model_path)
        test_results = evaluate_model(model, splits)
        probs = test_results["probs"]
        test_results_metrics = {
            "accuracy": test_results["accuracy"],
            "directional_accuracy": test_results["directional_accuracy"],
            "up_precision": test_results["up_precision"],
            "down_precision": test_results["down_precision"],
        }

    # ---- backtest ----
    log.info("[%s] backtesting", variant.name)
    bt = run_backtest(probs, splits,
                      overrides=variant.backtest_overrides,
                      name=variant.name)
    summary = bt["summary"]
    report = goals_mod.evaluate(summary)

    # ---- write logs ----
    changes_path = logs_dir / f"iteration_{variant.name}_changes.txt"
    results_path = logs_dir / f"iteration_{variant.name}_results.txt"
    _write(changes_path, "\n".join([
        f"Iteration: {variant.name}",
        f"Description: {variant.description}",
        f"Signature: {_summarise_signature(variant)}",
        f"Retrain: {variant.retrain}",
        f"Rebuild dataset: {variant.rebuild_dataset}",
    ]))
    _write(results_path, "\n".join([
        f"=== {variant.name} ===",
        f"Timestamp: {datetime.now().isoformat()}",
        f"Model path: {model_path}",
        "",
        "-- Model metrics --",
        f"  accuracy         : {test_results_metrics['accuracy']}",
        f"  directional_acc  : {test_results_metrics['directional_accuracy']}",
        f"  up_precision     : {test_results_metrics['up_precision']}",
        f"  down_precision   : {test_results_metrics['down_precision']}",
        "",
        "-- Backtest summary --",
    ] + [f"  {k:22s}: {v}" for k, v in summary.items()] + [
        "", *report.as_lines(),
    ]))

    return {
        "variant": variant.name,
        "summary": summary,
        "report": report,
        "model_path": str(model_path),
        "metrics": {**test_results_metrics, **summary},
    }


# =============================================================
#  Loop driver
# =============================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="AiVora iterative improvement loop")
    ap.add_argument("--max-iterations", type=int, default=len(VARIANTS))
    ap.add_argument("--list", action="store_true", help="Just list the variants and exit")
    ap.add_argument("--start-at", type=int, default=1,
                    help="1-based index to resume from (skips earlier variants)")
    ap.add_argument("--trials", type=int, default=None,
                    help="Override Optuna trials for every variant (default = config.model.optuna_trials)")
    args = ap.parse_args()

    # Fold the CLI --trials override into every variant that will retrain,
    # so a single flag can dial the loop between "quick smoke" and "full search".
    if args.trials is not None:
        for v in VARIANTS:
            v.train_overrides["n_trials"] = args.trials

    if args.list:
        for i, v in enumerate(VARIANTS, start=1):
            print(f"{i:2d}. {v.name:26s} {v.description}")
        return 0

    cfg = get_config()
    parquet = cfg.paths["parquet_path"]
    if not parquet.exists():
        print(
            f"\nERROR: training parquet not found at {parquet}.\n"
            "Run `python -m scripts.run_historical_load` first "
            "(or `python -m scripts.run_pipeline --mode historical`).\n"
        )
        return 2

    results: List[Dict[str, Any]] = []
    cache: Dict[str, Any] = {}
    best: Optional[Dict[str, Any]] = None

    variants_to_run = VARIANTS[args.start_at - 1 : args.start_at - 1 + args.max_iterations]
    for v in variants_to_run:
        try:
            result = run_one_iteration(v, cache)
        except Exception:
            tb = traceback.format_exc()
            log.error("Iteration %s crashed:\n%s", v.name, tb)
            _write(cfg.paths["logs_dir"] / f"iteration_{v.name}_error.txt", tb)
            continue

        results.append(result)
        # Track the best iteration by composite score, in case none pass.
        if best is None or result["report"].score > best["report"].score:
            best = result

        # Register with the model registry so it survives across sessions.
        try:
            reg_mod.register(Path(result["model_path"]), cache.get("model_meta") or {}, result["metrics"])
        except Exception as exc:
            log.warning("registry.register failed for %s: %s", v.name, exc)

        if result["report"].met:
            log.info("Goals met by variant %s — stopping loop early.", v.name)
            # Save the winning model as the canonical final model.
            joblib.dump(cache["model"], cfg.paths["models_dir"] / "final_model.pkl")
            break

    # ---- final report ----
    final_path = cfg.paths["logs_dir"] / "final_report.txt"
    lines: List[str] = [
        "=" * 70,
        "AiVora — iterative improvement final report",
        f"Timestamp: {datetime.now().isoformat()}",
        f"Iterations run: {len(results)}",
        "",
    ]
    if best is not None:
        lines.append(
            f"BEST variant: {best['variant']} "
            f"(composite score = {best['report'].score:+.4f})"
        )
        lines.append(f"Model: {best['model_path']}")
        lines.append("")
        lines.append("Best variant metrics:")
        for k, v in best["summary"].items():
            lines.append(f"  {k:22s}: {v}")
        lines.append("")
        lines.extend(best["report"].as_lines())
        if best["report"].met:
            lines.append("\nSTATUS: goals met.")
        else:
            lines.append("\nSTATUS: goals NOT met — closest attempt above.")
            lines.append("Remaining gaps:")
            for k, ok in best["report"].passes.items():
                if not ok:
                    lines.append(f"  MISS  {k}  gap={best['report'].gaps[k]:+.4f}")
    else:
        lines.append("No iterations completed successfully.")

    lines.append("")
    lines.append("Full leaderboard:")
    for r in sorted(results, key=lambda x: -x["report"].score):
        met = "PASS" if r["report"].met else "fail"
        lines.append(
            f"  [{met}] {r['variant']:26s} score={r['report'].score:+.4f}  "
            f"monthly={r['summary'].get('avg_monthly_return_pct', 0):+.3%}  "
            f"sharpe={r['summary'].get('sharpe', 0):+.2f}  "
            f"dd={r['summary'].get('max_drawdown', 0):+.3%}  "
            f"win={r['summary'].get('win_rate', 0):.2%}"
        )
    _write(final_path, "\n".join(lines))
    log.info("Final report -> %s", final_path)
    return 0 if best is not None and best["report"].met else 1


if __name__ == "__main__":
    raise SystemExit(main())
