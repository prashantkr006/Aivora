"""Financial-goal evaluation.

Given the ``summary`` dict returned by
:func:`aivora.backtest.backtester.run_backtest`, decide whether
the strategy meets the targets configured under ``goals:`` in
``config.yaml``.

Kept deliberately small and side-effect-free so the iteration
orchestrator can call it inside a loop without introducing state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..utils.config import get_config


@dataclass
class GoalReport:
    met: bool
    passes: Dict[str, bool]
    gaps: Dict[str, float]        # numeric distance from each goal
    score: float                  # weighted composite (higher = better)

    def as_lines(self) -> List[str]:
        out = ["=== Goal check ==="]
        for k, ok in self.passes.items():
            gap = self.gaps.get(k, 0.0)
            out.append(f"  {k:32s} {'PASS' if ok else 'FAIL'}  gap={gap:+.4f}")
        out.append(f"  {'composite score':32s}       {self.score:+.4f}")
        out.append(f"  {'ALL GOALS MET':32s} {'YES' if self.met else 'NO'}")
        return out


def evaluate(summary: Dict) -> GoalReport:
    """Compare a backtest summary against the configured goals."""
    goals = get_config().goals
    passes: Dict[str, bool] = {}
    gaps: Dict[str, float] = {}

    def _check(name: str, value: float, target: float, higher_better: bool):
        gap = (value - target) if higher_better else (target - value)
        passes[name] = gap >= 0
        gaps[name] = gap

    _check(
        "avg_monthly_return_pct >= min",
        summary.get("avg_monthly_return_pct", 0.0),
        float(goals.get("min_avg_monthly_return_pct", 0.03)),
        higher_better=True,
    )
    _check(
        "max_drawdown |x| <= cap",
        -summary.get("max_drawdown", 0.0),
        float(goals.get("max_drawdown_abs", 0.20)),
        higher_better=False,
    )
    _check(
        "sharpe >= min",
        summary.get("sharpe", 0.0),
        float(goals.get("min_sharpe", 1.0)),
        higher_better=True,
    )
    _check(
        "win_rate >= min",
        summary.get("win_rate", 0.0),
        float(goals.get("min_win_rate", 0.45)),
        higher_better=True,
    )
    _check(
        "reward_to_risk >= min",
        summary.get("reward_to_risk", 0.0),
        float(goals.get("min_reward_to_risk", 1.5)),
        higher_better=True,
    )
    _check(
        "months_positive_pct >= min",
        summary.get("months_positive_pct", 0.0),
        float(goals.get("min_months_positive_pct", 0.6)),
        higher_better=True,
    )

    # Composite score — weights lean on the top-line targets so the
    # orchestrator can rank iterations even when none pass yet.
    score = (
        3.0 * summary.get("avg_monthly_return_pct", 0.0)
        + 1.5 * summary.get("sharpe", 0.0) / 10.0
        + 1.0 * summary.get("months_positive_pct", 0.0)
        + 0.5 * summary.get("reward_to_risk", 0.0) / 3.0
        + 2.0 * summary.get("max_drawdown", 0.0)       # negative → hurts
    )
    return GoalReport(met=all(passes.values()), passes=passes, gaps=gaps, score=float(score))
