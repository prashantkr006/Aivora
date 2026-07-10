"""Step-based trailing stop-loss milestones.

Milestones (of the option premium since entry):

    ┌──────────────────────┬───────────────────────────────┐
    │ Premium move reaches │ Trailing SL rises to          │
    ├──────────────────────┼───────────────────────────────┤
    │ +10 %                │ Entry price (breakeven)       │
    │ +20 %                │ Entry × 1.10 (lock +10 %)     │
    │ +40 %                │ Entry × 1.25 (lock +25 %)     │
    │ +60 %                │ Take-profit triggers first    │
    └──────────────────────┴───────────────────────────────┘

The tracker checks the **peak premium since entry**, not the current
one — so a spike to +45 % locks in +25 % even if the price then
drops immediately.  Trailing SL only rises; it never falls.

Wired into both the live position-tracker and the backtester so
paper P&L and simulated P&L stay aligned.
"""

from __future__ import annotations

from typing import Optional, Tuple

# Milestone list — (peak-move threshold, locked-in gain).
# Ordered from highest to lowest so we pick the strongest applicable step.
_MILESTONES: Tuple[Tuple[float, float], ...] = (
    (0.40, 0.25),   # peak ≥ +40 % → SL at entry × 1.25
    (0.20, 0.10),   # peak ≥ +20 % → SL at entry × 1.10
    (0.10, 0.00),   # peak ≥ +10 % → SL at entry × 1.00 (breakeven)
)


def new_trailing_price(entry_premium: float, peak_premium: float) -> Optional[float]:
    """Return the SL price implied by the highest premium seen so far,
    or ``None`` if no milestone has been crossed yet."""
    if entry_premium <= 0:
        return None
    peak_move = (peak_premium - entry_premium) / entry_premium
    for threshold, lock in _MILESTONES:
        if peak_move >= threshold:
            return entry_premium * (1.0 + lock)
    return None


def update_trailing_sl(
    entry_premium: float,
    peak_premium: float,
    current_trailing: Optional[float],
) -> Optional[float]:
    """Compute the next trailing SL, respecting the "only rises" rule.

    Returns the new SL price, or the existing one if the peak hasn't
    unlocked a higher step.  ``None`` while below +10 % lifetime peak.
    """
    proposed = new_trailing_price(entry_premium, peak_premium)
    if proposed is None:
        return current_trailing
    if current_trailing is None:
        return proposed
    return max(float(current_trailing), proposed)


def would_stop_here(current_premium: float,
                    trailing_sl_price: Optional[float]) -> bool:
    """Return True if the current premium has fallen to or below the
    active trailing SL — i.e. the tracker should close the trade."""
    return trailing_sl_price is not None and current_premium <= float(trailing_sl_price)
