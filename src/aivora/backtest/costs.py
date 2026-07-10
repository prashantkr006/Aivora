"""Realistic Indian F&O (options) cost model.

A per-round-trip cost calculator that matches a retail Zerodha
account trading NSE index options.  Numbers are configurable via
``backtest.costs`` in ``config.yaml`` so you can swap in a
different broker's fee schedule without touching this file.

Applied on top of the model's exit premium so back-tested P&L
reflects what would actually land in your account.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class RoundTripCosts:
    """Line-item breakdown of the costs applied to one options round trip."""

    brokerage: float
    stt: float
    sebi: float
    exchange_txn: float
    gst: float
    stamp_duty: float
    slippage: float

    @property
    def total(self) -> float:
        return (
            self.brokerage
            + self.stt
            + self.sebi
            + self.exchange_txn
            + self.gst
            + self.stamp_duty
            + self.slippage
        )


def compute_round_trip(
    entry_premium: float,
    exit_premium: float,
    lots: int,
    lot_size: int,
    cfg: Dict,
) -> RoundTripCosts:
    """Compute all costs for a buy-then-sell options round trip.

    ``cfg`` should be ``config.yaml``'s ``backtest.costs`` block.
    We follow the Zerodha explainer numbers (accurate at time of
    writing — Q3 2025):

    * Brokerage      : min(₹20, 0.03 % of turnover) per order → so
                       a round trip costs at most ₹40 flat, less
                       for tiny turnovers.
    * STT            : 0.10 % on the SELL leg premium turnover.
    * SEBI turnover  : ₹10 / crore (≈ 0.00001) on both legs.
    * Exchange txn   : 0.03503 % on premium turnover, both legs.
    * GST            : 18 % on brokerage + SEBI + exchange txn.
    * Stamp duty     : 0.003 % on the BUY leg turnover.
    * Slippage       : ``slippage_pct`` × turnover, both legs
                       (models bid-ask crossing / adverse execution).
    """
    entry_turnover = entry_premium * lots * lot_size
    exit_turnover = exit_premium * lots * lot_size
    turnover = entry_turnover + exit_turnover

    per_order = min(
        float(cfg.get("brokerage_flat_per_order", 20.0)),
        entry_turnover * float(cfg.get("brokerage_pct_cap", 0.0003)),
    )
    brokerage = per_order + min(
        float(cfg.get("brokerage_flat_per_order", 20.0)),
        exit_turnover * float(cfg.get("brokerage_pct_cap", 0.0003)),
    )
    stt = exit_turnover * float(cfg.get("stt_pct_sell", 0.001))
    sebi = turnover * float(cfg.get("sebi_pct", 0.00001))
    exch = turnover * float(cfg.get("exchange_txn_pct", 0.0003503))
    stamp = entry_turnover * float(cfg.get("stamp_duty_pct_buy", 0.00003))
    gst = (brokerage + sebi + exch) * float(cfg.get("gst_pct", 0.18))
    slip = turnover * float(cfg.get("slippage_pct", 0.001))

    return RoundTripCosts(
        brokerage=brokerage,
        stt=stt,
        sebi=sebi,
        exchange_txn=exch,
        gst=gst,
        stamp_duty=stamp,
        slippage=slip,
    )
