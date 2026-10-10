"""Transaction and borrow cost model (CLAUDE.md §6). Rates come from config.py.

Per side (every buy or sell), as a fraction of the traded notional:
  HK leg: stamp duty + levies + commission + half bid-ask spread
  US leg:              commission + half bid-ask spread
Short borrow: BORROW_RATE_ANNUAL on the short leg's dollar value, accrued per
calendar day (borrow is charged over weekends and holidays too).
Everything is scaled by cost_multiplier (0, 0.5, 1, 2 for the sensitivity analysis).

The engine (backtest.backtest_pair) deducts these from pair equity as they occur,
so later trades are sized from equity after costs.
"""

from __future__ import annotations

from dataclasses import dataclass

import config


@dataclass(frozen=True)
class CostModel:
    """Cost rates for one pair at one cost multiplier."""
    us_side: float        # fraction of traded notional, per buy or sell
    hk_side: float
    borrow_annual: float  # fraction of short value per year

    def trade_cost(self, notional: float, leg: str) -> float:
        """Cost in dollars of trading `notional` dollars on leg 'us' or 'hk'."""
        return abs(notional) * (self.us_side if leg == "us" else self.hk_side)

    def borrow_cost(self, short_value: float, days: int) -> float:
        """Borrow fee in dollars for holding `short_value` dollars short for `days` calendar days."""
        return abs(short_value) * self.borrow_annual * days / 365.0


def cost_model(pair: str, multiplier: float = 1.0, half_spread: float | None = None,
               borrow_annual: float | None = None) -> CostModel:
    """Build the CostModel for a pair, scaled by `multiplier`.

    half_spread / borrow_annual default to config (HALF_SPREAD[pair], BORROW_RATE_ANNUAL);
    pass them explicitly for pairs outside the main universe or for a borrow sensitivity.
    """
    half_spread = config.HALF_SPREAD[pair] if half_spread is None else half_spread
    borrow = config.BORROW_RATE_ANNUAL if borrow_annual is None else borrow_annual
    us = config.COMMISSION + half_spread
    hk = config.HK_STAMP_DUTY + config.HK_LEVIES + config.COMMISSION + half_spread
    return CostModel(us_side=multiplier * us, hk_side=multiplier * hk,
                     borrow_annual=multiplier * borrow)


def round_trip_cost(pair: str, multiplier: float = 1.0) -> float:
    """Enter + exit on both legs, as a fraction of ONE leg's notional (no borrow)."""
    m = cost_model(pair, multiplier)
    return 2 * (m.us_side + m.hk_side)
