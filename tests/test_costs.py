"""Tests for src/costs.py and the costs inside the engine, against hand calculations."""

import numpy as np
import pandas as pd
import pytest

from src import costs
from src.backtest import backtest_pair, leg_returns
from tests.synthetic import _frame

# Fri 22 Apr .. Fri 29 Apr 2022 (no holidays)
DAYS = ["2022-04-22", "2022-04-25", "2022-04-26", "2022-04-27", "2022-04-28", "2022-04-29"]


def test_side_rates_by_hand():
    m = costs.cost_model("Alibaba")
    assert np.isclose(m.us_side, 0.0003 + 0.0005)                    # commission + half-spread
    assert np.isclose(m.hk_side, 0.0010 + 0.0001 + 0.0003 + 0.0005)  # + stamp duty + levies
    n = costs.cost_model("NetEase")
    assert np.isclose(n.us_side, 0.0013) and np.isclose(n.hk_side, 0.0024)


def test_multiplier_scales_everything():
    one, two = costs.cost_model("JD", 1.0), costs.cost_model("JD", 2.0)
    assert np.isclose(two.us_side, 2 * one.us_side) and np.isclose(two.hk_side, 2 * one.hk_side)
    assert np.isclose(two.borrow_annual, 2 * one.borrow_annual)
    assert costs.cost_model("JD", 0.0).trade_cost(1e6, "hk") == 0.0


def test_borrow_accrues_per_calendar_day():
    m = costs.cost_model("JD")
    assert np.isclose(m.borrow_cost(-1_000_000, 365), 10_000)          # 1% a year
    assert np.isclose(m.borrow_cost(-1_000_000, 3), 10_000 * 3 / 365)  # Fri -> Mon = 3 days


def test_round_trip_on_flat_prices_costs_exactly_the_fees_plus_borrow():
    """Flat prices: the only P&L is costs, so the final equity is known exactly.

    Decision +1 (long US / short HK) on Mon 25th, exit on Wed 27th.
    US: buys 0.5 at the Mon close, sells 0.5 at the Wed close.
    HK: shorts 0.5 at the Tue open, buys back at the Thu open.
    Borrow on the 0.5 HK short for the nights Tue->Wed and Wed->Thu = 2 days.
    """
    m = costs.cost_model("NetEase")
    flat = _frame(DAYS, 100.0, 100.0)
    us, hk = leg_returns(flat), leg_returns(flat)
    d = pd.Series([1, 1, 0, 0, 0], index=pd.DatetimeIndex(DAYS[1:]), dtype=float)
    out = backtest_pair(d, us.loc[DAYS[1]:], hk.loc[DAYS[1]:], cost=m)

    fees = 0.5 * 2 * m.us_side + 0.5 * 2 * m.hk_side
    borrow = 0.5 * 0.01 * 2 / 365
    assert np.isclose(out["equity"].iloc[-1], 1 - fees - borrow, atol=1e-15)
    e_prev = out["equity"].shift(1).fillna(1.0)
    assert np.isclose((out["borrow_cost"] * e_prev).sum(), borrow, atol=1e-15)   # dollars, not fractions
    assert (out["gross_ret"] == 0).all()


def test_zero_multiplier_equals_gross_and_double_doubles_costs():
    rng = np.random.default_rng(0)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(DAYS), 2)), axis=0))
    us, hk = leg_returns(_frame(DAYS, px[:, 0], px[:, 0])), leg_returns(_frame(DAYS, px[:, 1], px[:, 1]))
    d = pd.Series([1, 1, 0, -1, -1], index=pd.DatetimeIndex(DAYS[1:]), dtype=float)
    gross = backtest_pair(d, us.loc[DAYS[1]:], hk.loc[DAYS[1]:])
    zero = backtest_pair(d, us.loc[DAYS[1]:], hk.loc[DAYS[1]:], cost=costs.cost_model("JD", 0.0))
    pd.testing.assert_series_equal(gross["equity"], zero["equity"])
    one = backtest_pair(d, us.loc[DAYS[1]:], hk.loc[DAYS[1]:], cost=costs.cost_model("JD", 1.0))
    assert one["equity"].iloc[-1] < gross["equity"].iloc[-1]
    np.testing.assert_allclose(one["pair_ret"], one["gross_ret"] - one["us_cost"] - one["hk_cost"]
                               - one["borrow_cost"], atol=1e-15)


def test_costs_shrink_the_next_trade():
    """Sizing uses equity after costs: the second entry is 0.5 x (equity after the first round trip)."""
    m = costs.cost_model("Baidu")
    days = pd.bdate_range("2022-05-02", periods=9)
    flat = _frame(days, 100.0, 100.0)
    us, hk = leg_returns(flat), leg_returns(flat)
    d = pd.Series([1, 0, 0, 0, 1, 0, 0, 0], index=days[:8], dtype=float)
    out = backtest_pair(d, us, hk, cost=m)
    e_prev = out["equity"].shift(1).fillna(1.0)
    entries = out.index[out["us_trade"] > 0][[0, 2]]
    dollars = (out["us_trade"] * e_prev).loc[entries].to_numpy()
    assert np.isclose(dollars[0], 0.5)
    assert np.isclose(dollars[1], 0.5 * out["equity"].loc[:entries[1]].iloc[-2]) and dollars[1] < 0.5


@pytest.mark.parametrize("pair", ["Alibaba", "TripCom"])
def test_round_trip_cost_matches_analysis_preview(pair):
    from src.analysis import round_trip_cost
    assert np.isclose(costs.round_trip_cost(pair), round_trip_cost(pair))
