"""Tests for src/holdout.py: tier mapping (anchored on the original six) and the ratio check."""

import numpy as np
import pandas as pd

import config
from src import holdout


def test_tiers_reproduce_the_original_cost_table():
    # median HK value per day measured in Session 6 discussion (HKD), original six
    measured = {"YumChina": 88.6e6, "TripCom": 192.1e6, "Baidu": 664.4e6,
                "NetEase": 730.0e6, "JD": 1509.3e6, "Alibaba": 4187.1e6}
    for pair, med in measured.items():
        assert holdout.spread_tier(med) == config.HALF_SPREAD[pair]


def test_tier_edges():
    assert holdout.spread_tier(19.9e6) is None
    assert holdout.spread_tier(20e6) == 0.0020
    assert holdout.spread_tier(50e6) == 0.0010
    assert holdout.spread_tier(1_000e6) == 0.0005


def _legs(ratio_path):
    idx = pd.bdate_range("2021-05-03", periods=len(ratio_path))
    hk = pd.DataFrame({"close": 50.0, "volume": 1e6, "splits": 0.0}, index=idx)
    us = pd.DataFrame({"close": 50.0 * np.asarray(ratio_path) / 7.8, "volume": 1e6, "splits": 0.0}, index=idx)
    return us, hk, pd.Series(7.8, index=idx)


def test_ratio_check_passes_flat_and_catches_a_step():
    us, hk, fx = _legs([5.0] * 300)
    assert holdout.ratio_check(holdout.implied_ratio(us, hk, fx), 5)["ratio_ok"]
    us, hk, fx = _legs([5.0] * 150 + [15.0] * 150)              # 5 -> 15 mid-sample
    out = holdout.ratio_check(holdout.implied_ratio(us, hk, fx), 15)
    assert not out["ratio_ok"] and out["max_quarter_dev"] > 0.5


def test_median_hk_value_uses_in_sample_days_only():
    idx = pd.bdate_range("2023-12-01", "2024-02-29")
    hk = pd.DataFrame({"close": 10.0, "volume": np.where(idx > pd.Timestamp(config.IS_END), 1e9, 1e6)}, index=idx)
    assert holdout.median_hk_value(hk) == 10.0 * 1e6
