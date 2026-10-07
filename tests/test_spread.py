"""Tests for src/spread.py with known answers."""

import numpy as np
import pandas as pd

import config
from src import spread


def s(x):
    return pd.Series([x], dtype=float)


def test_identical_one_to_one_prices_give_zero_spread():
    assert spread.compute_spread(s(100), s(780), s(7.8), s(1)).iloc[0] == 0.0


def test_one_percent_adr_premium():
    out = spread.compute_spread(s(101), s(780), s(7.8), s(1)).iloc[0]
    assert np.isclose(out, np.log(1.01)) and np.isclose(out, 0.01, atol=1e-4)


def test_ratio_and_fx_scale_parity():
    # 8 HK shares at 97.5 HKD, FX 7.8 -> parity 100 USD
    assert np.isclose(spread.compute_spread(s(100), s(97.5), s(7.8), s(8)).iloc[0], 0.0)
    # HK rich: ADR at a 2% discount -> negative spread
    assert spread.compute_spread(s(98), s(97.5), s(7.8), s(8)).iloc[0] < 0


def _cleaned(dates, us_open, hk_close, div_adj_us=0.0, div_adj_hk=0.0,
             us_close_prev=100.0, hk_open=780.0, hk_open_reliable=True):
    return pd.DataFrame({"date": pd.to_datetime(dates), "pair": "Test", "ratio": 1.0,
                         "us_open": us_open, "hk_close": hk_close, "fx": 7.8,
                         "div_adj_us": div_adj_us, "div_adj_hk": div_adj_hk,
                         "us_close_prev": us_close_prev, "hk_open": hk_open,
                         "hk_open_reliable": hk_open_reliable,
                         "div_adj_us_eve": 0.0, "div_adj_hk_eve": 0.0})


def test_build_spreads_applies_dividend_add_back_and_backtest_flag():
    df = _cleaned([config.START, "2021-01-04"], 100.0, [772.2, 780.0], div_adj_hk=[7.8, 0.0])
    out = spread.build_spreads(df)
    np.testing.assert_allclose(out["spread"], 0.0, atol=1e-12)
    assert out["in_backtest"].tolist() == [True, False]


def test_flags_only_large_spreads():
    df = spread.build_spreads(_cleaned(["2022-01-03", "2022-01-04"], [100.0, 115.0], 780.0))
    flags = spread.flag_spreads(df)
    assert len(flags) == 1 and flags.loc[0, "date"] == pd.Timestamp("2022-01-04")


def test_evening_reading_and_two_reading_average():
    # morning: ADR 2% rich; evening: ADR 1% rich -> average ~1.5%
    df = spread.build_spreads(_cleaned(["2022-01-03"], 102.0, 780.0, us_close_prev=101.0, hk_open=780.0))
    row = df.iloc[0]
    assert np.isclose(row["spread"], np.log(1.02)) and np.isclose(row["spread_eve"], np.log(1.01))
    assert np.isclose(row["spread_2r"], (np.log(1.02) + np.log(1.01)) / 2) and row["n_readings"] == 2


def test_two_reading_falls_back_to_morning_when_hk_open_unreliable():
    df = spread.build_spreads(_cleaned(["2022-01-03", "2022-01-04"], 102.0, 780.0,
                                       hk_open_reliable=[True, False]))
    assert np.isnan(df.loc[1, "spread_eve"]) and df.loc[1, "n_readings"] == 1
    assert np.isclose(df.loc[1, "spread_2r"], df.loc[1, "spread"])
