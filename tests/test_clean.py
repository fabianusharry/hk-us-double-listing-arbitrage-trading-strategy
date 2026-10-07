"""Tests for src/clean.py on synthetic prices placed on real exchange dates.

Real dates are used so the XNYS/XHKG holiday labels can be checked:
2022-04-05 Ching Ming (HK), 2022-04-15 Good Friday (both), 2022-04-18 Easter
Monday (HK), 2022-05-30 Memorial Day (US), 2021-12-31 HK early close.
"""

import numpy as np
import pandas as pd
import pytest

import config
from src import clean
from src.data import COLUMNS

US, HK = "UUU", "1234.HK"


def frame(dates, close, open_=None, volume=1000.0, dividends=None) -> pd.DataFrame:
    """Synthetic snapshot frame; scalar or array prices, optional {date: amount} dividends."""
    idx = pd.DatetimeIndex(dates, name="date")
    close = np.broadcast_to(np.asarray(close, dtype=float), len(idx)).copy()
    open_ = close.copy() if open_ is None else np.broadcast_to(np.asarray(open_, dtype=float), len(idx)).copy()
    df = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 1, "low": np.minimum(open_, close) - 1,
                       "close": close, "adj_close": close, "volume": volume,
                       "dividends": 0.0, "splits": 0.0}, index=idx)[COLUMNS]
    for d, amt in (dividends or {}).items():
        df.loc[pd.Timestamp(d), "dividends"] = amt
    return df


def bdays(start, end, drop=()):
    return pd.bdate_range(start, end).difference(pd.DatetimeIndex(drop))


@pytest.fixture(autouse=True)
def one_to_one_pair(monkeypatch):
    monkeypatch.setattr(config, "PAIRS", {"Test": (US, HK, 1)})
    monkeypatch.setattr(config, "RATIO_CHANGES", {})


def fx_frame(start="2021-12-01", end="2022-06-30", value=7.8, drop=()):
    return frame(bdays(start, end, drop), value, volume=0.0)


def test_alignment_drops_and_labels_every_date():
    us = frame(bdays("2022-04-01", "2022-05-31", drop=["2022-04-15", "2022-04-20", "2022-05-30"]), 100)
    hk = frame(bdays("2022-04-01", "2022-05-31", drop=["2022-04-05", "2022-04-11", "2022-04-15", "2022-04-18"]), 780)
    aligned, dropped = clean.align_pair("Test", us, hk)
    reasons = dict(zip(pd.to_datetime(dropped["date"]).dt.strftime("%Y-%m-%d"), dropped["reason"]))
    assert reasons == {
        "2022-04-05": "HK holiday (XHKG closed)",
        "2022-04-11": "HK missing data (XHKG open)",
        "2022-04-18": "HK holiday (XHKG closed)",
        "2022-04-20": "US missing data (XNYS open)",
        "2022-05-30": "US holiday (XNYS closed)",
    }
    for d in reasons:
        assert pd.Timestamp(d) not in aligned.index
    assert pd.Timestamp("2022-04-15") not in aligned.index  # closed in both: not a pair date, not logged


def test_days_before_hk_listing_are_not_dropped():
    us = frame(bdays("2022-03-01", "2022-04-29"), 100)
    hk = frame(bdays("2022-04-01", "2022-04-29", drop=["2022-04-05", "2022-04-15", "2022-04-18"]), 780)
    aligned, dropped = clean.align_pair("Test", us, hk)
    assert aligned.index.min() == pd.Timestamp("2022-04-01")
    assert (pd.to_datetime(dropped["date"]) >= pd.Timestamp("2022-04-01")).all()


def test_single_price_bar_dropped_on_normal_day_kept_on_early_close():
    dates = bdays("2021-12-28", "2022-01-07")
    us, hk = frame(dates, 100), frame(dates, 780)
    for d in ["2021-12-29", "2021-12-31"]:  # 31 Dec 2021 = XHKG half day
        hk.loc[d, ["open", "high", "low", "close"]] = 777.0
        hk.loc[d, "volume"] = 0.0
    aligned, dropped = clean.align_pair("Test", us, hk)
    assert pd.Timestamp("2021-12-29") not in aligned.index
    assert dropped.set_index("date").loc[pd.Timestamp("2021-12-29").date(), "reason"] == "unreliable HK bar"
    assert not aligned.loc["2021-12-31", "hk_open_reliable"]
    assert aligned.drop(pd.Timestamp("2021-12-31"))["hk_open_reliable"].all()


def test_fx_is_previous_business_day_close():
    fx = fx_frame("2022-04-01", "2022-04-29")
    fx["close"] = np.arange(len(fx), dtype=float) + 7.80
    dates = pd.DatetimeIndex(["2022-04-05", "2022-04-11"])  # Tue, Mon
    out = clean.fx_for_dates(fx, dates)
    assert out.loc["2022-04-05", "fx"] == fx.loc["2022-04-04", "close"]
    assert out.loc["2022-04-11", "fx"] == fx.loc["2022-04-08", "close"]  # Monday uses Friday
    assert not out["fx_filled"].any()


def test_fx_fills_one_day_and_drops_after_two():
    fx = fx_frame(drop=["2022-04-12", "2022-04-19", "2022-04-20"])
    fx["close"] = np.arange(len(fx), dtype=float) + 7.80
    dates = pd.DatetimeIndex(["2022-04-13", "2022-04-20", "2022-04-21"])
    out = clean.fx_for_dates(fx, dates)
    assert out.loc["2022-04-13", "fx"] == fx.loc["2022-04-11", "close"] and out.loc["2022-04-13", "fx_filled"]
    assert out.loc["2022-04-20", "fx"] == fx.loc["2022-04-18", "close"] and out.loc["2022-04-20", "fx_filled"]
    assert np.isnan(out.loc["2022-04-21", "fx"])  # t-1 = 04-20 is the 2nd missing day


def test_clean_pair_logs_fx_fill_and_drops_missing_fx():
    dates = bdays("2022-04-01", "2022-04-29", drop=["2022-04-05", "2022-04-15", "2022-04-18"])
    frames = {US: frame(dates, 100), HK: frame(dates, 780),
              config.FX_TICKER: fx_frame(drop=["2022-04-12", "2022-04-19", "2022-04-20"])}
    cleaned, logs = clean.clean_pair("Test", frames)
    assert pd.Timestamp("2022-04-21") not in cleaned.index
    assert "FX missing (t-1, beyond 1-day fill)" in set(logs["dropped"]["reason"])
    assert set(pd.to_datetime(logs["fx_fills"]["date"])) == {pd.Timestamp("2022-04-13"), pd.Timestamp("2022-04-20")}


def test_ratio_change_applies_from_effective_date(monkeypatch):
    monkeypatch.setattr(config, "RATIO_CHANGES", {"Test": [("2022-04-11", 2)]})
    r = clean.ratio_on("Test", pd.DatetimeIndex(["2022-04-08", "2022-04-11", "2022-04-12"]))
    assert r.tolist() == [1.0, 2.0, 2.0]


def _dividend_frames(hk_ex, us_ex, hk_amt=7.8, us_amt=1.0):
    """1:1 pair, FX 7.8, prices drop by the dividend from each leg's own ex-date."""
    dates = bdays("2022-04-19", "2022-04-29")
    us_px = np.where(dates >= pd.Timestamp(us_ex), 100 - us_amt, 100.0) if us_ex else 100.0
    hk_px = np.where(dates >= pd.Timestamp(hk_ex), 780 - hk_amt, 780.0)
    us = frame(dates, us_px, dividends={us_ex: us_amt} if us_ex else None)
    hk = frame(dates, hk_px, dividends={hk_ex: hk_amt})
    return {US: us, HK: hk, config.FX_TICKER: fx_frame()}


def _spread(cleaned, col="spread"):
    from src.spread import build_spreads
    return build_spreads(cleaned.reset_index())[col].to_numpy()


def test_mismatched_ex_dates_corrected_hk_first():
    cleaned, logs = clean.clean_pair("Test", _dividend_frames("2022-04-21", "2022-04-25"))
    np.testing.assert_allclose(_spread(cleaned), 0.0, atol=1e-12)
    ev = logs["dividends"].iloc[0]
    assert ev["action"] == "add back HK dividend (HK ex first)" and ev["days_adjusted"] == 2  # 21st, 22nd
    raw = np.log(cleaned["us_open"]) - np.log(cleaned["hk_close"] / cleaned["fx"])
    assert raw.abs().max() > 0.009  # the fake jump really was there


def test_mismatched_ex_dates_corrected_us_first():
    cleaned, logs = clean.clean_pair("Test", _dividend_frames("2022-04-25", "2022-04-21"))
    np.testing.assert_allclose(_spread(cleaned), 0.0, atol=1e-12)
    assert logs["dividends"].iloc[0]["action"] == "add back US dividend (US ex first)"


def test_same_ex_date_needs_no_correction():
    cleaned, logs = clean.clean_pair("Test", _dividend_frames("2022-04-21", "2022-04-21"))
    assert (cleaned[["div_adj_us", "div_adj_hk"]] == 0).all().all()
    assert logs["dividends"].iloc[0]["action"] == "none (same ex-date)"
    np.testing.assert_allclose(_spread(cleaned), 0.0, atol=1e-12)


def test_evening_reading_corrected_even_for_same_ex_date():
    # HK open on ex-date 21st is ex, but the US close on the 20th is still cum.
    cleaned, logs = clean.clean_pair("Test", _dividend_frames("2022-04-21", "2022-04-21"))
    assert cleaned.loc["2022-04-21", "div_adj_hk_eve"] == 7.8
    assert logs["dividends"].iloc[0]["days_adjusted_eve"] == 1
    for col in ["spread_eve", "spread_2r"]:
        np.testing.assert_allclose(_spread(cleaned, col)[1:], 0.0, atol=1e-12)  # row 0 has no prev US close


@pytest.mark.parametrize("hk_ex,us_ex", [("2022-04-21", "2022-04-25"), ("2022-04-25", "2022-04-21")])
def test_evening_reading_flat_for_mismatched_ex_dates(hk_ex, us_ex):
    cleaned, _ = clean.clean_pair("Test", _dividend_frames(hk_ex, us_ex))
    np.testing.assert_allclose(_spread(cleaned, "spread_eve")[1:], 0.0, atol=1e-12)


def test_previous_us_session_skips_non_aligned_days():
    us = frame(bdays("2022-04-11", "2022-04-22", drop=["2022-04-15"]), np.arange(9, dtype=float) + 100)
    prev = clean.previous_us_session(us, pd.DatetimeIndex(["2022-04-11", "2022-04-19"]))
    assert pd.isna(prev.loc["2022-04-11", "us_prev_date"])
    assert prev.loc["2022-04-19", "us_prev_date"] == pd.Timestamp("2022-04-18")
    assert prev.loc["2022-04-19", "us_close_prev"] == us.loc["2022-04-18", "close"]


def test_unmatched_dividend_logged_not_corrected():
    cleaned, logs = clean.clean_pair("Test", _dividend_frames("2022-04-21", None))
    assert logs["dividends"].iloc[0]["status"] == "unmatched HK"
    assert (cleaned[["div_adj_us", "div_adj_hk"]] == 0).all().all()


def test_clean_all_writes_logs(tmp_path):
    out = clean.clean_all(_dividend_frames("2022-04-21", "2022-04-21"), log_dir=tmp_path)
    assert list(out.columns[:2]) == ["date", "pair"]
    for fname in clean.LOG_FILES.values():
        assert (tmp_path / fname).exists()
