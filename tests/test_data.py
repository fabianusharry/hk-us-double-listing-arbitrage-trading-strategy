"""Tests for src/data.py using synthetic snapshots in a temp directory (no network)."""

import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

import config
from src import data


def make_frame(start: str, periods: int) -> pd.DataFrame:
    """A clean synthetic daily snapshot frame: business days, sensible OHLCV."""
    idx = pd.bdate_range(start, periods=periods, name="date")
    close = 100 + np.arange(periods, dtype=float)
    return pd.DataFrame({
        "open": close - 0.5, "high": close + 1, "low": close - 1, "close": close,
        "adj_close": close, "volume": 1000.0, "dividends": 0.0, "splits": 0.0,
    }, index=idx)


@pytest.fixture
def snapshot_dir(tmp_path):
    """Write a synthetic snapshot spanning IS and OOS for two tickers + FX."""
    for t in ["AAA", "1111.HK", config.FX_TICKER]:
        make_frame("2023-12-01", 40).to_csv(data.snapshot_path(t, date(2026, 10, 7), tmp_path))
    return tmp_path


def test_round_trip_preserves_data(tmp_path):
    df = make_frame("2022-01-03", 10)
    path = data.snapshot_path("AAA", date(2026, 10, 7), tmp_path)
    df.to_csv(path)
    back = data.read_snapshot(path)
    pd.testing.assert_frame_equal(back, df, check_freq=False)
    assert back.index.name == "date"


def test_to_snapshot_frame_keeps_local_trading_date():
    # HK midnight local time = previous day in UTC; we must keep the HK date.
    idx = pd.DatetimeIndex(["2022-03-01", "2022-03-02"]).tz_localize("Asia/Hong_Kong")
    raw = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Adj Close": 1.0,
                        "Volume": 1, "Dividends": 0.0, "Stock Splits": 0.0}, index=idx)
    out = data.to_snapshot_frame(raw)
    assert list(out.index) == list(pd.to_datetime(["2022-03-01", "2022-03-02"]))
    assert out.index.tz is None and list(out.columns) == data.COLUMNS


def test_load_raw_defaults_to_in_sample(snapshot_dir, monkeypatch):
    monkeypatch.setattr(config, "ALLOW_OOS", False)
    frames = data.load_raw(["AAA"], raw_dir=snapshot_dir)
    assert frames["AAA"].index.max() <= pd.Timestamp(config.IS_END)


@pytest.mark.parametrize("start,end", [(None, "2024-01-01"), (None, "2025-06-30"), ("2024-02-01", "2024-03-01")])
def test_load_raw_refuses_oos_when_locked(snapshot_dir, monkeypatch, start, end):
    monkeypatch.setattr(config, "ALLOW_OOS", False)
    with pytest.raises(data.OOSAccessError):
        data.load_raw(["AAA"], start=start, end=end, raw_dir=snapshot_dir)


def test_load_raw_allows_oos_when_unlocked(snapshot_dir, monkeypatch):
    monkeypatch.setattr(config, "ALLOW_OOS", True)
    frames = data.load_raw(["AAA"], raw_dir=snapshot_dir)
    assert frames["AAA"].index.max() > pd.Timestamp(config.OOS_START)


def test_load_raw_works_offline(snapshot_dir, monkeypatch):
    # Any attempt to import yfinance fails: load_raw must not need it.
    monkeypatch.setitem(sys.modules, "yfinance", None)
    frames = data.load_raw(raw_dir=snapshot_dir, tickers=["AAA", "1111.HK"])
    assert set(frames) == {"AAA", "1111.HK"}


def test_latest_snapshot_picks_newest(tmp_path):
    make_frame("2022-01-03", 3).to_csv(data.snapshot_path("AAA", date(2026, 1, 1), tmp_path))
    make_frame("2022-01-03", 5).to_csv(data.snapshot_path("AAA", date(2026, 10, 7), tmp_path))
    assert data.latest_snapshot("AAA", tmp_path).name == "AAA_20261007.csv"


def test_missing_snapshot_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.latest_snapshot("NOPE", tmp_path)


def test_quality_report_finds_planted_defects():
    df = make_frame("2022-01-03", 10)
    d = df.index
    df.loc[d[1], "volume"] = 0                                   # zero volume
    df.loc[d[2], "open"] = df.loc[d[2], "high"] + 5              # open outside range
    df.loc[d[3], ["open", "high", "low"]] = df.loc[d[3], "close"]  # flat bar
    df.loc[d[4], "close"] = np.nan                               # missing price
    df.loc[d[5], "splits"] = 2.0                                 # split
    summary, detail = data.quality_report({"AAA": df})
    row = summary.iloc[0]
    assert row["zero_volume"] == 1 and row["open_outside_range"] == 1
    assert row["flat_bar"] == 1 and row["missing_close"] == 1
    assert row["n_splits"] == 1 and "2" in row["split_details"]
    assert set(detail["issue"]) == {"zero_volume", "open_outside_range", "flat_bar", "missing_price"}


def test_quality_report_skips_volume_checks_for_fx():
    df = make_frame("2022-01-03", 5)
    df["volume"] = 0.0
    for col in ["open", "high", "low"]:
        df[col] = df["close"]
    summary, _ = data.quality_report({config.FX_TICKER: df})
    assert summary.iloc[0]["zero_volume"] == 0 and summary.iloc[0]["flat_bar"] == 0
