"""Synthetic dual-listed pair for backtest tests.

One efficient log price E is observed at the four session times of each day, in
real-time order (ET):
    US close t-1 (16:00) -> HK open t (21:30) -> HK close t (04:00) -> US open t (09:30) -> US close t
Each step adds independent N(0, sigma) news. The HK leg trades at E; the US leg at
E + x(t), where x is an optional AR(1) mispricing (phi, stationary std mis_sd).
With mis_sd = 0 the two listings are priced perfectly, and the only "spread" is
non-synchronous timing noise: the placebo case.
"""

import numpy as np
import pandas as pd

from src.data import COLUMNS
from src.spread import compute_spread

FX = 7.8


def _frame(dates, open_, close) -> pd.DataFrame:
    df = pd.DataFrame({"open": open_, "close": close}, index=pd.DatetimeIndex(dates, name="date"))
    df["high"] = df[["open", "close"]].max(axis=1) * 1.001
    df["low"] = df[["open", "close"]].min(axis=1) * 0.999
    df["adj_close"] = df["close"]
    df["volume"] = 1e6
    df["dividends"] = 0.0
    df["splits"] = 0.0
    return df[COLUMNS]


def simulate_pair(n: int, sigma: float = 0.01, phi: float = 0.0, mis_sd: float = 0.0,
                  start: str = "1950-01-02", seed: int = 0) -> dict:
    """Return {'us', 'hk', 'fx', 'spreads'} for one 1:1 pair on n business days."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    steps = rng.normal(scale=sigma, size=(n, 4))            # HK open, HK close, US open, US close
    path = np.log(100.0) + np.cumsum(steps.ravel()).reshape(n, 4)
    hk_open, hk_close, us_open, us_close = path.T

    x = np.zeros(n)
    if mis_sd > 0:
        eps = rng.normal(scale=mis_sd * np.sqrt(1 - phi**2), size=n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + eps[t]

    us = _frame(dates, np.exp(us_open + x), np.exp(us_close + x))
    hk = _frame(dates, FX * np.exp(hk_open), FX * np.exp(hk_close))
    fx = _frame(dates, FX, FX)
    spreads = pd.DataFrame({"date": dates, "pair": "Test",
                            "spread": compute_spread(us["open"], hk["close"], fx["close"], 1.0).to_numpy()})
    return {"us": us, "hk": hk, "fx": fx, "spreads": spreads}
