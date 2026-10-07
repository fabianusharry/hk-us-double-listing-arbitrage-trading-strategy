"""Spread construction and sanity checks.

spread(t) = ln(P_US_open(t)) - ln(N * P_HK_close(t) / FX(t)); positive = ADR rich.
Timing: HK closes (16:00 HKT) before New York opens (09:30 ET), so the HK close
on day t is known at the US open on day t. Never compare the US *close* with the
HK close of the same date. FX(t) is the HKD=X close of t-1 (see clean.fx_for_dates).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config


def compute_spread(us_open: pd.Series, hk_close: pd.Series, fx: pd.Series, ratio: pd.Series) -> pd.Series:
    """Log spread of the ADR (at the US open) over its HK parity (at the HK close).

    Inputs (aligned, same index): US open in USD per ADR, HK close in HKD per
    share, FX in HKD per USD, ratio in HK shares per ADR.
    Output: ln(us_open) - ln(ratio * hk_close / fx). +0.01 means the ADR is ~1% rich.
    """
    parity_usd = ratio * hk_close / fx
    return np.log(us_open) - np.log(parity_usd)


def build_spreads(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Add parity_usd, spread and in_backtest to clean.clean_all() output.

    Dividend add-backs (div_adj_us, div_adj_hk) are applied here, so the spread
    has no artificial jump when the two legs go ex-dividend on different dates.
    in_backtest is True from config.START; earlier rows are rolling-window warm-up.
    """
    df = cleaned.copy()
    us_open = df["us_open"] + df["div_adj_us"]
    hk_close = df["hk_close"] + df["div_adj_hk"]
    df["parity_usd"] = df["ratio"] * hk_close / df["fx"]
    df["spread"] = compute_spread(us_open, hk_close, df["fx"], df["ratio"])
    df["in_backtest"] = df["date"] >= pd.Timestamp(config.START)
    return df


def flag_spreads(spreads: pd.DataFrame, threshold: float = config.SPREAD_FLAG_ABS) -> pd.DataFrame:
    """Rows with |spread| > threshold, for manual review (not removed from the data)."""
    cols = ["date", "pair", "spread", "us_open", "hk_close", "fx", "ratio", "in_backtest"]
    return spreads.loc[spreads["spread"].abs() > threshold, cols].reset_index(drop=True)


def save_spreads(
    spreads: pd.DataFrame,
    path: Path = config.PROCESSED_DIR / "spreads.csv",
    log_dir: Path = config.LOG_DIR,
) -> pd.DataFrame:
    """Write data/processed/spreads.csv and results/logs/spread_flags.csv; return the flags."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    spreads.to_csv(path, index=False)
    flags = flag_spreads(spreads)
    flags.to_csv(Path(log_dir) / "spread_flags.csv", index=False)
    return flags
