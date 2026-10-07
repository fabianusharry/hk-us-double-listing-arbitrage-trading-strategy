"""Download and load the raw price snapshot (unadjusted OHLCV, dividends, splits, FX).

download_raw() is the only function that touches the network; load_raw() reads
data/raw/ only and refuses OOS dates while config.ALLOW_OOS is False.

Snapshot format: one CSV per ticker, data/raw/{TICKER}_{YYYYMMDD}.csv, where the
date is the download date. Columns: open, high, low, close, adj_close, volume,
dividends, splits; index `date` = the trading date in the exchange's local time.

Price adjustment (important): with auto_adjust=False, Yahoo's open/high/low/close
are NOT dividend-adjusted but ARE split-adjusted; only adj_close includes
dividends. The spread uses the unadjusted open/close columns.
"""

from __future__ import annotations

import json
import time
from datetime import date as Date
from pathlib import Path

import pandas as pd

import config

COLUMNS = ["open", "high", "low", "close", "adj_close", "volume", "dividends", "splits"]
PRICE_COLUMNS = ["open", "high", "low", "close"]

_YAHOO_TO_OURS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Stock Splits": "splits",
}


class OOSAccessError(RuntimeError):
    """Raised when out-of-sample data is requested while config.ALLOW_OOS is False."""


# ---------------------------------------------------------------------------
# Out-of-sample firewall
# ---------------------------------------------------------------------------
def check_oos_allowed(start: str | None, end: str | None) -> None:
    """Raise OOSAccessError if [start, end] touches the OOS period while it is locked.

    Inputs: requested start/end dates (None = unbounded on that side).
    Output: None; raises if the request reaches OOS_START or later and
    config.ALLOW_OOS is False. An unbounded end counts as reaching OOS.
    """
    if config.ALLOW_OOS:
        return
    oos = pd.Timestamp(config.OOS_START)
    if end is None or pd.Timestamp(end) >= oos or (start is not None and pd.Timestamp(start) >= oos):
        raise OOSAccessError(
            f"Requested dates reach {config.OOS_START} or later (start={start}, end={end}) "
            "but config.ALLOW_OOS is False. OOS data is locked until Session 6."
        )


# ---------------------------------------------------------------------------
# Download (network)
# ---------------------------------------------------------------------------
def snapshot_path(ticker: str, download_date: Date, raw_dir: Path = config.RAW_DIR) -> Path:
    """Return data/raw/{TICKER}_{YYYYMMDD}.csv for a ticker and download date."""
    return Path(raw_dir) / f"{ticker}_{download_date:%Y%m%d}.csv"


def to_snapshot_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert a yfinance history frame to the snapshot format.

    Input: DataFrame from yf.Ticker.history(auto_adjust=False, actions=True),
    indexed by tz-aware timestamps in the exchange's time zone.
    Output: DataFrame with COLUMNS, indexed by a tz-naive DatetimeIndex `date`.
    Timing: the timestamp is midnight local exchange time, so dropping the time
    zone keeps the local trading date (HK rows stay on HK dates, US on US dates).
    """
    missing = set(_YAHOO_TO_OURS) - set(raw.columns)
    if missing:
        raise ValueError(f"yfinance frame is missing columns: {sorted(missing)}")
    df = raw.rename(columns=_YAHOO_TO_OURS)[COLUMNS].copy()
    idx = df.index
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = pd.DatetimeIndex(idx.normalize(), name="date")
    return df.sort_index()


def fetch_one(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download one ticker's daily history with polite retries.

    Inputs: Yahoo ticker, start (inclusive), end (exclusive, yfinance convention).
    Output: snapshot-format DataFrame. Raises RuntimeError after
    config.DOWNLOAD_RETRIES failed or empty attempts.
    keepna=True keeps rows Yahoo returns as all-NaN so the quality report can
    see them instead of yfinance silently dropping them; repair=False keeps
    Yahoo's numbers untouched (we log problems, we don't patch them here).
    """
    import yfinance as yf  # imported here so load_raw() never needs it

    last_error: Exception | None = None
    for attempt in range(1, config.DOWNLOAD_RETRIES + 1):
        try:
            raw = yf.Ticker(ticker).history(
                start=start, end=end, interval="1d",
                auto_adjust=False, actions=True, repair=False, keepna=True,
                raise_errors=True,
            )
            if raw.empty:
                raise RuntimeError("empty response")
            return to_snapshot_frame(raw)
        except Exception as exc:  # network errors, rate limits, empty frames
            last_error = exc
            if attempt < config.DOWNLOAD_RETRIES:
                wait = config.DOWNLOAD_BACKOFF_SEC * attempt
                print(f"  {ticker}: attempt {attempt} failed ({exc}); retrying in {wait:.0f}s")
                time.sleep(wait)
    raise RuntimeError(f"{ticker}: download failed after {config.DOWNLOAD_RETRIES} attempts") from last_error


def download_raw(
    tickers: list[str] | None = None,
    start: str = config.DOWNLOAD_START,
    end: str = config.DOWNLOAD_END,
    raw_dir: Path = config.RAW_DIR,
) -> list[Path]:
    """Fetch every ticker and write the snapshot CSVs plus manifest.json.

    Inputs: tickers (default: config.all_tickers()), download window, output dir.
    Output: list of CSV paths written.
    All tickers are fetched before anything is written, so a failure part-way
    leaves the existing snapshot untouched rather than half-replaced.
    This deliberately downloads the full window including OOS dates: the
    snapshot must be complete and frozen. The firewall is enforced on reading.
    """
    import yfinance as yf

    tickers = tickers or config.all_tickers()
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(config.DOWNLOAD_PAUSE_SEC)
        print(f"Downloading {ticker} ...")
        frames[ticker] = fetch_one(ticker, start, end)

    today = Date.today()
    paths = []
    for ticker, df in frames.items():
        path = snapshot_path(ticker, today, raw_dir)
        df.to_csv(path)
        paths.append(path)

    manifest = {
        "download_date": today.isoformat(),
        "download_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "source": "Yahoo Finance via yfinance",
        "yfinance_version": yf.__version__,
        "pandas_version": pd.__version__,
        "settings": {"auto_adjust": False, "actions": True, "repair": False, "keepna": True},
        "start_inclusive": start,
        "end_exclusive": end,
        "files": {t: p.name for t, p in zip(frames, paths)},
        "rows": {t: len(df) for t, df in frames.items()},
    }
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return paths


# ---------------------------------------------------------------------------
# Load (offline)
# ---------------------------------------------------------------------------
def latest_snapshot(ticker: str, raw_dir: Path = config.RAW_DIR) -> Path:
    """Return the most recent snapshot CSV for a ticker (by the date in its name)."""
    candidates = sorted(Path(raw_dir).glob(f"{ticker}_????????.csv"))
    if not candidates:
        raise FileNotFoundError(f"No snapshot for {ticker} in {raw_dir}. Run download_raw() first.")
    return candidates[-1]  # YYYYMMDD sorts chronologically


def read_snapshot(path: Path) -> pd.DataFrame:
    """Read one snapshot CSV into the standard frame (DatetimeIndex `date`, float columns)."""
    df = pd.read_csv(path, index_col="date", parse_dates=["date"])
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    return df[COLUMNS].astype(float)


def load_raw(
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    raw_dir: Path = config.RAW_DIR,
) -> dict[str, pd.DataFrame]:
    """Load the snapshot from disk only (no network).

    Inputs: tickers (default: all), start/end inclusive dates. While OOS is
    locked, end defaults to config.IS_END; otherwise to the full snapshot.
    Start defaults to the first row (including pre-START warm-up history).
    Output: {ticker: DataFrame} sliced to [start, end].
    Raises OOSAccessError for any request reaching OOS while ALLOW_OOS is False.
    """
    if end is None and not config.ALLOW_OOS:
        end = config.IS_END
    check_oos_allowed(start, end)

    tickers = tickers or config.all_tickers()
    out = {}
    for ticker in tickers:
        df = read_snapshot(latest_snapshot(ticker, raw_dir))
        out[ticker] = df.loc[start:end]
    return out


# ---------------------------------------------------------------------------
# Data-quality report
# ---------------------------------------------------------------------------
def currency_of(ticker: str) -> str:
    """Quote currency inferred from the ticker suffix (HK lines in HKD, ADRs in USD)."""
    if ticker == config.FX_TICKER:
        return "HKD per USD"
    return "HKD" if ticker.endswith(".HK") else "USD"


def suspicious_rows(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (date, issue) for bars that need a human look.

    Issues: missing_price, nonpositive_price, open_outside_range (open not within
    [low, high]), flat_bar (open=high=low=close with volume > 0, a common sign of
    a filled-in quote), zero_volume (not checked for FX, which has no volume).
    """
    is_fx = ticker == config.FX_TICKER
    px = df[PRICE_COLUMNS]
    tol = 1e-9
    checks = {
        "missing_price": px.isna().any(axis=1),
        "nonpositive_price": (px <= 0).any(axis=1),
        "open_outside_range": (df["open"] < df["low"] - tol) | (df["open"] > df["high"] + tol),
        "flat_bar": (px.nunique(axis=1) == 1) & (df["volume"] > 0) & (not is_fx),
        "zero_volume": (df["volume"] == 0) & (not is_fx),
    }
    rows = []
    for issue, mask in checks.items():
        for d in df.index[mask.fillna(False).to_numpy(bool)]:
            rows.append({"ticker": ticker, "date": d.date(), "issue": issue,
                         **{c: df.at[d, c] for c in PRICE_COLUMNS + ["volume"]}})
    return pd.DataFrame(rows, columns=["ticker", "date", "issue", *PRICE_COLUMNS, "volume"])


def quality_report(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise data quality per ticker.

    Input: {ticker: snapshot frame} (e.g. from load_raw()).
    Output: (summary with one row per ticker, detail with one row per suspicious bar).
    max_gap_days is the largest calendar-day gap between consecutive rows; values
    above ~4 days usually mean a holiday cluster (e.g. Lunar New Year) or missing data.
    """
    summary, detail = [], []
    for ticker, df in frames.items():
        bad = suspicious_rows(ticker, df)
        detail.append(bad)
        counts = bad["issue"].value_counts()
        splits = df.loc[df["splits"].fillna(0) != 0, "splits"]
        gaps = df.index.to_series().diff().dt.days
        summary.append({
            "ticker": ticker,
            "currency": currency_of(ticker),
            "first_date": df.index.min().date() if len(df) else None,
            "last_date": df.index.max().date() if len(df) else None,
            "n_rows": len(df),
            "duplicate_dates": int(df.index.duplicated().sum()),
            **{f"missing_{c}": int(df[c].isna().sum()) for c in PRICE_COLUMNS + ["volume"]},
            **{issue: int(counts.get(issue, 0)) for issue in
               ["nonpositive_price", "open_outside_range", "flat_bar", "zero_volume"]},
            "max_gap_days": int(gaps.max()) if len(df) > 1 else None,
            "max_gap_end": gaps.idxmax().date() if len(df) > 1 else None,
            "n_dividends": int((df["dividends"].fillna(0) != 0).sum()),
            "n_splits": len(splits),
            "split_details": "; ".join(f"{d.date()}:{v:g}" for d, v in splits.items()),
        })
    return pd.DataFrame(summary), pd.concat(detail, ignore_index=True)


def save_quality_report(frames: dict[str, pd.DataFrame], log_dir: Path = config.LOG_DIR) -> pd.DataFrame:
    """Write raw_quality.csv (summary) and raw_suspicious_rows.csv (detail); return the summary."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    summary, detail = quality_report(frames)
    summary.to_csv(log_dir / "raw_quality.csv", index=False)
    detail.to_csv(log_dir / "raw_suspicious_rows.csv", index=False)
    return summary
