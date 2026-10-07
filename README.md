# hk-us-double-listing-arbitrage-trading-strategy

Capstone (Data Structures and Algorithms, Imperial College Business School):
does the price gap between six Chinese companies' New York ADRs and Hong Kong
shares mean-revert profitably after realistic costs, out-of-sample?
See `CLAUDE.md` for the full specification.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Data provenance

All data are public daily prices from Yahoo Finance via `yfinance` 1.7.0,
downloaded on 2026-10-07 (details in `data/raw/manifest.json`):

- 6 US ADRs, 6 HK ordinary lines, and `HKD=X` (HKD per USD).
- `auto_adjust=False`: open/high/low/close are **not dividend-adjusted** but are
  **split-adjusted** by Yahoo; `adj_close` includes dividends. `repair=False`,
  `keepna=True` (Yahoo's numbers are kept as-is; problems are logged, not patched).
- Window 2019-01-01 to 2026-10-07: wider than the 2021-04-19 to 2026-09-30
  backtest so rolling windows can warm up and the last HK t+1 open exists.
- One CSV per ticker: `data/raw/{TICKER}_{YYYYMMDD}.csv` (download date).

The snapshot is the single source for all results. To reproduce offline:

```python
from src.data import load_raw          # reads data/raw/ only, no network
frames = load_raw()                    # in-sample only while config.ALLOW_OOS is False
```

To re-download (optional; creates new dated files, the newest is used):

```python
from src.data import download_raw
download_raw()
```
