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

All data are public daily prices from Yahoo Finance via `yfinance`
(unadjusted, `auto_adjust=False`). The snapshot in `data/raw/` is the single
source for all results; downloading is a separate, optional step.
