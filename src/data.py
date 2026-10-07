"""Download and load the raw price snapshot (unadjusted OHLCV, dividends, splits, FX).

download_raw() is the only function that touches the network; load_raw() reads
data/raw/ only and refuses OOS dates while config.ALLOW_OOS is False.

Implemented in Session 1. No code yet (Session 0 skeleton).
"""
