"""Calendar alignment, FX conversion, ADR ratios and dividend ex-date fixes.

Pipeline per pair (clean_pair):
  1. flag bad HK bars  -> drop single-price zero-volume bars on normal days,
                          keep them on early-close days but mark the open unreliable
  2. align the legs    -> keep dates on which BOTH legs traded; log every other date
  3. attach FX         -> HKD=X close of the previous FX business day (t-1)
  4. attach ADR ratio  -> date-dependent, from config
  5. dividend fixes    -> add-back amounts for mismatched ex-dates (signal only)

Every dropped date, FX fill, bar flag and dividend correction is logged to
results/logs/ by clean_all(). Nothing is dropped silently.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import exchange_calendars as xc
import pandas as pd

import config

PRICE_COLUMNS = ["open", "high", "low", "close"]

# Dividend matching tolerances (see match_dividends).
DIV_MATCH_MAX_DAYS = 14     # calendar days between the two legs' ex-dates
DIV_MATCH_REL_TOL = 0.15    # |US amount / implied US amount - 1|


@lru_cache(maxsize=None)
def calendar(code: str) -> xc.ExchangeCalendar:
    """Cached exchange calendar ('XNYS' or 'XHKG')."""
    return xc.get_calendar(code)


# ---------------------------------------------------------------------------
# 1. ADR ratio
# ---------------------------------------------------------------------------
def ratio_on(name: str, dates: pd.DatetimeIndex) -> pd.Series:
    """HK shares per ADR for each date.

    Uses config.PAIRS[name] as the base ratio and applies config.RATIO_CHANGES
    entries from their effective date (inclusive) onwards.
    """
    ratio = pd.Series(float(config.PAIRS[name][2]), index=dates, name="ratio")
    for effective, value in config.RATIO_CHANGES.get(name, []):
        ratio[dates >= pd.Timestamp(effective)] = float(value)
    return ratio


# ---------------------------------------------------------------------------
# 2. Bad HK bars
# ---------------------------------------------------------------------------
def hk_bar_flags(hk: pd.DataFrame) -> pd.DataFrame:
    """Classify Yahoo's single-price, zero-volume HK bars (open=high=low=close, volume=0).

    Output: DataFrame indexed like `hk` with boolean columns
      - drop_bar: single-price bar on a normal session -> the close itself is
        not trustworthy (e.g. 9888.HK on 2022-03-14), so the date is dropped.
      - open_unreliable: single-price bar on an XHKG early-close (half-day)
        session -> the close is kept, but the open is not a real open.
    """
    single = (hk[PRICE_COLUMNS].nunique(axis=1) == 1) & (hk["volume"] == 0)
    early = hk.index.isin(calendar("XHKG").early_closes)
    return pd.DataFrame({"drop_bar": single & ~early, "open_unreliable": single & early}, index=hk.index)


# ---------------------------------------------------------------------------
# 3. Calendar alignment
# ---------------------------------------------------------------------------
def drop_reason(d: pd.Timestamp, in_hk: bool, bad_hk_bar: bool) -> str:
    """Why date d is not an aligned pair date (exactly one leg has a usable row)."""
    if bad_hk_bar:
        return "unreliable HK bar"
    if not in_hk:
        return "HK missing data (XHKG open)" if calendar("XHKG").is_session(d) else "HK holiday (XHKG closed)"
    return "US missing data (XNYS open)" if calendar("XNYS").is_session(d) else "US holiday (XNYS closed)"


def align_pair(name: str, us: pd.DataFrame, hk: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the dates on which both legs have a usable row.

    Inputs: snapshot frames for the US and HK leg.
    Output: (aligned, dropped)
      - aligned: index `date`; us_open, us_close, hk_open, hk_close, hk_open_reliable.
      - dropped: one row per date in the pair window where only one leg has a
        usable row, with the reason (see drop_reason).
    The pair window runs from the later first date to the earlier last date, so
    days before the HK listing are not counted as "dropped".
    Timing: no shifting here; both legs keep their own local trading date t.
    """
    start = max(us.index.min(), hk.index.min())
    end = min(us.index.max(), hk.index.max())
    us = us.loc[start:end]
    hk = hk.loc[start:end]

    flags = hk_bar_flags(hk)
    bad = set(flags.index[flags["drop_bar"]])
    us_dates = set(us.index)
    hk_all = set(hk.index)
    hk_dates = hk_all - bad
    common = pd.DatetimeIndex(sorted(us_dates & hk_dates), name="date")

    dropped = [
        {"pair": name, "date": d.date(), "us_row": d in us_dates, "hk_row": d in hk_all,
         "reason": drop_reason(d, d in hk_dates, d in bad)}
        for d in sorted((us_dates | hk_all) - set(common))
    ]

    aligned = pd.DataFrame({
        "us_open": us.loc[common, "open"],
        "us_close": us.loc[common, "close"],
        "hk_open": hk.loc[common, "open"],
        "hk_close": hk.loc[common, "close"],
        "hk_open_reliable": ~flags.loc[common, "open_unreliable"],
    }, index=common)
    return aligned, pd.DataFrame(dropped, columns=["pair", "date", "us_row", "hk_row", "reason"])


# ---------------------------------------------------------------------------
# 4. FX
# ---------------------------------------------------------------------------
def fx_for_dates(fx: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """FX (HKD per USD) to use on each date t: the HKD=X close of the previous business day.

    Timing: Yahoo's FX close for day t is stamped at the London end of day, after
    the US open on t, so it is NOT known at decision time. The t-1 close is.
    A missing t-1 close is forward-filled from at most 1 business day earlier;
    beyond that the value is NaN and the caller drops the date.

    Output: DataFrame indexed by `dates` with
      fx (float or NaN), fx_date (date the value was observed), fx_filled (bool).
    """
    close = fx["close"]
    grid = pd.bdate_range(close.index.min(), max(close.index.max(), dates.max()))
    on_grid = close.reindex(grid)
    source_date = pd.Series(grid.where(on_grid.notna()), index=grid).ffill(limit=1)
    filled_close = on_grid.ffill(limit=1)

    prev_bday = dates - pd.offsets.BDay(1)
    out = pd.DataFrame({
        "fx": filled_close.reindex(prev_bday).to_numpy(),
        "fx_date": source_date.reindex(prev_bday).to_numpy(),
        "fx_filled": (on_grid.reindex(prev_bday).isna() & filled_close.reindex(prev_bday).notna()).to_numpy(),
    }, index=dates)
    return out


# ---------------------------------------------------------------------------
# 5. Dividends
# ---------------------------------------------------------------------------
def match_dividends(name: str, us: pd.DataFrame, hk: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """Pair each HK dividend with the corresponding US (ADR) dividend.

    HK amounts are HKD per share; US amounts are USD per ADR. A match needs
    ex-dates within DIV_MATCH_MAX_DAYS and |US / (N * HK / FX) - 1| <= DIV_MATCH_REL_TOL
    (the tolerance absorbs depositary fees and FX). Closest ex-date wins.
    Only dividends inside the pair window are considered.

    Output: one row per event: hk_ex, hk_amount, us_ex, us_amount, status
    ('matched', 'unmatched HK', 'unmatched US').
    """
    start = max(us.index.min(), hk.index.min())
    end = min(us.index.max(), hk.index.max())
    us_div = us.loc[start:end, "dividends"]
    hk_div = hk.loc[start:end, "dividends"]
    us_div, hk_div = us_div[us_div > 0], hk_div[hk_div > 0]
    fx_close = fx["close"].ffill()

    rows, used_us = [], set()
    for hk_ex, hk_amt in hk_div.items():
        n = ratio_on(name, pd.DatetimeIndex([hk_ex])).iloc[0]
        implied_usd = n * hk_amt / fx_close.asof(hk_ex)
        candidates = [
            (abs((us_ex - hk_ex).days), us_ex) for us_ex, us_amt in us_div.items()
            if us_ex not in used_us
            and abs((us_ex - hk_ex).days) <= DIV_MATCH_MAX_DAYS
            and abs(us_amt / implied_usd - 1) <= DIV_MATCH_REL_TOL
        ]
        if candidates:
            us_ex = min(candidates)[1]
            used_us.add(us_ex)
            rows.append({"hk_ex": hk_ex, "hk_amount": hk_amt, "us_ex": us_ex,
                         "us_amount": us_div[us_ex], "status": "matched"})
        else:
            rows.append({"hk_ex": hk_ex, "hk_amount": hk_amt, "us_ex": pd.NaT,
                         "us_amount": float("nan"), "status": "unmatched HK"})
    for us_ex, us_amt in us_div.items():
        if us_ex not in used_us:
            rows.append({"hk_ex": pd.NaT, "hk_amount": float("nan"), "us_ex": us_ex,
                         "us_amount": us_amt, "status": "unmatched US"})
    return pd.DataFrame(rows, columns=["hk_ex", "hk_amount", "us_ex", "us_amount", "status"])


def dividend_adjustments(
    name: str, dates: pd.DatetimeIndex, events: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add-back amounts that remove fake spread jumps from mismatched ex-dates.

    Timing: the spread compares the US open on t with the HK close on t, so the
    US leg is ex-dividend on t iff t >= us_ex, and the HK leg iff t >= hk_ex.
    On aligned dates in [first ex-date, second ex-date) one leg is ex and the
    other is not; there we add the declared dividend back to the leg that went
    ex first. Dividends are announced weeks ahead, so this uses no future data.
    Used for the spread SIGNAL only; P&L must use raw prices + cash dividends.

    Output: (adj, log)
      - adj: index `dates`; div_adj_us (USD per ADR), div_adj_hk (HKD per share).
      - log: one row per event with action and number of aligned days adjusted.
    """
    adj = pd.DataFrame(0.0, index=dates, columns=["div_adj_us", "div_adj_hk"])
    log = []
    for ev in events.itertuples(index=False):
        row = {"pair": name, "hk_ex": ev.hk_ex, "hk_amount_hkd": ev.hk_amount,
               "us_ex": ev.us_ex, "us_amount_usd": ev.us_amount, "status": ev.status,
               "action": "", "days_adjusted": 0, "first_adjusted": pd.NaT, "last_adjusted": pd.NaT}
        if ev.status != "matched":
            row["action"] = "none (unmatched; check manually)"
        elif ev.hk_ex == ev.us_ex:
            row["action"] = "none (same ex-date)"
        else:
            first, second = sorted([ev.hk_ex, ev.us_ex])
            window = dates[(dates >= first) & (dates < second)]
            if ev.hk_ex < ev.us_ex:
                adj.loc[window, "div_adj_hk"] += ev.hk_amount
                row["action"] = "add back HK dividend (HK ex first)"
            else:
                adj.loc[window, "div_adj_us"] += ev.us_amount
                row["action"] = "add back US dividend (US ex first)"
            row["days_adjusted"] = len(window)
            if len(window):
                row["first_adjusted"], row["last_adjusted"] = window.min(), window.max()
        log.append(row)
    return adj, pd.DataFrame(log)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def clean_pair(name: str, frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run steps 1-5 for one pair.

    Input: pair name and snapshot frames from data.load_raw() (already behind the
    OOS firewall).
    Output: (cleaned, logs). cleaned has index `date` and columns
      ratio, us_open, us_close, hk_open, hk_close, hk_open_reliable,
      fx, fx_date, fx_filled, div_adj_us, div_adj_hk.
    logs: dropped, fx_fills, dividends, hk_bars.
    """
    us_t, hk_t, _ = config.PAIRS[name]
    us, hk, fx = frames[us_t], frames[hk_t], frames[config.FX_TICKER]

    aligned, dropped = align_pair(name, us, hk)

    fxd = fx_for_dates(fx, aligned.index)
    no_fx = fxd.index[fxd["fx"].isna()]
    if len(no_fx):
        dropped = pd.concat([dropped, pd.DataFrame({
            "pair": name, "date": no_fx.date, "us_row": True, "hk_row": True,
            "reason": "FX missing (t-1, beyond 1-day fill)"})], ignore_index=True)
        aligned = aligned.drop(no_fx)
        fxd = fxd.drop(no_fx)

    events = match_dividends(name, us, hk, fx)
    adj, div_log = dividend_adjustments(name, aligned.index, events)

    cleaned = pd.concat([ratio_on(name, aligned.index), aligned, fxd, adj], axis=1)
    cleaned.index.name = "date"

    fills = fxd[fxd["fx_filled"]]
    fx_fills = pd.DataFrame({"pair": name, "date": fills.index.date,
                             "fx_date_used": pd.DatetimeIndex(fills["fx_date"]).date,
                             "fx": fills["fx"].to_numpy()})

    flags = hk_bar_flags(hk.loc[aligned.index.min():aligned.index.max()])
    flagged = flags[flags.any(axis=1)]
    hk_bars = pd.DataFrame({
        "pair": name, "hk_ticker": hk_t, "date": flagged.index.date,
        "action": ["dropped (close unreliable)" if f else "kept, open unreliable (early close)"
                   for f in flagged["drop_bar"]],
        "close": hk.loc[flagged.index, "close"].to_numpy(),
    })

    dropped = dropped.sort_values("date").reset_index(drop=True)
    return cleaned, {"dropped": dropped, "fx_fills": fx_fills, "dividends": div_log, "hk_bars": hk_bars}


LOG_FILES = {
    "dropped": "dropped_dates.csv",
    "fx_fills": "fx_fills.csv",
    "dividends": "dividend_corrections.csv",
    "hk_bars": "hk_bar_flags.csv",
}


def clean_all(frames: dict[str, pd.DataFrame], log_dir: Path | None = config.LOG_DIR) -> pd.DataFrame:
    """Clean every pair in config.PAIRS and write the logs.

    Output: long-format DataFrame (columns: date, pair, then clean_pair's columns).
    If log_dir is None, logs are not written (used in tests).
    """
    cleaned, logs = [], {k: [] for k in LOG_FILES}
    for name in config.PAIRS:
        df, pair_logs = clean_pair(name, frames)
        cleaned.append(df.reset_index().assign(pair=name))
        for k, v in pair_logs.items():
            logs[k].append(v)
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        for k, fname in LOG_FILES.items():
            pd.concat(logs[k], ignore_index=True).to_csv(Path(log_dir) / fname, index=False)
    out = pd.concat(cleaned, ignore_index=True)
    return out[["date", "pair"] + [c for c in out.columns if c not in ("date", "pair")]]
