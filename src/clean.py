"""Calendar alignment, FX conversion, ADR ratios and dividend ex-date fixes.

Pipeline per pair (clean_pair):
  1. flag bad HK bars  -> drop single-price zero-volume bars on normal days,
                          keep them on early-close days but mark the open unreliable
  2. align the legs    -> keep dates on which BOTH legs traded; log every other date
  3. attach FX         -> HKD=X close of the previous FX business day (t-1)
  4. attach ADR ratio  -> date-dependent, from config
  5. evening inputs   -> previous US session close (for the evening reading)
  6. dividend fixes    -> add-back amounts per reading (signal only)

Every dropped date, FX fill, bar flag and dividend correction is logged to
results/logs/ by clean_all(). Nothing is dropped silently.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import exchange_calendars as xc
import numpy as np
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
def ratio_on(name: str, dates: pd.DatetimeIndex, pairs: dict | None = None) -> pd.Series:
    """HK shares per ADR for each date.

    Uses pairs[name] (default config.PAIRS) as the base ratio and applies config.RATIO_CHANGES
    entries from their effective date (inclusive) onwards.
    """
    ratio = pd.Series(float((pairs or config.PAIRS)[name][2]), index=dates, name="ratio")
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
def match_dividends(name: str, us: pd.DataFrame, hk: pd.DataFrame, fx: pd.DataFrame,
                    pairs: dict | None = None) -> pd.DataFrame:
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
        n = ratio_on(name, pd.DatetimeIndex([hk_ex]), pairs).iloc[0]
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


def add_backs(
    index: pd.DatetimeIndex, us_obs: pd.DatetimeIndex, hk_obs: pd.DatetimeIndex, events: pd.DataFrame
) -> tuple[pd.DataFrame, list[int]]:
    """Dividend add-backs for one spread reading.

    A reading compares a US price observed on date us_obs[i] with an HK price
    observed on hk_obs[i]. The US price is ex-dividend iff us_obs >= us_ex; the
    HK price iff hk_obs >= hk_ex. When exactly one leg is ex, its declared
    dividend is added back so the reading has no artificial jump.
    Output: (adj with columns us, hk indexed by `index`; days adjusted per event).
    """
    adj = pd.DataFrame(0.0, index=index, columns=["us", "hk"])
    days = []
    for ev in events.itertuples(index=False):
        if ev.status != "matched":
            days.append(0)
            continue
        us_ex = np.asarray(us_obs >= ev.us_ex)
        hk_ex = np.asarray(hk_obs >= ev.hk_ex)
        adj.loc[hk_ex & ~us_ex, "hk"] += ev.hk_amount
        adj.loc[us_ex & ~hk_ex, "us"] += ev.us_amount
        days.append(int((hk_ex != us_ex).sum()))
    return adj, days


def dividend_adjustments(
    name: str, dates: pd.DatetimeIndex, us_prev_dates: pd.DatetimeIndex, events: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add-back amounts that remove fake spread jumps from ex-date mismatches.

    Morning reading (US open t vs HK close t): both legs observed on t, so a
    correction is needed only when the two ex-dates differ; on aligned dates in
    [first ex-date, second ex-date) the leg that went ex first gets its dividend
    added back.
    Evening reading (US close of the previous US session vs HK open t): the legs
    are observed on different dates, so even a same-day ex-date needs one day of
    correction (the HK open on ex-date t is ex, the US close the evening before
    is still cum).
    Dividends are announced weeks ahead, so this uses no future data.
    Used for the spread SIGNAL only; P&L must use raw prices + cash dividends.

    Output: (adj, log)
      - adj: index `dates`; div_adj_us, div_adj_hk (morning), div_adj_us_eve,
        div_adj_hk_eve (evening); US in USD per ADR, HK in HKD per share.
      - log: one row per event with the morning action and days adjusted per reading.
    """
    morning, days_m = add_backs(dates, dates, dates, events)
    evening, days_e = add_backs(dates, us_prev_dates, dates, events)
    adj = pd.DataFrame({
        "div_adj_us": morning["us"], "div_adj_hk": morning["hk"],
        "div_adj_us_eve": evening["us"], "div_adj_hk_eve": evening["hk"],
    }, index=dates)

    log = []
    for ev, dm, de in zip(events.itertuples(index=False), days_m, days_e):
        if ev.status != "matched":
            action = "none (unmatched; check manually)"
        elif ev.hk_ex == ev.us_ex:
            action = "none (same ex-date)"
        elif ev.hk_ex < ev.us_ex:
            action = "add back HK dividend (HK ex first)"
        else:
            action = "add back US dividend (US ex first)"
        log.append({"pair": name, "hk_ex": ev.hk_ex, "hk_amount_hkd": ev.hk_amount,
                    "us_ex": ev.us_ex, "us_amount_usd": ev.us_amount, "status": ev.status,
                    "action": action, "days_adjusted": dm, "days_adjusted_eve": de})
    return adj, pd.DataFrame(log)


def previous_us_session(us: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Last US close strictly before each date (the US side of the evening reading).

    Uses the raw US series, so a US session that is not an aligned pair date
    (e.g. an HK holiday) still counts as "the evening before".
    Output: DataFrame indexed by `dates` with us_prev_date and us_close_prev
    (NaT/NaN when no earlier US session exists).
    """
    pos = us.index.searchsorted(dates, side="left") - 1
    valid = pos >= 0
    prev = pd.DatetimeIndex([us.index[p] if ok else pd.NaT for p, ok in zip(pos, valid)])
    close = [us["close"].iloc[p] if ok else float("nan") for p, ok in zip(pos, valid)]
    return pd.DataFrame({"us_prev_date": prev, "us_close_prev": close}, index=dates)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def clean_pair(name: str, frames: dict[str, pd.DataFrame],
               pairs: dict | None = None) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run steps 1-6 for one pair.

    Input: pair name and snapshot frames from data.load_raw() (already behind the
    OOS firewall).
    Output: (cleaned, logs). cleaned has index `date` and columns
      ratio, us_open, us_close, hk_open, hk_close, hk_open_reliable,
      us_prev_date, us_close_prev, fx, fx_date, fx_filled,
      div_adj_us, div_adj_hk, div_adj_us_eve, div_adj_hk_eve.
    logs: dropped, fx_fills, dividends, hk_bars.
    """
    us_t, hk_t, _ = (pairs or config.PAIRS)[name]
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

    prev = previous_us_session(us, aligned.index)
    events = match_dividends(name, us, hk, fx, pairs)
    adj, div_log = dividend_adjustments(name, aligned.index, pd.DatetimeIndex(prev["us_prev_date"]), events)

    cleaned = pd.concat([ratio_on(name, aligned.index, pairs), aligned, prev, fxd, adj], axis=1)
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


def clean_all(frames: dict[str, pd.DataFrame], log_dir: Path | None = config.LOG_DIR,
              pairs: dict | None = None, log_prefix: str = "") -> pd.DataFrame:
    """Clean every pair in `pairs` (default config.PAIRS) and write the logs.

    log_prefix is prepended to every log file name (e.g. "holdout_"), so a second
    universe never overwrites the main logs.

    Output: long-format DataFrame (columns: date, pair, then clean_pair's columns).
    If log_dir is None, logs are not written (used in tests).
    """
    cleaned, logs = [], {k: [] for k in LOG_FILES}
    for name in (pairs or config.PAIRS):
        df, pair_logs = clean_pair(name, frames, pairs)
        cleaned.append(df.reset_index().assign(pair=name))
        for k, v in pair_logs.items():
            logs[k].append(v)
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        for k, fname in LOG_FILES.items():
            pd.concat(logs[k], ignore_index=True).to_csv(Path(log_dir) / f"{log_prefix}{fname}", index=False)
    out = pd.concat(cleaned, ignore_index=True)
    return out[["date", "pair"] + [c for c in out.columns if c not in ("date", "pair")]]
