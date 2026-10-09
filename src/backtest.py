"""Leg-level daily P&L with execution timing (no costs; costs.py adds them in Session 5).

Timing (CLAUDE.md §6), for a decision made at the US open on aligned day t:
  US leg (main)        trades at the US close on t        -> earns close-to-close from the next US session
  US leg (upper bound) trades at the US open on t         -> earns t's intraday part (NOT tradable; Session 6)
  HK leg               trades at the HK open of the next HK session after t

Per leg and session h:
  overnight return r_on(h) = (open_h + div_h) / close_{h-1} − 1   (dividend: the open on the ex-date is ex)
  intraday  return r_id(h) = close_h / open_h − 1
  close-to-close   r_cc(h) = (close_h + div_h) / close_{h-1} − 1
  leg return(h) = pos·r_cc(h)                     if the position did not change during session h
                = old·r_on(h) + new·r_id(h)       if it changed at the open of h
HK prices are converted to USD with the same day's HKD=X close.
Each leg runs on its own exchange calendar, so moves and dividends on days when
the other market is closed are counted. P&L is never computed from spread changes.

Weights: pair position +1 = long spread = US weight +1, HK weight −1; each leg is
config.LEG_WEIGHT of pair capital; each pair is config.PAIR_WEIGHT of the portfolio.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src import clean
from src.data import OOSAccessError
from src.strategy import Params, positions, zscore


# ---------------------------------------------------------------------------
# Leg returns
# ---------------------------------------------------------------------------
def leg_returns(raw: pd.DataFrame, fx: pd.Series | None = None, drop: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Per-session open, close, dividend and returns for one leg, in USD.

    Inputs: snapshot frame (open, close, dividends); optional FX close series
    (local currency per USD, forward-filled onto the leg's sessions); optional
    dates to remove (unreliable bars) - the next session's returns then span the gap.
    Output: index `date`; open, close, dividends (USD), r_on, r_id, r_cc.
    The first session has NaN returns (no previous close).
    """
    df = raw[["open", "close", "dividends"]].copy()
    if drop is not None and len(drop):
        df = df.drop(df.index.intersection(drop))
    if fx is not None:
        rate = fx.reindex(fx.index.union(df.index)).ffill().reindex(df.index)
        df = df.div(rate, axis=0)
    prev_close = df["close"].shift(1)
    df["r_on"] = (df["open"] + df["dividends"]) / prev_close - 1
    df["r_id"] = df["close"] / df["open"] - 1
    df["r_cc"] = (df["close"] + df["dividends"]) / prev_close - 1
    return df


def execution_sessions(decision_dates: pd.DatetimeIndex, sessions: pd.DatetimeIndex, rule: str) -> pd.DatetimeIndex:
    """Session at which each decision is executed on one leg.

    rule 'same'  : the session on the decision date itself (US leg; decision dates
                   are aligned days, so the US traded that day)
    rule 'next'  : the first session strictly after the decision date (HK leg)
    A decision with no later session maps to NaT (never executed in the window).
    """
    if rule == "same":
        pos = sessions.searchsorted(decision_dates, side="left")
    elif rule == "next":
        pos = sessions.searchsorted(decision_dates, side="right")
    else:
        raise ValueError(rule)
    return pd.DatetimeIndex([sessions[i] if i < len(sessions) else pd.NaT for i in pos])


def leg_pnl(target: pd.Series, legs: pd.DataFrame, rule: str, at_open: bool) -> pd.DataFrame:
    """Daily return of one leg for a given target position path.

    Inputs:
      target   - leg position (+1 long, −1 short, 0) after each decision, indexed by decision date
      legs     - leg_returns() output, already restricted to the window
      rule     - 'same' or 'next' (see execution_sessions)
      at_open  - True: the new position is in place from the OPEN of the execution session
                 False: from the CLOSE of the execution session (US main variant)
    Output: index = leg sessions; pos_overnight, pos_intraday, pos_end, trade (|change| in
    leg notional units at that session), ret (leg return per unit of leg notional).
    """
    exec_at = execution_sessions(target.index, legs.index, rule)
    keep = ~exec_at.isna()
    after_exec = pd.Series(target.to_numpy()[keep], index=exec_at[keep])
    after_exec = after_exec[~after_exec.index.duplicated(keep="last")]
    pos_end = after_exec.reindex(legs.index).ffill().fillna(0.0)
    pos_overnight = pos_end.shift(1).fillna(0.0)
    pos_intraday = pos_end if at_open else pos_overnight

    changed = pos_intraday != pos_overnight
    ret = np.where(changed,
                   pos_overnight * legs["r_on"] + pos_intraday * legs["r_id"],
                   pos_overnight * legs["r_cc"])
    ret = pd.Series(ret, index=legs.index).where(pos_overnight.ne(0) | pos_intraday.ne(0), 0.0)
    return pd.DataFrame({"pos_overnight": pos_overnight, "pos_intraday": pos_intraday, "pos_end": pos_end,
                         "trade": (pos_end - pos_overnight).abs(), "ret": ret.fillna(0.0)})


def backtest_pair(decisions: pd.Series, us: pd.DataFrame, hk: pd.DataFrame, us_exec: str = "close") -> pd.DataFrame:
    """Daily pair return from a spread-position path and the two legs' returns.

    Inputs: decisions = spread position after each decision (index: aligned decision
    dates); us, hk = leg_returns() output restricted to the window (HK already in USD);
    us_exec = 'close' (main) or 'open' (upper bound, not tradable).
    Output: index = union of both legs' sessions; us_*, hk_* columns and pair_ret
    (= LEG_WEIGHT·us_ret + LEG_WEIGHT·hk_ret, 0 when a market is closed).
    """
    us_leg = leg_pnl(decisions, us, rule="same", at_open=(us_exec == "open"))
    hk_leg = leg_pnl(-decisions, hk, rule="next", at_open=True)
    out = pd.concat([us_leg.add_prefix("us_"), hk_leg.add_prefix("hk_")], axis=1, sort=True)  # dates in order
    pos_cols = [c for c in out.columns if "pos" in c]
    out[pos_cols] = out[pos_cols].ffill().fillna(0.0)  # a closed market keeps its position
    out = out.fillna(0.0)                              # ...and has zero return and zero trades
    out["pair_ret"] = config.LEG_WEIGHT * out["us_ret"] + config.LEG_WEIGHT * out["hk_ret"]
    out.index.name = "date"
    return out


# ---------------------------------------------------------------------------
# One pair end to end: signal -> positions -> P&L
# ---------------------------------------------------------------------------
def run_pair(
    spreads: pd.DataFrame,
    us_raw: pd.DataFrame,
    hk_raw: pd.DataFrame,
    fx_raw: pd.DataFrame,
    p: Params,
    signal: str = "spread",
    start: str = config.START,
    end: str | None = None,
    us_exec: str = "close",
    hk_drop: pd.DatetimeIndex | None = None,
) -> dict[str, pd.DataFrame]:
    """Signal, positions and daily P&L for one pair over [start, end].

    Inputs: spreads for this pair only (columns date and `signal`, warm-up rows
    included for the rolling window); raw snapshot frames for both legs and FX.
    hk_drop: HK dates to skip as unreliable (default: clean.hk_bar_flags drop_bar).
    The window ends at the last aligned date <= end; positions are forced flat on
    the second-to-last decision day so the HK exit trades inside the window.
    Output: {'decisions', 'trades', 'daily'}.
    """
    s = spreads.set_index("date")[signal].sort_index()
    if end is not None:
        s = s.loc[:end]
    if not config.ALLOW_OOS and s.index.max() >= pd.Timestamp(config.OOS_START):
        raise OOSAccessError("run_pair received out-of-sample dates while ALLOW_OOS is False")

    z = zscore(s, p.L).loc[start:]
    window_end = z.index.max()
    force_flat_from = z.index[-2] if len(z) >= 2 else window_end
    decisions, trades = positions(z, p, force_flat_from=force_flat_from)

    if hk_drop is None:
        flags = clean.hk_bar_flags(hk_raw)
        hk_drop = flags.index[flags["drop_bar"]]
    us = leg_returns(us_raw).loc[z.index.min():window_end]
    hk = leg_returns(hk_raw, fx=fx_raw["close"], drop=hk_drop).loc[z.index.min():window_end]

    daily = backtest_pair(decisions["position"].astype(float), us, hk, us_exec=us_exec)
    return {"decisions": decisions, "trades": trades, "daily": daily}


def run_backtest(
    spreads: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    p: Params,
    signal: str = "spread",
    start: str = config.START,
    end: str | None = None,
    us_exec: str = "close",
) -> dict:
    """All pairs and the equal-weight portfolio.

    Inputs: build_spreads() output (long format) and load_raw() frames.
    Output: {'pairs': {name: run_pair output}, 'portfolio': daily DataFrame with one
    return column per pair and 'portfolio_ret' = sum of PAIR_WEIGHT * pair_ret}.
    """
    end = end or (config.END if config.ALLOW_OOS else config.IS_END)
    pairs = {}
    for name, (us_t, hk_t, _) in config.PAIRS.items():
        sp = spreads[spreads["pair"] == name]
        pairs[name] = run_pair(sp, frames[us_t], frames[hk_t], frames[config.FX_TICKER], p,
                               signal=signal, start=start, end=end, us_exec=us_exec)
    rets = pd.DataFrame({n: r["daily"]["pair_ret"] for n, r in pairs.items()}).fillna(0.0)
    rets["portfolio_ret"] = (rets[list(pairs)] * config.PAIR_WEIGHT).sum(axis=1)
    rets.index.name = "date"
    return {"pairs": pairs, "portfolio": rets}


# ---------------------------------------------------------------------------
# Run log (every parameter combination tried; CLAUDE.md rule 3)
# ---------------------------------------------------------------------------
GRID_LOG_COLUMNS = [
    "timestamp_utc", "run_type", "period", "signal", "us_exec", "L", "k", "exit_z", "H", "stop_z",
    "cost_multiplier", "n_trades", "avg_days_held", "pct_days_in_market",
    "cum_return", "sharpe", "max_drawdown", "hit_rate", "turnover", "cost_drag", "notes",
]


def log_run(row: dict, path: Path = config.GRID_LOG) -> None:
    """Append one run to results/grid_log.csv (header written on first use). Unknown keys raise."""
    unknown = set(row) - set(GRID_LOG_COLUMNS)
    if unknown:
        raise KeyError(f"unknown grid-log columns: {sorted(unknown)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GRID_LOG_COLUMNS)
        if new:
            w.writeheader()
        w.writerow({"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **row})
