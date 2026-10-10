"""Leg-level daily P&L with execution timing (no costs; costs.py adds them in Session 5).

Timing (CLAUDE.md §6), for a decision made at the US open on aligned day t:
  US leg (main)        trades at the US close on t        -> earns close-to-close from the next US session
  US leg (upper bound) trades at the US open on t         -> earns t's intraday part (NOT tradable; Session 6)
  HK leg               trades at the HK open of the next HK session after t

P&L is computed from a share-holding book (backtest_pair): each leg buys shares
worth LEG_WEIGHT x pair equity at entry and holds that share count until exit,
marked to market at every open and close it trades through. On the day a leg
trades at the open, the old shares earn the overnight move (incl. dividend) and
the new shares earn the intraday move. HK prices are converted to USD with the
same day's HKD=X close. Each leg runs on its own exchange calendar, so moves and
dividends on days when the other market is closed are counted. P&L is never
computed from spread changes.

Position: +1 = long spread = long US / short HK. Each pair is config.PAIR_WEIGHT of
the portfolio; the portfolio return is the PAIR_WEIGHT-weighted sum of pair returns
(cash re-allocated across pairs daily; no shares are traded for that).
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src import clean
from src.costs import CostModel, cost_model
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


def backtest_pair(decisions: pd.Series, us: pd.DataFrame, hk: pd.DataFrame, us_exec: str = "close",
                  cost: CostModel | None = None) -> pd.DataFrame:
    """Hold-shares book for one pair: daily P&L from share counts and USD prices.

    Inputs: decisions = spread position after each decision (index: aligned decision
    dates); us, hk = leg_returns() output restricted to the window (HK already in USD);
    us_exec = 'close' (main) or 'open' (upper bound, not tradable);
    cost = costs.CostModel or None (gross, no costs).

    Sizing: when a decision changes the position, each leg's target notional is
    N = LEG_WEIGHT x pair equity at the end of the previous date. The US leg buys or
    sells N dollars of shares at its execution price, the HK leg likewise at the next
    HK open. Share counts then stay FIXED until the exit (no rebalancing), exactly
    as a trader would hold them; leg values drift with prices during the trade.

    Order of events on each date d (real time): HK session first (it ends 04:00 ET),
    then the decision at the US open, then the US session.
      HK session: overnight P&L = shares x (open − prev close + dividend) [old shares]
                  execute any order due at this open
                  intraday P&L  = shares x (close − open)                 [new shares]
      US session: 'close' -> P&L = shares x (close − prev close + dividend), then execute at the close
                  'open'  -> overnight / execute at the open / intraday, as for HK
    Costs (if given): every trade pays cost.trade_cost(traded dollars) at execution;
    at the start of each date, borrow accrues on the short legs' dollar value at the
    previous date's close for the calendar days elapsed. Costs leave pair equity at
    once, so later trades are sized from equity after costs.
    One pass over the dates with O(1) work each: O(T).

    Output (index = union of both legs' sessions), all money columns as a fraction of
    pair equity at the end of the previous date:
      us_pos, hk_pos       sign of the leg after the date (+1 long, −1 short, 0)
      us_trade, hk_trade   traded notional (for transaction costs)
      us_value, hk_value   signed position value at the date's close (for borrow on shorts)
      us_pnl, hk_pnl       price P&L (incl. dividends); gross_ret = us_pnl + hk_pnl
      us_cost, hk_cost     transaction costs; borrow_cost = borrow fee
      pair_ret             net return = gross_ret − us_cost − hk_cost − borrow_cost
      equity               pair equity (net of costs), starting at 1.0
    """
    dates = us.index.union(hk.index)
    us_at = execution_sessions(decisions.index, us.index, "same")
    hk_at = execution_sessions(decisions.index, hk.index, "next")

    equity = 1.0
    us_sh = hk_sh = 0.0
    us_prev_close = hk_prev_close = np.nan
    current = 0.0
    us_orders: dict[pd.Timestamp, tuple[float, float]] = {}   # session -> (target sign, notional)
    hk_orders: dict[pd.Timestamp, tuple[float, float]] = {}
    decision_at = dict(zip(decisions.index, decisions.to_numpy()))
    us_exec_of = dict(zip(decisions.index, us_at))
    hk_exec_of = dict(zip(decisions.index, hk_at))
    rows = []
    prev_date = None

    for d in dates:
        e_prev = equity
        borrow = 0.0
        if cost is not None and prev_date is not None:
            short_value = min(us_sh * us_prev_close, 0.0) if us_sh else 0.0
            short_value += min(hk_sh * hk_prev_close, 0.0) if hk_sh else 0.0
            borrow = cost.borrow_cost(short_value, (d - prev_date).days)
        prev_date = d
        # decision at the US open of d: place orders sized from equity at the end of d-1
        if d in decision_at and decision_at[d] != current:
            current = decision_at[d]
            n = config.LEG_WEIGHT * e_prev
            if not pd.isna(us_exec_of[d]):
                us_orders[us_exec_of[d]] = (+current, n)   # US leg = +spread position
            if not pd.isna(hk_exec_of[d]):
                hk_orders[hk_exec_of[d]] = (-current, n)   # HK leg = −spread position

        hk_pnl = hk_trade = 0.0
        if d in hk.index:
            o, c, div = hk.at[d, "open"], hk.at[d, "close"], hk.at[d, "dividends"]
            if hk_sh != 0:
                hk_pnl += hk_sh * (o - hk_prev_close + div)
            if d in hk_orders:
                sign, n = hk_orders.pop(d)
                new = sign * n / o
                hk_trade, hk_sh = abs(new - hk_sh) * o, new
            hk_pnl += hk_sh * (c - o)
            hk_prev_close = c

        us_pnl = us_trade = 0.0
        if d in us.index:
            o, c, div = us.at[d, "open"], us.at[d, "close"], us.at[d, "dividends"]
            if us_exec == "open":
                if us_sh != 0:
                    us_pnl += us_sh * (o - us_prev_close + div)
                if d in us_orders:
                    sign, n = us_orders.pop(d)
                    new = sign * n / o
                    us_trade, us_sh = abs(new - us_sh) * o, new
                us_pnl += us_sh * (c - o)
            else:
                if us_sh != 0:
                    us_pnl += us_sh * (c - us_prev_close + div)
                if d in us_orders:
                    sign, n = us_orders.pop(d)
                    new = sign * n / c
                    us_trade, us_sh = abs(new - us_sh) * c, new
            us_prev_close = c

        us_cost = cost.trade_cost(us_trade, "us") if cost is not None else 0.0
        hk_cost = cost.trade_cost(hk_trade, "hk") if cost is not None else 0.0
        equity = e_prev + us_pnl + hk_pnl - us_cost - hk_cost - borrow
        rows.append({
            "date": d, "us_pos": np.sign(us_sh), "hk_pos": np.sign(hk_sh),
            "us_trade": us_trade / e_prev, "hk_trade": hk_trade / e_prev,
            "us_value": us_sh * us_prev_close / e_prev if us_sh else 0.0,
            "hk_value": hk_sh * hk_prev_close / e_prev if hk_sh else 0.0,
            "us_pnl": us_pnl / e_prev, "hk_pnl": hk_pnl / e_prev,
            "gross_ret": (us_pnl + hk_pnl) / e_prev,
            "us_cost": us_cost / e_prev, "hk_cost": hk_cost / e_prev, "borrow_cost": borrow / e_prev,
            "pair_ret": (equity - e_prev) / e_prev, "equity": equity,
        })
    return pd.DataFrame(rows).set_index("date")


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
    cost: CostModel | None = None,
) -> dict[str, pd.DataFrame]:
    """Signal, positions and daily P&L for one pair over [start, end].

    Inputs: spreads for this pair only (columns date and `signal`, warm-up rows
    included for the rolling window); raw snapshot frames for both legs and FX.
    hk_drop: HK dates to skip as unreliable (default: clean.hk_bar_flags drop_bar).
    cost: CostModel for this pair, or None for gross P&L.
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

    daily = backtest_pair(decisions["position"].astype(float), us, hk, us_exec=us_exec, cost=cost)
    return {"decisions": decisions, "trades": trades, "daily": daily}


def run_backtest(
    spreads: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    p: Params,
    signal: str = "spread",
    start: str = config.START,
    end: str | None = None,
    us_exec: str = "close",
    cost_multiplier: float = 1.0,
    pairs: dict | None = None,
    half_spreads: dict | None = None,
    borrow_annual: float | None = None,
) -> dict:
    """All pairs and the equal-weight portfolio.

    pairs: universe to run (default config.PAIRS); each pair gets 1/len(pairs) of capital.
    half_spreads / borrow_annual: cost overrides (default config) - used for the
    out-of-universe basket and its borrow sensitivity.

    Inputs: build_spreads() output (long format) and load_raw() frames;
    cost_multiplier scales every cost in config (0 = gross).
    Output: {'pairs': {name: run_pair output}, 'portfolio': daily DataFrame with one
    return column per pair and 'portfolio_ret' = sum of pair_ret / len(pairs)}.
    """
    end = end or (config.END if config.ALLOW_OOS else config.IS_END)
    universe = pairs or config.PAIRS
    weight = 1.0 / len(universe)
    results = {}
    for name, (us_t, hk_t, _) in universe.items():
        sp = spreads[spreads["pair"] == name]
        results[name] = run_pair(sp, frames[us_t], frames[hk_t], frames[config.FX_TICKER], p,
                               signal=signal, start=start, end=end, us_exec=us_exec,
                               cost=cost_model(name, cost_multiplier, (half_spreads or {}).get(name), borrow_annual)
                               if cost_multiplier else None)
    rets = pd.DataFrame({n: r["daily"]["pair_ret"] for n, r in results.items()}).fillna(0.0)
    rets["portfolio_ret"] = (rets[list(results)] * weight).sum(axis=1)
    rets.index.name = "date"
    return {"pairs": results, "portfolio": rets}


# ---------------------------------------------------------------------------
# Run log (every parameter combination tried; CLAUDE.md rule 3)
# ---------------------------------------------------------------------------
GRID_LOG_COLUMNS = [
    "timestamp_utc", "run_type", "period", "signal", "us_exec", "L", "k", "exit_z", "H", "stop_z",
    "cost_multiplier", "n_trades", "avg_days_held", "pct_days_in_market",
    "cum_return", "ann_return", "ann_vol", "sharpe", "sharpe_gross", "max_drawdown",
    "hit_rate", "turnover", "cost_drag", "elapsed_sec", "notes",
]


def log_run(row: dict, path: Path = config.GRID_LOG) -> None:
    """Append one run to results/grid_log.csv. Unknown keys raise.

    If the file was written with an older column set, it is first rewritten with
    the current columns (old rows kept; new columns left blank) - rows are never dropped.
    """
    unknown = set(row) - set(GRID_LOG_COLUMNS)
    if unknown:
        raise KeyError(f"unknown grid-log columns: {sorted(unknown)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open() as f:
            header = next(csv.reader(f), [])
        if header != GRID_LOG_COLUMNS:
            old = pd.read_csv(path)
            old.reindex(columns=GRID_LOG_COLUMNS).to_csv(path, index=False)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GRID_LOG_COLUMNS)
        if new:
            w.writeheader()
        w.writerow({"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **row})


def grid_params(grid: dict = config.GRID, stop_z: float = config.STOP_Z) -> list[Params]:
    """Every combination in the grid (3 x 3 x 2 x 2 = 36), in a fixed order."""
    from itertools import product
    return [Params(L=L, k=k, exit_z=e, H=H, stop_z=stop_z)
            for L, k, e, H in product(grid["L"], grid["k"], grid["exit_z"], grid["H"])]


def run_grid(
    spreads: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    signals: tuple[str, ...] = ("spread", "spread_2r"),
    cost_multiplier: float = 1.0,
    period: str = "IS",
    log_path: Path = config.GRID_LOG,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Run every grid cell for every signal, log each run, and time the whole loop.

    Output: (portfolio table: one row per run, per-pair table, total seconds).
    Every run is appended to log_path BEFORE the next one starts (CLAUDE.md rule 3).
    """
    import time
    from src import metrics

    rows, pair_rows = [], []
    t_total = time.perf_counter()
    for signal in signals:
        for p in grid_params():
            t0 = time.perf_counter()
            res = run_backtest(spreads, frames, p, signal=signal, cost_multiplier=cost_multiplier)
            summary, pairs = metrics.summarize_backtest(res)
            elapsed = time.perf_counter() - t0
            row = {"run_type": "grid", "period": period, "signal": signal, "us_exec": "close",
                   **p.as_dict(), "cost_multiplier": cost_multiplier, **summary, "elapsed_sec": round(elapsed, 4)}
            log_run(row, log_path)
            rows.append(row)
            pair_rows.append(pairs.assign(signal=signal, **p.as_dict()).rename_axis("pair").reset_index())
    total = time.perf_counter() - t_total
    return pd.DataFrame(rows), pd.concat(pair_rows, ignore_index=True), total


def break_even_multiplier(spreads: pd.DataFrame, frames: dict[str, pd.DataFrame], p: Params,
                          lo: float = 0.0, hi: float = 8.0, iters: int = 14, **kwargs) -> float:
    """Cost multiplier at which the portfolio's net Sharpe crosses zero (bisection).

    Sharpe falls as costs rise, so bisection on [lo, hi] works. Returns 0.0 if the strategy
    already loses before costs, inf if it still wins at `hi`. kwargs go to run_backtest
    (signal, start, end, pairs, ...). Precision after 14 halvings of [0, 8]: ~0.0005.
    """
    from src.metrics import sharpe

    def sr(m: float) -> float:
        return sharpe(run_backtest(spreads, frames, p, cost_multiplier=m, **kwargs)["portfolio"]["portfolio_ret"])

    if sr(lo) <= 0:
        return 0.0
    if sr(hi) > 0:
        return float("inf")
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sr(mid) > 0 else (lo, mid)
    return (lo + hi) / 2
