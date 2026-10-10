"""Performance metrics: cumulative return, Sharpe, max drawdown, trade statistics,
turnover, cost drag; plus the plateau score used to choose grid parameters.

All return inputs are daily NET returns (after costs) on the backtest's calendar
(union of both markets' sessions). Annualisation uses config.TRADING_DAYS = 252
per CLAUDE.md §6; years for per-year rates are measured in calendar time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Return metrics
# ---------------------------------------------------------------------------
def cumulative_return(r: pd.Series) -> float:
    """Total compounded return: prod(1 + r) − 1."""
    return float((1 + r).prod() - 1)


def years(r: pd.Series) -> float:
    """Calendar length of the series in years (first to last date)."""
    return max((r.index.max() - r.index.min()).days, 1) / 365.25


def annualised_return(r: pd.Series) -> float:
    """Geometric annual return over the series' calendar length."""
    return float((1 + cumulative_return(r)) ** (1 / years(r)) - 1)


def annualised_vol(r: pd.Series) -> float:
    return float(r.std() * np.sqrt(config.TRADING_DAYS))


def sharpe(r: pd.Series) -> float:
    """mean / std of daily returns × √252, risk-free = 0. NaN if the series never moves."""
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(config.TRADING_DAYS)) if sd > 0 else float("nan")


def max_drawdown(r: pd.Series) -> float:
    """Largest peak-to-trough fall of the equity curve (a negative number, 0 if none)."""
    equity = (1 + r).cumprod()
    peak = np.maximum(equity.cummax(), 1.0)  # the starting capital counts as a peak
    return float((equity / peak - 1).min())


# ---------------------------------------------------------------------------
# Trade metrics
# ---------------------------------------------------------------------------
def trade_returns(daily: pd.DataFrame, trades: pd.DataFrame) -> pd.Series:
    """Net return of each round trip, from the pair's equity curve.

    A trade runs from the close before its entry decision date to the date its
    HK exit fills (the first HK trade after the exit decision). On a day where one
    trade's HK exit and the next trade's US entry coincide, that day's costs go to
    the earlier trade (a small attribution approximation).
    """
    eq = daily["equity"]
    hk_fills = daily.index[daily["hk_trade"] > 0]
    out = []
    for t in trades.itertuples(index=False):
        before = eq.loc[:t.entry_date].iloc[:-1]
        start_eq = before.iloc[-1] if len(before) else 1.0
        fills = hk_fills[hk_fills > t.exit_date]
        end_date = fills[0] if len(fills) else eq.index[-1]
        out.append(eq.loc[end_date] / start_eq - 1)
    return pd.Series(out, index=trades["entry_date"] if len(trades) else None, dtype=float)


def turnover(daily: pd.DataFrame) -> float:
    """Traded notional per year as a multiple of equity (both legs, entries and exits)."""
    return float((daily["us_trade"] + daily["hk_trade"]).sum() / years(daily["pair_ret"]))


def cost_drag(daily: pd.DataFrame) -> float:
    """Costs per year as a fraction of equity (transaction + borrow)."""
    total = daily[["us_cost", "hk_cost", "borrow_cost"]].sum().sum()
    return float(total / years(daily["pair_ret"]))


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def summarize_returns(r: pd.Series, gross: pd.Series | None = None) -> dict:
    out = {"cum_return": cumulative_return(r), "ann_return": annualised_return(r),
           "ann_vol": annualised_vol(r), "sharpe": sharpe(r), "max_drawdown": max_drawdown(r)}
    if gross is not None:
        out["sharpe_gross"] = sharpe(gross)
    return out


def summarize_pair(res: dict) -> dict:
    """Metrics for one pair from a backtest.run_pair() result."""
    daily, trades, dec = res["daily"], res["trades"], res["decisions"]
    tr = trade_returns(daily, trades)
    return {**summarize_returns(daily["pair_ret"], daily["gross_ret"]),
            "n_trades": len(trades), "hit_rate": float((tr > 0).mean()) if len(tr) else float("nan"),
            "avg_days_held": float(trades["days_held"].mean()) if len(trades) else float("nan"),
            "pct_days_in_market": float((dec["position"] != 0).mean()),
            "turnover": turnover(daily), "cost_drag": cost_drag(daily)}


def summarize_backtest(result: dict) -> tuple[dict, pd.DataFrame]:
    """(portfolio metrics, per-pair table) from a backtest.run_backtest() result.

    Portfolio turnover and cost drag are the equally weighted pair figures; hit rate
    and holding days are pooled over all trades of all pairs.
    """
    pairs = pd.DataFrame({n: summarize_pair(r) for n, r in result["pairs"].items()}).T
    w = 1.0 / len(result["pairs"])  # equal weight, as in run_backtest
    port = result["portfolio"]
    gross = sum(w * r["daily"]["gross_ret"].reindex(port.index).fillna(0.0)
                for r in result["pairs"].values())
    all_tr = pd.concat([trade_returns(r["daily"], r["trades"]) for r in result["pairs"].values()])
    all_trades = pd.concat([r["trades"] for r in result["pairs"].values()])
    summary = {**summarize_returns(port["portfolio_ret"], gross),
               "n_trades": len(all_trades),
               "hit_rate": float((all_tr > 0).mean()) if len(all_tr) else float("nan"),
               "avg_days_held": float(all_trades["days_held"].mean()) if len(all_trades) else float("nan"),
               "pct_days_in_market": float(pairs["pct_days_in_market"].mean()),
               "turnover": float((pairs["turnover"] * w).sum()),
               "cost_drag": float((pairs["cost_drag"] * w).sum())}
    return summary, pairs


# ---------------------------------------------------------------------------
# Plateau score for parameter choice
# ---------------------------------------------------------------------------
def plateau_scores(grid: pd.DataFrame, value: str = "sharpe", dims=("L", "k", "exit_z", "H")) -> pd.DataFrame:
    """Score each grid cell by how good its NEIGHBOURHOOD is, not just the cell itself.

    Neighbours of a cell = cells that differ by one step in exactly one parameter
    (e.g. L 60 -> 20 or 120, k 2.0 -> 1.5 or 2.5, the other exit_z, the other H).
    Output adds: nbhd_mean (mean of the cell and its neighbours), nbhd_min (worst of
    them), n_nbrs. A setting on a plateau has nbhd_mean close to its own value and a
    high nbhd_min; an isolated peak has a much lower nbhd_mean / nbhd_min.
    Input: one row per cell (a single signal) with the dims and `value` columns.
    """
    levels = {d: sorted(grid[d].unique()) for d in dims}
    pos = {d: {v: i for i, v in enumerate(levels[d])} for d in dims}
    key = {tuple(pos[d][row[d]] for d in dims): row[value] for _, row in grid.iterrows()}
    means, mins, counts = [], [], []
    for _, row in grid.iterrows():
        here = tuple(pos[d][row[d]] for d in dims)
        vals = [key[here]]
        for i in range(len(dims)):
            for step in (-1, 1):
                nb = list(here)
                nb[i] += step
                if tuple(nb) in key:
                    vals.append(key[tuple(nb)])
        means.append(np.nanmean(vals))
        mins.append(np.nanmin(vals))
        counts.append(len(vals) - 1)
    return grid.assign(nbhd_mean=means, nbhd_min=mins, n_nbrs=counts)


def trade_table(result: dict) -> pd.DataFrame:
    """Every round trip of a run_backtest() result with its net return and its contribution
    to the portfolio (= trade return x pair weight), largest contribution first."""
    w = 1.0 / len(result["pairs"])
    rows = []
    for name, r in result["pairs"].items():
        tr = trade_returns(r["daily"], r["trades"])
        for (_, t), ret in zip(r["trades"].iterrows(), tr.to_numpy()):
            rows.append({"pair": name, "entry": t.entry_date.date(), "direction": int(t.direction),
                         "days_held": int(t.days_held), "exit_reason": t.exit_reason,
                         "trade_return": ret, "contribution": ret * w})
    return pd.DataFrame(rows).sort_values("contribution", ascending=False).reset_index(drop=True)


def concentration(trades: pd.DataFrame, top: int = 3) -> dict:
    """How much of the total profit the `top` best trades provide, and what is left without them."""
    total = trades["contribution"].sum()
    best = trades["contribution"].head(top).sum()
    return {"n_trades": len(trades), "total_contribution": total, f"top{top}_share": best / total if total else float("nan"),
            f"total_without_top{top}": total - best, "median_trade_return": trades["trade_return"].median()}
