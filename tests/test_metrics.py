"""Tests for src/metrics.py with hand-checkable numbers."""

import numpy as np
import pandas as pd

from src import metrics


def series(values, start="2022-01-03") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_cumulative_return_compounds():
    assert np.isclose(metrics.cumulative_return(series([0.10, -0.10])), -0.01)


def test_sharpe_by_hand_and_flat_series():
    r = series([0.02, 0.0] * 50)
    assert np.isclose(metrics.sharpe(r), r.mean() / r.std() * np.sqrt(252))
    assert np.isnan(metrics.sharpe(series([0.0] * 10)))


def test_max_drawdown_by_hand():
    # equity 1.1 -> 0.55 -> 0.66: worst fall is from 1.1 to 0.55 = -50%
    assert np.isclose(metrics.max_drawdown(series([0.10, -0.50, 0.20])), -0.5)
    assert metrics.max_drawdown(series([0.01, 0.02])) == 0.0
    assert np.isclose(metrics.max_drawdown(series([-0.10, 0.05])), -0.10)  # losing from the start counts


def _daily(equity, us_trade, hk_trade):
    idx = pd.bdate_range("2022-01-03", periods=len(equity))
    eq = pd.Series(equity, index=idx)
    return pd.DataFrame({"equity": eq, "pair_ret": eq.pct_change().fillna(eq.iloc[0] - 1),
                         "us_trade": us_trade, "hk_trade": hk_trade,
                         "us_cost": 0.001, "hk_cost": 0.0, "borrow_cost": 0.0}, index=idx)


def test_trade_return_runs_from_close_before_entry_to_hk_exit_fill():
    # entry decided day 1, exit decided day 3, HK exit fills day 4
    daily = _daily([1.00, 1.00, 1.02, 1.05, 1.04, 1.04], [0, .5, 0, .5, 0, 0], [0, 0, .5, 0, .5, 0])
    trades = pd.DataFrame({"entry_date": [daily.index[1]], "exit_date": [daily.index[3]], "days_held": [2]})
    tr = metrics.trade_returns(daily, trades)
    assert np.isclose(tr.iloc[0], 1.04 / 1.00 - 1)


def test_turnover_and_cost_drag_per_year():
    idx = pd.date_range("2022-01-01", "2023-01-01", freq="D")   # exactly 365 days ~ 0.9993 years
    daily = pd.DataFrame({"pair_ret": 0.0, "us_trade": 0.0, "hk_trade": 0.0,
                          "us_cost": 0.0, "hk_cost": 0.0, "borrow_cost": 0.0}, index=idx)
    daily.iloc[10, daily.columns.get_loc("us_trade")] = 0.5
    daily.iloc[11, daily.columns.get_loc("hk_trade")] = 0.5
    daily.iloc[10, daily.columns.get_loc("us_cost")] = 0.002
    yrs = 365 / 365.25
    assert np.isclose(metrics.turnover(daily), 1.0 / yrs)
    assert np.isclose(metrics.cost_drag(daily), 0.002 / yrs)


def test_plateau_score_prefers_plateau_over_isolated_peak():
    rows = []
    for L in (20, 60, 120):
        for k in (1.5, 2.0, 2.5):
            for e in (0.0, 0.5):
                for H in (5, 10):
                    v = 0.0
                    if (L, k) in {(60, 2.0), (60, 2.5), (120, 2.0), (120, 2.5)}:
                        v = 0.6                      # broad plateau
                    if (L, k, e, H) == (20, 1.5, 0.0, 5):
                        v = 1.0                      # isolated spike
                    rows.append({"L": L, "k": k, "exit_z": e, "H": H, "sharpe": v})
    scored = metrics.plateau_scores(pd.DataFrame(rows))
    best = scored.sort_values("nbhd_mean", ascending=False).iloc[0]
    assert (best["L"], best["k"]) in {(60, 2.0), (60, 2.5), (120, 2.0), (120, 2.5)}
    spike = scored[(scored.L == 20) & (scored.k == 1.5) & (scored.exit_z == 0.0) & (scored.H == 5)].iloc[0]
    assert spike["sharpe"] == scored["sharpe"].max() and spike["nbhd_mean"] < best["nbhd_mean"]
    assert scored["n_nbrs"].between(3, 6).all()


def test_concentration_by_hand():
    t = pd.DataFrame({"contribution": [0.03, 0.01, 0.005, -0.005, -0.01], "trade_return": [0.18, 0.06, 0.03, -0.03, -0.06]})
    c = metrics.concentration(t, top=2)
    assert np.isclose(c["total_contribution"], 0.03) and np.isclose(c["top2_share"], 0.04 / 0.03)
    assert np.isclose(c["total_without_top2"], -0.01) and np.isclose(c["median_trade_return"], 0.03)
