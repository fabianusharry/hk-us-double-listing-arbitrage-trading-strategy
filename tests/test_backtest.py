"""Tests for src/backtest.py: hand-computed timing, calendar cases, look-ahead,
synthetic profit and the mandatory zero-mispricing placebo."""

import numpy as np
import pandas as pd
import pytest

import config
from src import backtest
from src.backtest import backtest_pair, leg_returns, run_backtest, run_pair
from src.data import COLUMNS
from src.strategy import Params
from tests.synthetic import simulate_pair


def raw(dates, open_, close, dividends=0.0) -> pd.DataFrame:
    df = pd.DataFrame({"open": open_, "close": close, "dividends": dividends},
                      index=pd.DatetimeIndex(dates, name="date"))
    df["high"], df["low"] = df[["open", "close"]].max(axis=1), df[["open", "close"]].min(axis=1)
    df["adj_close"], df["volume"], df["splits"] = df["close"], 1e6, 0.0
    return df[COLUMNS]


def decisions(dates, values) -> pd.Series:
    return pd.Series(values, index=pd.DatetimeIndex(dates), dtype=float)


def sharpe(r: pd.Series) -> float:
    return r.mean() / r.std() * np.sqrt(252)


# ---------------------------------------------------------------------------
# 1. Hand-computed 5-day example (walked through in the Session 4 notes)
# ---------------------------------------------------------------------------
DAYS = ["2022-04-22", "2022-04-25", "2022-04-26", "2022-04-27", "2022-04-28", "2022-04-29"]
#        day 0 (prev)  day 1         day 2         day 3         day 4         day 5


def test_hand_computed_five_day_example():
    # US: closes 100,100,100,110,99,99. The US opens are deliberately odd (105 on day 3):
    # with execution at the close they must not matter.
    us = leg_returns(raw(DAYS, [100, 100, 100, 105, 112, 99], [100, 100, 100, 110, 99, 99]))
    # HK (FX = 1): day 3 gaps to 102 then closes 108; day 4 closes 99; day 5 opens 98 and
    # rallies to 120 AFTER our exit at the open (must not count).
    fx = pd.Series(1.0, index=pd.DatetimeIndex(DAYS))
    hk = leg_returns(raw(DAYS, [100, 100, 100, 102, 107, 98], [100, 100, 100, 108, 99, 120]), fx=fx)
    # Spread position decided at the US open: enter +1 (long US / short HK) on day 2, exit on day 4.
    d = decisions(DAYS[1:], [0, 1, 1, 0, 0])
    out = backtest_pair(d, us.loc[DAYS[1]:], hk.loc[DAYS[1]:])

    us_exp = [0, 0, 110 / 100 - 1, 99 / 110 - 1, 0]                  # long from close day 2 to close day 4
    hk_exp = [0, 0, -(108 / 102 - 1), -(99 / 108 - 1), -(98 / 99 - 1)]  # short from open day 3 to open day 5
    np.testing.assert_allclose(out["us_ret"], us_exp, atol=1e-12)
    np.testing.assert_allclose(out["hk_ret"], hk_exp, atol=1e-12)
    np.testing.assert_allclose(out["pair_ret"], 0.5 * np.array(us_exp) + 0.5 * np.array(hk_exp), atol=1e-12)
    assert out["us_trade"].tolist() == [0, 1, 0, 1, 0]   # US trades at the close of days 2 and 4
    assert out["hk_trade"].tolist() == [0, 0, 1, 0, 1]   # HK trades at the open of days 3 and 5


def test_upper_bound_variant_trades_us_at_the_open():
    us = leg_returns(raw(DAYS, [100, 100, 100, 105, 112, 99], [100, 100, 100, 110, 99, 99]))
    hk = leg_returns(raw(DAYS, 100.0, 100.0))
    out = backtest_pair(decisions(DAYS[1:], [0, 1, 1, 0, 0]), us.loc[DAYS[1]:], hk.loc[DAYS[1]:], us_exec="open")
    # day 2: enters at the open (100 -> 100: 0); day 3 close-to-close +10%; day 4 exits at the
    # open: old position earns the overnight 110 -> 112, new (flat) earns nothing intraday.
    np.testing.assert_allclose(out["us_ret"], [0, 0, 0.10, 112 / 110 - 1, 0], atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Calendar cases
# ---------------------------------------------------------------------------
def test_hk_trades_on_next_hk_session_even_if_us_closed():
    # Thanksgiving 2022-11-24: US closed, HK open. Decision Wed 23rd -> HK trades Thu 24th open.
    us_days = ["2022-11-21", "2022-11-22", "2022-11-23", "2022-11-25"]
    hk_days = ["2022-11-21", "2022-11-22", "2022-11-23", "2022-11-24", "2022-11-25"]
    us, hk = leg_returns(raw(us_days, 100.0, 100.0)), leg_returns(raw(hk_days, 100.0, 100.0))
    out = backtest_pair(decisions(us_days[1:3], [0, 1]), us.loc["2022-11-22":], hk.loc["2022-11-22":])
    assert out.loc["2022-11-24", "hk_pos_intraday"] == -1 and out.loc["2022-11-24", "hk_trade"] == 1
    assert out.loc["2022-11-23", "us_pos_end"] == 1


@pytest.mark.parametrize("position,sign", [(1, +1), (-1, -1)])
def test_dividend_on_non_aligned_day_is_credited_or_debited(position, sign):
    # 2022-04-05: HK holiday (Ching Ming), US open and the ADR goes ex with a $1 dividend.
    us_days = ["2022-04-01", "2022-04-04", "2022-04-05", "2022-04-06"]
    us = leg_returns(raw(us_days, [100, 100, 99.5, 99.5], [100, 100, 99.5, 99.5], [0, 0, 1.0, 0]))
    hk = leg_returns(raw(["2022-04-01", "2022-04-04", "2022-04-06"], 100.0, 100.0))
    out = backtest_pair(decisions(["2022-04-01", "2022-04-04"], [position, position]),
                        us.loc["2022-04-01":], hk.loc["2022-04-01":])
    # price 100 -> 99.5 plus a $1 dividend: total return +0.5% to a holder, -0.5% to a short
    assert np.isclose(out.loc["2022-04-05", "us_ret"], sign * 0.005)
    assert "2022-04-05" not in hk.index.strftime("%Y-%m-%d")  # HK closed: not an aligned day


def test_dividend_debited_to_short_when_price_does_not_drop():
    us = leg_returns(raw(DAYS[:3], 100.0, 100.0, [0, 0, 2.0]))
    hk = leg_returns(raw(DAYS[:3], 100.0, 100.0))
    out = backtest_pair(decisions(DAYS[:2], [-1, -1]), us, hk)  # short US
    assert np.isclose(out.loc[DAYS[2], "us_ret"], -0.02)


def test_half_day_hk_execution_happens_at_the_close():
    # Day 3 is an HK half-day: Yahoo bar open = close = 105. Exit decided day 2 (short HK).
    hk = leg_returns(raw(DAYS[:4], [100, 100, 100, 105], [100, 100, 100, 105]))
    us = leg_returns(raw(DAYS[:4], 100.0, 100.0))
    out = backtest_pair(decisions(DAYS[1:3], [1, 0]), us.loc[DAYS[1]:], hk.loc[DAYS[1]:])
    # the old (short) position earns the whole half-day move: -5%
    assert np.isclose(out.loc[DAYS[3], "hk_ret"], -0.05)


def test_dropped_hk_bar_is_skipped_and_trade_moves_to_next_session():
    days = ["2022-03-10", "2022-03-11", "2022-03-14", "2022-03-15", "2022-03-16"]
    hk_raw = raw(days, [100, 100, 101, 80, 90], [100, 100, 101, 85, 90])  # 14th = fake bar
    hk = leg_returns(hk_raw, drop=pd.DatetimeIndex(["2022-03-14"]))
    us = leg_returns(raw(days, 100.0, 100.0))
    # decision on the 11th: short HK; would execute at the 14th open, moves to the 15th
    out = backtest_pair(decisions(days[1:2], [1]), us.loc[days[1]:], hk.loc[days[1]:])
    assert "2022-03-14" not in hk.index.strftime("%Y-%m-%d")
    assert out.loc["2022-03-15", "hk_trade"] == 1
    assert np.isclose(out.loc["2022-03-15", "hk_ret"], -(85 / 80 - 1))     # entered at the 15th open
    assert np.isclose(hk.loc["2022-03-15", "r_cc"], 85 / 100 - 1)           # return spans 11th -> 15th


# ---------------------------------------------------------------------------
# 3. Look-ahead
# ---------------------------------------------------------------------------
P = Params(L=40, k=1.5, exit_z=0.0, H=10)


def _run(sim, spreads=None):
    return run_pair(spreads if spreads is not None else sim["spreads"], sim["us"], sim["hk"], sim["fx"], P,
                    start=str(sim["spreads"]["date"].iloc[0].date()), hk_drop=pd.DatetimeIndex([]))


def test_changing_the_future_does_not_change_the_past():
    sim = simulate_pair(600, sigma=0.005, phi=0.9, mis_sd=0.02, seed=1)
    base = _run(sim)
    d = sim["spreads"]["date"].iloc[400]

    future_spreads = sim["spreads"].copy()
    future_spreads.loc[future_spreads["date"] > d, "spread"] = np.random.default_rng(9).normal(0, 0.05, 199)
    alt = _run(sim, future_spreads)
    pd.testing.assert_frame_equal(base["decisions"].loc[:d], alt["decisions"].loc[:d])
    pd.testing.assert_frame_equal(base["daily"].loc[:d], alt["daily"].loc[:d])
    assert not base["decisions"]["position"].equals(alt["decisions"]["position"])  # the change did bite

    future_prices = {k: (v.copy() if k != "spreads" else v) for k, v in sim.items()}
    for leg in ("us", "hk"):
        future_prices[leg].loc[future_prices[leg].index > d, ["open", "close"]] *= 1.3
    alt2 = _run(future_prices)
    pd.testing.assert_frame_equal(base["daily"].loc[:d], alt2["daily"].loc[:d])


# ---------------------------------------------------------------------------
# 4. Synthetic profit: a persistent (OU / AR(1)) mispricing is profitable at zero cost
# ---------------------------------------------------------------------------
def test_persistent_mispricing_is_profitable_at_zero_cost():
    sim = simulate_pair(3000, sigma=0.003, phi=0.93, mis_sd=0.02, seed=3)  # half-life ~10 days
    out = _run(sim)
    ret = out["daily"]["pair_ret"]
    assert len(out["trades"]) > 50
    assert ret.sum() > 0 and sharpe(ret) > 1.0


# ---------------------------------------------------------------------------
# 5. Mandatory placebo: zero mispricing, non-synchronous observation -> no profit
# ---------------------------------------------------------------------------
def test_placebo_zero_mispricing_earns_nothing_gross():
    sim = simulate_pair(10_000, sigma=0.01, seed=4)          # ~40 years, no mispricing at all
    out = _run(sim)
    ret = out["daily"]["pair_ret"]
    assert len(out["trades"]) > 500                            # the strategy DOES trade on timing noise
    assert abs(sharpe(ret)) < 0.6                              # ... but earns nothing (noise s.e. ~0.16)

    # Positive control: the WRONG way (P&L from spread changes) shows a large fake profit on the
    # same data. If it did not, this placebo could not detect the bug it exists for.
    s = sim["spreads"].set_index("date")["spread"]
    pos = out["decisions"]["position"]
    naive = (pos.shift(1) * s.diff()).loc[pos.index].fillna(0.0)
    assert sharpe(naive) > 3.0


# ---------------------------------------------------------------------------
# 6. Portfolio aggregation and the run log
# ---------------------------------------------------------------------------
def test_portfolio_is_sum_of_pair_weight_times_pair_returns(monkeypatch):
    a, b = simulate_pair(300, phi=0.9, mis_sd=0.02, seed=5), simulate_pair(300, phi=0.9, mis_sd=0.02, seed=6)
    monkeypatch.setattr(config, "PAIRS", {"A": ("UA", "1.HK", 1), "B": ("UB", "2.HK", 1)})
    spreads = pd.concat([a["spreads"].assign(pair="A"), b["spreads"].assign(pair="B")])
    frames = {"UA": a["us"], "1.HK": a["hk"], "UB": b["us"], "2.HK": b["hk"], config.FX_TICKER: a["fx"]}
    start = str(a["spreads"]["date"].iloc[0].date())
    res = run_backtest(spreads, frames, Params(L=40, k=1.5, exit_z=0.0, H=10), start=start, end="1951-12-31")
    port = res["portfolio"]
    np.testing.assert_allclose(port["portfolio_ret"], config.PAIR_WEIGHT * (port["A"] + port["B"]))


def test_run_pair_refuses_oos_dates_when_locked(monkeypatch):
    monkeypatch.setattr(config, "ALLOW_OOS", False)
    sim = simulate_pair(100, start="2023-10-02")
    with pytest.raises(backtest.OOSAccessError):
        _run(sim)


def test_log_run_appends_rows_and_rejects_unknown_columns(tmp_path):
    path = tmp_path / "grid_log.csv"
    backtest.log_run({"run_type": "test", "L": 20}, path)
    backtest.log_run({"run_type": "test", "L": 60}, path)
    assert pd.read_csv(path)["L"].tolist() == [20, 60]
    with pytest.raises(KeyError):
        backtest.log_run({"bogus": 1}, path)
