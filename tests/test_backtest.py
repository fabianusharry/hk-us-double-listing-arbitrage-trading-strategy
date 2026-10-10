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
    out = backtest_pair(decisions(DAYS[1:], [0, 1, 1, 0, 0]), us.loc[DAYS[1]:], hk.loc[DAYS[1]:])

    # On paper, pair equity starts at 1. Entry decided day 2: N = 0.5 x equity(end day 1) = 0.5.
    us_sh = 0.5 / 100          # bought at the day-2 close
    hk_sh = -0.5 / 102         # sold short at the day-3 open
    e3 = 1 + us_sh * (110 - 100) + hk_sh * (108 - 102)       # day 3: US close-to-close, HK intraday only
    e4 = e3 + us_sh * (99 - 110) + hk_sh * (99 - 108)        # day 4: US sold at the close, HK held all day
    e5 = e4 + hk_sh * (98 - 99)                              # day 5: HK bought back at the open
    np.testing.assert_allclose(out["us_pnl"], [0, 0, us_sh * 10, us_sh * -11 / e3, 0], atol=1e-12)
    np.testing.assert_allclose(out["hk_pnl"], [0, 0, hk_sh * 6, hk_sh * -9 / e3, hk_sh * -1 / e4], atol=1e-12)
    np.testing.assert_allclose(out["equity"], [1, 1, e3, e4, e5], atol=1e-12)
    np.testing.assert_allclose(out["us_trade"], [0, 0.5, 0, us_sh * 99 / e3, 0], atol=1e-12)     # US trades at closes of days 2, 4
    np.testing.assert_allclose(out["hk_trade"], [0, 0, 0.5, 0, -hk_sh * 98 / e4], atol=1e-12)   # HK at opens of days 3, 5


def test_engine_matches_independent_dollar_ledger():
    """Engine vs a from-scratch shares-and-dollars ledger with the same (hold-shares) sizing rule."""
    from tests.timing_example import dollar_ledger, engine_result

    eng = engine_result()
    ledger = dollar_ledger(sizing="hold")
    np.testing.assert_allclose(eng["pair_ret"].to_numpy(), ledger["day_return"].to_numpy(), atol=1e-12)
    np.testing.assert_allclose(eng["equity"].to_numpy(), ledger["equity"].to_numpy() / 1e6, atol=1e-12)


def test_walkthrough_numbers_quoted_in_notes():
    """Pins every number quoted in the Session 4 walkthrough, so the text cannot drift from the code."""
    from tests.timing_example import dollar_ledger, engine_result

    eng = engine_result()
    np.testing.assert_allclose(eng["us_pnl"] * 100, [0, 0, 5.0000, -5.3890, 0], atol=5e-5)
    np.testing.assert_allclose(eng["hk_pnl"] * 100, [0, 0, -2.9412, 4.3228, 0.4855], atol=5e-5)
    np.testing.assert_allclose(eng["pair_ret"] * 100, [0, 0, 2.0588, -1.0663, 0.4855], atol=5e-5)
    assert np.isclose((eng["equity"].iloc[-1] - 1) * 100, 1.460784, atol=5e-7)            # trade P&L
    old = dollar_ledger(sizing="rebalanced")["equity"].iloc[-1] / 1e6 - 1
    assert np.isclose(old * 100, 1.719487, atol=5e-7)                                      # old assumption
    assert np.isclose(eng["pair_ret"].sum() * 100, 1.478025, atol=5e-7)                   # sum: not a P&L


def test_upper_bound_variant_trades_us_at_the_open():
    us = leg_returns(raw(DAYS, [100, 100, 100, 105, 112, 99], [100, 100, 100, 110, 99, 99]))
    hk = leg_returns(raw(DAYS, 100.0, 100.0))
    out = backtest_pair(decisions(DAYS[1:], [0, 1, 1, 0, 0]), us.loc[DAYS[1]:], hk.loc[DAYS[1]:], us_exec="open")
    # bought at the day-2 OPEN (100); day 3: 100 -> 110; day 4 sold at the open 112 (the
    # overnight 110 -> 112 counts, the slide to 99 does not).
    sh = 0.5 / 100
    np.testing.assert_allclose(out["us_pnl"], [0, 0, sh * 10, sh * 2 / (1 + sh * 10), 0], atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Calendar cases (pair equity starts at 1, so a leg holds 0.5 / price shares)
# ---------------------------------------------------------------------------
def test_hk_trades_on_next_hk_session_even_if_us_closed():
    # Thanksgiving 2022-11-24: US closed, HK open. Decision Wed 23rd -> HK trades Thu 24th open.
    us_days = ["2022-11-21", "2022-11-22", "2022-11-23", "2022-11-25"]
    hk_days = ["2022-11-21", "2022-11-22", "2022-11-23", "2022-11-24", "2022-11-25"]
    us, hk = leg_returns(raw(us_days, 100.0, 100.0)), leg_returns(raw(hk_days, 100.0, 100.0))
    out = backtest_pair(decisions(us_days[1:3], [0, 1]), us.loc["2022-11-22":], hk.loc["2022-11-22":])
    assert out.loc["2022-11-24", "hk_pos"] == -1 and np.isclose(out.loc["2022-11-24", "hk_trade"], 0.5)
    assert out.loc["2022-11-23", "us_pos"] == 1


@pytest.mark.parametrize("position,sign", [(1, +1), (-1, -1)])
def test_dividend_on_non_aligned_day_is_credited_or_debited(position, sign):
    # 2022-04-05: HK holiday (Ching Ming), US open and the ADR goes ex with a $1 dividend.
    us_days = ["2022-04-01", "2022-04-04", "2022-04-05", "2022-04-06"]
    us = leg_returns(raw(us_days, [100, 100, 99.5, 99.5], [100, 100, 99.5, 99.5], [0, 0, 1.0, 0]))
    hk = leg_returns(raw(["2022-04-01", "2022-04-04", "2022-04-06"], 100.0, 100.0))
    out = backtest_pair(decisions(["2022-04-01", "2022-04-04"], [position, position]),
                        us.loc["2022-04-01":], hk.loc["2022-04-01":])
    # 0.005 shares: price -0.5 plus $1 dividend = +0.0025 to a holder, -0.0025 to a short
    assert np.isclose(out.loc["2022-04-05", "us_pnl"], sign * 0.0025)
    assert "2022-04-05" not in hk.index.strftime("%Y-%m-%d")  # HK closed: not an aligned day


def test_dividend_debited_to_short_when_price_does_not_drop():
    us = leg_returns(raw(DAYS[:3], 100.0, 100.0, [0, 0, 2.0]))
    hk = leg_returns(raw(DAYS[:3], 100.0, 100.0))
    out = backtest_pair(decisions(DAYS[:2], [-1, -1]), us, hk)  # short 0.005 US shares
    assert np.isclose(out.loc[DAYS[2], "us_pnl"], -0.01)


def test_half_day_hk_execution_happens_at_the_close():
    # Day 3 is an HK half-day: Yahoo bar open = close = 105. Exit decided day 2 (short HK).
    hk = leg_returns(raw(DAYS[:4], [100, 100, 100, 105], [100, 100, 100, 105]))
    us = leg_returns(raw(DAYS[:4], 100.0, 100.0))
    out = backtest_pair(decisions(DAYS[1:3], [1, 0]), us.loc[DAYS[1]:], hk.loc[DAYS[1]:])
    # the old short (0.005 shares from the day-2 open) earns the whole half-day move: -0.005 x 5
    assert np.isclose(out.loc[DAYS[3], "hk_pnl"], -0.025)


def test_dropped_hk_bar_is_skipped_and_trade_moves_to_next_session():
    days = ["2022-03-10", "2022-03-11", "2022-03-14", "2022-03-15", "2022-03-16"]
    hk_raw = raw(days, [100, 100, 101, 80, 90], [100, 100, 101, 85, 90])  # 14th = fake bar
    hk = leg_returns(hk_raw, drop=pd.DatetimeIndex(["2022-03-14"]))
    us = leg_returns(raw(days, 100.0, 100.0))
    # decision on the 11th: short HK; would execute at the 14th open, moves to the 15th
    out = backtest_pair(decisions(days[1:2], [1]), us.loc[days[1]:], hk.loc[days[1]:])
    assert "2022-03-14" not in hk.index.strftime("%Y-%m-%d")
    assert np.isclose(out.loc["2022-03-15", "hk_trade"], 0.5)
    assert np.isclose(out.loc["2022-03-15", "hk_pnl"], -(0.5 / 80) * (85 - 80))   # entered at the 15th open
    assert np.isclose(hk.loc["2022-03-15", "r_cc"], 85 / 100 - 1)                  # bar spans 11th -> 15th


def test_both_legs_enter_with_equal_dollars_and_hold_share_counts():
    sim = simulate_pair(400, sigma=0.01, phi=0.9, mis_sd=0.02, seed=7)
    daily = _run(sim)["daily"]
    e_prev = daily["equity"].shift(1).fillna(1.0)
    us_in = daily.index[daily["us_trade"] > 0][::2]
    hk_in = daily.index[daily["hk_trade"] > 0][::2]
    assert len(us_in) > 5
    np.testing.assert_allclose((daily["us_trade"] * e_prev).loc[us_in].to_numpy(),
                               (daily["hk_trade"] * e_prev).loc[hk_in].to_numpy())
    # between entry and exit a leg never trades again (no rebalancing)
    held = daily["us_pos"].ne(0) & daily["us_pos"].shift(1).ne(0)
    assert (daily.loc[held, "us_trade"] == 0).all()


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
    res = run_backtest(spreads, frames, Params(L=40, k=1.5, exit_z=0.0, H=10), start=start, end="1951-12-31",
                       cost_multiplier=0.0)
    port = res["portfolio"]
    np.testing.assert_allclose(port["portfolio_ret"], 0.5 * (port["A"] + port["B"]))   # 1 / len(pairs)
    assert config.PAIR_WEIGHT == 1 / 6   # the main universe keeps its 1/6 weights


def test_run_backtest_uses_the_universe_it_is_given():
    """Passing pairs= must run exactly that universe (not config.PAIRS) with its own cost overrides."""
    a, b = simulate_pair(300, phi=0.9, mis_sd=0.02, seed=5), simulate_pair(300, phi=0.9, mis_sd=0.02, seed=6)
    universe = {"A": ("UA", "1.HK", 1), "B": ("UB", "2.HK", 1)}
    spreads = pd.concat([a["spreads"].assign(pair="A"), b["spreads"].assign(pair="B")])
    frames = {"UA": a["us"], "1.HK": a["hk"], "UB": b["us"], "2.HK": b["hk"], config.FX_TICKER: a["fx"]}
    start = str(a["spreads"]["date"].iloc[0].date())
    p = Params(L=40, k=1.5, exit_z=0.0, H=10)
    res = run_backtest(spreads, frames, p, start=start, end="1951-12-31", pairs=universe,
                       half_spreads={"A": 0.001, "B": 0.002}, cost_multiplier=1.0)
    assert set(res["pairs"]) == {"A", "B"}
    cheap = run_backtest(spreads, frames, p, start=start, end="1951-12-31", pairs=universe,
                         half_spreads={"A": 0.0, "B": 0.0}, cost_multiplier=1.0)
    assert cheap["portfolio"]["portfolio_ret"].sum() > res["portfolio"]["portfolio_ret"].sum()


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


def test_break_even_multiplier_brackets_zero_sharpe():
    a = simulate_pair(600, sigma=0.003, phi=0.93, mis_sd=0.02, seed=3)
    universe = {"A": ("UA", "1.HK", 1)}
    spreads = a["spreads"].assign(pair="A")
    frames = {"UA": a["us"], "1.HK": a["hk"], config.FX_TICKER: a["fx"]}
    kw = dict(start=str(a["spreads"]["date"].iloc[0].date()), end="1952-12-31", pairs=universe,
              half_spreads={"A": 0.001})
    p = Params(L=40, k=1.5, exit_z=0.0, H=10)
    be = backtest.break_even_multiplier(spreads, frames, p, **kw)
    from src.metrics import sharpe
    s = lambda m: sharpe(run_backtest(spreads, frames, p, cost_multiplier=m, **kw)["portfolio"]["portfolio_ret"])
    assert 0 < be < 8 and s(be * 0.9) > 0 > s(be * 1.1)
