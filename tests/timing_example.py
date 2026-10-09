"""The 5-day timing example, computed two independent ways.

1. The engine (src/backtest.py), which works in returns.
2. A dollar ledger written here from scratch: it holds share counts, executes
   trades at the stated prices and marks the book to market at every open and
   close. It uses no code or formulas from src/.

tests/test_backtest.py asserts the two agree. Run this file to print the
walkthrough table, so every number quoted in the notes comes from code:

    .venv/bin/python -m tests.timing_example
"""

import pandas as pd

# Day 0 is history only (provides the previous close). FX = 1, ratio 1:1.
DAYS = ["2022-04-22", "2022-04-25", "2022-04-26", "2022-04-27", "2022-04-28", "2022-04-29"]
US_OPEN = [100, 100, 100, 105, 112, 99]
US_CLOSE = [100, 100, 100, 110, 99, 99]
HK_OPEN = [100, 100, 100, 102, 107, 98]
HK_CLOSE = [100, 100, 100, 108, 99, 120]
# Spread position decided at the US open of each day (day 0 has no decision).
DECISION = [None, 0, 1, 1, 0, 0]          # +1 = long US / short HK


def dollar_ledger(start_equity: float = 1_000_000.0, sizing: str = "rebalanced") -> pd.DataFrame:
    """Walk the trade in dollars and shares, day by day.

    US leg trades at the close of the decision day; HK leg at the open of the next day.
    Each leg's target notional is 50% of pair equity.
    sizing = 'rebalanced': every day each open leg is resized to 50% of the equity at the
             previous close (what the engine assumes: constant-notional, daily, at no cost).
    sizing = 'hold':       shares are bought once at entry (50% of equity then) and held
             unchanged until exit (what a trader actually does).
    Output: one row per day with dollar P&L per leg and end-of-day equity.
    """
    equity = start_equity
    us_shares = hk_shares = 0.0
    us_target = hk_target = 0                     # leg positions: US = +spread, HK = -spread
    rows = []
    for i in range(1, len(DAYS)):
        prev_equity = equity
        if sizing == "rebalanced":                # resize open legs to 50% of equity at last close
            us_shares = us_target * 0.5 * prev_equity / US_CLOSE[i - 1]
            hk_shares = hk_target * 0.5 * prev_equity / HK_CLOSE[i - 1]

        # --- HK session (Asia, happens first) ---
        hk_pnl = hk_shares * (HK_OPEN[i] - HK_CLOSE[i - 1])            # overnight, old position
        hk_new = -DECISION[i - 1] if DECISION[i - 1] is not None else 0  # yesterday's decision executes now
        if hk_new != hk_target:
            hk_shares = hk_new * 0.5 * (prev_equity if sizing == "rebalanced" else equity + hk_pnl) / HK_OPEN[i]
            hk_target = hk_new
        hk_pnl += hk_shares * (HK_CLOSE[i] - HK_OPEN[i])               # intraday, new position

        # --- US session ---
        us_pnl = us_shares * (US_CLOSE[i] - US_CLOSE[i - 1])           # held close-to-close
        equity = prev_equity + hk_pnl + us_pnl
        us_new = DECISION[i]                                           # today's decision executes at the close
        if us_new != us_target:
            us_shares = us_new * 0.5 * equity / US_CLOSE[i]
            us_target = us_new

        rows.append({"date": DAYS[i], "decision": DECISION[i], "us_pnl": us_pnl, "hk_pnl": hk_pnl,
                     "equity": equity, "day_return": equity / prev_equity - 1})
    return pd.DataFrame(rows).set_index("date")


def engine_result() -> pd.DataFrame:
    """The same example run through the engine."""
    from src.backtest import backtest_pair, leg_returns
    from tests.synthetic import _frame

    fx = pd.Series(1.0, index=pd.DatetimeIndex(DAYS))
    us = leg_returns(_frame(DAYS, US_OPEN, US_CLOSE))
    hk = leg_returns(_frame(DAYS, HK_OPEN, HK_CLOSE), fx=fx)
    decisions = pd.Series(DECISION[1:], index=pd.DatetimeIndex(DAYS[1:]), dtype=float)
    return backtest_pair(decisions, us.loc[DAYS[1]:], hk.loc[DAYS[1]:])


def walkthrough() -> pd.DataFrame:
    """Per-day table in percent: engine legs, engine pair, and both ledgers."""
    eng = engine_result()
    reb, hold = dollar_ledger(sizing="rebalanced"), dollar_ledger(sizing="hold")
    t = pd.DataFrame({
        "decision": reb["decision"].to_numpy(),
        "US leg %": eng["us_ret"].to_numpy() * 100,
        "HK leg %": eng["hk_ret"].to_numpy() * 100,
        "pair (engine) %": eng["pair_ret"].to_numpy() * 100,
        "pair (ledger, rebalanced) %": reb["day_return"].to_numpy() * 100,
        "pair (ledger, hold shares) %": hold["day_return"].to_numpy() * 100,
    }, index=reb.index)
    return t


if __name__ == "__main__":
    t = walkthrough()
    pd.set_option("display.width", 200)
    print(t.round(4).to_string())
    eng = engine_result()["pair_ret"]
    print(f"\nSum of daily returns (NOT a P&L):        {eng.sum() * 100:+.4f}%")
    print(f"Compounded, engine (= rebalanced ledger): {((1 + eng).prod() - 1) * 100:+.4f}%")
    print(f"Hold-shares ledger (what a trader gets):  {(dollar_ledger(sizing='hold')['equity'].iloc[-1] / 1e6 - 1) * 100:+.4f}%")
