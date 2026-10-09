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


def dollar_ledger(start_equity: float = 1_000_000.0, sizing: str = "hold") -> pd.DataFrame:
    """Walk the trade in dollars and shares, day by day.

    Decision on day i (at the US open). The US leg trades at the close of day i; the
    HK leg at the open of day i+1. Both legs are sized from N = 50% of the pair equity
    at the end of day i-1.
    sizing = 'hold':       shares bought once at entry and held unchanged until exit
                           (what a trader does; what the engine now implements).
    sizing = 'rebalanced': every day each open leg is reset to 50% of the equity at the
                           previous close, at no cost (the engine's earlier assumption;
                           kept for comparison only).
    Output: one row per day with dollar P&L per leg and end-of-day equity.
    """
    equity = start_equity
    us_shares = hk_shares = 0.0
    us_target = hk_target = 0                     # leg signs: US = +spread, HK = -spread
    pending_hk = None                             # (target sign, notional) to execute at the next HK open
    rows = []
    for i in range(1, len(DAYS)):
        prev_equity = equity
        if sizing == "rebalanced":                # reset open legs to 50% of equity at the last close
            us_shares = us_target * 0.5 * prev_equity / US_CLOSE[i - 1]
            hk_shares = hk_target * 0.5 * prev_equity / HK_CLOSE[i - 1]

        # --- HK session (Asia, happens first) ---
        hk_pnl = hk_shares * (HK_OPEN[i] - HK_CLOSE[i - 1])            # overnight: old shares
        if pending_hk is not None:                                     # yesterday's decision executes now
            hk_target, notional = pending_hk
            hk_shares = hk_target * notional / HK_OPEN[i]
            pending_hk = None
        hk_pnl += hk_shares * (HK_CLOSE[i] - HK_OPEN[i])               # intraday: new shares

        # --- decision at the US open, then the US session ---
        us_pnl = us_shares * (US_CLOSE[i] - US_CLOSE[i - 1])           # held close-to-close
        if DECISION[i] != us_target:
            notional = 0.5 * prev_equity                               # sized from yesterday's close
            us_target = DECISION[i]
            us_shares = us_target * notional / US_CLOSE[i]             # executes at today's close
            pending_hk = (-DECISION[i], notional)
        equity = prev_equity + hk_pnl + us_pnl

        rows.append({"date": DAYS[i], "decision": DECISION[i], "us_shares": us_shares, "hk_shares": hk_shares,
                     "us_pnl": us_pnl, "hk_pnl": hk_pnl, "equity": equity,
                     "day_return": equity / prev_equity - 1})
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
    """Per-day table: engine P&L by leg (% of pair equity), engine vs ledger pair return."""
    eng = engine_result()
    hold, reb = dollar_ledger(sizing="hold"), dollar_ledger(sizing="rebalanced")
    return pd.DataFrame({
        "decision": hold["decision"].to_numpy(),
        "US P&L %": eng["us_pnl"].to_numpy() * 100,
        "HK P&L %": eng["hk_pnl"].to_numpy() * 100,
        "pair (engine) %": eng["pair_ret"].to_numpy() * 100,
        "pair (ledger, hold) %": hold["day_return"].to_numpy() * 100,
        "pair (old rebalanced) %": reb["day_return"].to_numpy() * 100,
        "equity (engine)": eng["equity"].to_numpy(),
    }, index=hold.index)


if __name__ == "__main__":
    t = walkthrough()
    pd.set_option("display.width", 200)
    print(t.round(4).to_string())
    eng = engine_result()
    print(f"\nTrade P&L, engine (equity end - 1):    {(eng['equity'].iloc[-1] - 1) * 100:+.6f}%")
    print(f"Trade P&L, hold-shares ledger:          {(dollar_ledger(sizing='hold')['equity'].iloc[-1] / 1e6 - 1) * 100:+.6f}%")
    print(f"Old engine assumption (daily rebalance): {(dollar_ledger(sizing='rebalanced')['equity'].iloc[-1] / 1e6 - 1) * 100:+.6f}%")
    print(f"Sum of engine daily returns (not a P&L): {eng['pair_ret'].sum() * 100:+.6f}%")
