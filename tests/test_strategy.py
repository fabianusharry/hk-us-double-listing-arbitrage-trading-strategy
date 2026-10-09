"""Tests for src/strategy.py: z-score timing and every state-machine rule."""

import numpy as np
import pandas as pd

from src.strategy import Params, positions, zscore

P = Params(L=3, k=2.0, exit_z=0.0, H=5, stop_z=4.0)


def zs(values) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2022-01-03", periods=len(values)), dtype=float)


def run(values, p=P, force_flat_from=None):
    dec, trades = positions(zs(values), p, force_flat_from)
    return dec["position"].tolist(), dec, trades


def test_zscore_uses_only_past_window():
    s = zs([1, 2, 3, 10, 5])
    z = zscore(s, 3)
    assert z.iloc[:3].isna().all()  # window not yet full
    window = np.array([1, 2, 3])
    assert np.isclose(z.iloc[3], (10 - window.mean()) / window.std(ddof=1))
    # changing a future value never changes today's z
    s2 = s.copy()
    s2.iloc[4] = 1e6
    assert zscore(s2, 3).iloc[3] == z.iloc[3]


def test_entry_thresholds_and_no_entry_beyond_stop():
    assert run([4.5])[0] == [0]      # beyond stop_z: news, not mispricing
    assert run([4.0])[0] == [-1]     # k < z <= stop_z
    assert run([2.0])[0] == [0]      # strictly above k needed
    assert run([2.1])[0] == [-1]     # ADR rich -> short spread
    assert run([-2.1])[0] == [1]     # ADR cheap -> long spread
    assert run([-4.5])[0] == [0]


def test_revert_exit_is_direction_aware():
    assert run([2.5, 1.0, 0.1, -0.1])[0] == [-1, -1, -1, 0]  # exit_z=0: exit once z < 0
    p = Params(L=3, k=2.0, exit_z=0.5, H=5)
    assert run([2.5, 1.0, 0.4], p)[0] == [-1, -1, 0]          # exit once z < 0.5
    assert run([-2.5, -1.0, -0.4], p)[0] == [1, 1, 0]         # long exits once z > -0.5


def test_stop_and_time_exits_with_reasons():
    pos, _, trades = run([2.5, 4.5])
    assert pos == [-1, 0] and trades.loc[0, "exit_reason"] == "stop"
    pos, dec, trades = run([2.5, 2.5, 2.5, 2.5, 2.5, 2.5], Params(L=3, k=2.0, exit_z=0.0, H=3))
    assert pos == [-1, -1, -1, 0, -1, -1]       # held 3 decision days, then re-entered the day after
    assert trades.loc[0, "exit_reason"] == "time" and trades.loc[0, "days_held"] == 3


def test_no_entry_on_exit_day():
    pos, dec, _ = run([2.5, -2.5, -2.5])
    assert pos == [-1, 0, 1]                    # day 2 exits (revert) but may not flip
    assert dec["event"].tolist() == ["enter:short", "exit:revert", "enter:long"]


def test_missing_z_holds_position_and_blocks_entry():
    assert run([np.nan, 2.5, np.nan, np.nan, -0.1])[0] == [0, -1, -1, -1, 0]


def test_force_flat_exits_and_blocks_entries():
    idx = pd.bdate_range("2022-01-03", periods=4)
    pos, _, trades = run([2.5, 2.5, 2.5, 2.5], force_flat_from=idx[2])
    assert pos == [-1, -1, 0, 0] and trades.loc[0, "exit_reason"] == "end"
