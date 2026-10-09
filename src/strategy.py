"""Rolling z-score signal and the per-pair position state machine.

Position convention: the position is held IN THE SPREAD (spread = ln US − ln HK).
  +1 = long spread  = long US / short HK   (entered when the ADR is cheap, z < −k)
  −1 = short spread = short US / long HK   (entered when the ADR is rich,  z > +k)
   0 = flat
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

import config


@dataclass(frozen=True)
class Params:
    """One strategy setting (one cell of the grid)."""
    L: int            # rolling window, in aligned trading days
    k: float          # entry threshold
    exit_z: float     # reversion exit level (direction-aware, see positions())
    H: int            # maximum holding period, in decision days
    stop_z: float = config.STOP_Z

    def as_dict(self) -> dict:
        return asdict(self)


def zscore(s: pd.Series, L: int) -> pd.Series:
    """z(t) = (s(t) − mean of s over t−L..t−1) / std of s over t−L..t−1.

    Timing: .shift(1) after .rolling() means the mean and std on day t use data
    up to t−1 only; s(t) itself is known at the US open on t. The first L days
    are NaN (min_periods=L): no signal until the window is full.
    """
    mean = s.rolling(L, min_periods=L).mean().shift(1)
    std = s.rolling(L, min_periods=L).std().shift(1)
    return (s - mean) / std


def positions(z: pd.Series, p: Params, force_flat_from: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the position state machine over the decision days in z's index.

    Rules on each day t (decision known at the US open on t):
      1. In a position: days_held += 1, then exit to 0 if (first match wins)
           end    - t >= force_flat_from
           stop   - |z| > stop_z
           revert - position −1 and z < exit_z, or position +1 and z > −exit_z
           time   - days_held >= H
         No new entry on an exit day.
      2. Flat: enter −1 if k < z <= stop_z; enter +1 if −stop_z <= z < −k
         (never on or after force_flat_from).
      3. z is NaN: keep the position (time and end exits still apply); no entry.

    One pass over T days with O(1) work per day: O(T) time, O(T) memory. A loop is
    clearer than vectorising here because each day depends on the previous state.

    Output: (decisions, trades)
      decisions: index `date`; z, position (after today's decision), days_held, event
      trades: one row per round trip; entry_date, exit_date, direction, days_held,
              exit_reason, z_entry, z_exit
    """
    pos, held = 0, 0
    entry_date, z_entry = None, np.nan
    out_pos, out_held, out_event, trades = [], [], [], []

    for date, zt in z.items():
        event = ""
        has_z = not np.isnan(zt)
        flat_period = force_flat_from is not None and date >= force_flat_from
        if pos != 0:
            held += 1
            reason = None
            if flat_period:
                reason = "end"
            elif has_z and abs(zt) > p.stop_z:
                reason = "stop"
            elif has_z and ((pos == -1 and zt < p.exit_z) or (pos == 1 and zt > -p.exit_z)):
                reason = "revert"
            elif held >= p.H:
                reason = "time"
            if reason:
                trades.append({"entry_date": entry_date, "exit_date": date, "direction": pos,
                               "days_held": held, "exit_reason": reason,
                               "z_entry": z_entry, "z_exit": zt})
                pos, held, event = 0, 0, f"exit:{reason}"
        elif has_z and not flat_period:
            if p.k < zt <= p.stop_z:
                pos, event = -1, "enter:short"
            elif -p.stop_z <= zt < -p.k:
                pos, event = 1, "enter:long"
            if pos != 0:
                held, entry_date, z_entry = 0, date, zt
        out_pos.append(pos)
        out_held.append(held)
        out_event.append(event)

    decisions = pd.DataFrame({"z": z.to_numpy(), "position": out_pos, "days_held": out_held,
                              "event": out_event}, index=z.index)
    decisions.index.name = "date"
    trade_cols = ["entry_date", "exit_date", "direction", "days_held", "exit_reason", "z_entry", "z_exit"]
    return decisions, pd.DataFrame(trades, columns=trade_cols)
