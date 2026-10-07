"""Spread construction and sanity checks.

spread(t) = ln(P_US_open(t)) - ln(N * P_HK_close(t) / FX(t)); positive = ADR rich.
Timing: HK closes (16:00 HKT) before New York opens (09:30 ET), so the HK close
on day t is known at the US open on day t. Never compare the US *close* with the
HK close of the same date.

Implemented in Session 2. No code yet (Session 0 skeleton).
"""
