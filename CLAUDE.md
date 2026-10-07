# CLAUDE.md — Capstone: US–Hong Kong Dual-Listing Arbitrage

This file gives Claude Code the full context for this project. Read it at the start of every session, then follow the session task in `docs/SESSION_PLAN.md`.

## 1. Project in one paragraph

Individual capstone (Data Structures and Algorithms, Imperial College Business School; deadline 23 Oct 2026, target submission 20 Oct 2026). Hypothesis: for six Chinese companies listed both as ADRs in New York and as ordinary shares in Hong Kong, the price gap between the two listings (adjusted for ADR ratio and USD/HKD) mean-reverts within a few days, because the shares are fully fungible but arbitrage is limited by non-overlapping trading hours, conversion delays, short-borrow costs and segmented investor bases (notably Stock Connect, which lets mainland investors buy only the HK leg). We test whether a market-neutral z-score strategy profits after realistic costs, out-of-sample.

**The student must be able to explain and defend every line.** Prefer clear, readable code over clever code. After each task, explain in plain language what the code does, why it is built that way, and what would break if a key assumption were wrong.

## 2. Non-negotiable rules

1. **No look-ahead.** Every value used for a decision on day t must be knowable at that decision time.
   - The spread uses the **US open on day t** against the **HK close on day t**. Never compare same-date closes.
   - Rolling mean/std for the z-score use data **up to t−1 only** (`.shift(1)` after `.rolling(...)`).
   - Use **unadjusted** prices (`auto_adjust=False`). Never use back-adjusted closes for the spread level.
2. **Out-of-sample firewall.** In-sample = 2021-04-19 to 2023-12-31. Out-of-sample = 2024-01-01 to 2026-09-30. `config.ALLOW_OOS = False` until Session 6; data loaders must raise an error if asked for OOS dates while the flag is False. The OOS period is run **once** with frozen parameters.
3. **Log every parameter combination tried** to `results/grid_log.csv` (needed for the overfitting checklist).
4. **Never silently drop data.** Every dropped date, flagged spread or corrected dividend is written to a log file in `results/logs/` with a reason.
5. **No credentials** anywhere in the repo. All data sources are public.
6. **Reproducible offline.** After Session 1, everything runs from the CSV snapshot in `data/raw/`; downloading is a separate, optional step.

## 3. Tech stack

Python 3.11+. Use: `pandas`, `numpy`, `matplotlib` (no seaborn), `yfinance`, `statsmodels` (OLS, ADF test), `exchange_calendars` (XNYS, XHKG), `pytest`, `jupyter`. Pin versions in `requirements.txt`. No other dependencies without asking.

## 4. Repository structure

```
.
├── CLAUDE.md
├── README.md                 # how to run; data provenance
├── requirements.txt
├── config.py                 # all constants: pairs, dates, costs, grid
├── src/
│   ├── data.py               # download + load raw snapshot
│   ├── clean.py              # calendar alignment, FX, ratios, dividends, logs
│   ├── spread.py             # spread + sanity checks
│   ├── analysis.py           # summary stats, AR(1) half-life, ADF, correlations
│   ├── strategy.py           # z-score, position state machine
│   ├── backtest.py           # leg-level P&L with execution timing
│   ├── costs.py              # transaction + borrow cost model
│   ├── metrics.py            # cumulative return, Sharpe, max drawdown, trade stats
│   └── plots.py              # matplotlib figures, saved to figures/
├── tests/                    # pytest; synthetic-data tests for every module
├── notebook.ipynb            # Data Acquisition → Cleaning → Strategy → Backtest
├── data/raw/                 # snapshot as downloaded, filenames include date
├── data/processed/
├── results/                  # tables, grid_log.csv, logs/
├── figures/
└── docs/SESSION_PLAN.md
```

Note: the brief lists the submission as report + `notebook.ipynb` + `data/`. Keep `src/` but in Session 8 decide whether to submit it alongside the notebook or inline it into the notebook.

## 5. Configuration (`config.py`)

```python
PAIRS = {  # name: (US ticker, HK ticker, HK shares per ADR)
    "Alibaba":  ("BABA", "9988.HK", 8),
    "JD":       ("JD",   "9618.HK", 2),
    "NetEase":  ("NTES", "9999.HK", 5),
    "YumChina": ("YUMC", "9987.HK", 1),
    "Baidu":    ("BIDU", "9888.HK", 8),
    "TripCom":  ("TCOM", "9961.HK", 1),
}
FX_TICKER = "HKD=X"            # HKD per USD
START, END = "2021-04-19", "2026-09-30"
IS_END    = "2023-12-31"
OOS_START = "2024-01-01"
ALLOW_OOS = False
CONNECT_EVENTS = {"Alibaba": "2024-09-10", "Baidu": "2026-09-07"}
```
ADR ratios verified 2026-10-07 (Deutsche Bank DR directory; see Progress log). Make ratios easy to change, and support date-dependent ratios if a ratio changed mid-sample.

## 6. Core definitions

**Parity and spread** (per pair, day t):
- `parity_usd = N * P_HK_close(t) / FX(t)`
- `spread(t) = ln(P_US_open(t)) − ln(parity_usd)`; positive = ADR rich.

**Signal:** `z(t) = (spread(t) − mean_L(t−1)) / std_L(t−1)` with window L.

**Positions** (state machine per pair, +1 = long HK / short US, −1 = short HK / long US, 0 = flat):
- Enter −1 when z > k (ADR rich); enter +1 when z < −k.
- Exit to 0 when |z| < exit_z, or holding days ≥ H, or |z| > stop_z.
- No new entry on the same day as an exit.

**Execution timing:**
- Signal is known at US open on day t.
- US leg trades at **US close on day t**: US position decided on day t earns close-to-close returns from day t+1 onward.
- HK leg trades at **HK open on day t+1** (next aligned trading day). Split HK daily returns into overnight (close→open) and intraday (open→close): on a change day, the old position earns the overnight part and the new position earns the intraday part.
- Sensitivity variant only (labelled "upper bound, not tradable"): US leg at US open on day t.

**Sizing:** each pair gets 1/6 of capital; within a pair each leg's notional = 50% of pair capital (gross exposure 1×, no leverage). Idle capital earns 0.

**Returns:** leg returns are total returns in USD (dividends credited to longs on the ex-date, debited to shorts).

**Costs** (per side, as fraction of traded notional; in `config.py`):
| Component | Value | Leg |
|---|---|---|
| HK stamp duty | 0.0010 | HK |
| HK levies/fees | 0.0001 | HK |
| Commission | 0.0003 | both |
| Half bid-ask | 0.0005 (BABA, JD), 0.0010 (others) | both |
| Short borrow | 0.01 per year, daily accrual | short leg |

Support a `cost_multiplier` in {0, 0.5, 1, 2} and a break-even search.

**Metrics:** cumulative return, annualised Sharpe (mean/std of daily returns × √252, risk-free = 0), max drawdown; plus trade count, hit rate, average holding days, turnover, cost drag. Report per pair and for the portfolio, in-sample vs out-of-sample.

**Parameter grid (in-sample only, 36 combos):** L ∈ {20, 60, 120}, k ∈ {1.5, 2.0, 2.5}, exit_z ∈ {0, 0.5}, H ∈ {5, 10}; stop_z = 4 fixed. Choose a setting on a plateau, not an isolated peak.

## 7. Coding conventions

- Small, pure functions with type hints and a docstring stating inputs, outputs, and the timing assumption if relevant.
- Vectorise with pandas/numpy where clear; a simple loop is fine for the position state machine (explain its O(T) cost).
- Dates as a `DatetimeIndex` named `date`; columns named `{TICKER}_{field}` or tidy long format, consistently.
- Every module gets pytest tests using **synthetic data** with a known answer (e.g. a 1:1 pair with an artificial gap; an Ornstein–Uhlenbeck spread the strategy should profit from at zero cost).
- Figures: matplotlib, titled, labelled axes with units, saved as PNG to `figures/`.
- Commit at the end of each session with a descriptive message.

## 8. Progress log

Update this section at the end of every session (date, what was built, open issues, decisions taken and why).

- **2026-10-07 — Session 0 (setup).** Created folder skeleton (§4), `config.py` (§5 constants plus costs, grid, sizing, paths), docstring-only `src/` modules, `tests/conftest.py`, `.gitignore`, pinned `requirements.txt`, `.venv` on Python 3.14.6. `import config` works; `pytest` runs (0 tests).
  - Decisions: pins are the versions pip resolved together on 2026-10-07 (pandas 3.0 — note copy-on-write semantics). Date-dependent ADR ratios go in `config.RATIO_CHANGES` (empty for now). `FROZEN_PARAMS = None` until Session 5. Empty data/results/figures folders kept in git via `.gitkeep`; `data/` is not ignored.
  - Open: ~~ADR ratios unverified~~ → resolved 2026-10-07: all six ratios confirmed by the student via the Deutsche Bank DR directory (adr.db.com): BABA 8, JD 2, NTES 5, BIDU 8, TCOM 1, YUMC 1. These are current ratios; Session 1's split/ratio check must confirm no change inside 2021-04-19 → 2026-09-30.
