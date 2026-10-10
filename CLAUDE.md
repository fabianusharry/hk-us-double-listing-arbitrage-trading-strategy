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
- `parity_usd = N * P_HK_close(t) / FX(t)`, where **FX(t) = HKD=X close of the previous FX business day (t−1)**: Yahoo's FX close for day t is stamped at the London end of day, after the US open, so using it would be look-ahead (decided Session 2). Forward-fill at most 1 day (logged); otherwise drop the date (logged).
- Dividend ex-date mismatches: on aligned dates between the first and second ex-date, the declared dividend is added back to the leg that went ex first (spread signal only; P&L uses raw prices + cash dividends).
- `spread(t) = ln(P_US_open(t)) − ln(parity_usd)`; positive = ADR rich.

**Two signal variants (decided Session 3, before any backtest):**
- **Primary ("as pitched"):** the spread above (the *morning reading*: HK close t vs US open t; HK is the stale leg).
- **Pre-registered secondary ("two-reading"):** `spread_2r(t) = ½[spread(t) + spread_eve(t)]`, where the *evening reading* `spread_eve(t) = ln(P_US_close(prev US session)) − ln(N * P_HK_open(t) / FX(t))` (US is the stale leg; HK opens 21:30 ET, before the US open on t, so it is known at decision time). The two readings carry timing noise from different news windows, so averaging cuts the noise while a persistent mispricing appears in both. If the evening reading is unavailable (unreliable HK open, no previous US close) `spread_2r` = the morning reading, and `n_readings` records it. Dividend add-backs are applied to each reading using each leg's own observation date.
- Reason: Session 2–3 diagnostics show the morning reading is dominated by non-synchronous ("stale HK") noise (AR(1) β ≈ −0.95). Both variants are reported; the primary remains the test of the pitch.

**Signal:** `z(t) = (s(t) − mean_L(t−1)) / std_L(t−1)` with window L, where s is the variant's spread.

**Positions** (state machine per pair; the position is held **in the spread** = ln US − ln HK: +1 = long US / short HK, −1 = short US / long HK, 0 = flat). *(Labels corrected Session 4: the earlier labels were the reverse of the entry rules, which would have bet on the gap widening.)*
- Enter −1 when k < z ≤ stop_z (ADR rich); enter +1 when −stop_z ≤ z < −k. No entry when |z| > stop_z: such moves are news, not mispricing (decided Session 3).
- Exit to 0 when the spread has reverted past exit_z — direction-aware: a −1 position exits when z < exit_z, a +1 position when z > −exit_z (so exit_z = 0 means "z crossed zero"); or days held ≥ H (counted in decision days); or |z| > stop_z. A missing z keeps an open position and blocks entries. *(Decided Session 4: the literal "|z| < exit_z" never fires for exit_z = 0.)*
- End of window: every position is forced flat on the second-to-last decision day (reason "end"), so the last HK trade happens inside the window (protects the OOS firewall).
- No new entry on the same day as an exit.

**Execution timing:**
- Signal is known at US open on day t.
- US leg trades at **US close on day t**: US position decided on day t earns close-to-close returns from day t+1 onward.
- HK leg trades at **the HK open of the next HK session after t** (even if the US is closed that day; decided Session 4). Split HK daily returns into overnight (close→open, incl. dividend) and intraday (open→close): on a change day, the old position earns the overnight part and the new position earns the intraday part; on other days the position earns close-to-close. On HK half-days Yahoo's bar has open = close, so execution is effectively at the half-day close. HK bars dropped as unreliable (2022-03-14) are skipped; a trade due then moves to the next session.
- Each leg's P&L runs on its own exchange calendar (so dividends and moves on non-aligned days are counted). **Hold shares** (decided Session 4, replacing daily rebalancing): when a decision changes the position, each leg's notional N = 50% of pair equity at the end of the previous date; each leg buys/sells N dollars of shares at its own execution price and holds that share count unchanged until exit (leg values drift; no free rebalancing). HK prices are converted with that day's HKD=X close (realised P&L, not a signal input). Pair P&L is marked to market from share counts; portfolio return = PAIR_WEIGHT-weighted sum of pair returns (cash re-allocated across pairs daily).
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

**Parameter grid (in-sample only, 36 combos per signal variant = 72 logged runs):** L ∈ {20, 60, 120}, k ∈ {1.5, 2.0, 2.5}, exit_z ∈ {0, 0.5}, H ∈ {5, 10}; stop_z = 4 fixed. Choose a setting on a plateau, not an isolated peak.

**Mandatory placebo test (Session 4):** simulate two listings of one efficient price with **zero mispricing**, observed non-synchronously (HK earlier than US). The spread will look mean-reverting and the strategy will trade; gross P&L at execution prices must be ≈ 0. Profit here = a timing bug. P&L is always computed per leg from execution prices, never from spread changes.

**Stock Connect analysis (pre-registered 2026-10-09, before OOS).** Formal test: **Alibaba only** (effective 2024-09-10, verified by the student). Compare the spread's mean, std, AR(1) half-life and the two-reading persistence measure over 120 trading days before vs after the event, against the same change for comparison pairs whose Southbound Connect status did not change in that window (`config.CONNECT_CONTROLS`, verified by the student 2026-10-09: Baidu, JD, TripCom, NetEase outside Connect throughout; YumChina inside throughout since 2022-10-24). **NetEase** (eligible 2026-06-30, ~65 post-event days) and **Baidu** (2026-09-07, unverified, ~17 days) are too recent: descriptive plots only, no inference; the sample is not extended past 2026-09-30.

**Out-of-universe robustness test (pre-registered 2026-10-09, before any OOS result; Session 6b).** Same frozen parameters, code and cost table, applied to HK–US dual listings never used before. Reported as a **separate portfolio**; the six-pair OOS result remains the headline test.
- *Candidates:* `config.HOLDOUT_CANDIDATES` (15 names). Ratios are unverified until the student confirms them (`docs/HOLDOUT_VERIFICATION.md`); the pipeline refuses to run unverified pairs.
- *Structural exclusion:* exclude any candidate whose ADR ratio, US ticker or domicile changed between 2021-04-19 and 2026-09-30, or whose HK listing is after 2022-12-31, or which stopped being dual-listed before 2026-09-30. Evidence: the student's verification plus a data check (implied ratio US close × FX / HK close, in-sample only).
- *Liquidity rule (measured from yfinance, in-sample days only: max(2021-04-19, HK listing) to 2023-12-29; no 2024+ data):* median daily HK value traded (close × volume, HKD) → ≥ HKD 1,000m: half-spread 0.05%; HKD 50m–1,000m: 0.10%; HKD 20m–50m: 0.20%; < HKD 20m: excluded as too thin. Thresholds chosen so that the rule reproduces the existing cost table for the original six (Yum China HKD 89m … Alibaba HKD 4,187m).
- *Borrow:* 1% p.a. like the main test, plus a sensitivity at 5% p.a. for the out-of-universe portfolio.
- *Every inclusion/exclusion is logged with its reason* (`results/logs/holdout_selection.csv`). Order: pre-registration committed → Session 6 main OOS run → Session 6b.

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
- **2026-10-07 — Session 1 (data acquisition).** `src/data.py`: `download_raw()` (yfinance, `auto_adjust=False`, `repair=False`, `keepna=True`, 3 retries with backoff, fetch-all-then-write, `manifest.json`), `load_raw()` (offline; OOS firewall raises `OOSAccessError`, defaults to `end=IS_END`), `quality_report()` → `results/logs/raw_quality.csv` + `raw_suspicious_rows.csv`. 12 tests in `tests/test_data.py` (round trip, tz→local date, firewall, offline with yfinance blocked, planted defects). Snapshot: 13 CSVs, 2019-01-01 → 2026-10-07.
  - Decisions: download from 2019-01-01 (pre-dates all HK listings) for warm-up, but backtest starts 2021-04-19 for all pairs; download end 2026-10-08 (exclusive) so the HK t+1 open after END exists; quality report is **in-sample only** (rerun for OOS in Session 6); one CSV per ticker.
  - Findings (in-sample): no missing/duplicate rows in any equity series. NTES split 5.0 on 2020-10-02 = ADS ratio change 25→5, before START and back-adjusted by Yahoo, so ratio 5 is right for the whole sample; no other splits. HK zero-volume single-price bars (O=H=L=C) on half-days 2020-12-24/31, 2021-02-11, 2021-12-24/31, 2022-01-31, and 2022-03-14 (9888, 9961 only) — the open is not a real open. HKD=X: 1 missing day (2019-05-22, pre-sample) and 17 bars with open outside [low, high] (open = close; differences < 0.4%).
  - Open for Session 2: treat HK opens on those single-price bars as unreliable (affects HK t+1 execution in Session 4) and log them; choose which HKD=X field to use (close) and note the FX bar's timestamp is not the HK close time; investigate 2022-03-14.
- **2026-10-07 — Session 2 (cleaning + spread).** `src/clean.py` (bad-HK-bar flags, calendar alignment with XNYS/XHKG reasons, FX t−1 with 1-day fill, date-dependent ratios, dividend matching + add-back), `src/spread.py` (`compute_spread`, `build_spreads`, `flag_spreads`, `save_spreads`), `src/plots.py` (`plot_spreads`). 17 new tests (29 total). Outputs: `data/processed/spreads.csv` (in-sample + warm-up, 4,779 rows), logs `dropped_dates.csv`, `fx_fills.csv`, `dividend_corrections.csv`, `hk_bar_flags.csv`, `spread_flags.csv`; figure `figures/spreads_in_sample.png`.
  - Decisions (approved by student): FX(t) = HKD=X close of t−1 (CLAUDE.md §6 updated); zero-volume single-price HK bars dropped on normal days (Baidu, TripCom 2022-03-14) but kept on XHKG early-close days with `hk_open_reliable=False`; mismatched dividend ex-dates fixed by adding the dividend back to the leg that went ex first, for the signal only.
  - Results (2021-04-19 → 2023-12-29, ~647 days per pair): spread means −0.13% to −0.49% (ADR slightly cheap; TripCom most), std 1.2–1.4%. Dropped per pair: 32 HK holidays, 19 US holidays, 2 unexplained HK gaps (2023-09-01 Typhoon Saola, 2023-09-08 black rainstorm: real closures missing from XHKG calendar), 1 unreliable bar (Baidu/TripCom). FX fills: none. Dividends: 31 matched, 0 unmatched; 3 ex-date mismatches (JD 2023-04-04 corrected 1 day; NetEase 2020-08-26 corrected 1 day; YumChina 2023-05-26/30 needed 0 days). Spread flags (|s|>10%): Alibaba 2023-03-28 +10.5%, JD 2022-03-10 −13.6%, NetEase 2022-04-11 +10.8%, Baidu 2021-03-26 −14.4% (warm-up); all one-day spikes that close next day = news between HK close and US open, not data errors. Kept.
  - Open for Session 3–4: (a) much of the spread variance is HK being "stale" vs news in the 5.5–6.5h gap, which HK absorbs at its next open — expect very short half-lives; the strategy trades HK at the t+1 open, i.e. AFTER that catch-up, so Session 4 timing tests must prove no phantom profit from it. (b) P&L must use raw prices + cash dividends and handle dividends falling on non-aligned dates (e.g. YUMC US ex 2023-05-26, an HK holiday). (c) Respect `hk_open_reliable=False` on HK half-days in execution. (d) `exchange_calendars` emits a harmless NumPy DeprecationWarning.
- **2026-10-07 — Session 3 (descriptive analysis, in-sample).** Discussion first: the morning spread is dominated by non-synchronous "stale HK" noise. Student approved: (1) mandatory zero-mispricing placebo test in Session 4, (2) no entry when |z| > stop_z, (3) report the persistence split, (4) pre-registered secondary signal `spread_2r` (two-reading average; §6 updated; grid = 72 runs). Built: `clean.previous_us_session`, per-reading dividend add-backs (`div_adj_*_eve`), `spread_eve`/`spread_2r`/`n_readings` in `spread.py`; `src/analysis.py` (summary stats, AR(1) half-life, ADF, ARMA(1,1) persistence split, timing diagnostics, Δspread correlations + effective bets, capture-vs-cost); 3 new figures. 47 tests.
  - Results (2021-04-19 → 2023-12-29): morning std 1.2–1.4%, two-reading 0.9–1.1%. AR(1) β −0.75 to −0.98 (half-life 0.2–0.5 d); two-reading −0.66 to −0.93 (0.3–0.6 d). ADF rejects a unit root everywhere (p≈0), which is uninformative for near-white-noise. Persistent component: morning detected only for JD (share 7%, HL 3.4 d) and YumChina (48%, 1.1 d); two-reading detected for 5/6 (shares 18–61%, HL 0.5–1.7 d; none for TripCom). corr(morning t, evening t+1) −0.01 to 0.28. Cross-pair Δspread avg ρ 0.46 → 1.8 effective bets of 6 (two-reading 0.36 → 2.1). Persistent ADR discount: mean −0.1% to −0.5% in BOTH readings (so a real level, not timing drift); removed by the rolling mean.
  - Falsification test (a) (half-life > ~20 d): not triggered for any pair, but the short half-life is mostly timing noise, so this is weak support. Capture-vs-cost (ρ1 × 2σ vs round trip 0.54–0.74%): below cost for every pair; YumChina closest (0.62% morning / 0.72% two-reading vs 0.74%). Prediction for Session 5: little or no net profit except possibly YumChina.
  - Open: Session 4 placebo test + `k < |z| ≤ stop_z` entry; z-scores must handle `spread_2r` from `n_readings`; long-lag ACF bars (~0.1 at lags 4–10 for some pairs) hint at a slowly moving mean — not tested formally (multiple-testing risk).
- **2026-10-09 — Session 4 (strategy + backtest engine, no costs).** `src/strategy.py` (`Params`, `zscore` with `.shift(1)`, O(T) `positions` state machine with exit reasons and trade list), `src/backtest.py` (`leg_returns` overnight/intraday/close-to-close in USD, `execution_sessions`, `leg_pnl`, `backtest_pair`, `run_pair`, `run_backtest`, `log_run` → `results/grid_log.csv`), `tests/synthetic.py` (one efficient price observed at the four session times + optional AR(1) mispricing). 21 new tests (68 total): state-machine rules, hand-computed 5-day example (exact to 1e-12), upper-bound US-open variant, HK-only session execution (Thanksgiving), dividends on non-aligned days (credit/debit), HK half-day = execution at close, dropped-bar skip, look-ahead (future spreads and future prices), OU profit (Sharpe 1.36), placebo (Sharpe +0.02 over 40 simulated years, 1,119 trades; the wrong spread-change P&L shows +5.76 on the same trades), portfolio weights, OOS refusal, log.
  - Spec fixes/decisions (approved): **position labels corrected** — position is in the spread, +1 = long US / short HK (the old labels contradicted the entry rules and would have bet on divergence); HK executes at the next HK session (not next aligned day); exit_z direction-aware; forced flat on the second-to-last decision day ("end"); half-day HK execution = at the close; HK converted with same-day FX close. **Follow-up (same day, student request):** the 5-day example is now verified against an independent shares-and-dollars ledger (`tests/timing_example.py`; `python -m tests.timing_example` prints the walkthrough). This exposed that the original daily-rebalanced engine gave +1.7195% on the example vs +1.4608% for a trader holding shares (and that the first walkthrough summed daily returns, +1.73%). Engine switched to **hold shares** (§6); engine == ledger to 1e-12; walkthrough numbers pinned in tests. Placebo Sharpe +0.05, OU profit Sharpe +1.37 after the switch; equal entry dollars per leg verified on real data; 6 pairs run in ~0.3 s.
  - Smoke run (logged as `smoke`, L=60, k=2, exit_z=0, H=10, both signals; returns deliberately not inspected): 19–30 trades per pair, average 1.6–3.3 days held, in the market 6.5–14% of days, almost all exits `revert` (consistent with the short half-life). Baidu/TripCom first decision 2021-06-23/07-19 (no warm-up history). All timing invariants held.
  - Open for Session 5: costs on `us_trade`/`hk_trade` (traded notional as a fraction of pair equity at the previous close) + borrow on the short leg's `us_value`/`hk_value`; metrics; 72-run grid. Note for interpretation: average holding ~2 days ⇒ round-trip costs (0.54–0.74% per leg) are paid very often.
- **2026-10-09 — Session 5 (costs, metrics, in-sample grid).** `src/costs.py` (`CostModel`, `cost_model(pair, multiplier)`, per-side rates, borrow per calendar day), costs deducted **inside** the engine loop (later trades sized from equity after costs; `gross_ret`, `us_cost`, `hk_cost`, `borrow_cost`, net `pair_ret`); `src/metrics.py` (cumulative/annual return, vol, Sharpe ×√252, max drawdown, per-trade returns from equity, hit rate, turnover, cost drag, `plateau_scores`); `backtest.grid_params`/`run_grid`; grid-log schema widened (old smoke rows kept); `plots.plot_sharpe_heatmaps`. `analysis.round_trip_cost` now delegates to `costs`. 14 new tests (85 total), incl. exact round-trip cost on flat prices and multiplier 0 = gross.
  - Grid (IS, costs 1×): 72 runs logged, **21.2 s total (~0.29 s/run)**. Outputs `results/grid_results.csv`, `grid_pairs.csv`, `grid_plateau.csv`, `figures/sharpe_grid_spread.png`, `sharpe_grid_spread_2r.png`.
  - Results: **gross Sharpe positive in all 72 cells** (morning 0.33–0.57, two-reading 0.42–0.78) but **net Sharpe −0.66…+0.06 (morning) and −0.90…+0.15 (two-reading)**; net improves monotonically towards L=120, k=2.5 (fewer, larger-deviation trades: fixed cost per trade, capture ∝ entry gap — as predicted in Session 3). H barely matters (avg hold 1.1–2.7 days; time exit rarely binds). Only 9–17 trades per pair at the best cells. Sharpe standard error over 2.69 years ≈ 0.61, so no cell is distinguishable from zero.
  - **Frozen 2026-10-09 (student's choice, accepting the suggestion):** `config.FROZEN_PARAMS` = one common setting for both signals, L=120, k=2.5, exit_z=0.5, H=10, stop_z=4 (morning net 0.06 / gross 0.36; two-reading net 0.08 / gross 0.51; highest worst-neighbour for two-reading). Caveat: it sits at the edge of the pre-registered grid; the grid is not being extended.
- **2026-10-09 — Session 6 (OOS, sensitivity, Stock Connect).** Commits before the run: freeze `efd5e08` → out-of-universe pre-registration `d1a3f6c` → Connect design `20f84c1` → `ALLOW_OOS=True` `fef6322`. Spreads rebuilt on full data; **all 4,779 IS rows identical** (max diff 3e-14). OOS quality: half-day fake bars handled; 2024-01-15 fake bars (JD, YumChina, Baidu, TripCom) dropped by the existing rule; 1 date lost to two consecutive missing FX days (2025-04-22); FX fills 2025-01-02, 2026-01-02. New: `analysis.event_windows/segment_stats/connect_event_study/difference_vs_controls` (+2 tests, 87 total), `plots.plot_equity_is_oos`, `plots.plot_connect_event`. Outputs: `results/is_vs_oos.csv`, `sensitivity.csv`, `connect_alibaba_study.csv`, `connect_alibaba_vs_controls.csv`, `raw_quality_oos.csv`; figures `equity_is_oos.png`, `connect_alibaba.png`, `connect_netease_descriptive.png`, `connect_baidu_descriptive.png`. All runs logged (`oos_main`, `is_frozen`, `sensitivity`, `breakeven`, `upper_bound`).
  - **OOS (run once, costs 1×):** morning net Sharpe **0.22** (gross 0.88), ann. return 0.4%, max DD −1.5%, 60 trades; two-reading net **0.50** (gross 1.16), ann. return 1.3%, ann. vol 2.5%, max DD −1.9%, 80 trades. t-stats 0.37 and 0.83 (SE ≈ 0.6): neither distinguishable from zero. Morning: top 3 trades = 131% of profit (without them −0.29%), median trade ≈ 0; two-reading: top 3 = 82%, bootstrap P(mean trade ≤ 0) = 0.06. Profits arrive in jumps on dislocation episodes (Sep–Oct 2024 China stimulus rally; Apr 2025 tariff shock). Pair results flip sign between IS and OOS (e.g. NetEase best IS → worst OOS; JD worst IS → best OOS).
  - **Cost sensitivity / break-even:** net Sharpe at 0/0.5/1/2× = morning OOS 0.88/0.55/0.22/−0.43, two-reading OOS 1.16/0.83/0.50/−0.14. Break-even multiplier: IS 1.18× / 1.19×, OOS 1.34× / 1.78×. 2× costs make every variant negative. (IS stamp duty was really 0.13% Aug 2021–Nov 2023 ≈ 1.1× costs, which uses most of the IS margin.)
  - **Execution upper bound (US at open on t, not tradable):** morning OOS net 0.78 (vs 0.22), IS 0.31 (vs 0.06); two-reading OOS 0.53 (vs 0.50). Waiting for the US close costs the morning signal most of its edge.
  - **Stock Connect (Alibaba, ±120 days, 2024-03-08 → 2025-03-13):** spread mean change +0.03 pp (morning) / −0.17 pp (two-reading), inside the controls' range (rank 3/6 and 5/6): **no detectable HK premium from mainland demand**. Std rose most for Alibaba (+0.56 pp, rank 1/6) but YumChina (already in Connect) rose almost as much (+0.50 pp) and the window contains the 24 Sep 2024 stimulus rally two weeks after inclusion: confounded, not attributable. Persistence fell (two-reading ACF 0.19 → −0.02), also within controls' range. NetEase/Baidu: descriptive plots only.
  - **Verdict:** falsification test (c) — primary signal: not supported (net Sharpe not distinguishable from zero; profit = 3 trades). Secondary: weakly suggestive, not significant. Connect channel: not visible in the data. Open: Session 6b (out-of-universe), notebook (Session 7).
- **2026-10-10 — Session 6b, part 1 (out-of-universe selection).** Student read all 15 ratios on Citi DR pages; Claude fetched Citi's programme-details tables (`results/logs/holdout_citi_programs.csv`: ratio, tickers, country, depositary, milestones). Downloaded the 15 pairs to `data/raw/holdout/` (main snapshot untouched). New `src/holdout.py` (`spread_tier`, `median_hk_value`, `implied_ratio`, `ratio_check`, `us_splits`, `select`) + `tests/test_holdout.py` (tiers reproduce the original six's cost table; ratio step detection). 91 tests. Selection log: `results/logs/holdout_selection.csv`.
  - **Included (6, then 7 with Hutchmed):** Bilibili (HKD 398m/day → 0.10%), XPeng (500m → 0.10%), LiAuto (587m → 0.10%), NIO (85m → 0.10%), GDS (57m → 0.10%), ZTO (23m → 0.20%). Implied ratios flat at the verified ratio; no US splits in sample.
  - **Excluded (9):** too thin — Autohome 1.4m, Weibo 3.3m, TencentMusic 1.2m, KEHoldings 18.4m (< 20m); structural — HWorld (ratio 1→10 2021-06-29), ZaiLab (1→10 2022-03-30), BeiGene (BGNE→ONC 2025-01-02, Swiss), NewOriental (ADR ratio → 1:10 Apr 2022 per Yahoo split records; Citi's 4:1 contradicts market-implied 10); Hutchmed — **resolved 2026-10-10: included** at ratio 5 (company investor FAQ: "each ADR represents five ordinary shares"; depositary Deutsche Bank; Citi's third-party page showed 15 in error), HKD 45m/day → 0.20%. **Final out-of-universe basket: 7 pairs** (Bilibili, XPeng, LiAuto, Hutchmed, NIO, ZTO, GDS).
  - Lessons: Citi's directory is third-party for DB programmes and was wrong/stale for HCM and EDU; market-implied ratios caught it. Limitation: the implied-ratio check confirms the data are consistent with the configured ratio but cannot detect an in-sample ratio change that Yahoo back-adjusted on both lines (HWorld/ZaiLab read a flat 10); change detection relies on depositary milestones and Yahoo split records.
