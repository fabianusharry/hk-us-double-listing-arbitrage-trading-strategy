# Session Plan — Coding the Capstone with Claude Code

Nine sessions of 45–60 minutes, 7–19 October 2026. Each session has a goal, tasks, a "done when" check, questions you should be able to answer afterwards, and a starter prompt to paste into Claude Code.

General workflow for every session:
1. Start Claude Code in the repo root (it reads `CLAUDE.md` automatically).
2. Paste the session's starter prompt. For bigger sessions, ask it to **plan first** and approve the plan before it writes code.
3. Run the tests, look at the outputs and figures yourself.
4. Ask Claude Code to explain anything you can't explain yourself.
5. Have it update the Progress log in `CLAUDE.md`, then commit.

---

## Session 0 — Setup (7 Oct, ~30 min)

**Goal:** an empty but runnable project skeleton.

Tasks:
- Create the folder structure from `CLAUDE.md` §4, `requirements.txt` with pinned versions, `config.py` from §5, empty `src/` modules with docstrings, `tests/` folder, `.gitignore` (ignore `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`; do **not** ignore `data/`).
- Set up a virtual environment and install requirements.

Done when: `pytest` runs (0 tests is fine) and `python -c "import config"` works.

You should understand: why all constants live in one `config.py`; what `ALLOW_OOS` protects against.

Starter prompt:
> Read CLAUDE.md. Do Session 0 from docs/SESSION_PLAN.md: create the repository skeleton, requirements.txt with pinned versions, config.py and .gitignore. Don't write any analysis code yet.

---

## Session 1 — Data acquisition (7–8 Oct)

**Goal:** a raw data snapshot on disk, and a loader that works offline.

Tasks:
- `src/data.py`: `download_raw()` fetches daily OHLCV for the 12 tickers and `HKD=X` with `auto_adjust=False`, plus dividends and splits; saves one CSV per ticker to `data/raw/` with the download date in the filename. Retries politely on failure.
- `load_raw()` reads from the snapshot only (no network).
- A data-quality report saved to `results/logs/raw_quality.csv`: first/last date, row count, missing values, zero-volume days, and any splits per ticker.

Done when: snapshot exists for all 13 series; `load_raw()` runs with Wi-Fi off; quality report reviewed by you.

You should understand: what unadjusted vs adjusted prices mean and why the spread needs unadjusted ones; what an ADR ratio is.

Starter prompt:
> Session 1: implement download_raw() and load_raw() in src/data.py per CLAUDE.md, run the download once, and produce the raw quality report. Show me the report and point out anything suspicious (gaps, zero volumes, splits, odd opening prices).

---

## Session 2 — Cleaning and spread construction (9 Oct)

**Goal:** one clean, aligned spread series per pair, with every adjustment logged.

Tasks:
- `src/clean.py`:
  - Align each pair on dates when **both** legs traded. Use `exchange_calendars` (XNYS, XHKG) to label each dropped date: US holiday, HK holiday, or unexplained missing data. Save to `results/logs/dropped_dates.csv`.
  - Convert HK prices to USD with `HKD=X` (forward-fill FX at most 1 day; log any fill).
  - Apply ADR ratios (support date-dependent ratios).
  - Detect dividend ex-date mismatches between the two legs and correct the artificial one-day jump; log each correction.
- `src/spread.py`: compute spread per CLAUDE.md §6; flag |spread| > 10% to `results/logs/spread_flags.csv`.
- Save to `data/processed/spreads.csv`.
- Tests: a synthetic 1:1 pair with identical prices gives spread 0; a known 1% gap gives ≈0.01; a missing day in one leg is dropped and logged.

Done when: tests pass; a plot of all six spreads looks centred near zero; you have read every flagged row.

You should understand: why we pair US open with HK close; why a mismatched ex-dividend date creates a fake signal; why the HKD peg makes FX timing a minor issue.

Starter prompt:
> Session 2: plan first, then implement src/clean.py and src/spread.py per CLAUDE.md with tests on synthetic data. Then plot the six spreads and summarise the logs (dropped dates by reason, FX fills, dividend corrections, flagged spreads).

---

## Session 3 — Descriptive analysis, in-sample only (10–11 Oct)

**Goal:** evidence on whether spreads mean-revert, before any strategy exists.

Tasks:
- `src/analysis.py`:
  - Summary stats per pair: mean, std, percentiles, share of days with |spread| > 0.5% and > 1%.
  - AR(1) regression Δs(t) = α + β·s(t−1) + ε with `statsmodels` OLS; half-life = −ln 2 / ln(1+β); report β, its t-stat and the half-life.
  - ADF test per pair.
  - Correlation matrix of daily spread changes across pairs.
- Figures: spread time series with ±2 std bands; correlation heatmap (matplotlib `imshow`).
- Save tables to `results/`.

Done when: you can say, per pair, whether the in-sample data supports falsification test (a) of your pitch (half-life above ~20 days).

You should understand: what the half-life formula means; why highly correlated spreads mean fewer independent bets.

Starter prompt:
> Session 3: implement src/analysis.py on in-sample data only (ALLOW_OOS is False). Produce the stats table, AR(1) half-lives, ADF results and correlation heatmap. Then explain the results to me as if I need to defend them to an examiner.

---

## Session 4 — Strategy and backtest engine (12 Oct)

**Goal:** a correct, tested backtest engine with the right execution timing. Correctness first; costs come next session.

Tasks:
- `src/strategy.py`: rolling z-score with `.shift(1)`; position state machine (entry, exit, max holding, stop-loss, no same-day re-entry).
- `src/backtest.py`: leg-level daily P&L with the execution timing in CLAUDE.md §6 (US leg at US close day t; HK leg at HK open day t+1 using overnight/intraday split); total returns including dividends; pair and portfolio aggregation with the sizing rule.
- Tests:
  - **Look-ahead test:** changing spread values after date d leaves all positions up to d unchanged.
  - **Synthetic profit test:** on a simulated Ornstein–Uhlenbeck spread with zero costs, the strategy is profitable.
  - **Timing test:** a hand-built 5-day example where you can compute the P&L on paper matches the engine.

Done when: all three tests pass and you can walk through the 5-day example yourself.

You should understand: why the HK leg's first-day return is open-to-close; why the US position from day t only earns returns from t+1; the time complexity of the state machine.

Starter prompt:
> Session 4: plan first. Implement src/strategy.py and src/backtest.py per CLAUDE.md §6 without costs, plus the look-ahead, synthetic-profit and hand-computed timing tests. Walk me through the 5-day timing example step by step.

---

## Session 5 — Costs and parameter search, in-sample only (13 Oct)

**Goal:** frozen parameters chosen honestly.

Tasks:
- `src/costs.py`: per-side costs on each position change per leg, daily borrow on the short leg, `cost_multiplier`.
- `src/metrics.py`: cumulative return, Sharpe, max drawdown, trade count, hit rate, average holding days, turnover, cost drag.
- Run the 36-combination grid on in-sample data with costs at 1×; append every run to `results/grid_log.csv`; time it with `time.perf_counter()` (for the efficiency touchpoint).
- Figure: Sharpe heatmap over (L, k) for each (exit_z, H) slice.
- Choose the frozen parameters on a plateau; write the choice and reasoning into `config.py` and the Progress log.

Done when: parameters frozen and committed **before** Session 6 starts.

You should understand: the difference between picking the best cell and picking a plateau; why logging all 36 runs matters.

Starter prompt:
> Session 5: implement src/costs.py and src/metrics.py with tests, run the in-sample grid, log every run, time it, and show me the Sharpe heatmaps. Suggest a plateau setting and explain why, but let me make the final choice.

---

## Session 6 — Out-of-sample, sensitivity and Stock Connect (14 Oct)

**Goal:** the headline results.

Tasks:
- Set `ALLOW_OOS = True` (commit this change separately so the history shows when the OOS data was first touched).
- Run the frozen strategy **once** on OOS. Produce the in-sample vs out-of-sample results table (per pair and portfolio).
- Sensitivity: cost multipliers {0, 0.5, 1, 2}; break-even multiplier where OOS Sharpe = 0; the "upper bound, not tradable" US-open execution variant.
- Stock Connect analysis: Alibaba spread mean, std and half-life 120 trading days before vs after 2024-09-10, compared with the same change for the other pairs over the same windows.
- Figures: equity curves (IS and OOS shaded differently), drawdown chart, Alibaba spread around the Connect date.

Done when: results table and figures saved to `results/` and `figures/`.

You should understand: how to interpret the IS vs OOS gap; what the break-even multiplier says about your falsification test (c).

Starter prompt:
> Session 6: I've frozen parameters. Enable OOS, run it once, then the cost sensitivity, break-even search, execution upper bound and the Stock Connect comparison. Summarise the results honestly, including if the hypothesis fails.

---

## Session 6b — Out-of-universe robustness test (after Session 6)

**Goal:** check the frozen strategy on dual listings that were never used, under the rule pre-registered in CLAUDE.md §6.

Tasks:
- Student verifies ADR ratios (`docs/HOLDOUT_VERIFICATION.md`) and sets `HOLDOUT_RATIO_VERIFIED`.
- Download the candidates (yfinance, same settings), quality report, implied-ratio check on in-sample data.
- Apply the structural and liquidity rules on in-sample data only; log every inclusion/exclusion to `results/logs/holdout_selection.csv`.
- Run the frozen strategy on the survivors as a separate portfolio, IS and OOS periods, borrow 1% and 5%.

Done when: selection log and holdout results table saved; nothing re-tuned.

## Session 7 — Notebook assembly (15 Oct)

**Goal:** `notebook.ipynb` that runs top to bottom and tells the story.

Tasks:
- Markdown headings in order: **Data Acquisition**, **Cleaning**, **Strategy**, **Backtest** (then Results, Sensitivity, Stock Connect).
- Each section: short markdown explanation, the function calls, key tables and figures.
- Default to loading the snapshot; downloading behind a `REFRESH_DATA = False` flag.
- Restart kernel and run all; no errors.

Done when: "Restart & Run All" succeeds in a fresh environment.

Starter prompt:
> Session 7: build notebook.ipynb with the section headings from the brief, calling src/ functions, loading from the snapshot by default. Then run it end-to-end from a clean kernel and fix any errors.

---

## Session 8 — Final verification (19 Oct)

**Goal:** everything you'll submit is consistent and defensible.

Tasks:
- Fresh clone, new virtual environment, install, run tests and notebook.
- Cross-check each pitfall-checklist answer in your report against the actual code (ask Claude Code to point to the exact lines).
- Confirm no credentials anywhere; `README.md` documents data provenance and how to run.
- Decide on `src/`: submit alongside the notebook, or inline it into the notebook.
- Package: report, `notebook.ipynb`, `data/`.

Starter prompt:
> Session 8: do a final verification. Fresh environment run, tests, notebook end-to-end. Then for each item in my pitfall checklist, show me the code lines that implement it, and flag anything in my report that doesn't match the code.

---

## Report writing (15–18 Oct, outside Claude Code)

Write alongside Sessions 7–8, using the brief's exact headings: 1. Hypothesis Pitch, 2. Data Pipeline, 3. Strategy and Backtest, 4. Methodology Defense, 5. Pitfall Checklist, 6. Efficiency Touchpoint, 7. Risk Touchpoint, 8. Reflection.
