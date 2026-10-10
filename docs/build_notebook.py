"""Builds notebook.ipynb (Session 7). Markdown carries the story; numbers come from code."""
import sys
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# US–Hong Kong Dual-Listing Arbitrage

**Question.** Six Chinese companies trade both as ADRs in New York and as ordinary shares in Hong Kong. After
adjusting for the ADR ratio and USD/HKD, the two prices should be equal. Does the gap mean-revert fast enough
for a market-neutral z-score strategy to profit **after realistic costs, out-of-sample**?

**How to read this notebook.** Each section states what it does and why in a few lines; all logic lives in
`src/` (unit-tested in `tests/` on synthetic data). Every number quoted at the end is printed from the computations above it.

**Discipline built in:** no look-ahead (signal at the US open uses only data known then); parameters frozen
on 2021-04-19 → 2023-12-31 before the out-of-sample period (2024-01-01 → 2026-09-30) was opened, and the OOS
period was run once; every parameter combination tried is logged (`results/grid_log.csv`); every dropped or
corrected data point is logged (`results/logs/`); a placebo test proves the engine cannot profit from pure
timing noise.
""")

code("""
REFRESH_DATA = False   # True: re-download from Yahoo Finance (creates a new dated snapshot); default = offline snapshot
RERUN_GRID   = False   # True: re-run the 72-run in-sample grid (logged to results/grid_log_rerun.csv, not the main log)

import warnings; warnings.filterwarnings("ignore", category=DeprecationWarning)
import json
import numpy as np, pandas as pd
from IPython.display import Image, display
import config
from src import data, clean, spread, analysis, strategy, backtest, metrics, plots, holdout
from src.strategy import Params
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
pct = lambda x: f"{x * 100:+.2f}%"
FROZEN = Params(**config.FROZEN_PARAMS)
print("OOS unlocked:", config.ALLOW_OOS, "| frozen parameters:", FROZEN)
""")

# ---------------------------------------------------------------- Data Acquisition
md("""
## 1. Data Acquisition

Daily prices for the 6 ADRs, their 6 Hong Kong lines and HKD=X from Yahoo Finance (`yfinance`,
`auto_adjust=False`: split-adjusted but **not** dividend-adjusted, so the spread level is real). Downloaded once
into `data/raw/` (one dated CSV per ticker); everything below reads that snapshot offline.
""")
code("""
if REFRESH_DATA:
    data.download_raw()
frames = data.load_raw()                       # offline; OOS dates allowed because config.ALLOW_OOS is True
manifest = json.load(open(config.RAW_DIR / "manifest.json"))
print(f"Snapshot downloaded {manifest['download_date']} with yfinance {manifest['yfinance_version']};",
      f"window {manifest['start_inclusive']} to {manifest['end_exclusive']} (exclusive)")
pd.DataFrame(config.PAIRS, index=["US ADR", "HK line", "HK shares per ADR"])
""")
code("""
q_is, _ = data.quality_report({t: f.loc[config.START:config.IS_END] for t, f in frames.items()})
q_oos = pd.read_csv(config.LOG_DIR / "raw_quality_oos.csv")
cols = ["ticker", "first_date", "last_date", "n_rows", "missing_close", "zero_volume", "open_outside_range", "n_dividends", "n_splits"]
print("In-sample data quality (2021-04-19 .. 2023-12-31). Zero-volume rows are Yahoo's single-price bars on HK half-days; FX has no volume.")
q_is[cols]
""")

# ---------------------------------------------------------------- Cleaning
md("""
## 2. Cleaning and the spread

- **Align** each pair on days both markets traded; every other day is logged with a reason (US/HK holiday from
  `exchange_calendars`, or unexplained missing data — e.g. HK typhoon closures).
- **Bad HK bars**: Yahoo's single-price, zero-volume bars are dropped on normal days (e.g. 2022-03-14) and kept on
  half-days with the open marked unreliable.
- **FX**: HKD=X close of the *previous* day — Yahoo stamps its FX close after the US open, so same-day FX would be look-ahead.
- **Dividends**: when the two legs go ex on different days, the dividend is added back to the leg that went first
  (signal only; P&L uses raw prices plus cash dividends).

**Spread (morning reading):** `ln(US open_t) − ln(N × HK close_t / FX)`. HK has closed before New York opens, so
both prices are known at the US open. Positive = ADR rich.
""")
code("""
spreads = spread.build_spreads(clean.clean_all(frames))      # writes results/logs/*.csv
flags = spread.save_spreads(spreads)
is_spreads = analysis.in_sample(spreads)
dropped = pd.read_csv(config.LOG_DIR / "dropped_dates.csv", parse_dates=["date"])
dropped = dropped[dropped["date"].between(config.START, config.END)]
print("Dropped pair-dates, 2021-04-19 .. 2026-09-30, by reason:")
display(dropped.groupby(["reason", "pair"]).size().unstack(fill_value=0))
div = pd.read_csv(config.LOG_DIR / "dividend_corrections.csv")
print("Dividend events by action:", div.groupby("action").size().to_dict())
print(f"Spreads with |s| > 10% (kept, reviewed): {len(flags)}")
display(flags[["date", "pair", "spread", "spread_eve"]].round({"spread": 3, "spread_eve": 3}))
plots.plot_spreads(spreads[spreads["date"] <= config.IS_END])
display(Image(str(config.FIGURES_DIR / "spreads_in_sample.png")))
""")
md("""
Every flagged spike is a one-day jump that disappears the next day — news arriving after Hong Kong closed and
before New York opened. That observation drives the next section.
""")

# ---------------------------------------------------------------- Diagnosis
md("""
## 3. Does the gap mean-revert? (in-sample only)

The two markets never trade at the same time. The morning spread compares an HK price that is 5½–6½ hours old
with a fresh US price, so it mixes **a real mispricing** with **timing noise** that HK absorbs at its next open
(and that nobody can trade). Diagnostics:

- AR(1) on the spread → half-life; ADF test.
- A persistence split (ARMA(1,1) = persistent AR(1) + noise) → how much of the variance actually persists.
- The **two-reading spread**: average the morning reading with an evening reading (US close of the previous
  session vs HK open today, where the *US* side is stale). Timing noise comes from different news windows and
  partly cancels; a real mispricing appears in both. This was **pre-registered** as a secondary signal before any backtest.
""")
code("""
tables = analysis.run_analysis(spreads)
mr = tables["mean_reversion"].set_index(["measure", "pair"])
display(mr[["ar1_beta", "ar1_half_life_days", "adf_p", "acf_lag1", "persistent_share", "persistent_half_life_days"]].round(3))
td = tables["timing_diagnostics"].set_index("pair")
display(td[["corr_morning_vs_next_evening", "std_spread", "std_spread_2r", "common_r2_spread"]].round(3))
plots.plot_acf(is_spreads, config.FIGURES_DIR / "spread_acf.png")
display(Image(str(config.FIGURES_DIR / "spread_acf.png")))
corrs = {m: analysis.change_correlations(is_spreads, m) for m in ("spread", "spread_2r")}
plots.plot_corr_heatmaps(corrs, config.FIGURES_DIR / "spread_change_corr.png")
display(Image(str(config.FIGURES_DIR / "spread_change_corr.png")))
""")
md("""
**Reading.** β ≈ −1 (half-life well under a day) looks like fast arbitrage but is mostly timing noise: two
readings of the same pair 12 hours apart barely correlate, and ~half of each pair's variance is shared
China-wide news. ADF rejects a unit root everywhere, which is uninformative for near-white noise. A small
persistent component exists (clearest for Yum China), and the six pairs behave like only ~2 independent bets.
The capture-vs-cost estimate below predicted, *before any backtest*, that costs would roughly match the edge.
""")
code("""
tables["capture_vs_cost"].assign(**{c: lambda d, c=c: d[c] * 100 for c in ["entry_deviation", "expected_capture", "round_trip_cost", "capture_minus_cost"]}).round(2)
""")

# ---------------------------------------------------------------- Strategy
md("""
## 4. Strategy

**Signal:** `z_t = (s_t − mean_L) / std_L`, rolling mean and std over the previous L days **up to t−1**.

**Positions are held in the spread** (+1 = long US / short HK, −1 = short US / long HK):

| Situation | Action |
|---|---|
| flat, k < z ≤ 4 (ADR rich) | enter −1 |
| flat, −4 ≤ z < −k (ADR cheap) | enter +1 |
| −1 and z < exit_z, or +1 and z > −exit_z | exit (reverted) |
| \\|z\\| > 4 | exit, and never enter (news, not mispricing) |
| held H decision days | exit |
| exit day | no re-entry the same day |

**Execution:** signal at the US open on day t → US leg trades at the **US close on t**, HK leg at the **next HK
open**. Each leg buys 50% of pair equity worth of shares and **holds the share count** until exit; each leg is
marked on its own exchange calendar in USD, with dividends. Costs are deducted as they occur (HK stamp duty
0.10%, levies 0.01%, commission 0.03%, half-spread 0.05–0.10% per side; borrow 1% p.a. on the short leg).
The position state machine is one loop over T days with O(1) work per day: **O(T)**.

**Timing, checked by hand.** A 5-day example (enter on day 2, exit on day 4) computed by the engine and by an
independent shares-and-dollars ledger:
""")
code("""
from tests.timing_example import walkthrough, engine_result, dollar_ledger
display(walkthrough().round(4))
eq = engine_result()["equity"]
print(f"Trade P&L: engine {pct(eq.iloc[-1] - 1)}  |  independent ledger {pct(dollar_ledger()['equity'].iloc[-1] / 1e6 - 1)}")
""")
md("""
**Placebo.** Two listings of one perfectly priced stock, observed at the real session times (HK before US), with
**zero** mispricing. The spread *looks* mean-reverting and the strategy trades — but an honest engine must earn
nothing. Computing P&L from spread changes (the common mistake) shows a large fake profit on the same trades.
""")
code("""
from tests.synthetic import simulate_pair
sim = simulate_pair(10_000, sigma=0.01, seed=4)            # ~40 simulated years, no mispricing
pl = backtest.run_pair(sim["spreads"], sim["us"], sim["hk"], sim["fx"], Params(L=40, k=1.5, exit_z=0.0, H=10),
                       start=str(sim["spreads"]["date"].iloc[0].date()), hk_drop=pd.DatetimeIndex([]))
s = sim["spreads"].set_index("date")["spread"]; pos = pl["decisions"]["position"]
naive = (pos.shift(1) * s.diff()).loc[pos.index].fillna(0.0)
print(f"Trades: {len(pl['trades'])} | engine gross Sharpe: {metrics.sharpe(pl['daily']['pair_ret']):+.2f} "
      f"| wrong spread-change P&L Sharpe: {metrics.sharpe(naive):+.2f}")
""")

# ---------------------------------------------------------------- Backtest
md("""
## 5. Backtest: in-sample grid and frozen parameters

36 settings (L ∈ {20, 60, 120}, k ∈ {1.5, 2.0, 2.5}, exit_z ∈ {0, 0.5}, H ∈ {5, 10}) × 2 signals = 72 runs on
in-sample data at 1× costs, every run logged. The setting is chosen on a **plateau** (good neighbourhood), not the
single best cell, then frozen in `config.FROZEN_PARAMS` and committed before the OOS data were opened.
""")
code("""
if RERUN_GRID:
    grid, grid_pairs, secs = backtest.run_grid(spreads, frames, log_path=config.RESULTS_DIR / "grid_log_rerun.csv")
else:
    grid = pd.read_csv(config.RESULTS_DIR / "grid_results.csv"); secs = grid["elapsed_sec"].sum()
print(f"72 runs, {secs:.1f} s in total ({grid['elapsed_sec'].mean():.2f} s per run)")
for sig in ("spread", "spread_2r"):
    g = grid[grid["signal"] == sig]
    print(f"{sig:9s}: gross Sharpe {g['sharpe_gross'].min():+.2f}..{g['sharpe_gross'].max():+.2f}, "
          f"net Sharpe {g['sharpe'].min():+.2f}..{g['sharpe'].max():+.2f}, net > 0 in {(g['sharpe'] > 0).sum()}/36 cells")
vlim = float(np.nanmax(np.abs(grid["sharpe"])))
for sig in ("spread", "spread_2r"):
    plots.plot_sharpe_heatmaps(grid, sig, config.FIGURES_DIR / f"sharpe_grid_{sig}.png", vlim=vlim, mark=config.FROZEN_PARAMS)
    display(Image(str(config.FIGURES_DIR / f"sharpe_grid_{sig}.png")))
scored = pd.concat([metrics.plateau_scores(grid[grid["signal"] == s].reset_index(drop=True)) for s in ("spread", "spread_2r")])
f = config.FROZEN_PARAMS
scored[(scored.L == f["L"]) & (scored.k == f["k"]) & (scored.exit_z == f["exit_z"]) & (scored.H == f["H"])][
    ["signal", "sharpe", "sharpe_gross", "nbhd_mean", "nbhd_min", "n_trades"]].round(3)
""")
md("""
Gross Sharpe is positive in every cell; after costs almost everything is ≈ 0. Net Sharpe improves steadily toward
long windows and high thresholds (fewer, larger-gap trades against a fixed cost per trade). Frozen choice:
**L = 120, k = 2.5, exit_z = 0.5, H = 10** for both signals (outlined), on the plateau — not the H = 5 spike.
""")

# ---------------------------------------------------------------- Results
md("""
## 6. Results: in-sample vs out-of-sample (OOS run once, frozen parameters, costs 1×)
""")
code("""
PERIODS = {"IS": (config.START, config.IS_END), "OOS": (config.OOS_START, config.END)}
runs, rows = {}, []
for per, (a, b) in PERIODS.items():
    for sig in ("spread", "spread_2r"):
        runs[per, sig] = backtest.run_backtest(spreads, frames, FROZEN, signal=sig, start=a, end=b)
        summ, by_pair = metrics.summarize_backtest(runs[per, sig])
        rows.append({"period": per, "signal": sig, **summ})
res = pd.DataFrame(rows).set_index(["signal", "period"])
display(res[["sharpe", "sharpe_gross", "ann_return", "ann_vol", "max_drawdown", "n_trades", "hit_rate", "avg_days_held", "cost_drag"]].round(3))
per_pair = pd.concat({(p, s): metrics.summarize_backtest(runs[p, s])[1]["sharpe"] for p, s in runs}, axis=1).astype(float)
print("Net Sharpe by pair:"); display(per_pair.round(2))
oos_t = {s: res.loc[(s, "OOS"), "sharpe"] * np.sqrt(metrics.years(runs["OOS", s]["portfolio"]["portfolio_ret"])) for s in ("spread", "spread_2r")}
print("OOS t-statistic (Sharpe x sqrt(years)):", ", ".join(f"{k} {v:.2f}" for k, v in oos_t.items()))
plots.plot_equity_is_oos({plots.MEASURE_LABELS[s]: (runs["IS", s]["portfolio"]["portfolio_ret"], runs["OOS", s]["portfolio"]["portfolio_ret"])
                          for s in ("spread", "spread_2r")}, config.FIGURES_DIR / "equity_is_oos.png")
display(Image(str(config.FIGURES_DIR / "equity_is_oos.png")))
""")
md("""
**Is the OOS profit real?** Look at where it comes from: how much the best three trades contribute, and what is
left without them.
""")
code("""
conc = {s: metrics.concentration(metrics.trade_table(runs["OOS", s])) for s in ("spread", "spread_2r")}
display(pd.DataFrame(conc).T.round(4))
print("Largest OOS trades (two-reading):")
metrics.trade_table(runs["OOS", "spread_2r"]).head(5).round(4)
""")
md("""
**Known data gaps.** Yahoo has no US dividend matching some HK ex-dates (logged as "unmatched" in the dividend
logs). The prices are real, so the signal is unaffected; the P&L would only miss the cash dividend if a position
was open across the ex-date. Check:
""")
code("""
def open_over(run, pair, ex):
    d = run["pairs"][pair]["daily"]
    held = d.loc[:pd.Timestamp(ex) - pd.Timedelta(days=1)].iloc[-1]
    return bool(held["us_pos"] != 0 or held["hk_pos"] != 0)
gaps = pd.read_csv(config.LOG_DIR / "dividend_corrections.csv").query("status != 'matched'")
gaps = gaps[gaps["hk_ex"].fillna(gaps["us_ex"]) >= config.START]
pd.DataFrame([{"pair": g.pair, "ex_date": g.hk_ex if isinstance(g.hk_ex, str) else g.us_ex, "status": g.status,
               **{f"position open ({s})": open_over(runs["OOS" if (g.hk_ex or "") >= config.OOS_START else "IS", s], g.pair,
                                                     g.hk_ex if isinstance(g.hk_ex, str) else g.us_ex) for s in ("spread", "spread_2r")}}
              for g in gaps.itertuples()])
""")
md("""
The OOS gains arrive in jumps on market shocks (Sep–Oct 2024 China stimulus rally, Apr 2025 tariff shock), are
concentrated in a handful of trades, have t-statistics below 1, and pair-level results flip sign between periods.
""")

# ---------------------------------------------------------------- Sensitivity
md("""
## 7. Sensitivity: costs, break-even and an execution upper bound

Cost multipliers 0 / 0.5 / 1 / 2; the multiplier at which net Sharpe hits zero; and an **upper bound that is not
tradable** — the US leg filled at the US open (the same moment the signal is computed).
""")
code("""
sens = []
for per, (a, b) in PERIODS.items():
    for sig in ("spread", "spread_2r"):
        row = {"period": per, "signal": sig}
        for m in config.COST_MULTIPLIERS:
            row[f"Sharpe @{m:g}x"] = metrics.sharpe(backtest.run_backtest(spreads, frames, FROZEN, signal=sig, start=a, end=b,
                                                                           cost_multiplier=m)["portfolio"]["portfolio_ret"])
        row["break-even x"] = backtest.break_even_multiplier(spreads, frames, FROZEN, signal=sig, start=a, end=b)
        row["upper bound @1x (US at open)"] = metrics.sharpe(backtest.run_backtest(spreads, frames, FROZEN, signal=sig, start=a, end=b,
                                                                                    us_exec="open")["portfolio"]["portfolio_ret"])
        sens.append(row)
sens = pd.DataFrame(sens).set_index(["signal", "period"]); sens.round(2)
""")
md("""
Costs only modestly above the modelled level erase the edge (see the break-even column), and the modelled costs are optimistic (no market
impact, perfect auction fills, constant spreads; HK stamp duty was actually 0.13% from Aug 2021 to Nov 2023).
Filling the US leg at the open would help the morning signal a lot — most of its edge is gone by the US close.
""")

# ---------------------------------------------------------------- Stock Connect
md("""
## 8. Stock Connect

Southbound Stock Connect lets mainland investors buy only the HK leg. If segmentation matters, Alibaba's inclusion
(2024-09-10) should make HK relatively expensive (spread falls) or change how the gap behaves. Pre-registered
design: ±120 trading days before/after, compared with pairs whose Connect status did **not** change in that
window (Baidu, JD, Trip.com, NetEase out throughout; Yum China in throughout). NetEase (2026-06-30) and Baidu
(2026-09-07) are too recent: plots only.
""")
code("""
sp_end = spreads[spreads["date"] <= config.END]
study = analysis.connect_event_study(sp_end, config.CONNECT_FORMAL, config.CONNECT_EVENTS[config.CONNECT_FORMAL], config.CONNECT_CONTROLS)
vs = pd.concat([analysis.difference_vs_controls(study, st, config.CONNECT_STATUS_IN_WINDOW) for st in ("mean", "std", "acf_lag1")])
display(vs.round(4))
plots.plot_connect_event(sp_end, "Alibaba", config.CONNECT_EVENTS["Alibaba"], config.FIGURES_DIR / "connect_alibaba.png",
                         config.CONNECT_CONTROLS, config.CONNECT_WINDOW, config.END, " (formal test, ±120 trading days)")
display(Image(str(config.FIGURES_DIR / "connect_alibaba.png")))
""")
md("""
Alibaba's change in spread level sits inside the control pairs' range: no HK premium appeared. Its volatility
rose most, but so did Yum China's (already in Connect), and the window contains the stimulus rally two weeks after
inclusion — confounded, not attributable to Connect.
""")

# ---------------------------------------------------------------- Out-of-universe
md("""
## 9. Robustness: seven dual listings never used before

Pre-registered before any OOS result: same frozen parameters, code and costs, applied to other HK–US dual
listings chosen by rule — exclude ratio/ticker/domicile changes in the sample (Citi programme milestones, Yahoo
split records, company sources) and HK lines trading < HKD 20m/day; half-spread by HK liquidity tier.
""")
code("""
sel = pd.read_csv(config.LOG_DIR / "holdout_selection.csv")
display(sel.assign(half_spread_pct=sel["half_spread"] * 100)[["name", "us", "hk", "ratio", "median_hk_value_hkd_m", "half_spread_pct",
                                                               "implied_ratio_median", "decision", "reasons"]].round(2).fillna(""))
H_PAIRS, H_SPREADS = holdout.universe()
h_frames = holdout.load_frames(H_PAIRS)
h_spreads = spread.build_spreads(clean.clean_all(h_frames, pairs=H_PAIRS, log_prefix="holdout_"))
h_rows, h_runs = [], {}
for per, (a, b) in PERIODS.items():
    for sig in ("spread", "spread_2r"):
        for case, m, br in [("gross", 0.0, None), ("net, borrow 1%", 1.0, None), ("net, borrow 5%", 1.0, config.HOLDOUT_BORROW_SENSITIVITY)]:
            r = backtest.run_backtest(h_spreads, h_frames, FROZEN, signal=sig, start=a, end=b, cost_multiplier=m,
                                      pairs=H_PAIRS, half_spreads=H_SPREADS, borrow_annual=br)
            h_runs[per, sig, case] = r
            h_rows.append({"period": per, "signal": sig, "case": case, "sharpe": metrics.sharpe(r["portfolio"]["portfolio_ret"])})
h_res = pd.DataFrame(h_rows).pivot_table(index=["signal", "period"], columns="case", values="sharpe")
display(h_res.round(2))
hg = pd.read_csv(config.LOG_DIR / "holdout_dividend_corrections.csv").query("status != 'matched'")
for g in hg.itertuples():
    ex = g.hk_ex if isinstance(g.hk_ex, str) else g.us_ex
    per = "OOS" if ex >= config.OOS_START else "IS"
    print(f"Data gap: {g.pair} {g.status} on {ex}; position open across it:",
          {s: open_over(h_runs[per, s, "net, borrow 1%"], g.pair, ex) for s in ("spread", "spread_2r")})
plots.plot_equity_is_oos({plots.MEASURE_LABELS[s]: (h_runs["IS", s, "net, borrow 1%"]["portfolio"]["portfolio_ret"],
                                                    h_runs["OOS", s, "net, borrow 1%"]["portfolio"]["portfolio_ret"]) for s in ("spread", "spread_2r")},
                         config.FIGURES_DIR / "holdout_equity_is_oos.png",
                         title="Out-of-universe 7-pair portfolio, frozen parameters, costs 1x")
display(Image(str(config.FIGURES_DIR / "holdout_equity_is_oos.png")))
""")

# ---------------------------------------------------------------- Conclusion
md("""
## 10. Conclusion

The summary below is printed from the results above, so its numbers cannot drift from the code.
""")
code("""
g = lambda sig, per: res.loc[(sig, per)]
print("MAIN BASKET (6 pairs), frozen parameters, costs 1x")
for sig, label in (("spread", "morning (primary)"), ("spread_2r", "two-reading (pre-registered secondary)")):
    print(f"  {label:40s} net Sharpe IS {g(sig, 'IS')['sharpe']:+.2f} -> OOS {g(sig, 'OOS')['sharpe']:+.2f} "
          f"(gross OOS {g(sig, 'OOS')['sharpe_gross']:+.2f}; OOS t = {oos_t[sig]:.2f}; OOS annual return {pct(g(sig, 'OOS')['ann_return'])}; "
          f"break-even costs {sens.loc[(sig, 'OOS'), 'break-even x']:.2f}x)")
    c = conc[sig]; print(f"  {'':40s} top 3 OOS trades = {c['top3_share']:.0%} of profit; without them {pct(c['total_without_top3'])}")
print("OUT-OF-UNIVERSE (7 unseen pairs), costs 1x, borrow 1%")
for sig in ("spread", "spread_2r"):
    print(f"  {sig:40s} net Sharpe IS {h_res.loc[(sig, 'IS'), 'net, borrow 1%']:+.2f}, OOS {h_res.loc[(sig, 'OOS'), 'net, borrow 1%']:+.2f} "
          f"(gross OOS {h_res.loc[(sig, 'OOS'), 'gross']:+.2f})")
alb = vs[(vs["stat"] == "mean")].set_index("measure")
print("STOCK CONNECT (Alibaba): change in spread level vs controls (rank among 6):",
      {m: alb.loc[m, "treated_rank_of_n"] for m in alb.index})
""")
md("""
**Verdict.**

1. The ADR–HK gap is mostly **timing noise** from non-overlapping trading hours, with a small genuine mispricing on
   top. The placebo test shows the engine does not turn the noise into profit.
2. That mispricing produces a **real gross edge** — in the main basket and in seven unseen pairs — but **realistic
   costs absorb it**; break-even sits only modestly above modelled costs, and those costs are optimistic.
3. Out-of-sample net Sharpe ratios are **not distinguishable from zero**, profits are concentrated in a few
   shock-driven trades, and the best main-basket result does not replicate in the unseen pairs.
   **Falsification test (c) fails: the hypothesis is not supported after costs.**
4. **Stock Connect** inclusion left no detectable mark on Alibaba's spread.

**Limitations.** Daily data only (no intraday fills or market impact); costs held constant (HK stamp duty was 0.13%
until Nov 2023); borrow assumed available at the open; ~2.7 years per period gives Sharpe standard errors of ~0.6;
one formal Connect event. **What would change the answer:** materially lower costs (e.g. stamp-duty-exempt market
makers) or intraday execution closer to the moment the gap is measured.
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, sys.argv[1])
print("cells:", len(C))
