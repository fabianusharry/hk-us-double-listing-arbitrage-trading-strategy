"""Descriptive analysis on in-sample data: summary stats, AR(1) half-life, ADF test,
persistence split, timing diagnostics, cross-pair correlation of spread changes.

All functions take the long-format output of spread.build_spreads(). Nothing here
makes trading decisions, so using the whole in-sample period at once is fine
(no look-ahead question) - but OOS rows are refused while config.ALLOW_OOS is False.

Spread measures analysed:
  spread     morning reading (US open t vs HK close t)  - primary signal
  spread_2r  two-reading average                         - pre-registered secondary
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

import config
from src.data import OOSAccessError

MEASURES = ["spread", "spread_2r"]


def in_sample(spreads: pd.DataFrame) -> pd.DataFrame:
    """Rows inside the backtest's in-sample window [config.START, config.IS_END].

    Raises OOSAccessError if the input contains OOS rows while ALLOW_OOS is False
    (defence in depth: load_raw() should already have stopped them).
    """
    if not config.ALLOW_OOS and (spreads["date"] >= pd.Timestamp(config.OOS_START)).any():
        raise OOSAccessError("analysis input contains out-of-sample rows while ALLOW_OOS is False")
    mask = (spreads["date"] >= pd.Timestamp(config.START)) & (spreads["date"] <= pd.Timestamp(config.IS_END))
    return spreads.loc[mask]


def wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Pivot one measure to a date x pair table (NaN where a pair has no row)."""
    return df.pivot(index="date", columns="pair", values=col)[list(config.PAIRS)]


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summary_stats(s: pd.Series) -> dict:
    """Distribution of one spread series (in log units, i.e. 0.01 = ~1%).

    Output: n, mean, std, percentiles, and the share of days with |s| above
    0.5% and 1% (how often the gap is big enough to matter after costs).
    """
    s = s.dropna()
    q = s.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "n": len(s), "mean": s.mean(), "std": s.std(),
        "p01": q[0.01], "p05": q[0.05], "p25": q[0.25], "median": q[0.5],
        "p75": q[0.75], "p95": q[0.95], "p99": q[0.99],
        "share_abs_gt_0.5pct": (s.abs() > 0.005).mean(),
        "share_abs_gt_1pct": (s.abs() > 0.01).mean(),
    }


# ---------------------------------------------------------------------------
# Mean reversion
# ---------------------------------------------------------------------------
def half_life(beta: float) -> float:
    """Half-life in trading days implied by Δs(t) = α + β s(t−1) + ε.

    The deviation shrinks by a factor (1 + β) per day, so it halves after
    h = −ln 2 / ln(1 + β) days. Returns inf if β ≥ 0 (no reversion) and NaN if
    β ≤ −1 (1 + β ≤ 0: the series overshoots and flips sign each day, so a
    "half-life" is not defined).
    """
    if beta >= 0:
        return float("inf")
    if beta <= -1:
        return float("nan")
    return -np.log(2) / np.log(1 + beta)


def ar1(s: pd.Series) -> dict:
    """OLS of Δs(t) = α + β·s(t−1) + ε.

    t−1 is the previous aligned trading day of the pair. β near 0 = persistent
    (slow reversion); β near −1 = no memory (each day starts afresh).
    Note: under the null of a unit root the t-stat of β is not t-distributed,
    which is why the ADF test below uses its own critical values.
    """
    s = s.dropna()
    y = s.diff().iloc[1:]
    x = sm.add_constant(s.shift(1).iloc[1:].rename("s_lag"))
    fit = sm.OLS(y, x).fit()
    beta = fit.params["s_lag"]
    return {"ar1_alpha": fit.params["const"], "ar1_beta": beta, "ar1_beta_se": fit.bse["s_lag"],
            "ar1_beta_t": fit.tvalues["s_lag"], "ar1_half_life_days": half_life(beta), "ar1_n": int(fit.nobs)}


def adf(s: pd.Series) -> dict:
    """Augmented Dickey-Fuller test with a constant; lags chosen by AIC.

    H0: unit root (the spread wanders and never reverts). A small p-value rejects
    H0, i.e. the spread is stationary / mean-reverting.
    """
    stat, p, lags, nobs, crit, _ = adfuller(s.dropna(), regression="c", autolag="AIC", result_object=False)
    return {"adf_stat": stat, "adf_p": p, "adf_lags": lags, "adf_crit_5pct": crit["5%"]}


def persistence_split(s: pd.Series) -> dict:
    """Split the spread into a persistent part and day-to-day noise.

    Model: s(t) = x(t) + e(t), where x is AR(1) with coefficient φ (a mispricing
    that decays slowly) and e is white noise (timing noise that is gone the next
    day). This model is an ARMA(1,1); we fit it by maximum likelihood.
    Under this model the lag-1 autocorrelation equals φ × (persistent share of
    variance), so persistent_share = ρ1 / φ.
    Returned as NaN when no persistent component is detected (φ ≤ 0, or the
    ratio falls outside (0, 1]). Caveat: when the persistent part is small, φ
    is weakly identified; read the numbers as indicative.
    """
    s = s.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ARIMA(s.to_numpy(), order=(1, 0, 1)).fit()
    phi, theta = float(fit.params[1]), float(fit.params[2])
    rho1 = s.autocorr(lag=1)
    share = rho1 / phi if phi > 0 else float("nan")
    if not (0 < share <= 1):
        share = float("nan")
    hl = -np.log(2) / np.log(phi) if 0 < phi < 1 else float("nan")
    return {"acf_lag1": rho1, "arma_phi": phi, "arma_theta": theta,
            "persistent_half_life_days": hl, "persistent_share": share}


def mean_reversion_table(df: pd.DataFrame) -> pd.DataFrame:
    """AR(1), ADF and persistence split for every pair and measure."""
    rows = []
    for measure in MEASURES:
        for pair, g in df.groupby("pair", sort=False):
            s = g.set_index("date")[measure]
            rows.append({"pair": pair, "measure": measure, **ar1(s), **adf(s), **persistence_split(s)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Timing diagnostics (non-synchronous trading)
# ---------------------------------------------------------------------------
def timing_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """How much of the spread is timing noise rather than a shared mispricing?

    - corr_morning_vs_next_evening: corr( spread(t), spread_eve(t+1) ). The
      evening reading on t+1 is US close t vs HK open t+1, i.e. the same pair
      measured ~12 hours later with the OTHER leg stale. A persistent mispricing
      would show up in both (high correlation); pure timing noise would not.
      DIAGNOSTIC ONLY - it uses t+1 data and must never feed a signal.
    - common_r2_{measure}: share of a pair's spread variance explained by the
      cross-pair average on the same date (market-wide news in the time gap).
    """
    rows = []
    commons = {m: wide(df, m).dropna().mean(axis=1) for m in MEASURES}
    for pair, g in df.groupby("pair", sort=False):
        g = g.set_index("date")
        row = {"pair": pair,
               "corr_morning_vs_next_evening": g["spread"].corr(g["spread_eve"].shift(-1)),
               "std_spread": g["spread"].std(), "std_spread_eve": g["spread_eve"].std(),
               "std_spread_2r": g["spread_2r"].std(),
               "days_one_reading": int((g["n_readings"] < 2).sum())}
        for m in MEASURES:
            joined = pd.concat([g[m], commons[m]], axis=1, join="inner").dropna()
            row[f"common_r2_{m}"] = joined.iloc[:, 0].corr(joined.iloc[:, 1]) ** 2
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-pair correlation
# ---------------------------------------------------------------------------
def change_correlations(df: pd.DataFrame, measure: str = "spread") -> pd.DataFrame:
    """Correlation matrix of daily spread changes across pairs.

    Changes are taken within each pair (vs its previous aligned day), then
    matched on dates where all six pairs have a change. High correlation means
    the six pairs are fewer independent bets than they look.
    """
    changes = wide(df.assign(d=df.groupby("pair")[measure].diff()), "d")
    return changes.dropna().corr()


def effective_bets(corr: pd.DataFrame) -> float:
    """Effective number of independent bets for an equal-weight portfolio.

    N_eff = N / (1 + (N − 1)·ρ̄), with ρ̄ the average off-diagonal correlation.
    """
    n = len(corr)
    rho = corr.to_numpy()[np.triu_indices(n, 1)].mean()
    return n / (1 + (n - 1) * rho)


# ---------------------------------------------------------------------------
# Back-of-envelope: expected capture vs round-trip cost
# ---------------------------------------------------------------------------
def round_trip_cost(pair: str) -> float:
    """Enter + exit, both legs, as a fraction of one leg's notional (config costs at 1x).

    Delegates to costs.round_trip_cost so the cost formula lives in one place.
    """
    from src.costs import round_trip_cost as rtc
    return rtc(pair, 1.0)


def capture_vs_cost(mean_rev: pd.DataFrame, summary: pd.DataFrame, entry_z: float = 2.0) -> pd.DataFrame:
    """Rough expected gross capture per trade vs round-trip cost.

    Under "persistent AR(1) + noise", the best forecast of tomorrow's spread is
    rho1 * s(t). Our HK leg executes at the t+1 open, so roughly rho1 * s(t) is the
    deviation still left to capture after entry. For an entry at entry_z
    standard deviations: capture ~= rho1 * entry_z * std.
    Ignores borrow, rolling vs full-sample std, and any slower persistence.
    """
    m = mean_rev.merge(summary[["pair", "measure", "std"]], on=["pair", "measure"])
    out = pd.DataFrame({"pair": m["pair"], "measure": m["measure"], "acf_lag1": m["acf_lag1"],
                        "entry_deviation": entry_z * m["std"]})
    out["expected_capture"] = out["acf_lag1"] * out["entry_deviation"]
    out["round_trip_cost"] = out["pair"].map(round_trip_cost)
    out["capture_minus_cost"] = out["expected_capture"] - out["round_trip_cost"]
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_analysis(spreads: pd.DataFrame, out_dir: Path = config.RESULTS_DIR) -> dict[str, pd.DataFrame]:
    """Compute every table on in-sample data and save them as CSV in out_dir."""
    df = in_sample(spreads)
    summary = pd.DataFrame([{"pair": p, "measure": m, **summary_stats(g[m])}
                            for m in MEASURES for p, g in df.groupby("pair", sort=False)])
    mean_rev = mean_reversion_table(df)
    tables = {
        "spread_summary": summary,
        "mean_reversion": mean_rev,
        "timing_diagnostics": timing_diagnostics(df),
        "capture_vs_cost": capture_vs_cost(mean_rev, summary),
    }
    for m in MEASURES:
        tables[f"change_corr_{m}"] = change_correlations(df, m)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv", index=name.startswith("change_corr"))
    return tables
