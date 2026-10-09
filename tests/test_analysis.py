"""Tests for src/analysis.py on simulated series with known properties."""

import numpy as np
import pandas as pd
import pytest

import config
from src import analysis
from src.data import OOSAccessError

RNG = np.random.default_rng(42)


def ar1_series(phi: float, n: int, sigma: float = 1.0, rng=RNG) -> np.ndarray:
    x = np.zeros(n)
    eps = rng.normal(scale=sigma, size=n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def test_half_life_formula_and_edge_cases():
    assert np.isclose(analysis.half_life(-0.5), 1.0)  # halves every day
    assert analysis.half_life(0.0) == float("inf")
    assert np.isnan(analysis.half_life(-1.0))


def test_ar1_recovers_known_coefficient():
    s = pd.Series(ar1_series(0.9, 20_000))
    out = analysis.ar1(s)
    assert abs(out["ar1_beta"] - (-0.1)) < 0.01
    assert abs(out["ar1_half_life_days"] - np.log(2) / -np.log(0.9)) < 0.8


def test_adf_separates_random_walk_from_stationary():
    rw = pd.Series(np.cumsum(RNG.normal(size=2000)))
    stationary = pd.Series(ar1_series(0.5, 2000))
    assert analysis.adf(rw)["adf_p"] > 0.05
    assert analysis.adf(stationary)["adf_p"] < 0.01


def test_persistence_split_recovers_ar1_plus_noise():
    # persistent AR(1) with variance 1 plus white noise with variance 1 -> share 0.5
    phi, n = 0.8, 20_000
    x = ar1_series(phi, n, sigma=np.sqrt(1 - phi**2))
    s = pd.Series(x + RNG.normal(size=n))
    out = analysis.persistence_split(s)
    assert abs(out["arma_phi"] - phi) < 0.05
    assert abs(out["persistent_share"] - 0.5) < 0.1


def test_summary_stats_shares():
    out = analysis.summary_stats(pd.Series([0.002, -0.007, 0.02, -0.003]))
    assert out["share_abs_gt_0.5pct"] == 0.5 and out["share_abs_gt_1pct"] == 0.25


def test_effective_bets_extremes():
    assert np.isclose(analysis.effective_bets(pd.DataFrame(np.eye(6))), 6.0)
    assert np.isclose(analysis.effective_bets(pd.DataFrame(np.ones((6, 6)))), 1.0)


def _six_pair_frame(persistent_scale: float, n: int = 3000) -> pd.DataFrame:
    """Spreads for all six config pairs: persistent AR(1) part + independent timing noise per reading."""
    dates = pd.bdate_range("2021-05-03", periods=n)
    rows = []
    for name in config.PAIRS:
        x = persistent_scale * ar1_series(0.95, n + 1)
        morning = x[:-1] + RNG.normal(size=n)
        evening = x + RNG.normal(size=n + 1)  # evening(t+1) sees the same mispricing as morning(t)
        rows.append(pd.DataFrame({"date": dates, "pair": name, "spread": morning,
                                  "spread_eve": np.r_[np.nan, evening[1:-1]], "n_readings": 2}))
    df = pd.concat(rows, ignore_index=True)
    df["spread_2r"] = df[["spread", "spread_eve"]].mean(axis=1)
    return df


def test_timing_diagnostic_zero_for_pure_noise_high_for_persistent_mispricing():
    noise_only = analysis.timing_diagnostics(_six_pair_frame(0.0))
    mostly_mispricing = analysis.timing_diagnostics(_six_pair_frame(3.0))
    assert noise_only["corr_morning_vs_next_evening"].abs().max() < 0.1
    assert mostly_mispricing["corr_morning_vs_next_evening"].min() > 0.7


def test_change_correlations_identical_pairs_are_perfectly_correlated():
    df = _six_pair_frame(1.0, n=200)
    base = df.loc[df["pair"] == "Alibaba", "spread"].to_numpy()
    df.loc[df["pair"] == "JD", "spread"] = base
    corr = analysis.change_correlations(df, "spread")
    assert np.isclose(corr.loc["Alibaba", "JD"], 1.0)


def test_in_sample_refuses_oos_rows_when_locked(monkeypatch):
    monkeypatch.setattr(config, "ALLOW_OOS", False)
    df = pd.DataFrame({"date": pd.to_datetime(["2023-12-29", "2024-01-02"]), "pair": "Alibaba", "spread": 0.0})
    with pytest.raises(OOSAccessError):
        analysis.in_sample(df)


def test_in_sample_drops_warm_up_rows():
    df = pd.DataFrame({"date": pd.to_datetime(["2021-01-04", config.START, "2023-12-29"]),
                       "pair": "Alibaba", "spread": 0.0})
    assert analysis.in_sample(df)["date"].min() == pd.Timestamp(config.START)


def test_round_trip_cost_matches_config_by_hand():
    # Alibaba: HK side 0.0010+0.0001+0.0003+0.0005, US side 0.0003+0.0005 -> 2 * 0.0027
    assert np.isclose(analysis.round_trip_cost("Alibaba"), 0.0054)
    assert np.isclose(analysis.round_trip_cost("NetEase"), 0.0074)


def test_capture_vs_cost_arithmetic():
    mr = pd.DataFrame({"pair": ["JD"], "measure": ["spread"], "acf_lag1": [0.25]})
    sm_ = pd.DataFrame({"pair": ["JD"], "measure": ["spread"], "std": [0.01]})
    out = analysis.capture_vs_cost(mr, sm_).iloc[0]
    assert np.isclose(out["expected_capture"], 0.25 * 2 * 0.01)
    assert np.isclose(out["capture_minus_cost"], 0.005 - 0.0054)


def _event_frame(shift_treated: float) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=300)
    rows = []
    for i, pair in enumerate(["T", "C1", "C2"]):
        s = RNG.normal(0, 0.01, len(dates))
        if pair == "T":
            s[dates >= pd.Timestamp("2024-07-01")] += shift_treated
        rows.append(pd.DataFrame({"date": dates, "pair": pair, "spread": s, "spread_2r": s}))
    return pd.concat(rows, ignore_index=True)


def test_event_windows_use_aligned_days():
    dates = pd.bdate_range("2024-01-01", periods=50)
    start, ev, end = analysis.event_windows(dates, "2024-01-20", 5)  # a Saturday -> next aligned day
    assert ev == pd.Timestamp("2024-01-22") and start == dates[dates.get_loc(ev) - 5]
    assert end == dates[dates.get_loc(ev) + 4]


def test_connect_study_detects_a_level_shift_in_the_treated_pair_only():
    study = analysis.connect_event_study(_event_frame(-0.02), "T", "2024-07-01", ["C1", "C2"], window=100)
    d = analysis.difference_vs_controls(study, "mean").set_index("measure").loc["spread"]
    assert abs(d["treated_change"] - (-0.02)) < 0.005 and abs(d["controls_mean_change"]) < 0.005
    assert d["treated_rank_of_n"] == "3 of 3"   # most negative change -> ranked last
    assert (study.loc[study.role == "treated", "before_n"] == 100).all()
