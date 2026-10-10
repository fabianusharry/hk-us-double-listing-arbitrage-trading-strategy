"""Out-of-universe candidate selection (pre-registered rule: CLAUDE.md §6, Session 6b).

Three data checks on the yfinance snapshot in data/raw/holdout/:
  1. Liquidity  - median daily HK value traded (close x volume, HKD) on IN-SAMPLE days
                  only, max(START, first HK date) .. IS_END -> spread tier or exclusion.
  2. Listing    - first HK trading date must be on or before HOLDOUT_HK_LISTED_BY.
  3. Ratio      - implied ratio N_hat = US close x FX / HK close (in-sample, quarterly
                  medians) should be flat at the verified ratio; and Yahoo must record no
                  ADR split (= ratio change) on the US ticker between START and END.
The structural verdicts from Citi's programme milestones (config.HOLDOUT_STRUCTURAL) are
combined with these; every decision is logged with its reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def spread_tier(median_hkd: float) -> float | None:
    """Half-spread for a median daily HK value traded (HKD); None = excluded as too thin."""
    if median_hkd < config.HOLDOUT_EXCLUDE_BELOW_HKD:
        return None
    for floor, half_spread in config.HOLDOUT_SPREAD_TIERS:
        if median_hkd >= floor:
            return half_spread
    return None


def median_hk_value(hk: pd.DataFrame, start: str = config.START, end: str = config.IS_END) -> float:
    """Median of close x volume (HKD) over in-sample days from max(start, first HK date)."""
    h = hk.loc[max(pd.Timestamp(start), hk.index.min()):end]
    return float((h["close"] * h["volume"]).median())


def implied_ratio(us: pd.DataFrame, hk: pd.DataFrame, fx_close: pd.Series,
                  start: str = config.START, end: str = config.IS_END) -> pd.Series:
    """Daily N_hat = US close (USD) x FX (HKD per USD) / HK close (HKD) on dates both traded.

    The two closes are ~12 hours apart, so single days are noisy (a few %); medians over
    a quarter are stable and reveal a ratio change as a step.
    """
    idx = us.index.intersection(hk.index)
    idx = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
    fx = fx_close.reindex(fx_close.index.union(idx)).ffill().reindex(idx)
    return us.loc[idx, "close"] * fx / hk.loc[idx, "close"]


def ratio_check(n_hat: pd.Series, ratio: float, tol: float = 0.10) -> dict:
    """Quarterly medians of N_hat vs the verified ratio.

    Passes if every quarter's median is within tol (relative) of the ratio. The tolerance
    absorbs the ~12-hour gap between the closes and the ADR premium/discount; a genuine
    ratio change (e.g. 5 -> 15) is a step of 100%+ and cannot hide inside it.
    """
    q = n_hat.groupby(n_hat.index.to_period("Q")).median()
    dev = (q / ratio - 1).abs()
    return {"implied_ratio_median": float(n_hat.median()), "quarter_min": float(q.min()),
            "quarter_max": float(q.max()), "max_quarter_dev": float(dev.max()),
            "ratio_ok": bool((dev <= tol).all()), "worst_quarter": str(dev.idxmax()) if len(dev) else ""}


def us_splits(us: pd.DataFrame, start: str = config.START, end: str = config.END) -> str:
    """Yahoo 'splits' on the US ticker in [start, end] (an ADR ratio change shows up here)."""
    s = us.loc[start:end, "splits"]
    s = s[s.fillna(0) != 0]
    return "; ".join(f"{d.date()}:{v:g}" for d, v in s.items())


def select(frames: dict[str, pd.DataFrame], fx_close: pd.Series) -> pd.DataFrame:
    """Apply the pre-registered rule to every candidate; one row per candidate with the reason."""
    rows = []
    for name, (us_t, hk_t, ratio) in config.HOLDOUT_CANDIDATES.items():
        us, hk = frames[us_t], frames[hk_t]
        first_hk = hk.index.min()
        med = median_hk_value(hk)
        tier = spread_tier(med)
        rc = ratio_check(implied_ratio(us, hk, fx_close), ratio)
        splits = us_splits(us)
        structural = config.HOLDOUT_STRUCTURAL[name]

        reasons = []
        if structural.startswith("exclude"):
            reasons.append(f"structural (Citi): {structural[len('exclude: '):]}")
        if not config.HOLDOUT_RATIO_VERIFIED[name]:
            reasons.append("ratio not verified")
        if first_hk > pd.Timestamp(config.HOLDOUT_HK_LISTED_BY):
            reasons.append(f"HK listing {first_hk.date()} after {config.HOLDOUT_HK_LISTED_BY}")
        if splits:
            reasons.append(f"US ADR split(s) in sample: {splits}")
        if not rc["ratio_ok"]:
            reasons.append(f"implied ratio off by {rc['max_quarter_dev']:.0%} in {rc['worst_quarter']}")
        if tier is None:
            reasons.append(f"too thin: median HK value HKD {med / 1e6:.1f}m < {config.HOLDOUT_EXCLUDE_BELOW_HKD / 1e6:.0f}m")

        rows.append({"name": name, "us": us_t, "hk": hk_t, "ratio": ratio, "first_hk_date": first_hk.date(),
                     "median_hk_value_hkd_m": med / 1e6, "half_spread": tier, **rc, "us_splits_in_sample": splits,
                     "structural_citi": structural, "decision": "include" if not reasons else "exclude",
                     "reasons": "; ".join(reasons)})
    return pd.DataFrame(rows)


def universe(selection_path=None) -> tuple[dict, dict]:
    """(pairs, half_spreads) of the INCLUDED out-of-universe pairs, read from the selection log."""
    sel = pd.read_csv(selection_path or config.LOG_DIR / "holdout_selection.csv")
    inc = sel[sel["decision"] == "include"]
    pairs = {r["name"]: (r["us"], r["hk"], int(r["ratio"])) for _, r in inc.iterrows()}
    return pairs, {r["name"]: float(r["half_spread"]) for _, r in inc.iterrows()}


def load_frames(pairs: dict) -> dict[str, pd.DataFrame]:
    """Snapshot frames for the given holdout pairs plus the main FX series (offline)."""
    from src.data import load_raw
    frames = load_raw([t for us, hk, _ in pairs.values() for t in (us, hk)], raw_dir=config.RAW_DIR / "holdout")
    frames[config.FX_TICKER] = load_raw([config.FX_TICKER])[config.FX_TICKER]
    return frames
