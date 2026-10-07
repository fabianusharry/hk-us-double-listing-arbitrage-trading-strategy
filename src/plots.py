"""Matplotlib figures (titled, labelled with units), saved as PNG to figures/."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

SERIES = "#2a78d6"   # single-series line colour
INK = "#3d3d3a"      # titles and labels
MUTED = "#8a8980"    # axes, ticks, reference lines
GRID = "#e6e5df"
WARMUP = "#f0efea"   # shading for rolling-window warm-up


def _style(ax: plt.Axes) -> None:
    """Recessive axes: no top/right spines, light horizontal grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_spreads(spreads: pd.DataFrame, path: Path = config.FIGURES_DIR / "spreads_in_sample.png") -> Path:
    """Six spreads as small multiples on a shared y-axis (in % of parity).

    Input: build_spreads() output (long format: date, pair, spread, in_backtest).
    Warm-up rows (before config.START) are shaded; dashed lines mark the
    +/- config.SPREAD_FLAG_ABS review threshold.
    """
    pairs = list(config.PAIRS)
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.5), sharex=True, sharey=True)
    start = pd.Timestamp(config.START)
    flag = config.SPREAD_FLAG_ABS * 100

    for ax, name in zip(axes.flat, pairs):
        s = spreads[spreads["pair"] == name].set_index("date")["spread"] * 100
        us, hk, n = config.PAIRS[name]
        if s.index.min() < start:
            ax.axvspan(s.index.min(), start, color=WARMUP, linewidth=0)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        for y in (flag, -flag):
            ax.axhline(y, color=MUTED, linewidth=0.6, linestyle=(0, (3, 3)))
        ax.plot(s.index, s.to_numpy(), color=SERIES, linewidth=0.9)
        ax.set_title(f"{name}  ({us} vs {hk}, {n}:1)", fontsize=10, color=INK, loc="left")
        _style(ax)

    for ax in axes[:, 0]:
        ax.set_ylabel("Spread (% of HK parity)", fontsize=9, color=INK)
    for ax in axes[-1, :]:
        ax.set_xlabel("Date", fontsize=9, color=INK)

    fig.suptitle("ADR vs HK spread, in-sample: ln(US open) − ln(N × HK close / FX)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.945,
             "Positive = ADR rich. Shaded = warm-up before 2021-04-19 (not traded). "
             f"Dashed = ±{flag:.0f}% review threshold.",
             fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


SERIES_2 = "#eb6834"  # second series (two-reading average)
MEASURE_LABELS = {"spread": "Morning reading (US open t vs HK close t)",
                  "spread_2r": "Two-reading average"}


def plot_spread_bands(df: pd.DataFrame, measure: str, path: Path) -> Path:
    """In-sample spread per pair with its in-sample mean ± 2 std (descriptive, not a signal).

    Input: analysis.in_sample() rows; measure = 'spread' or 'spread_2r'.
    """
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.5), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, config.PAIRS):
        s = df[df["pair"] == name].set_index("date")[measure] * 100
        mu, sd = s.mean(), s.std()
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.plot(s.index, s.to_numpy(), color=SERIES, linewidth=0.8)
        for y in (mu - 2 * sd, mu + 2 * sd):
            ax.axhline(y, color=INK, linewidth=0.8, linestyle=(0, (4, 3)))
        out = (s - mu).abs() > 2 * sd
        ax.set_title(f"{name}: mean {mu:+.2f}%, std {sd:.2f}%, {out.mean():.0%} of days outside ±2σ",
                     fontsize=9.5, color=INK, loc="left")
        _style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Spread (% of HK parity)", fontsize=9, color=INK)
    for ax in axes[-1, :]:
        ax.set_xlabel("Date", fontsize=9, color=INK)
    fig.suptitle(f"{MEASURE_LABELS[measure]}, in-sample 2021-04-19 to 2023-12-29",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.945, "Dashed = in-sample mean ± 2 std (descriptive; the strategy uses rolling "
             "windows up to t−1 instead).", fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


def plot_acf(df: pd.DataFrame, path: Path, max_lag: int = 10) -> Path:
    """Autocorrelation of each measure at lags 1..max_lag, per pair.

    Bars near zero = no memory (timing noise); slowly decaying bars = a
    persistent mispricing. Grey band = ±1.96/√n, the 95% band for white noise.
    """
    lags = range(1, max_lag + 1)
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True, sharey=True)
    width = 0.38
    for ax, name in zip(axes.flat, config.PAIRS):
        g = df[df["pair"] == name].set_index("date")
        band = 1.96 / np.sqrt(len(g))
        ax.axhspan(-band, band, color=WARMUP, linewidth=0)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        for offset, measure, color in [(-width / 2, "spread", SERIES), (width / 2, "spread_2r", SERIES_2)]:
            acf = [g[measure].autocorr(lag=k) for k in lags]
            ax.bar(np.array(lags) + offset, acf, width=width - 0.04, color=color,
                   label=MEASURE_LABELS[measure])
        ax.set_title(name, fontsize=10, color=INK, loc="left")
        ax.set_xticks(list(lags))
        _style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Autocorrelation", fontsize=9, color=INK)
    for ax in axes[-1, :]:
        ax.set_xlabel("Lag (trading days)", fontsize=9, color=INK)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle("Spread autocorrelation, in-sample", fontsize=12, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.945, "Grey band = 95% range for pure noise (±1.96/√n).",
             fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


def plot_corr_heatmaps(corrs: dict[str, pd.DataFrame], path: Path) -> Path:
    """Correlation of daily spread changes across pairs, one panel per measure (imshow).

    Diverging scale fixed at [−1, 1] with a neutral grey midpoint at 0.
    """
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#f0efea", "#c4471b"])
    fig, axes = plt.subplots(1, len(corrs), figsize=(5.6 * len(corrs), 5.2))
    axes = np.atleast_1d(axes)
    for ax, (measure, c) in zip(axes, corrs.items()):
        im = ax.imshow(c.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
        ax.set_xticks(range(len(c)), c.columns, rotation=45, ha="right", fontsize=8.5, color=INK)
        ax.set_yticks(range(len(c)), c.index, fontsize=8.5, color=INK)
        for i in range(len(c)):
            for j in range(len(c)):
                ax.text(j, i, f"{c.iat[i, j]:.2f}", ha="center", va="center", fontsize=8, color=INK)
        n = len(c)
        rho = c.to_numpy()[np.triu_indices(n, 1)].mean()
        ax.set_title(f"{MEASURE_LABELS[measure]}\navg ρ = {rho:.2f}, effective bets = "
                     f"{n / (1 + (n - 1) * rho):.1f} of {n}", fontsize=9.5, color=INK, loc="left")
        for side in ax.spines.values():
            side.set_visible(False)
    cb = fig.colorbar(im, ax=axes.tolist(), shrink=0.8)
    cb.set_label("Correlation of daily spread changes", fontsize=9, color=INK)
    cb.ax.tick_params(labelsize=8, colors=MUTED, labelcolor=INK)
    fig.suptitle("Cross-pair correlation of daily spread changes, in-sample",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return Path(path)
