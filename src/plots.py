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


def plot_sharpe_heatmaps(grid: pd.DataFrame, signal: str, path: Path, value: str = "sharpe",
                         vlim: float | None = None, mark: dict | None = None) -> Path:
    """Sharpe over (L, k) for each (exit_z, H) slice of the grid, one panel per slice.

    Diverging colour scale centred at 0 (grey); pass the same vlim for both signals so
    the figures are comparable. `mark` = {L, k, exit_z, H} outlines one cell (the suggestion).
    """
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    cmap = LinearSegmentedColormap.from_list("div", ["#c4471b", "#f0efea", "#1c5cab"])  # red < 0 < blue
    g = grid[grid["signal"] == signal]
    Ls, ks = sorted(g["L"].unique()), sorted(g["k"].unique())
    exits, Hs = sorted(g["exit_z"].unique()), sorted(g["H"].unique())
    vlim = vlim or float(np.nanmax(np.abs(g[value])))
    fig, axes = plt.subplots(len(exits), len(Hs), figsize=(9, 7.6), sharex=True, sharey=True)
    for i, e in enumerate(exits):
        for j, H in enumerate(Hs):
            ax = axes[i, j]
            m = (g[(g["exit_z"] == e) & (g["H"] == H)]
                 .pivot(index="L", columns="k", values=value).reindex(index=Ls, columns=ks))
            im = ax.imshow(m.to_numpy(), cmap=cmap, vmin=-vlim, vmax=vlim, aspect="auto")
            for a in range(len(Ls)):
                for b in range(len(ks)):
                    ax.text(b, a, f"{m.iat[a, b]:+.2f}", ha="center", va="center", fontsize=9, color=INK)
            if mark and mark["exit_z"] == e and mark["H"] == H:
                ax.add_patch(Rectangle((ks.index(mark["k"]) - 0.5, Ls.index(mark["L"]) - 0.5), 1, 1,
                                       fill=False, edgecolor=INK, linewidth=2.2))
            ax.set_title(f"exit_z = {e:g}, H = {H} days", fontsize=9.5, color=INK, loc="left")
            ax.set_xticks(range(len(ks)), [f"{k:g}" for k in ks], fontsize=8.5, color=INK)
            ax.set_yticks(range(len(Ls)), [str(L) for L in Ls], fontsize=8.5, color=INK)
            for side in ax.spines.values():
                side.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Entry threshold k (|z|)", fontsize=9, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("Window L (trading days)", fontsize=9, color=INK)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    cb.set_label("Annualised Sharpe, net of costs (1x)", fontsize=9, color=INK)
    cb.ax.tick_params(labelsize=8, colors=MUTED, labelcolor=INK)
    label = MEASURE_LABELS.get(signal, signal)
    fig.suptitle(f"In-sample Sharpe grid: {label}", fontsize=12, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.945, "Portfolio of 6 pairs, 2021-04-19 to 2023-12-29, costs at 1x. "
             "Outlined = suggested setting.", fontsize=8.5, color=MUTED, ha="left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return Path(path)


def plot_equity_is_oos(curves: dict[str, tuple[pd.Series, pd.Series]], path: Path) -> Path:
    """Cumulative net return (top) and drawdown (bottom) for IS then OOS, OOS shaded.

    curves: {label: (IS daily returns, OOS daily returns)}. The OOS curve continues from the
    IS end value (positions are flat at the boundary: the two runs are separate).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    colors = [SERIES, SERIES_2]
    oos_start = None
    for (label, (r_is, r_oos)), color in zip(curves.items(), colors):
        r = pd.concat([r_is, r_oos])
        eq = (1 + r).cumprod()
        dd = eq / np.maximum(eq.cummax(), 1.0) - 1
        ax1.plot(eq.index, (eq - 1) * 100, color=color, linewidth=1.4, label=label)
        ax2.plot(dd.index, dd * 100, color=color, linewidth=1.0)
        oos_start = r_oos.index.min()
    for ax in (ax1, ax2):
        ax.axvspan(oos_start, ax.get_xlim()[1] if False else r.index.max(), color=WARMUP, linewidth=0)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        _style(ax)
    ax1.text(oos_start, ax1.get_ylim()[1], "  out-of-sample (run once)", va="top", fontsize=8.5, color=MUTED)
    ax1.set_ylabel("Cumulative net return (%)", fontsize=9, color=INK)
    ax2.set_ylabel("Drawdown (%)", fontsize=9, color=INK)
    ax2.set_xlabel("Date", fontsize=9, color=INK)
    ax1.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
    fig.suptitle("Six-pair portfolio, frozen parameters (L=120, k=2.5, exit_z=0.5, H=10), costs 1x",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


def plot_connect_event(spreads: pd.DataFrame, pair: str, event: str, path: Path, controls: list[str],
                       window: int, end: str, title_note: str = "") -> Path:
    """Spread of `pair` around its Stock Connect date (20-day rolling mean in bold), with the
    controls' average 20-day rolling mean for comparison. Two panels: morning and two-reading."""
    ev = pd.Timestamp(event)
    g = spreads[spreads["pair"] == pair].set_index("date").sort_index().loc[:end]
    i = g.index.searchsorted(ev)
    lo, hi = g.index[max(i - window, 0)], g.index[min(i + window - 1, len(g) - 1)]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, m in zip(axes, ["spread", "spread_2r"]):
        s = g.loc[lo:hi, m] * 100
        ctl = (spreads[spreads["pair"].isin(controls)].pivot(index="date", columns="pair", values=m)
               .loc[lo:hi].mean(axis=1) * 100)
        ax.plot(s.index, s.to_numpy(), color=SERIES, linewidth=0.6, alpha=0.45)
        ax.plot(s.index, s.rolling(20, min_periods=5).mean(), color=SERIES, linewidth=2, label=f"{pair}, 20-day mean")
        ax.plot(ctl.index, ctl.rolling(20, min_periods=5).mean(), color=SERIES_2, linewidth=1.6,
                label="Comparison pairs, 20-day mean")
        ax.axvline(ev, color=INK, linewidth=1, linestyle=(0, (4, 3)))
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_title(MEASURE_LABELS[m], fontsize=10, color=INK, loc="left")
        ax.set_ylabel("Spread (% of HK parity)", fontsize=9, color=INK)
        _style(ax)
    axes[0].text(ev, axes[0].get_ylim()[1], f"  Connect {ev.date()}", va="top", fontsize=8.5, color=INK)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower left")
    axes[-1].set_xlabel("Date", fontsize=9, color=INK)
    fig.suptitle(f"{pair}: spread around Southbound Stock Connect inclusion{title_note}",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)
