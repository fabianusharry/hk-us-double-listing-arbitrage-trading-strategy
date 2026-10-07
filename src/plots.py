"""Matplotlib figures (titled, labelled with units), saved as PNG to figures/."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed
import matplotlib.pyplot as plt
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
