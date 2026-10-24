"""Actual-versus-synthetic history charts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _label(instrument: tuple) -> str:
    option, strike, expiry = instrument
    return f"{option} K={float(strike):g} {pd.Timestamp(expiry):%Y-%m-%d}"


def actual_vs_synthetic(
    result: pd.DataFrame,
    greek: str,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 6),
):
    """Plot the long frame produced by ``analytics.history.actual_vs_synthetic``."""
    if result.empty:
        raise ValueError("no comparison rows to plot")

    fig, ax = plt.subplots(figsize=figsize)
    for instrument, group in result.groupby("instrument"):
        group = group.sort_values("date")
        line = ax.plot(
            group["date"],
            group["actual"],
            marker=".",
            markersize=4,
            linewidth=1.5,
            label=f"{_label(instrument)} actual",
        )[0]
        ax.plot(
            group["date"],
            group["synthetic"],
            linestyle="--",
            linewidth=1.4,
            color=line.get_color(),
            label=f"{_label(instrument)} synthetic",
        )

    ax.set_xlabel("Observation date")
    ax.set_ylabel(greek)
    ax.set_title(title or f"{greek.capitalize()} — actual vs synthetic")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def comparison_scatter(
    result: pd.DataFrame,
    greek: str,
    figsize: tuple[float, float] = (6, 6),
):
    """Scatter of synthetic against actual values with a 45-degree line."""
    valid = result.dropna(subset=["actual", "synthetic"])
    if valid.empty:
        raise ValueError("no matched actual/synthetic pairs")

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(valid["actual"], valid["synthetic"], s=12, alpha=0.6)
    low = float(np.nanmin([valid["actual"].min(), valid["synthetic"].min()]))
    high = float(np.nanmax([valid["actual"].max(), valid["synthetic"].max()]))
    ax.plot([low, high], [low, high], color="black", linewidth=1, linestyle="--")
    ax.set_xlabel(f"Actual {greek}")
    ax.set_ylabel(f"Synthetic {greek}")
    ax.set_title(f"Actual vs synthetic {greek}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
