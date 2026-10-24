"""Implied-volatility smile, surface and grid charts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"CE": "#1f77b4", "PE": "#d62728"}


def iv_smile(
    frame: pd.DataFrame,
    date: str,
    days_to_exp: float | None = None,
    option_types: tuple[str, ...] = ("CE", "PE"),
    figsize: tuple[float, float] = (10, 5),
):
    """IV against strike for one maturity."""
    subset = frame[frame["date"] == date]
    if subset.empty:
        raise ValueError(f"no rows for date {date}")
    available = np.sort(subset["days_to_exp"].dropna().unique())
    tenor = available[0] if days_to_exp is None else available[np.argmin(np.abs(available - days_to_exp))]

    fig, ax = plt.subplots(figsize=figsize)
    for option in option_types:
        slice_ = subset[
            (subset["option_type"] == option) & np.isclose(subset["days_to_exp"], tenor)
        ].sort_values("strike")
        slice_ = slice_[slice_["valid_iv"]] if "valid_iv" in slice_ else slice_
        if slice_.empty:
            continue
        ax.plot(
            slice_["strike"],
            slice_["iv"],
            marker=".",
            markersize=4,
            color=COLORS.get(option),
            label=option,
        )

    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied volatility")
    ax.set_title(f"IV Smile — {date} (DTE={tenor:.0f})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def iv_surface(
    frame: pd.DataFrame,
    date: str,
    option_type: str = "CE",
    figsize: tuple[float, float] = (12, 7),
):
    """3-D IV surface over strike and maturity."""
    subset = frame[(frame["date"] == date) & (frame["option_type"] == option_type)]
    if "valid_iv" in subset.columns:
        subset = subset[subset["valid_iv"]]
    if subset.empty:
        raise ValueError(f"no valid IV rows for {date}/{option_type}")

    pivot = subset.pivot_table(index="days_to_exp", columns="strike", values="iv", aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1).dropna(how="all").dropna(axis=1, how="all")
    if pivot.empty:
        raise ValueError("surface has no data")

    X, Y = np.meshgrid(pivot.columns.to_numpy(dtype=float), pivot.index.to_numpy(dtype=float))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, pivot.to_numpy(dtype=float), cmap="viridis", alpha=0.85)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Days to expiry")
    ax.set_zlabel("IV")
    ax.set_title(f"IV Surface — {date} ({option_type})")
    fig.tight_layout()
    return fig


def surface_heatmap(
    surface: pd.DataFrame,
    date: str,
    figsize: tuple[float, float] = (10, 5),
):
    """Heatmap of the processed 10 x 8 IV grid for one date."""
    subset = surface[surface["date"] == date]
    if subset.empty:
        raise ValueError(f"no surface rows for {date}")
    pivot = subset.pivot_table(index="tau_days", columns="moneyness", values="iv").sort_index()
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{value:.2f}" for value in pivot.columns], rotation=45, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{value:.0f}" for value in pivot.index], fontsize=8)
    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Days to expiry")
    ax.set_title(f"IV Grid — {date}")
    fig.colorbar(image, ax=ax, label="IV")
    fig.tight_layout()
    return fig
