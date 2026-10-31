"""Greek charts (line, grid and 3-D surface)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"CE": "#1f77b4", "PE": "#d62728"}


def _pick_tenor(frame: pd.DataFrame, days_to_exp: float | None) -> float:
    available = np.sort(frame["days_to_exp"].dropna().unique())
    if len(available) == 0:
        raise ValueError("no maturities available")
    if days_to_exp is None:
        return float(available[0])
    return float(available[np.argmin(np.abs(available - float(days_to_exp)))])


def greek_vs_strike(
    frame: pd.DataFrame,
    date: str,
    greek: str,
    days_to_exp: float | None = None,
    option_types: tuple[str, ...] = ("CE", "PE"),
    figsize: tuple[float, float] = (10, 5),
):
    """Greek against strike for one date and maturity."""
    subset = frame[frame["date"] == date]
    if subset.empty:
        raise ValueError(f"no rows for date {date}")
    tenor = _pick_tenor(subset, days_to_exp)

    fig, ax = plt.subplots(figsize=figsize)
    for option in option_types:
        slice_ = subset[
            (subset["option_type"] == option) & np.isclose(subset["days_to_exp"], tenor)
        ].sort_values("strike")
        if slice_.empty:
            continue
        ax.plot(
            slice_["strike"],
            slice_[greek],
            marker=".",
            markersize=4,
            linewidth=1.4,
            color=COLORS.get(option),
            label=option,
        )

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Strike")
    ax.set_ylabel(greek)
    ax.set_title(f"{greek.capitalize()} vs Strike — {date} (DTE={tenor:.0f})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def greek_by_expiry(
    frame: pd.DataFrame,
    date: str,
    greek: str,
    strikes: list[float] | None = None,
    option_type: str = "CE",
    figsize: tuple[float, float] = (10, 5),
):
    """Greek against maturity for a few strikes."""
    subset = frame[(frame["date"] == date) & (frame["option_type"] == option_type)]
    if subset.empty:
        raise ValueError(f"no rows for {date}/{option_type}")

    available = np.sort(subset["strike"].unique())
    if strikes is None:
        picks = available[np.linspace(0, len(available) - 1, min(5, len(available))).astype(int)]
    else:
        picks = [available[np.argmin(np.abs(available - strike))] for strike in strikes]

    fig, ax = plt.subplots(figsize=figsize)
    for strike in picks:
        slice_ = subset[np.isclose(subset["strike"], strike)].sort_values("days_to_exp")
        ax.plot(slice_["days_to_exp"], slice_[greek], marker=".", label=f"K={strike:g}")

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Days to expiry")
    ax.set_ylabel(greek)
    ax.set_title(f"{greek.capitalize()} vs Maturity — {date} ({option_type})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def greek_surface(
    frame: pd.DataFrame,
    date: str,
    greek: str,
    option_type: str = "CE",
    figsize: tuple[float, float] = (12, 7),
):
    """3-D Greek surface over strike and maturity."""
    subset = frame[(frame["date"] == date) & (frame["option_type"] == option_type)]
    if subset.empty:
        raise ValueError(f"no rows for {date}/{option_type}")

    pivot = subset.pivot_table(
        index="days_to_exp", columns="strike", values=greek, aggfunc="mean"
    ).sort_index()
    pivot = pivot.sort_index(axis=1)
    if pivot.empty:
        raise ValueError("surface has no data")

    X, Y = np.meshgrid(pivot.columns.to_numpy(dtype=float), pivot.index.to_numpy(dtype=float))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, pivot.to_numpy(dtype=float), cmap="viridis", alpha=0.85)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Days to expiry")
    ax.set_zlabel(greek)
    ax.set_title(f"{greek.capitalize()} Surface — {date} ({option_type})")
    fig.tight_layout()
    return fig
