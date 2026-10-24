"""Portfolio totals, contributions and payoff charts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def portfolio_totals(portfolio: pd.DataFrame, greek: str, figsize: tuple[float, float] = (12, 6)):
    if portfolio.empty:
        raise ValueError("portfolio is empty")
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(portfolio.index, portfolio["actual"], linewidth=2, label=f"Actual {greek}")
    ax.plot(
        portfolio.index,
        portfolio["synthetic"],
        linewidth=2,
        linestyle="--",
        label=f"Synthetic {greek}",
    )
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Observation date")
    ax.set_ylabel(f"Net {greek}")
    ax.set_title(f"Portfolio {greek.capitalize()}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def contributions(contributions: pd.DataFrame, greek: str, figsize: tuple[float, float] = (13, 6)):
    if contributions.empty:
        raise ValueError("no contributions to plot")
    fig, ax = plt.subplots(figsize=figsize)
    for _, group in contributions.groupby("instrument"):
        group = group.sort_values("date")
        label = group["label"].iloc[0]
        side = "Long" if group["direction"].iloc[0] > 0 else "Short"
        quantity = group["quantity"].iloc[0]
        line = ax.plot(
            group["date"],
            group["actual_contribution"],
            linewidth=1.5,
            label=f"{label} {side} {quantity} actual",
        )[0]
        ax.plot(
            group["date"],
            group["synthetic_contribution"],
            linewidth=1.4,
            linestyle="--",
            color=line.get_color(),
            label=f"{label} {side} {quantity} synthetic",
        )
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Observation date")
    ax.set_ylabel(f"Contribution to {greek}")
    ax.set_title(f"Signed instrument contributions — {greek}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def payoff(payoff_frame: pd.DataFrame, figsize: tuple[float, float] = (11, 5)):
    if payoff_frame.empty:
        raise ValueError("payoff frame is empty")
    fig, ax = plt.subplots(figsize=figsize)
    for column in payoff_frame.columns:
        if column in {"underlying_price", "portfolio_payoff"}:
            continue
        ax.plot(payoff_frame["underlying_price"], payoff_frame[column], linewidth=1, alpha=0.5, label=column)
    ax.plot(
        payoff_frame["underlying_price"],
        payoff_frame["portfolio_payoff"],
        color="black",
        linewidth=2.4,
        label="Net portfolio",
    )
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Underlying at expiry")
    ax.set_ylabel("Payoff")
    ax.set_title("Portfolio payoff at expiry")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
