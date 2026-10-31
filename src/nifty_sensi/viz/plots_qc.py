"""Data-quality charts: coverage, rates, repricing and surface completeness."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def coverage_by_date(frame: pd.DataFrame, figsize: tuple[float, float] = (12, 4)):
    if frame.empty:
        raise ValueError("no rows to plot")
    grouped = (
        frame.assign(_date=pd.to_datetime(frame["date"]))
        .groupby("_date")
        .agg(rows=("date", "size"), usable=("valid_iv", "sum"))
    )
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(grouped.index, grouped["rows"], linewidth=1, label="contracts")
    ax.plot(grouped.index, grouped["usable"], linewidth=1, label="usable IV")
    ax.set_xlabel("Date")
    ax.set_ylabel("Contracts")
    ax.set_title("Contract coverage by observation date")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def rate_history(rates: pd.DataFrame, figsize: tuple[float, float] = (12, 4)):
    if rates.empty:
        raise ValueError("no rates to plot")
    frame = rates.copy()
    frame["date_dt"] = pd.to_datetime(frame["date_dt"])
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(frame["date_dt"], frame["rate"], linewidth=1, marker=".", markersize=2)
    fallback = frame[frame["rate_source"] != "parity"]
    if not fallback.empty:
        ax.scatter(fallback["date_dt"], fallback["rate"], s=18, color="tab:orange", label="imputed")
        ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Implied risk-free rate")
    ax.set_title("Put-call-parity risk-free rate")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def repricing_error(frame: pd.DataFrame, figsize: tuple[float, float] = (9, 4)):
    errors = pd.to_numeric(frame.get("repricing_error"), errors="coerce").dropna()
    if errors.empty:
        raise ValueError("no repricing errors available")
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(errors.clip(-5, 5), bins=80)
    ax.set_yscale("log")
    ax.set_xlabel("Model price - market price")
    ax.set_ylabel("Contracts (log)")
    ax.set_title("Black-76 repricing residuals")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def surface_coverage(surface_qc: pd.DataFrame, figsize: tuple[float, float] = (12, 4)):
    if surface_qc.empty:
        raise ValueError("no surface QC rows")
    frame = surface_qc.copy()
    frame["date_dt"] = pd.to_datetime(frame["date"])
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(frame["date_dt"], frame["grid_points"] - frame["grid_missing"], width=1.0, label="filled")
    ax.bar(
        frame["date_dt"],
        frame["grid_missing"],
        width=1.0,
        bottom=frame["grid_points"] - frame["grid_missing"],
        label="missing",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Grid points")
    ax.set_title("IV grid coverage")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def iv_status_breakdown(frame: pd.DataFrame, figsize: tuple[float, float] = (8, 4)):
    if frame.empty or "iv_status_label" not in frame.columns:
        raise ValueError("no IV status column")
    counts = frame["iv_status_label"].value_counts()
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(counts.index, counts.to_numpy())
    ax.set_yscale("log")
    ax.set_ylabel("Contracts (log)")
    ax.set_title("IV solver status")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig
