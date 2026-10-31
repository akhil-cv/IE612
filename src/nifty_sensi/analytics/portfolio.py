"""Portfolio-level Greek aggregation and expiry payoff."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings
from ..models.conventions import GREEKS
from .history import SyntheticCache, instrument_history, prepare_history, select_synthetic_greek


def normalise_positions(positions) -> dict[tuple, dict]:
    """Accept ``{instrument: {'direction': 1, 'quantity': 10}}`` or a list of dicts."""
    if isinstance(positions, dict):
        result = {}
        for instrument, value in positions.items():
            if isinstance(value, dict):
                result[instrument] = {
                    "direction": int(value.get("direction", 1)),
                    "quantity": int(value.get("quantity", 1)),
                }
            else:
                result[instrument] = {"direction": 1, "quantity": int(value)}
        return result
    result = {}
    for item in positions:
        instrument = (item["option"], float(item["strike"]), pd.Timestamp(item["expiry"]))
        result[instrument] = {
            "direction": int(item.get("direction", 1)),
            "quantity": int(item.get("quantity", 1)),
        }
    return result


def calculate_contributions(
    data: pd.DataFrame,
    positions,
    greek: str,
    as_of,
    settings: Settings,
    cache: SyntheticCache | None = None,
) -> pd.DataFrame:
    """Signed actual and synthetic Greek contributions through *as_of*."""
    if greek not in GREEKS:
        raise ValueError(f"unknown Greek: {greek}")
    positions = normalise_positions(positions)
    cache = cache or SyntheticCache(data, settings)

    records = []
    for instrument, position in positions.items():
        signed_quantity = position["direction"] * position["quantity"]
        history = instrument_history(data, instrument, as_of, greek)
        for _, row in history.iterrows():
            synthetic = np.nan
            if row["_days_to_exp"] > 0:
                synthetic = select_synthetic_greek(
                    cache.get(row["_date"]),
                    option=instrument[0],
                    strike=float(instrument[1]),
                    target_days_to_exp=float(row["_days_to_exp"]),
                    target_rate=row["_r"],
                    greek=greek,
                )
            records.append(
                {
                    "date": pd.Timestamp(row["_date"]),
                    "instrument": instrument,
                    "label": f"{instrument[0]} K={float(instrument[1]):g} Exp={pd.Timestamp(instrument[2]):%Y-%m-%d}",
                    "direction": position["direction"],
                    "quantity": position["quantity"],
                    "signed_quantity": signed_quantity,
                    "actual": float(row[greek]),
                    "synthetic": synthetic,
                    "actual_contribution": signed_quantity * float(row[greek]),
                    "synthetic_contribution": signed_quantity * synthetic,
                }
            )
    return pd.DataFrame(records)


def aggregate_portfolio(contributions: pd.DataFrame, n_instruments: int) -> pd.DataFrame:
    """Portfolio totals; partial dates are returned as NaN."""
    if contributions.empty:
        return pd.DataFrame(columns=["actual", "synthetic"])
    portfolio = contributions.groupby("date").agg(
        actual=("actual_contribution", "sum"),
        synthetic=("synthetic_contribution", "sum"),
        actual_count=("actual_contribution", "count"),
        synthetic_count=("synthetic_contribution", "count"),
    )
    portfolio.loc[portfolio["actual_count"] < n_instruments, "actual"] = np.nan
    portfolio.loc[portfolio["synthetic_count"] < n_instruments, "synthetic"] = np.nan
    return portfolio.sort_index()


def payoff_curve(
    positions,
    spot_min: float | None = None,
    spot_max: float | None = None,
    n_points: int = 500,
    lot_size: int = 1,
) -> pd.DataFrame:
    """Expiry payoff for a set of option positions."""
    positions = normalise_positions(positions)
    if not positions:
        raise ValueError("no positions supplied")

    strikes = [float(key[1]) for key in positions]
    spot_min = spot_min if spot_min is not None else min(strikes) * 0.8
    spot_max = spot_max if spot_max is not None else max(strikes) * 1.2
    spots = np.linspace(spot_min, spot_max, n_points)
    total = np.zeros_like(spots)

    per_instrument = {}
    for (option, strike, expiry), position in positions.items():
        strike = float(strike)
        signed = position["direction"] * position["quantity"] * lot_size
        if str(option).upper() in {"CE", "CALL", "C"}:
            intrinsic = np.maximum(spots - strike, 0.0)
        elif str(option).upper() in {"PE", "PUT", "P"}:
            intrinsic = np.maximum(strike - spots, 0.0)
        else:
            continue
        contribution = signed * intrinsic
        per_instrument[f"{option} K={strike:g}"] = contribution
        total += contribution

    frame = pd.DataFrame({"underlying_price": spots, "portfolio_payoff": total})
    for label, values in per_instrument.items():
        frame[label] = values
    return frame


def prepare_portfolio_input(df: pd.DataFrame, expiry_col: str = "expiry", date_col: str = "date"):
    """Convenience wrapper used by the notebook and widgets."""
    return prepare_history(df, expiry_col=expiry_col, date_col=date_col)
