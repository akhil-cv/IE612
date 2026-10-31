"""Historical actual-versus-synthetic Greek comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings
from ..models.conventions import GREEKS
from ..models.synthetic import generate_synthetic_greeks


def prepare_history(
    df: pd.DataFrame,
    expiry_col: str = "expiry",
    date_col: str = "date",
) -> pd.DataFrame:
    """Add consistent helper columns without mutating the input."""
    data = df.copy()
    required = [date_col, expiry_col, "strike", "option_type", "days_to_exp", "r"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")

    data["_date"] = pd.to_datetime(data[date_col], errors="coerce")
    data["_expiry"] = pd.to_datetime(data[expiry_col], errors="coerce")
    data["_option"] = data["option_type"].astype(str).str.upper()
    data["_strike"] = pd.to_numeric(data["strike"], errors="coerce")
    data["_days_to_exp"] = pd.to_numeric(data["days_to_exp"], errors="coerce")
    data["_r"] = pd.to_numeric(data["r"], errors="coerce")
    data = data.dropna(subset=["_date", "_expiry", "_option", "_strike"])
    if data.empty:
        raise ValueError("no valid historical option observations")
    return data


def get_available_greeks(data: pd.DataFrame) -> list[str]:
    available = [greek for greek in GREEKS if greek in data.columns]
    if not available:
        raise ValueError("no Greek columns found")
    return available


def build_instrument_options(data: pd.DataFrame, greeks: list[str] | None = None) -> list[tuple[str, tuple]]:
    """List ``(label, instrument)`` pairs that have at least one Greek value.

    Contracts that are always filtered out (for example in-the-money contracts
    when ``otm_only`` is enabled) are excluded so widgets and reports never
    offer a series that is entirely NaN.
    """
    available = [name for name in (greeks or GREEKS) if name in data.columns]
    candidates = data
    if available:
        candidates = data[data[available].notna().any(axis=1)]
    instruments = (
        candidates[["_option", "_strike", "_expiry"]]
        .drop_duplicates()
        .sort_values(["_expiry", "_option", "_strike"])
    )
    options = []
    for option, strike, expiry in instruments.itertuples(index=False, name=None):
        expiry = pd.Timestamp(expiry)
        strike = float(strike)
        label = f"{option} | Strike: {strike:g} | Expiry: {expiry:%Y-%m-%d}"
        options.append((label, (option, strike, expiry)))
    return options


def select_synthetic_greek(
    synthetic: pd.DataFrame | None,
    option: str,
    strike: float,
    target_days_to_exp: float,
    target_rate: float | None,
    greek: str,
) -> float:
    """Nearest synthetic value: maturity, then strike, then rate."""
    if synthetic is None or synthetic.empty or greek not in synthetic.columns:
        return np.nan

    candidates = synthetic[synthetic["option_type"].astype(str).str.upper() == str(option).upper()].copy()
    if candidates.empty:
        return np.nan

    for column in ("strike", "ttm", greek):
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce")
    if "r" in candidates.columns:
        candidates["r"] = pd.to_numeric(candidates["r"], errors="coerce")
    candidates = candidates.dropna(subset=["strike", "ttm", greek])
    if candidates.empty:
        return np.nan

    target_ttm = max(float(target_days_to_exp), 0.0) / 365.0
    distance = (candidates["ttm"] - target_ttm).abs()
    candidates = candidates[np.isclose(distance, distance.min())]

    distance = (candidates["strike"] - float(strike)).abs()
    candidates = candidates[np.isclose(distance, distance.min())]

    if target_rate is not None and np.isfinite(target_rate) and "r" in candidates.columns:
        rate_distance = (candidates["r"] - float(target_rate)).abs()
        candidates = candidates.loc[[rate_distance.idxmin()]]

    return float(candidates.iloc[0][greek])


class SyntheticCache:
    """Lazily build and cache the synthetic surface for each date."""

    def __init__(self, df_with_greeks: pd.DataFrame, settings: Settings):
        self.df = df_with_greeks
        self.settings = settings
        self._cache: dict[pd.Timestamp, pd.DataFrame] = {}
        self._originals = (
            df_with_greeks.drop_duplicates("date")
            .assign(_parsed=lambda f: pd.to_datetime(f["date"]))
            .set_index("_parsed")["date"]
            .to_dict()
        )

    def get(self, date) -> pd.DataFrame:
        key = pd.Timestamp(date)
        if key not in self._cache:
            original = self._originals.get(key, key)
            self._cache[key] = generate_synthetic_greeks(self.df, original, self.settings)
        return self._cache[key]


def instrument_history(
    data: pd.DataFrame,
    instrument: tuple,
    as_of,
    greek: str,
) -> pd.DataFrame:
    option, strike, expiry = instrument
    history = data[
        (data["_option"] == option)
        & np.isclose(data["_strike"], float(strike))
        & (data["_expiry"] == pd.Timestamp(expiry))
        & (data["_date"] <= pd.Timestamp(as_of))
    ].copy()
    if history.empty:
        return history
    history[greek] = pd.to_numeric(history[greek], errors="coerce")
    history = history.dropna(subset=[greek, "_days_to_exp"])
    history = history.sort_index().groupby("_date", as_index=False).tail(1)
    return history.sort_values("_date")


def actual_vs_synthetic(
    data: pd.DataFrame,
    instruments: list[tuple],
    greek: str,
    as_of,
    settings: Settings,
    cache: SyntheticCache | None = None,
) -> pd.DataFrame:
    """Long frame with actual and synthetic values for each instrument."""
    cache = cache or SyntheticCache(data, settings)
    records = []
    for instrument in instruments:
        history = instrument_history(data, instrument, as_of, greek)
        for _, row in history.iterrows():
            value = np.nan
            if row["_days_to_exp"] > 0:
                value = select_synthetic_greek(
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
                    "actual": float(row[greek]),
                    "synthetic": value,
                }
            )
    return pd.DataFrame(records)
