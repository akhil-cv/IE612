"""Stage 2: put-call-parity risk-free rate per observation date."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings
from ..logging_utils import get_logger

log = get_logger("rate")


def parity_rates(day_df: pd.DataFrame, settings: Settings | None = None) -> pd.Series:
    """Implied rates from matched call/put pairs on one date.

    ``C - P = S - K exp(-rT)`` with ``F = S exp(rT)`` gives
    ``r = log(K / (S - C + P)) / T``.
    """
    settings = settings or Settings()
    calls = day_df[day_df["option_type"] == "CE"]
    puts = day_df[day_df["option_type"] == "PE"]
    pairs = calls.merge(
        puts,
        on=["date_dt", "exdate_dt", "strike", "spot", "ttm"],
        suffixes=("_call", "_put"),
    )
    if pairs.empty:
        return pd.Series(dtype=float)

    denominator = pairs["spot"] - pairs["price_call"] + pairs["price_put"]
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = np.log(pairs["strike"] / denominator) / pairs["ttm"]

    valid = (
        np.isfinite(rates)
        & (denominator > 0)
        & (pairs["ttm"] > 0)
        & (rates >= settings.rate.min)
        & (rates <= settings.rate.max)
    )
    return rates[valid]


def estimate_rates(prepared: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Median parity rate per date with a documented fallback chain."""
    dates = pd.DatetimeIndex(sorted(prepared["date_dt"].unique()))
    rows = []
    for date, day in prepared.groupby("date_dt"):
        rates = parity_rates(day, settings)
        rows.append(
            {
                "date_dt": pd.Timestamp(date),
                "rate": float(rates.median()) if len(rates) else np.nan,
                "rate_pairs": int(len(rates)),
            }
        )

    table = pd.DataFrame(rows).set_index("date_dt").reindex(dates).sort_index()
    table["rate_source"] = np.where(table["rate"].notna(), "parity", "missing")

    window = max(int(settings.rate.rolling_window), 1)
    rolling = table["rate"].rolling(window, min_periods=1, center=True).median()
    fill_rolling = table["rate"].isna() & rolling.notna()
    table.loc[fill_rolling, "rate"] = rolling[fill_rolling]
    table.loc[fill_rolling, "rate_source"] = "rolling_median"

    fill_constant = table["rate"].isna()
    table.loc[fill_constant, "rate"] = settings.rate.fallback
    table.loc[fill_constant, "rate_source"] = "fallback_constant"

    counts = table["rate_source"].value_counts().to_dict()
    log.info("rate: sources %s", counts)
    table.index.name = "date_dt"
    return table.reset_index()


def attach_rates(prepared: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge the daily rate onto every option row."""
    rates = estimate_rates(prepared, settings)
    merged = prepared.merge(rates, on="date_dt", how="left")
    # The documented pipeline column is ``r``; ``rate`` is kept for traceability.
    merged["r"] = merged["rate"].astype(float)
    return merged, rates


def data_day(date, niftyoptions_df, niftyspot_df, settings: Settings | None = None):
    """Legacy-compatible per-day helper.

    Returns ``(subset_df, r)``. Retains the legacy behaviour of keeping both
    calls and puts (no OTM filter); the new pipeline applies the filter later.
    """
    settings = settings or Settings()
    frame = niftyoptions_df
    if "date_dt" not in frame.columns:
        from .prepare import optiondata_transform

        frame, _, _, _ = optiondata_transform(frame, niftyspot_df)

    day = frame[frame["date_dt"] == pd.Timestamp(date)].copy()
    if day.empty:
        return day, np.nan

    spot_df = niftyspot_df.copy()
    if "date_dt" not in spot_df.columns:
        spot_df["date_dt"] = pd.to_datetime(spot_df["date"])
    closing = spot_df.drop_duplicates("date_dt", keep="last").set_index("date_dt")["close"]
    day["spot"] = day["date_dt"].map(closing).astype(float)
    day["moneyness"] = day["strike"] / day["spot"]
    day["mid_price"] = day[settings.options.price_column]
    day["price"] = day["mid_price"]

    rates = parity_rates(day, settings)
    rate = float(rates.median()) if len(rates) else np.nan
    day["r"] = rate
    return day.reset_index(drop=True), rate
