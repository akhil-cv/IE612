"""Stage 1: normalise the raw option and spot feeds.

Backward compatible with ``datacleaning.optiondata_transform`` but never
mutates the caller's DataFrames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings
from ..logging_utils import get_logger
from ..schema import validate_spot

log = get_logger("prepare")


def optiondata_transform(niftyoptions_df, niftyspot_df):
    """Add date/TTM columns and align spot prices to observation dates.

    Returns ``(options, spot_df, spot_array, observation_dates)`` exactly like
    the legacy implementation.
    """
    options = niftyoptions_df.copy()
    spot_df = niftyspot_df.copy()

    options["date_dt"] = pd.to_datetime(options["date"], errors="raise")
    options["exdate_dt"] = pd.to_datetime(options["expiry"], errors="raise")
    options["time_to_expiry"] = options["exdate_dt"] - options["date_dt"]
    options["days_to_exp"] = options["time_to_expiry"].dt.days
    options["ttm"] = options["days_to_exp"] / 365.0

    spot_df["date_dt"] = pd.to_datetime(spot_df["date"], errors="raise")
    observation_dates = pd.DatetimeIndex(options["date_dt"].unique()).sort_values()

    closing = (
        spot_df.drop_duplicates("date_dt", keep="last")
        .set_index("date_dt")["close"]
        .astype(float)
    )
    missing = observation_dates.difference(closing.index)
    if len(missing):
        preview = ", ".join(str(d.date()) for d in missing[:5])
        raise KeyError(
            f"spot price missing for {len(missing)} observation date(s): {preview}"
            + (" ..." if len(missing) > 5 else "")
        )

    spot = closing.reindex(observation_dates).to_numpy(dtype=float)
    return options, spot_df, spot, observation_dates


def _apply_date_window(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    start, end = settings.data.start, settings.data.end
    if start:
        frame = frame[frame["date_dt"] >= pd.Timestamp(start)]
    if end:
        frame = frame[frame["date_dt"] <= pd.Timestamp(end)]
    return frame


def prepare_options(options_df, spot_df, settings: Settings) -> pd.DataFrame:
    """Validate, align and filter raw options into the analysis frame."""
    validate_spot(spot_df)
    options, spot_df, _, _ = optiondata_transform(options_df, spot_df)
    options = _apply_date_window(options, settings)

    closing = spot_df.drop_duplicates("date_dt", keep="last").set_index("date_dt")["close"]
    options["spot"] = options["date_dt"].map(closing).astype(float)
    options["moneyness"] = options["strike"].astype(float) / options["spot"]
    options["price"] = options[settings.options.price_column].astype(float)

    before = len(options)
    options = options[options["price"] > 0]
    options = options[options["strike"] > 0]
    options = options[options["spot"] > 0]
    options = options[
        options["days_to_exp"].between(
            settings.options.min_days, settings.options.max_days, inclusive="both"
        )
    ]
    dropped_window = before - len(options)

    if settings.options.dedupe:
        key = ["date", "expiry", "strike", "option_type"]
        before = len(options)
        options = options.drop_duplicates(key, keep="last")
        dropped_dupes = before - len(options)
    else:
        dropped_dupes = 0

    options = options[options["price"] >= settings.options.min_price]
    options = options[options["option_type"].astype(str).str.upper().isin(["CE", "PE"])]
    options["option_type"] = options["option_type"].astype(str).str.upper()

    options = options.sort_values(["date_dt", "exdate_dt", "strike", "option_type"])
    options = options.reset_index(drop=True)
    options["row_id"] = np.arange(len(options), dtype=np.int64)

    log.info(
        "prepare: %d rows (dropped %d outside day window, %d duplicate contracts)",
        len(options),
        dropped_window,
        dropped_dupes,
    )
    return options


def prepare_report(raw_df: pd.DataFrame, prepared: pd.DataFrame) -> dict[str, float]:
    """Small QC summary for the prepare stage."""
    return {
        "raw_rows": int(len(raw_df)),
        "prepared_rows": int(len(prepared)),
        "dates": int(prepared["date"].nunique()),
        "expiries": int(prepared["expiry"].nunique()),
        "strikes": int(prepared["strike"].nunique()),
        "calls": int((prepared["option_type"] == "CE").sum()),
        "puts": int((prepared["option_type"] == "PE").sum()),
    }
