"""Data contracts for the raw NIFTY option and spot feeds."""

from __future__ import annotations

import pandas as pd

OPTION_COLUMNS = ["date", "expiry", "strike", "option_type", "close"]
SPOT_COLUMNS = ["date", "close"]


class DataValidationError(ValueError):
    """Raised when an input feed violates the expected contract."""


def _require_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise DataValidationError(f"{label} is missing required columns: {missing}")


def validate_options(frame: pd.DataFrame) -> dict[str, int]:
    """Validate the raw option feed and return a small QC summary."""
    _require_columns(frame, OPTION_COLUMNS, "options")
    if frame.empty:
        raise DataValidationError("options feed is empty")

    parsed = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["date"], errors="coerce"),
            "expiry": pd.to_datetime(frame["expiry"], errors="coerce"),
            "strike": pd.to_numeric(frame["strike"], errors="coerce"),
            "price": pd.to_numeric(frame["close"], errors="coerce"),
            "option_type": frame["option_type"].astype(str).str.upper(),
        }
    )

    summary = {
        "rows": int(len(frame)),
        "bad_date": int(parsed["date"].isna().sum()),
        "bad_expiry": int(parsed["expiry"].isna().sum()),
        "bad_strike": int((parsed["strike"].isna() | (parsed["strike"] <= 0)).sum()),
        "bad_price": int((parsed["price"].isna() | (parsed["price"] < 0)).sum()),
        "bad_type": int((~parsed["option_type"].isin(["CE", "PE"])).sum()),
        "expiry_not_after_date": int((parsed["expiry"] <= parsed["date"]).sum()),
        "duplicate_contracts": int(
            frame.duplicated(["date", "expiry", "strike", "option_type"]).sum()
        ),
    }
    if summary["rows"] and summary["bad_date"] == summary["rows"]:
        raise DataValidationError("every row has an unparseable date")
    return summary


def validate_spot(frame: pd.DataFrame) -> dict[str, int]:
    _require_columns(frame, SPOT_COLUMNS, "spot")
    if frame.empty:
        raise DataValidationError("spot feed is empty")
    parsed = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["date"], errors="coerce"),
            "close": pd.to_numeric(frame["close"], errors="coerce"),
        }
    )
    summary = {
        "rows": int(len(frame)),
        "bad_date": int(parsed["date"].isna().sum()),
        "bad_close": int((parsed["close"].isna() | (parsed["close"] <= 0)).sum()),
        "duplicate_dates": int(parsed["date"].duplicated().sum()),
    }
    if summary["bad_date"] == summary["rows"]:
        raise DataValidationError("every spot row has an unparseable date")
    return summary
