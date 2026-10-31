"""Stage 4: attach the full Greek set to every valid contract."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings
from ..logging_utils import get_logger
from ..models import black76
from ..models.conventions import GREEKS, forward_price

log = get_logger("greeks")


def attach_greeks(iv_frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Compute all 15 Greeks for rows with ``valid_iv``."""
    frame = iv_frame.copy()
    mask = frame["valid_iv"].fillna(False).to_numpy(dtype=bool)

    for name in GREEKS:
        frame[name] = np.nan

    if not mask.any():
        log.warning("greeks: no valid rows, returning empty Greek columns")
        return frame

    subset = frame.loc[mask]
    flag = np.where(subset["option_type"].eq("CE"), "c", "p")
    forward = forward_price(
        subset["spot"].to_numpy(dtype=float),
        subset["r"].to_numpy(dtype=float),
        subset["ttm"].to_numpy(dtype=float),
    )
    values = black76.greeks(
        flag,
        forward,
        subset["strike"].to_numpy(dtype=float),
        subset["ttm"].to_numpy(dtype=float),
        subset["r"].to_numpy(dtype=float),
        subset["iv"].to_numpy(dtype=float),
        settings=settings.greeks,
    )
    for name in GREEKS:
        frame.loc[mask, name] = values[name]

    log.info("greeks: computed %d rows x %d Greeks", int(mask.sum()), len(GREEKS))
    return frame


def greeks_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-Greek coverage and magnitude summary."""
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for name in GREEKS:
        series = frame[name]
        rows.append(
            {
                "greek": name,
                "non_null": int(series.notna().sum()),
                "coverage_pct": 100.0 * series.notna().mean(),
                "median": float(series.median()),
                "p05": float(series.quantile(0.05)),
                "p95": float(series.quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)
