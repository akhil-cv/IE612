"""Stage 3: per-contract implied volatility with solver diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings
from ..logging_utils import get_logger
from ..models import black76
from ..models.conventions import forward_price, moneyness

log = get_logger("iv")


def _flag(option_type: pd.Series) -> np.ndarray:
    return np.where(option_type.astype(str).str.upper().eq("CE"), "c", "p")


def compute_iv(rated: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Solve Black-76 IV for every row and flag quality issues."""
    frame = rated.copy()
    frame["forward"] = forward_price(frame["spot"], frame["r"], frame["ttm"])
    flag = _flag(frame["option_type"])

    iv, status = black76.implied_vol(
        frame["price"].to_numpy(dtype=float),
        frame["forward"].to_numpy(dtype=float),
        frame["strike"].to_numpy(dtype=float),
        frame["ttm"].to_numpy(dtype=float),
        frame["r"].to_numpy(dtype=float),
        flag,
        lower=settings.iv.lower,
        upper=settings.iv.upper,
        tolerance=settings.iv.tolerance,
        max_iter=settings.iv.max_iter,
    )
    frame["iv"] = iv
    frame["iv_status"] = status
    frame["iv_status_label"] = [black76.STATUS_LABELS.get(int(code), "unknown") for code in status]
    frame["model_price"] = black76.price(
        flag,
        frame["forward"].to_numpy(dtype=float),
        frame["strike"].to_numpy(dtype=float),
        frame["ttm"].to_numpy(dtype=float),
        frame["r"].to_numpy(dtype=float),
        frame["iv"].to_numpy(dtype=float),
    )
    frame["repricing_error"] = frame["model_price"] - frame["price"]
    frame["moneyness"] = moneyness(frame["strike"], frame["spot"])

    call = frame["option_type"].eq("CE")
    frame["is_otm"] = np.where(call, frame["moneyness"] >= 1.0, frame["moneyness"] < 1.0)

    usable = (
        frame["iv"].notna()
        & (frame["iv_status"] == black76.IV_OK)
        & (frame["iv"] > settings.iv.lower)
        & (frame["iv"] < settings.iv.upper)
    )
    if settings.options.otm_only:
        usable &= frame["is_otm"]
    frame["valid_iv"] = usable

    log.info(
        "iv: %d rows, %d usable (%.1f%%), statuses %s",
        len(frame),
        int(usable.sum()),
        100.0 * usable.mean() if len(frame) else 0.0,
        frame["iv_status_label"].value_counts().to_dict(),
    )
    return frame


def iv_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate solver and repricing quality by date."""
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby("date", as_index=False).agg(
        rows=("iv", "size"),
        solved=("iv", lambda s: int(s.notna().sum())),
        usable=("valid_iv", "sum"),
        median_iv=("iv", "median"),
        p95_abs_repricing=("repricing_error", lambda s: float(np.nanpercentile(np.abs(s), 95))),
    )
    grouped["usable_pct"] = 100.0 * grouped["usable"] / grouped["rows"].clip(lower=1)
    return grouped
