"""Synthetic Greek surface on a dense strike x maturity grid.

Rebuilds and fixes the legacy ``generate_synthetic_greeks`` helper:

* no dead ``sigma_range`` branch,
* ``moneyness`` is ``K / S`` (same as the pipeline),
* ``speed`` is a true third spot derivative, not gamma,
* every Greek comes from the tested :mod:`nifty_sensi.models.black76` engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings, SyntheticSettings
from ..models import black76
from ..models.conventions import GREEKS, forward_price, moneyness


def _strike_grid(subset: pd.DataFrame, spot: float, cfg: SyntheticSettings) -> np.ndarray:
    low = max(spot - cfg.spot_pad, 1.0)
    high = spot + cfg.spot_pad
    grid = np.arange(low, high + 1e-9, cfg.strike_step)
    if cfg.strike_round > 0:
        grid = np.round(grid / cfg.strike_round) * cfg.strike_round
    return np.unique(grid)


def _attach_iv(grid: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    """Nearest observed IV for each (option type, maturity, strike)."""
    iv_data = observed[["strike", "option_type", "ttm", "iv"]].dropna(subset=["strike", "iv"]).copy()
    iv_data = iv_data.sort_values("strike")
    iv_data["strike"] = iv_data["strike"].astype(float)
    grid = grid.sort_values("strike")
    merged = pd.merge_asof(
        grid,
        iv_data,
        on="strike",
        by=["option_type", "ttm"],
        direction="nearest",
    )
    return merged


def generate_synthetic_greeks(
    df_with_greeks: pd.DataFrame,
    dt: str | pd.Timestamp,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Build a synthetic Greek surface for one observation date."""
    settings = settings or Settings()
    cfg = settings.synthetic
    subset = df_with_greeks[df_with_greeks["date"] == str(dt)]
    if subset.empty:
        subset = df_with_greeks[pd.to_datetime(df_with_greeks["date"]) == pd.Timestamp(dt)]
    if subset.empty:
        return pd.DataFrame()

    spot = float(subset["spot"].median())
    base_rate = float(subset["r"].median())
    rate_grid = np.round(base_rate + np.asarray(cfg.rate_offsets, dtype=float), 6)
    tenors = np.sort(subset["ttm"].dropna().unique())
    if len(tenors) == 0:
        return pd.DataFrame()

    grid = pd.MultiIndex.from_product(
        [[spot], _strike_grid(subset, spot, cfg), rate_grid, tenors, ["CE", "PE"]],
        names=["spot", "strike", "r", "ttm", "option_type"],
    ).to_frame(index=False)

    merged = _attach_iv(grid, subset)
    merged = merged.dropna(subset=["iv"]).reset_index(drop=True)
    if merged.empty:
        return pd.DataFrame()

    flag = np.where(merged["option_type"].eq("CE"), "c", "p")
    forward = forward_price(
        merged["spot"].to_numpy(dtype=float),
        merged["r"].to_numpy(dtype=float),
        merged["ttm"].to_numpy(dtype=float),
    )
    merged["forward"] = forward
    merged["model_price"] = black76.price(
        flag,
        forward,
        merged["strike"].to_numpy(dtype=float),
        merged["ttm"].to_numpy(dtype=float),
        merged["r"].to_numpy(dtype=float),
        merged["iv"].to_numpy(dtype=float),
    )
    values = black76.greeks(
        flag,
        forward,
        merged["strike"].to_numpy(dtype=float),
        merged["ttm"].to_numpy(dtype=float),
        merged["r"].to_numpy(dtype=float),
        merged["iv"].to_numpy(dtype=float),
        settings=settings.greeks,
    )
    for name in GREEKS:
        merged[name] = values[name]

    merged["implied_volatility"] = merged["iv"]
    merged["moneyness"] = moneyness(merged["strike"], merged["spot"])
    merged["days_to_exp"] = merged["ttm"] * 365.0

    columns = [
        "spot",
        "strike",
        "r",
        "ttm",
        "days_to_exp",
        "option_type",
        "implied_volatility",
        "moneyness",
        "forward",
        "model_price",
        *GREEKS,
    ]
    return merged[columns].sort_values(["option_type", "ttm", "strike"]).reset_index(drop=True)
