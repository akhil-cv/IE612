"""Stage 5: interpolate valid IVs onto the fixed moneyness x maturity grid."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings
from ..logging_utils import get_logger

log = get_logger("surface")


def _interp(x, y, target):
    """Linear interpolation with flat extrapolation."""
    order = np.argsort(x)
    return np.interp(target, np.asarray(x)[order], np.asarray(y)[order])


def _date_surface(day: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, dict]:
    m_grid = np.asarray(settings.grid.moneyness, dtype=float)
    tau_grid = np.asarray(settings.grid.tau_days, dtype=float) / 365.0

    tenors = sorted(day["ttm"].dropna().unique())
    matrix = np.full((len(tenors), len(m_grid)), np.nan)
    extrapolated = np.zeros((len(tenors), len(m_grid)), dtype=bool)
    used_tenors = []

    for i, tau in enumerate(tenors):
        expiry_slice = day[np.isclose(day["ttm"], tau)]
        expiry_slice = (
            expiry_slice.groupby("moneyness", as_index=False)["iv"].mean().sort_values("moneyness")
        )
        if len(expiry_slice) < settings.grid.min_points:
            continue
        m_obs = expiry_slice["moneyness"].to_numpy(dtype=float)
        iv_obs = expiry_slice["iv"].to_numpy(dtype=float)
        matrix[i, :] = _interp(m_obs, iv_obs, m_grid)
        extrapolated[i, :] = (m_grid < m_obs.min()) | (m_grid > m_obs.max())
        used_tenors.append(tau)

    keep = [i for i, tau in enumerate(tenors) if tau in used_tenors]
    matrix = matrix[keep, :]
    extrapolated = extrapolated[keep, :]
    tenors = np.asarray(used_tenors, dtype=float)

    surface = np.full((len(m_grid), len(tau_grid)), np.nan)
    if len(tenors) >= 2:
        for j in range(len(m_grid)):
            column = matrix[:, j]
            ok = np.isfinite(column)
            if ok.sum() >= 2:
                surface[j, :] = _interp(tenors[ok], column[ok], tau_grid)
    elif len(tenors) == 1:
        surface[:, :] = matrix[0, :][:, None]

    # Calendar-spread check on total variance w = iv^2 * tau.
    variance = surface**2 * tau_grid[None, :]
    diffs = np.diff(variance, axis=1)
    valid_pairs = np.isfinite(diffs)
    violations = int((diffs[valid_pairs] < -1e-8).sum()) if valid_pairs.any() else 0
    pairs = int(valid_pairs.sum())

    long = pd.DataFrame(
        {
            "moneyness": np.repeat(m_grid, len(tau_grid)),
            "tau_days": np.tile(np.asarray(settings.grid.tau_days, dtype=float), len(m_grid)),
            "tau": np.tile(tau_grid, len(m_grid)),
            "iv": surface.reshape(-1),
            "extrapolated": np.repeat(extrapolated.any(axis=0) if len(extrapolated) else np.zeros(len(m_grid), bool), len(tau_grid)),
        }
    )
    qc = {
        "tenors_used": int(len(tenors)),
        "grid_points": int(len(long)),
        "grid_missing": int(long["iv"].isna().sum()),
        "calendar_violations": violations,
        "calendar_pairs": pairs,
        "calendar_violation_pct": 100.0 * violations / pairs if pairs else 0.0,
    }
    return long, qc


def build_surfaces(iv_frame: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return long-format surfaces and a per-date QC table."""
    usable = iv_frame[iv_frame["valid_iv"].fillna(False)].copy()
    if usable.empty:
        return pd.DataFrame(), pd.DataFrame()

    surfaces = []
    reports = []
    for date, day in usable.groupby("date"):
        long, qc = _date_surface(day, settings)
        long.insert(0, "date", date)
        long.insert(1, "date_dt", pd.Timestamp(day["date_dt"].iloc[0]))
        surfaces.append(long)
        reports.append({"date": date, **qc})

    surface_df = pd.concat(surfaces, ignore_index=True)
    report_df = pd.DataFrame(reports)
    log.info(
        "surface: %d dates, %.1f%% grid coverage, median calendar violations %d",
        len(report_df),
        100.0 * (1 - surface_df["iv"].isna().mean()) if len(surface_df) else 0.0,
        int(report_df["calendar_violations"].median()) if len(report_df) else 0,
    )
    return surface_df, report_df
