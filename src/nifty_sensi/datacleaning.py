"""Legacy-compatible cleaning helpers.

The original ``datacleaning.py`` module is untouched. This shim mirrors its
public functions so new code can ``from nifty_sensi import datacleaning`` while
old notebooks keep importing the root module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline, interp1d
from scipy import interpolate as _interpolate

from .pipeline.prepare import optiondata_transform
from .pipeline.rate import data_day


def kernel(x, y, h1, h2):
    return (np.exp(-x * x / (2 * h1)) * np.exp(-y * y / (2 * h2))) / (2 * np.pi)


def kernelVega(x, y, h1, h2, vega):
    return vega * kernel(x, y, h1, h2)


def Smooth(I_in, m_in, tau_in, m_want, tau_want, h1, h2):
    out = np.zeros((len(m_want), len(tau_want)))
    for i, m in enumerate(m_want):
        for j, tau in enumerate(tau_want):
            weights = kernel(m - m_in, tau - tau_in, h1, h2)
            out[i, j] = np.sum(I_in * weights) / np.sum(weights)
    return out


def SmoothVega(I_in, m_in, tau_in, m_want, tau_want, h1, h2, vega):
    out = np.zeros((len(m_want), len(tau_want)))
    for i, m in enumerate(m_want):
        for j, tau in enumerate(tau_want):
            weights = kernelVega(m - m_in, tau - tau_in, h1, h2, vega)
            out[i, j] = np.sum(I_in * weights) / np.sum(weights)
    return out


def interpolate_m(m_known, iv_known, m_want, k=1):
    tck = _interpolate.splrep(m_known, iv_known, k=k)
    return BSpline(tck[0], tck[1], tck[2])(m_want)


def interpolate_tau(tau_known, iv_known, tau_want, k=1):
    tck = _interpolate.splrep(tau_known, iv_known, k=k)
    return BSpline(tck[0], tck[1], tck[2])(tau_want)


def interpolate_and_extrapolate(x_data, y_data, x):
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length")
    order = np.argsort(x_data)
    return interp1d(
        np.asarray(x_data)[order],
        np.asarray(y_data)[order],
        kind="linear",
        fill_value="extrapolate",
    )(x)


def interpolate_data(otm_df, m_want, tau_want, ttm_available):
    iv_interpol1 = np.zeros((len(m_want), len(ttm_available)))
    for i, tau in enumerate(ttm_available):
        slice_ = otm_df[np.isclose(otm_df["ttm"], tau)]
        order = np.argsort(slice_["moneyness"])
        iv_interpol1[:, i] = interpolate_and_extrapolate(
            slice_["moneyness"].to_numpy()[order],
            slice_["impl_volatility"].to_numpy()[order],
            m_want,
        )
    iv_interpol2 = np.zeros((len(m_want), len(tau_want)))
    order = np.argsort(ttm_available)
    for i in range(len(m_want)):
        iv_interpol2[i, :] = interpolate_and_extrapolate(
            np.asarray(ttm_available)[order], iv_interpol1[i, order], tau_want
        )
    return iv_interpol2


def interpolate_surface(iv_in, m_in, tau_in, m_out, tau_out):
    iv_interpol1 = np.zeros((len(m_out), len(tau_in)))
    for i in range(len(tau_in)):
        iv_interpol1[:, i] = interpolate_and_extrapolate(m_in, iv_in[:, i], m_out)
    iv_interpol2 = np.zeros((len(m_out), len(tau_out)))
    for i in range(len(m_out)):
        iv_interpol2[i, :] = interpolate_and_extrapolate(tau_in, iv_interpol1[i, :], tau_out)
    return iv_interpol2


def interpolate_dataBspline(otm_df, m_want, tau_want, ttm_available):
    iv_interpol1 = np.zeros((len(m_want), len(ttm_available)))
    for i, tau in enumerate(ttm_available):
        slice_ = otm_df[np.isclose(otm_df["ttm"], tau)]
        order = np.argsort(slice_["moneyness"])
        iv_interpol1[:, i] = interpolate_m(
            slice_["moneyness"].to_numpy()[order],
            slice_["impl_volatility"].to_numpy()[order],
            m_want,
            k=1,
        )
    iv_interpol2 = np.zeros((len(m_want), len(tau_want)))
    order = np.argsort(ttm_available)
    for i in range(len(m_want)):
        iv_interpol2[i, :] = interpolate_tau(
            np.asarray(ttm_available)[order], iv_interpol1[i, order], tau_want, k=1
        )
    return iv_interpol2


__all__ = [
    "optiondata_transform",
    "data_day",
    "kernel",
    "kernelVega",
    "Smooth",
    "SmoothVega",
    "interpolate_m",
    "interpolate_tau",
    "interpolate_and_extrapolate",
    "interpolate_data",
    "interpolate_surface",
    "interpolate_dataBspline",
]
