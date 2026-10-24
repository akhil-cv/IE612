"""Black-76 pricing, implied volatility and the full Greek set.

All functions are vectorised and accept scalars or NumPy arrays. Prices are
discounted, so the correct input is the forward ``F`` (see
:func:`nifty_sensi.models.conventions.forward_price`). Greeks are expressed in
spot terms (``F = S exp(rT)``) with the units documented in
:mod:`nifty_sensi.models.conventions`.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .conventions import (
    DAYS_PER_YEAR,
    RATE_POINT,
    VOL_POINT,
    GREEKS,
    is_call,
)

# IV solver status codes
IV_OK = 0
IV_AT_LOWER = 1
IV_AT_UPPER = 2
IV_INVALID = 3
IV_NO_CONVERGENCE = 4
IV_BELOW_INTRINSIC = 5

STATUS_LABELS = {
    IV_OK: "ok",
    IV_AT_LOWER: "at_lower_bound",
    IV_AT_UPPER: "at_upper_bound",
    IV_INVALID: "invalid_input",
    IV_NO_CONVERGENCE: "no_convergence",
    IV_BELOW_INTRINSIC: "below_intrinsic",
}


# --------------------------------------------------------------------------- #
# Core pricing
# --------------------------------------------------------------------------- #
def _d1_d2(F, K, t, sigma):
    F, K, t, sigma = np.broadcast_arrays(
        np.asarray(F, dtype=float),
        np.asarray(K, dtype=float),
        np.asarray(t, dtype=float),
        np.asarray(sigma, dtype=float),
    )
    sqrt_t = np.sqrt(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * sigma * sigma * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def price(flag, F, K, t, r, sigma):
    """Discounted Black-76 option price. ``flag`` may be c/p/CE/PE."""
    F, K, t, r, sigma = (
        np.asarray(F, dtype=float),
        np.asarray(K, dtype=float),
        np.asarray(t, dtype=float),
        np.asarray(r, dtype=float),
        np.asarray(sigma, dtype=float),
    )
    call = is_call(flag)
    d1, d2 = _d1_d2(F, K, t, sigma)
    disc = np.exp(-r * t)
    with np.errstate(invalid="ignore"):
        value = np.where(
            call,
            disc * (F * norm.cdf(d1) - K * norm.cdf(d2)),
            disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1)),
        )
    return value


def price_from_spot(flag, S, K, t, r, sigma, q=0.0):
    """Price from spot, building the forward internally."""
    F = np.asarray(S, dtype=float) * np.exp(
        (np.asarray(r, dtype=float) - q) * np.asarray(t, dtype=float)
    )
    return price(flag, F, K, t, r, sigma)


def _core(flag, S, K, t, r, sigma):
    """Analytic first/second order Greeks in spot terms (unnormalised)."""
    S, K, t, r, sigma = np.broadcast_arrays(
        np.asarray(S, dtype=float),
        np.asarray(K, dtype=float),
        np.asarray(t, dtype=float),
        np.asarray(r, dtype=float),
        np.asarray(sigma, dtype=float),
    )
    call = is_call(flag)
    F = S * np.exp(r * t)
    d1, d2 = _d1_d2(F, K, t, sigma)
    pdf = norm.pdf(d1)
    disc = np.exp(-r * t)
    sqrt_t = np.sqrt(t)

    delta = np.where(call, norm.cdf(d1), norm.cdf(d1) - 1.0)
    gamma = pdf / (S * sigma * sqrt_t)
    vega = S * pdf * sqrt_t
    theta = -S * pdf * sigma / (2.0 * sqrt_t) - np.where(
        call,
        r * K * disc * norm.cdf(d2),
        -r * K * disc * norm.cdf(-d2),
    )
    rho = np.where(
        call,
        t * K * disc * norm.cdf(d2),
        -t * K * disc * norm.cdf(-d2),
    )
    vanna = -pdf * d2 / sigma
    vomma = S * sqrt_t * pdf * d1 * d2 / sigma
    dual_delta = np.where(call, -disc * norm.cdf(d2), disc * norm.cdf(-d2))
    dual_gamma = disc * norm.pdf(d2) / (K * sigma * sqrt_t)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
        "vanna": vanna,
        "vomma": vomma,
        "dual_delta": dual_delta,
        "dual_gamma": dual_gamma,
        "d1": d1,
        "d2": d2,
        "pdf": pdf,
    }


def _central_diff(func, bump: float):
    return (func(bump) - func(-bump)) / (2.0 * bump)


def greeks(flag, F, K, t, r, sigma, settings=None):
    """Return a DataFrame-like dict with all 15 Greeks, in documented units.

    ``settings`` is ``GreekSettings`` (theta/vega/rho normalisation). Higher
    order Greeks use central finite differences of the analytic sensitivities.
    """
    from ..config import GreekSettings

    settings = settings or GreekSettings()
    F, K, t, r, sigma = (
        np.asarray(F, dtype=float),
        np.asarray(K, dtype=float),
        np.asarray(t, dtype=float),
        np.asarray(r, dtype=float),
        np.asarray(sigma, dtype=float),
    )
    # Broadcast once so every derivative uses the same shape.
    F, K, t, r, sigma = np.broadcast_arrays(F, K, t, r, sigma)
    valid = (
        np.isfinite(F)
        & np.isfinite(K)
        & np.isfinite(t)
        & np.isfinite(r)
        & np.isfinite(sigma)
        & (F > 0)
        & (K > 0)
        & (t > 1e-6)
        & (sigma > 1e-6)
    )
    t_safe = np.where(valid, t, 1.0)
    sigma_safe = np.where(valid, sigma, 0.2)
    S = F * np.exp(-r * t_safe)

    base = _core(flag, S, K, t_safe, r, sigma_safe)

    def gamma_at(s):
        return _core(flag, s, K, t_safe, r, sigma_safe)["gamma"]

    def vega_at(v):
        return _core(flag, S, K, t_safe, r, v)["vega"]

    def vomma_at(v):
        return _core(flag, S, K, t_safe, r, v)["vomma"]

    def delta_at(tt):
        return _core(flag, S, K, tt, r, sigma_safe)["delta"]

    def gamma_at_t(tt):
        return _core(flag, S, K, tt, r, sigma_safe)["gamma"]

    def vega_at_t(tt):
        return _core(flag, S, K, tt, r, sigma_safe)["vega"]

    h_s = np.maximum(S * 1e-3, 0.5)
    h_v = 1e-3
    h_t = np.maximum(t_safe * 1e-3, 1e-6)

    speed = _central_diff(lambda x: gamma_at(S + x), h_s)
    zomma = _central_diff(lambda x: gamma_at_sigma(flag, S, K, t_safe, r, sigma_safe + x), h_v)
    charm = _central_diff(lambda x: delta_at(t_safe + x), h_t) / DAYS_PER_YEAR
    color = _central_diff(lambda x: gamma_at_t(t_safe + x), h_t) / DAYS_PER_YEAR
    veta = _central_diff(lambda x: vega_at_t(t_safe + x), h_t) / DAYS_PER_YEAR
    ultima = _central_diff(lambda x: vomma_at(sigma_safe + x), h_v)

    vega_scale = VOL_POINT if settings.vega_per_point else 1.0
    theta_scale = DAYS_PER_YEAR if settings.theta_per_day else 1.0
    rho_scale = RATE_POINT if settings.rho_per_point else 1.0

    result = {
        "delta": base["delta"],
        "gamma": base["gamma"],
        "vega": base["vega"] * vega_scale,
        "theta": base["theta"] / theta_scale,
        "rho": base["rho"] * rho_scale,
        "vanna": base["vanna"] * VOL_POINT,
        "charm": charm,
        "vomma": base["vomma"] * VOL_POINT * VOL_POINT,
        "speed": speed,
        "zomma": zomma * VOL_POINT,
        "color": color,
        "veta": veta * VOL_POINT,
        "ultima": ultima * VOL_POINT**3,
        "dual_delta": base["dual_delta"],
        "dual_gamma": base["dual_gamma"],
    }
    for name in GREEKS:
        result[name] = np.where(valid, result[name], np.nan)
    return result


def gamma_at_sigma(flag, S, K, t, r, sigma):
    """Internal helper: gamma as a function of sigma."""
    return _core(flag, S, K, t, r, sigma)["gamma"]


# --------------------------------------------------------------------------- #
# Implied volatility
# --------------------------------------------------------------------------- #
def _iv_scalar(target, F, K, t, r, flag, lo, hi, tol):
    def objective(sigma):
        return float(price(flag, F, K, t, r, sigma)) - target

    try:
        return brentq(objective, lo, hi, xtol=tol, maxiter=200), IV_OK
    except (ValueError, RuntimeError):
        return np.nan, IV_NO_CONVERGENCE


def implied_vol(
    market_price,
    F,
    K,
    t,
    r,
    flag,
    lower=1e-4,
    upper=5.0,
    tolerance=1e-8,
    max_iter=60,
):
    """Vectorised implied volatility.

    Returns ``(iv, status)`` where status is one of the ``IV_*`` codes.
    Newton iterations run first; non-converged rows fall back to Brent.
    """
    target, F, K, t, r = (
        np.asarray(market_price, dtype=float),
        np.asarray(F, dtype=float),
        np.asarray(K, dtype=float),
        np.asarray(t, dtype=float),
        np.asarray(r, dtype=float),
    )
    scalar_input = target.ndim == 0
    target, F, K, t, r = (
        np.atleast_1d(x) for x in (target, F, K, t, r)
    )
    target, F, K, t, r = np.broadcast_arrays(target, F, K, t, r)
    call = np.broadcast_to(is_call(flag), target.shape)

    disc = np.exp(-r * t)
    with np.errstate(invalid="ignore", divide="ignore"):
        intrinsic = np.where(call, np.maximum(disc * (F - K), 0.0), np.maximum(disc * (K - F), 0.0))

    valid = (
        np.isfinite(target)
        & np.isfinite(F)
        & np.isfinite(K)
        & np.isfinite(t)
        & np.isfinite(r)
        & (target > 0)
        & (F > 0)
        & (K > 0)
        & (t > 1e-8)
    )
    iv = np.full(target.shape, np.nan)
    status = np.full(target.shape, IV_INVALID, dtype=np.int8)

    below = valid & (target < intrinsic - tolerance)
    status[below] = IV_BELOW_INTRINSIC

    # Brenner-Subrahmanyam style starting point from the time value.
    tv = np.maximum(target - intrinsic, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        guess = np.sqrt(2.0 * np.pi / t) * tv / np.maximum(disc * F, 1e-12)
    guess = np.clip(np.where(np.isfinite(guess), guess, 0.3), lower + 1e-6, upper - 1e-6)

    active = valid & ~below
    sigma = np.where(active, guess, lower)
    tol_vec = tolerance * (1.0 + np.abs(target))

    for _ in range(max_iter):
        if not active.any():
            break
        model = price(flag, F, K, t, r, sigma)
        d1, _ = _d1_d2(F, K, t, sigma)
        vega_raw = disc * F * norm.pdf(d1) * np.sqrt(t)
        vega_raw = np.maximum(vega_raw, 1e-12)
        step = np.clip((model - target) / vega_raw, -0.5, 0.5)
        candidate = np.clip(sigma - step, lower, upper)
        converged = np.abs(model - target) <= tol_vec
        sigma = np.where(active & ~converged, candidate, sigma)
        active = active & ~converged

    iv = np.where(valid & ~below, sigma, np.nan)
    status = np.where(valid, status, IV_INVALID)
    status = np.where((status == IV_INVALID) & valid, IV_NO_CONVERGENCE, status)

    model = price(flag, F, K, t, r, sigma)
    with np.errstate(invalid="ignore"):
        residual = np.abs(model - target)
    solvable = valid & ~below
    ok = solvable & (residual <= tol_vec)
    status = np.where(ok, IV_OK, status)
    status = np.where(solvable & ~ok & (sigma <= lower + 1e-12), IV_AT_LOWER, status)
    status = np.where(solvable & ~ok & (sigma >= upper - 1e-12), IV_AT_UPPER, status)

    # Brent fallback for the few rows Newton could not solve. ``xtol`` is in
    # volatility units, so it is tightened to keep the price residual small.
    retry = np.where(solvable & ~ok & (status == IV_NO_CONVERGENCE))[0]
    for idx in retry:
        value, code = _iv_scalar(
            target.flat[idx],
            F.flat[idx],
            K.flat[idx],
            t.flat[idx],
            r.flat[idx],
            "c" if call.flat[idx] else "p",
            lower,
            upper,
            max(tolerance * 1e-3, 1e-12),
        )
        if np.isfinite(value):
            iv.flat[idx] = value
            status.flat[idx] = code

    if scalar_input:
        return iv[0], status[0]
    return iv, status
