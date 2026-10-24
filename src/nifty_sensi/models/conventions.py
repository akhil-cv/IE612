"""Single source of truth for pricing conventions and Greek units.

Conventions
-----------
* Forward: ``F = S * exp((r - q) * T)`` with ``q = 0`` for NIFTY (price index).
* Pricing: Black-76, ``V = exp(-rT) * [F N(d1) - K N(d2)]``.
* Moneyness: ``K / S`` (strike over spot), matching the legacy pipeline.
* Time: calendar days / 365.
* Delta: per 1.0 change in spot.
* Gamma: per 1.0 change in spot (second derivative).
* Vega: per 1 volatility point (1%) -- divide the raw derivative by 100.
* Theta: per calendar day -- divide the raw derivative by 365.
* Rho: per 1 percentage point of rate -- divide the raw derivative by 100.
* Vanna: per volatility point.
* Vomma: per volatility point squared.
* Zomma: per volatility point.
* Ultima: per volatility point cubed.
* Charm: change in delta per calendar day.
* Color: change in gamma per calendar day.
* Veta: change in vega (per vol point) per calendar day.
* Dual delta / dual gamma: derivatives with respect to strike.
"""

from __future__ import annotations

import numpy as np

GREEKS: tuple[str, ...] = (
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "vanna",
    "charm",
    "vomma",
    "speed",
    "zomma",
    "color",
    "veta",
    "ultima",
    "dual_delta",
    "dual_gamma",
)

DAYS_PER_YEAR = 365.0
VOL_POINT = 0.01
RATE_POINT = 0.01


def forward_price(spot, rate, ttm, dividend_yield=0.0):
    """Black-76 forward from spot."""
    return np.asarray(spot, dtype=float) * np.exp(
        (np.asarray(rate, dtype=float) - dividend_yield) * np.asarray(ttm, dtype=float)
    )


def moneyness(strike, spot) -> np.ndarray:
    """Strike-over-spot moneyness (legacy convention)."""
    return np.asarray(strike, dtype=float) / np.asarray(spot, dtype=float)


def is_call(flag) -> np.ndarray:
    """Accept ``c``/``p`` and legacy ``CE``/``PE`` flags."""
    values = np.asarray(flag, dtype=object)
    text = np.char.upper(values.astype(str))
    return (text == "C") | (text == "CE") | (text == "CALL")
