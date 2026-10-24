"""Generate the small, deterministic test fixture used by ``tests/``.

The fixture is synthetic but realistic: prices come from the package's own
Black-76 engine at a known rate and moneyness-dependent volatility, then rounded
to the INR 0.05 tick used by NSE. Tests therefore know the true answer.

Usage::

    python scripts/make_fixtures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nifty_sensi.models import black76  # noqa: E402
from nifty_sensi.models.conventions import forward_price  # noqa: E402

FIXTURE_DIR = ROOT / "tests" / "fixtures"
RATE = 0.065
DATES = ("2025-01-01", "2025-01-02", "2025-01-03")
SPOTS = {"2025-01-01": 21000.0, "2025-01-02": 21150.0, "2025-01-03": 21050.0}
TENOR_DAYS = (9, 37)
MONEYNESS = np.round(np.arange(0.90, 1.101, 0.02), 4)
TICK = 0.05


def _vol(moneyness: float, tau: float) -> float:
    """Skewed smile with a mild term structure."""
    return 0.15 + 1.2 * (moneyness - 1.0) ** 2 - 0.25 * (moneyness - 1.0) + 0.03 * np.sqrt(
        tau / 0.1
    )


def _rounded(value: float) -> float:
    return max(round(value / TICK) * TICK, TICK)


def build_options() -> pd.DataFrame:
    rows = []
    for date in DATES:
        spot = SPOTS[date]
        for days in TENOR_DAYS:
            expiry = pd.Timestamp(date) + pd.Timedelta(days=days)
            tau = days / 365.0
            forward = float(forward_price(spot, RATE, tau))
            for moneyness in MONEYNESS:
                strike = float(round(spot * moneyness / 50.0) * 50.0)
                sigma = _vol(moneyness, tau)
                for option_type in ("CE", "PE"):
                    value = black76.price(option_type, forward, strike, tau, RATE, sigma)
                    rows.append(
                        {
                            "date": date,
                            "timestamp": date,
                            "expiry": expiry.strftime("%Y-%m-%d"),
                            "strike": strike,
                            "option_type": option_type,
                            "close": _rounded(float(value)),
                            "_true_iv": sigma,
                            "_true_r": RATE,
                        }
                    )
    frame = pd.DataFrame(rows)

    # Dirty rows: the cleaner must drop or de-duplicate each of these.
    first = frame.iloc[0].to_dict()
    dirty = pd.DataFrame(
        [
            first,  # exact duplicate contract
            {**first, "close": 0.0},
            {**first, "close": -5.0},
            {**first, "strike": 0.0},
            {**first, "option_type": "XX"},
            {
                **first,
                "date": "2025-01-01",
                "expiry": "2024-12-31",
                "timestamp": "2025-01-01",
            },
        ]
    )
    return pd.concat([frame, dirty], ignore_index=True)


def build_spot() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": list(SPOTS),
            "open": [value - 20 for value in SPOTS.values()],
            "high": [value + 80 for value in SPOTS.values()],
            "low": [value - 90 for value in SPOTS.values()],
            "close": list(SPOTS.values()),
        }
    )


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    options = build_options()
    spot = build_spot()
    truth = (
        options.loc[options["close"] > 0]
        .drop_duplicates(["date", "expiry", "strike", "option_type"])
        [["date", "expiry", "strike", "option_type", "close", "_true_iv", "_true_r"]]
        .reset_index(drop=True)
    )
    # The raw feed must look exactly like the production input (no truth columns).
    raw = options.drop(columns=["_true_iv", "_true_r"])
    raw.to_csv(FIXTURE_DIR / "options_sample.csv", index=False)
    spot.to_csv(FIXTURE_DIR / "spot_sample.csv", index=False)
    truth.to_csv(FIXTURE_DIR / "truth_sample.csv", index=False)
    print(f"options: {len(raw)} rows -> {FIXTURE_DIR / 'options_sample.csv'}")
    print(f"spot: {len(spot)} rows -> {FIXTURE_DIR / 'spot_sample.csv'}")
    print(f"truth: {len(truth)} rows -> {FIXTURE_DIR / 'truth_sample.csv'}")


if __name__ == "__main__":
    main()
