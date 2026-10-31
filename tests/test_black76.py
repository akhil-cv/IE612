"""Black-76 pricing, implied volatility and Greek conventions."""

from __future__ import annotations

import unittest

import numpy as np

from tests.helpers import SRC  # noqa: F401  (imports put ``src`` on sys.path)

from nifty_sensi.config import GreekSettings
from nifty_sensi.models import black76
from nifty_sensi.models.conventions import GREEKS, forward_price


class PriceTests(unittest.TestCase):
    def test_reference_values(self):
        # Standard textbook example: S=K=100, r=5%, sigma=20%, T=1.
        forward = float(forward_price(100.0, 0.05, 1.0))
        call = float(black76.price("c", forward, 100.0, 1.0, 0.05, 0.2))
        put = float(black76.price("p", forward, 100.0, 1.0, 0.05, 0.2))
        self.assertAlmostEqual(call, 10.4506, places=3)
        self.assertAlmostEqual(put, 5.5735, places=3)

    def test_put_call_parity(self):
        forward = float(forward_price(23742.9, 0.07, 0.25))
        call = black76.price("CE", forward, 24000.0, 0.25, 0.07, 0.18)
        put = black76.price("PE", forward, 24000.0, 0.25, 0.07, 0.18)
        expected = np.exp(-0.07 * 0.25) * (forward - 24000.0)
        self.assertAlmostEqual(float(call - put), float(expected), places=10)

    def test_vector_flags_match_scalars(self):
        forward = forward_price(np.array([100.0, 120.0]), 0.05, np.array([0.5, 1.0]))
        strikes = np.array([100.0, 100.0])
        ttm = np.array([0.5, 1.0])
        values = black76.price(
            np.array(["CE", "PE"]), forward, strikes, ttm, 0.05, np.array([0.2, 0.25])
        )
        self.assertEqual(values.shape, (2,))
        self.assertGreater(values[0], 0.0)
        self.assertGreater(values[1], 0.0)


class ImpliedVolTests(unittest.TestCase):
    def test_round_trip_vector(self):
        forward = forward_price(np.array([100.0, 105.0, 95.0]), 0.05, np.array([0.5, 1.0, 0.25]))
        strikes = np.array([100.0, 100.0, 100.0])
        ttm = np.array([0.5, 1.0, 0.25])
        sigma = np.array([0.15, 0.3, 0.45])
        flags = np.array(["c", "p", "c"])
        prices = black76.price(flags, forward, strikes, ttm, 0.05, sigma)
        iv, status = black76.implied_vol(prices, forward, strikes, ttm, 0.05, flags)
        np.testing.assert_allclose(iv, sigma, atol=1e-7)
        self.assertTrue((status == black76.IV_OK).all())

    def test_scalar_round_trip(self):
        forward = float(forward_price(100.0, 0.05, 1.0))
        price = float(black76.price("c", forward, 100.0, 1.0, 0.05, 0.2))
        iv, status = black76.implied_vol(price, forward, 100.0, 1.0, 0.05, "c")
        self.assertAlmostEqual(float(iv), 0.2, places=8)
        self.assertEqual(int(status), black76.IV_OK)

    def test_below_intrinsic_is_flagged_and_returns_nan(self):
        # Deep in-the-money call priced below its discounted intrinsic value.
        forward = 120.0
        iv, status = black76.implied_vol(5.0, forward, 100.0, 0.5, 0.05, "c")
        self.assertTrue(np.isnan(float(iv)))
        self.assertEqual(int(status), black76.IV_BELOW_INTRINSIC)

    def test_non_convergence_uses_upper_bound(self):
        forward = 100.0
        price = float(black76.price("c", forward, 80.0, 1.0, 0.05, 4.9))
        iv, status = black76.implied_vol(price, forward, 80.0, 1.0, 0.05, "c", upper=1.0)
        self.assertEqual(int(status), black76.IV_AT_UPPER)
        self.assertAlmostEqual(float(iv), 1.0, places=10)


class GreekTests(unittest.TestCase):
    SPOT, STRIKE, TTM, RATE, SIGMA = 100.0, 100.0, 0.75, 0.05, 0.25

    def setUp(self):
        self.forward = float(forward_price(self.SPOT, self.RATE, self.TTM))
        self.greeks = black76.greeks(
            "c", self.forward, self.STRIKE, self.TTM, self.RATE, self.SIGMA
        )

    def _price(self, spot=None, sigma=None, ttm=None, rate=None):
        spot = self.SPOT if spot is None else spot
        sigma = self.SIGMA if sigma is None else sigma
        ttm = self.TTM if ttm is None else ttm
        rate = self.RATE if rate is None else rate
        return black76.price(
            "c", forward_price(spot, rate, ttm), self.STRIKE, ttm, rate, sigma
        )

    def test_all_greeks_present_and_finite(self):
        for name in GREEKS:
            self.assertIn(name, self.greeks)
            self.assertTrue(np.isfinite(float(self.greeks[name])), name)

    def test_first_order_greeks_match_finite_differences(self):
        h, hv, ht = 1e-3, 1e-4, 1e-5
        checks = {
            "delta": (self._price(spot=self.SPOT + h) - self._price(spot=self.SPOT - h)) / (2 * h),
            "gamma": (
                self._price(spot=self.SPOT + h)
                - 2 * self._price()
                + self._price(spot=self.SPOT - h)
            )
            / h**2,
            "vega": (self._price(sigma=self.SIGMA + hv) - self._price(sigma=self.SIGMA - hv))
            / (2 * hv)
            * 0.01,
            "theta": -(self._price(ttm=self.TTM + ht) - self._price(ttm=self.TTM - ht))
            / (2 * ht)
            / 365.0,
            "rho": (self._price(rate=self.RATE + hv) - self._price(rate=self.RATE - hv))
            / (2 * hv)
            * 0.01,
        }
        for name, expected in checks.items():
            self.assertAlmostEqual(float(self.greeks[name]), expected, places=6, msg=name)

    def test_higher_order_greeks_match_finite_differences(self):
        h, hv, ht = 1e-4, 1e-3, 1e-4

        def core(spot=None, sigma=None, ttm=None):
            return black76._core(
                "c",
                self.SPOT if spot is None else spot,
                self.STRIKE,
                self.TTM if ttm is None else ttm,
                self.RATE,
                self.SIGMA if sigma is None else sigma,
            )

        speed = (core(spot=self.SPOT + h)["gamma"] - core(spot=self.SPOT - h)["gamma"]) / (2 * h)
        charm = (
            (core(ttm=self.TTM + ht)["delta"] - core(ttm=self.TTM - ht)["delta"])
            / (2 * ht)
            / 365.0
        )
        color = (
            (core(ttm=self.TTM + ht)["gamma"] - core(ttm=self.TTM - ht)["gamma"])
            / (2 * ht)
            / 365.0
        )
        veta = (
            (core(ttm=self.TTM + ht)["vega"] - core(ttm=self.TTM - ht)["vega"])
            / (2 * ht)
            / 365.0
            * 0.01
        )
        zomma = (
            (core(sigma=self.SIGMA + hv)["gamma"] - core(sigma=self.SIGMA - hv)["gamma"])
            / (2 * hv)
            * 0.01
        )
        vomma = (
            (core(sigma=self.SIGMA + hv)["vega"] - core(sigma=self.SIGMA - hv)["vega"])
            / (2 * hv)
            * 0.01**2
        )
        ultima = (
            (core(sigma=self.SIGMA + hv)["vomma"] - core(sigma=self.SIGMA - hv)["vomma"])
            / (2 * hv)
            * 0.01**3
        )
        relative = {
            name: abs(float(self.greeks[name]) - expected) / max(abs(expected), 1e-12)
            for name, expected in {
                "speed": speed,
                "charm": charm,
                "color": color,
                "veta": veta,
                "zomma": zomma,
                "vomma": vomma,
                "ultima": ultima,
            }.items()
        }
        for name, error in relative.items():
            self.assertLess(error, 2e-3, name)

    def test_configurable_units(self):
        raw = black76.greeks(
            "c",
            self.forward,
            self.STRIKE,
            self.TTM,
            self.RATE,
            self.SIGMA,
            settings=GreekSettings(theta_per_day=False, vega_per_point=False, rho_per_point=False),
        )
        self.assertAlmostEqual(float(raw["vega"]), float(self.greeks["vega"]) * 100.0, places=6)
        self.assertAlmostEqual(float(raw["theta"]), float(self.greeks["theta"]) * 365.0, places=6)
        self.assertAlmostEqual(float(raw["rho"]), float(self.greeks["rho"]) * 100.0, places=6)

    def test_invalid_inputs_produce_nan(self):
        values = black76.greeks("c", 100.0, 100.0, 0.0, 0.05, 0.2)
        for name in GREEKS:
            self.assertTrue(np.isnan(float(values[name])), name)


if __name__ == "__main__":
    unittest.main()
