"""Stage 2: put-call-parity rates and their fallback chain."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tests.helpers import fixture_frame, make_settings, truth_frame

from nifty_sensi.pipeline.prepare import prepare_options
from nifty_sensi.pipeline.rate import attach_rates, data_day, estimate_rates, parity_rates


class RateTests(unittest.TestCase):
    def setUp(self):
        options, spot = fixture_frame()
        self.spot = spot
        self.settings = make_settings()
        self.prepared = prepare_options(options, spot, self.settings)

    def test_parity_recovers_known_rate(self):
        rates = parity_rates(self.prepared[self.prepared["date"] == "2025-01-01"], self.settings)
        self.assertGreater(len(rates), 0)
        self.assertAlmostEqual(float(rates.median()), 0.065, delta=0.01)

    def test_attach_rates_adds_r_column(self):
        merged, rates = attach_rates(self.prepared, self.settings)
        self.assertIn("r", merged.columns)
        self.assertTrue(merged["r"].notna().all())
        np.testing.assert_allclose(merged["r"], merged["rate"])
        self.assertTrue((rates["rate_source"] == "parity").all())
        self.assertEqual(len(rates), self.prepared["date"].nunique())

    def test_fallback_chain(self):
        # Remove all puts: parity is impossible, so the constant fallback must apply.
        calls_only = self.prepared[self.prepared["option_type"] == "CE"].copy()
        settings = make_settings(rate={"fallback": 0.042, "rolling_window": 5})
        table = estimate_rates(calls_only, settings)
        self.assertTrue((table["rate_source"] == "fallback_constant").all())
        np.testing.assert_allclose(table["rate"], 0.042)

    def test_rolling_median_fills_single_gap(self):
        prepared = self.prepared.copy()
        last_date = prepared["date_dt"].max()
        # Drop the puts on the last date: parity is undefined there.
        prepared = prepared[
            ~((prepared["date_dt"] == last_date) & (prepared["option_type"] == "PE"))
        ]
        table = estimate_rates(prepared, make_settings(rate={"rolling_window": 3}))
        gap = table[table["date_dt"] == last_date]
        self.assertEqual(len(gap), 1)
        self.assertEqual(gap["rate_source"].iloc[0], "rolling_median")
        self.assertAlmostEqual(float(gap["rate"].iloc[0]), 0.065, delta=0.02)

    def test_legacy_data_day(self):
        day, rate = data_day(
            "2025-01-02", self.prepared, self.spot, self.settings
        )
        self.assertFalse(day.empty)
        self.assertAlmostEqual(rate, 0.065, delta=0.01)
        self.assertTrue(np.allclose(day["r"], rate))
        self.assertIn("mid_price", day.columns)

    def test_rate_matches_truth(self):
        truth = truth_frame()
        self.assertTrue(np.allclose(truth["_true_r"].unique(), 0.065))


if __name__ == "__main__":
    unittest.main()
