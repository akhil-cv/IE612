"""Stage 5: IV grid interpolation and calendar-spread checks."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tests.helpers import fixture_frame, make_settings

from nifty_sensi.pipeline.prepare import prepare_options
from nifty_sensi.pipeline.surface import build_surfaces


class SurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        options, spot = fixture_frame()
        cls.settings = make_settings()
        prepared = prepare_options(options, spot, cls.settings)
        prepared["r"] = 0.065
        prepared["iv"] = 0.2
        prepared["valid_iv"] = True
        cls.frame = prepared
        cls.surface, cls.report = build_surfaces(prepared, cls.settings)

    def test_grid_shape_matches_config(self):
        n_dates = self.frame["date"].nunique()
        expected = n_dates * len(self.settings.grid.moneyness) * len(self.settings.grid.tau_days)
        self.assertEqual(len(self.surface), expected)
        self.assertEqual(len(self.report), n_dates)

    def test_grid_values_are_finite_and_plausible(self):
        self.assertTrue(self.surface["iv"].notna().all())
        self.assertTrue(self.surface["iv"].between(0.0, 5.0).all())
        np.testing.assert_allclose(
            np.sort(self.surface["tau_days"].unique()),
            np.asarray(self.settings.grid.tau_days, dtype=float),
        )
        np.testing.assert_allclose(
            np.sort(self.surface["moneyness"].unique()),
            np.asarray(self.settings.grid.moneyness, dtype=float),
        )

    def test_flat_iv_surface_has_no_calendar_violations(self):
        self.assertTrue((self.report["calendar_violations"] == 0).all())
        self.assertTrue((self.report["calendar_pairs"] > 0).all())

    def test_missing_iv_returns_empty_frames(self):
        frame = self.frame.copy()
        frame["valid_iv"] = False
        surface, report = build_surfaces(frame, self.settings)
        self.assertTrue(surface.empty)
        self.assertTrue(report.empty)

    def test_monotone_variance_has_no_violations(self):
        # Deliberately invert the term structure of *volatility* but keep total
        # variance increasing so the calendar check stays quiet.
        frame = pd.DataFrame(
            {
                "date": ["2025-01-01"] * 24,
                "date_dt": [pd.Timestamp("2025-01-01")] * 24,
                "ttm": [9 / 365] * 12 + [37 / 365] * 12,
                "moneyness": np.tile(np.linspace(0.9, 1.0, 12), 2),
                "iv": np.concatenate([np.full(12, 0.24), np.full(12, 0.18)]),
                "valid_iv": [True] * 24,
            }
        )
        surface, report = build_surfaces(frame, self.settings)
        self.assertFalse(surface.empty)
        self.assertEqual(int(report["calendar_violations"].iloc[0]), 0)


if __name__ == "__main__":
    unittest.main()
