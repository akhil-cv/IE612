"""Stage 1: cleaning, alignment and filtering of the raw option feed."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tests.helpers import fixture_frame, make_settings

from nifty_sensi.pipeline.prepare import optiondata_transform, prepare_options


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.options, self.spot = fixture_frame()
        self.settings = make_settings()

    def test_legacy_transform_contract(self):
        options, spot_df, spot, dates = optiondata_transform(self.options, self.spot)
        for column in ("date_dt", "exdate_dt", "days_to_exp", "ttm"):
            self.assertIn(column, options.columns)
        self.assertEqual(len(spot), len(dates))
        self.assertEqual(list(dates), sorted(dates))
        expected = {
            pd.Timestamp(date): close
            for date, close in zip(self.spot["date"], self.spot["close"])
        }
        np.testing.assert_allclose(
            spot, [expected[date] for date in dates]
        )
        self.assertIn("date_dt", spot_df.columns)

    def test_cleaning_drops_dirty_rows(self):
        prepared = prepare_options(self.options, self.spot, self.settings)
        self.assertEqual(len(prepared), 132)
        self.assertTrue((prepared["price"] > 0).all())
        self.assertTrue((prepared["strike"] > 0).all())
        self.assertTrue(prepared["option_type"].isin(["CE", "PE"]).all())
        self.assertTrue(prepared["days_to_exp"].between(1, 365).all())
        self.assertFalse(
            prepared.duplicated(["date", "expiry", "strike", "option_type"]).any()
        )

    def test_inputs_are_not_mutated(self):
        before_options = self.options.copy(deep=True)
        before_spot = self.spot.copy(deep=True)
        prepare_options(self.options, self.spot, self.settings)
        pd.testing.assert_frame_equal(self.options, before_options)
        pd.testing.assert_frame_equal(self.spot, before_spot)

    def test_spot_and_moneyness_alignment(self):
        prepared = prepare_options(self.options, self.spot, self.settings)
        expected_spot = dict(zip(self.spot["date"], self.spot["close"]))
        for date, group in prepared.groupby("date"):
            self.assertTrue(np.allclose(group["spot"], expected_spot[date]))
            self.assertTrue(np.allclose(group["moneyness"], group["strike"] / group["spot"]))

    def test_day_window_filter(self):
        settings = make_settings(options={"min_days": 1, "max_days": 10})
        prepared = prepare_options(self.options, self.spot, settings)
        self.assertTrue(prepared["days_to_exp"].between(1, 10).all())
        self.assertEqual(set(prepared["days_to_exp"]), {9})

    def test_duplicate_contracts_collapse(self):
        options = pd.concat([self.options, self.options.iloc[[1]]], ignore_index=True)
        prepared = prepare_options(options, self.spot, self.settings)
        self.assertEqual(len(prepared), 132)

    def test_missing_spot_raises(self):
        options = pd.concat(
            [
                self.options,
                self.options.iloc[[0]].assign(date="2025-02-03", timestamp="2025-02-03"),
            ],
            ignore_index=True,
        )
        with self.assertRaises(KeyError):
            optiondata_transform(options, self.spot)


if __name__ == "__main__":
    unittest.main()
