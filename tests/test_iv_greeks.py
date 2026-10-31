"""Stages 3 and 4: implied volatility recovery and Greek coverage."""

from __future__ import annotations

import unittest

import numpy as np

from tests.helpers import fixture_frame, make_settings, truth_frame

from nifty_sensi.models.conventions import GREEKS
from nifty_sensi.pipeline.greeks import attach_greeks
from nifty_sensi.pipeline.implied_vol import compute_iv
from nifty_sensi.pipeline.prepare import prepare_options
from nifty_sensi.pipeline.rate import attach_rates


class ImpliedVolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        options, spot = fixture_frame()
        cls.settings = make_settings()
        cls.prepared = prepare_options(options, spot, cls.settings)
        cls.rated, _ = attach_rates(cls.prepared, cls.settings)
        cls.iv_frame = compute_iv(cls.rated, cls.settings)

    def test_recovers_generating_volatility(self):
        truth = truth_frame()
        merged = self.iv_frame.merge(
            truth[["date", "expiry", "strike", "option_type", "_true_iv"]],
            on=["date", "expiry", "strike", "option_type"],
            how="inner",
        )
        merged = merged[merged["valid_iv"]]
        self.assertGreater(len(merged), 50)
        error = (merged["iv"] - merged["_true_iv"]).abs()
        # Market prices are rounded to the INR 0.05 tick, so cheap options carry
        # a comparatively large IV round-trip error by construction.
        self.assertLess(float(error.median()), 1e-4)
        self.assertLess(float(error.quantile(0.95)), 6e-3)
        self.assertLess(float(error.max()), 3e-2)

    def test_solver_and_repricing_quality(self):
        statuses = set(self.iv_frame["iv_status_label"].unique())
        self.assertTrue(statuses <= {"ok", "below_intrinsic"}, statuses)
        solved = self.iv_frame[self.iv_frame["iv_status_label"] == "ok"]
        relative = (solved["repricing_error"].abs() / (1.0 + solved["price"])).max()
        self.assertLess(float(relative), 5e-8)
        below = self.iv_frame[self.iv_frame["iv_status_label"] == "below_intrinsic"]
        if len(below):
            # Below-intrinsic quotes are moneyness artefacts of the 0.05 tick on
            # deep in-the-money contracts; they are never usable.
            self.assertTrue((below["is_otm"] == False).all())  # noqa: E712
            self.assertTrue(below["iv"].isna().all())
        self.assertGreater(int(self.iv_frame["valid_iv"].sum()), 60)

    def test_only_otm_contracts_are_usable(self):
        usable = self.iv_frame[self.iv_frame["valid_iv"]]
        calls = usable[usable["option_type"] == "CE"]
        puts = usable[usable["option_type"] == "PE"]
        self.assertTrue((calls["moneyness"] >= 1.0).all())
        self.assertTrue((puts["moneyness"] < 1.0).all())

    def test_every_status_has_a_label(self):
        self.assertFalse(self.iv_frame["iv_status_label"].isna().any())
        self.assertTrue(self.iv_frame["iv_status_label"].isin(["ok"]).any())


class GreekAttachmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        options, spot = fixture_frame()
        cls.settings = make_settings()
        prepared = prepare_options(options, spot, cls.settings)
        rated, _ = attach_rates(prepared, cls.settings)
        iv_frame = compute_iv(rated, cls.settings)
        cls.frame = attach_greeks(iv_frame, cls.settings)

    def test_all_greeks_are_added(self):
        for name in GREEKS:
            self.assertIn(name, self.frame.columns)
        computed = self.frame["delta"].notna()
        self.assertEqual(int(computed.sum()), int(self.frame["valid_iv"].sum()))

    def test_greek_signs_and_magnitudes(self):
        frame = self.frame[self.frame["valid_iv"]]
        calls = frame[frame["option_type"] == "CE"]
        puts = frame[frame["option_type"] == "PE"]
        self.assertTrue((calls["delta"] >= 0).all() and (calls["delta"] <= 1).all())
        self.assertTrue((puts["delta"] <= 0).all() and (puts["delta"] >= -1).all())
        self.assertTrue((frame["gamma"] > 0).all())
        self.assertTrue((frame["vega"] > 0).all())
        self.assertTrue((frame["theta"] < 0).mean() > 0.9)

    def test_no_greek_for_invalid_rows(self):
        invalid = self.frame[~self.frame["valid_iv"]]
        self.assertTrue(invalid["delta"].isna().all())
        self.assertTrue(invalid["vega"].isna().all())

    def test_analytic_greeks_agree_with_bumped_prices(self):
        from nifty_sensi.models import black76
        from nifty_sensi.models.conventions import forward_price

        sample = self.frame[self.frame["valid_iv"]].sample(25, random_state=3)
        flag = np.where(sample["option_type"].eq("CE"), "c", "p")
        spot = sample["spot"].to_numpy(float)
        strike = sample["strike"].to_numpy(float)
        ttm = sample["ttm"].to_numpy(float)
        rate = sample["r"].to_numpy(float)
        sigma = sample["iv"].to_numpy(float)

        def price(spot_=None, sigma_=None):
            spot_ = spot if spot_ is None else spot_
            sigma_ = sigma if sigma_ is None else sigma_
            return black76.price(
                flag, forward_price(spot_, rate, ttm), strike, ttm, rate, sigma_
            )

        bump = np.maximum(spot * 1e-4, 0.01)
        delta = (price(spot_=spot + bump) - price(spot_=spot - bump)) / (2 * bump)
        self.assertLess(float(np.max(np.abs(delta / sample["delta"] - 1))), 1e-3)


if __name__ == "__main__":
    unittest.main()
