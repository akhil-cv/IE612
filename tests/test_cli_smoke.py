"""End-to-end CLI test: build -> cache -> report on a temporary root."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tests.helpers import FIXTURES

from nifty_sensi.cli import main
from nifty_sensi.config import load_settings

CONFIG = {
    "data": {
        "options_file": "data/options.csv",
        "spot_file": "data/spot.csv",
        "interim_dir": "data/interim",
        "processed_dir": "data/processed",
        "artifacts_dir": "artifacts",
    },
    "report": {"dpi": 60, "max_dates": 2},
}


class CliSmokeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "data").mkdir()
        shutil.copyfile(FIXTURES / "options_sample.csv", self.root / "data" / "options.csv")
        shutil.copyfile(FIXTURES / "spot_sample.csv", self.root / "data" / "spot.csv")
        (self.root / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
        self.args = ["--root", str(self.root), "--config", "config.json"]

    def tearDown(self):
        self._tmp.cleanup()

    def test_full_build_is_cached_and_reports(self):
        self.assertEqual(main(self.args + ["build"]), 0)

        greeks = self.root / "data" / "processed" / "greeks.parquet"
        surface = self.root / "data" / "processed" / "surfaces.parquet"
        self.assertTrue(greeks.exists())
        self.assertTrue(surface.exists())

        import pandas as pd

        frame = pd.read_parquet(greeks)
        self.assertGreater(len(frame), 100)
        self.assertGreater(int(frame["delta"].notna().sum()), 50)
        self.assertIn("dual_gamma", frame.columns)

        # Second run must reuse the stage caches.
        self.assertEqual(main(self.args + ["build"]), 0)
        settings = load_settings(self.root / "config.json", root=self.root)
        pointer = json.loads(
            (settings.artifacts_dir / "latest.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (settings.artifacts_dir / "runs" / pointer["run_id"] / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        statuses = [stage["status"] for stage in manifest["stages"]]
        self.assertEqual(statuses, ["cached"] * 5)

        report = self.root / "report.html"
        self.assertEqual(main(self.args + ["report", "--out", str(report)]), 0)
        html = report.read_text(encoding="utf-8")
        self.assertIn("data:image/png;base64,", html)
        self.assertIn("NIFTY 50 Option Sensitivity Report", html)
        self.assertGreater(len(html), 50_000)

        self.assertEqual(main(self.args + ["qc"]), 0)

    def test_unknown_stage_is_rejected(self):
        self.assertEqual(main(self.args + ["build", "--stages", "nope"]), 1)


if __name__ == "__main__":
    unittest.main()
