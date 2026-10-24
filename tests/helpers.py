"""Shared helpers: make the src layout importable and locate fixtures."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nifty_sensi.config import Settings, load_settings  # noqa: E402


def fixture_frame():
    """Raw option and spot frames from the committed fixture."""
    options = pd.read_csv(FIXTURES / "options_sample.csv")
    spot = pd.read_csv(FIXTURES / "spot_sample.csv")
    return options, spot


def truth_frame() -> pd.DataFrame:
    """Fixture rows with the known IV/rate used to generate the prices."""
    return pd.read_csv(FIXTURES / "truth_sample.csv")


def make_settings(root: Path | None = None, **overrides) -> Settings:
    """Settings with fixture-friendly paths (defaults otherwise)."""
    base = {
        "spot_file": "data/spot.csv",
        "options_file": "data/options.csv",
        "interim_dir": "data/interim",
        "processed_dir": "data/processed",
        "artifacts_dir": "artifacts",
    }
    base.update(overrides.pop("data", {}) or {})
    payload = {"data": base}
    payload.update(overrides)
    settings = Settings.from_dict(payload, root=root or ROOT)
    return settings


def stage_fixture_root(root: Path) -> Settings:
    """Materialise the fixture into ``root`` and return settings for it."""
    (root / "data").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXTURES / "options_sample.csv", root / "data" / "options.csv")
    shutil.copyfile(FIXTURES / "spot_sample.csv", root / "data" / "spot.csv")
    settings = make_settings(root)
    settings.ensure_dirs()
    return settings


__all__ = [
    "ROOT",
    "SRC",
    "FIXTURES",
    "fixture_frame",
    "truth_frame",
    "make_settings",
    "load_settings",
    "stage_fixture_root",
]
