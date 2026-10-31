"""NIFTY 50 option implied volatility and Greeks analytics.

Public API
----------
The package intentionally exposes the legacy ``datacleaning`` function names so
existing notebooks keep working::

    import nifty_sensi as ns
    df, spot_df, spot, dates = ns.optiondata_transform(options, spot_df)
    day_df, r = ns.data_day(dates[0], df, spot_df)

New code should prefer the pipeline stages in :mod:`nifty_sensi.pipeline`.
"""

from __future__ import annotations

__version__ = "1.0.0"

from .config import Settings, load_settings
from .pipeline.prepare import optiondata_transform, prepare_options
from .pipeline.rate import data_day, estimate_rates, attach_rates

__all__ = [
    "__version__",
    "Settings",
    "load_settings",
    "optiondata_transform",
    "prepare_options",
    "data_day",
    "estimate_rates",
    "attach_rates",
]
