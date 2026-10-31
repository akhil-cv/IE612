"""Analytics over the processed option and surface datasets."""

from .history import (
    SyntheticCache,
    actual_vs_synthetic,
    build_instrument_options,
    get_available_greeks,
    instrument_history,
    prepare_history,
    select_synthetic_greek,
)
from .portfolio import (
    aggregate_portfolio,
    calculate_contributions,
    normalise_positions,
    payoff_curve,
    prepare_portfolio_input,
)

__all__ = [
    "SyntheticCache",
    "actual_vs_synthetic",
    "build_instrument_options",
    "get_available_greeks",
    "instrument_history",
    "prepare_history",
    "select_synthetic_greek",
    "aggregate_portfolio",
    "calculate_contributions",
    "normalise_positions",
    "payoff_curve",
    "prepare_portfolio_input",
]
