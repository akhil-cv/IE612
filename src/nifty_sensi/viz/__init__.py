"""Static figures (always available) and optional widget explorers."""

from .plots_greeks import greek_by_expiry, greek_surface, greek_vs_strike
from .plots_history import actual_vs_synthetic, comparison_scatter
from .plots_iv import iv_smile, iv_surface, surface_heatmap
from .plots_portfolio import contributions, payoff, portfolio_totals
from .plots_qc import (
    coverage_by_date,
    iv_status_breakdown,
    rate_history,
    repricing_error,
    surface_coverage,
)
from .report import ReportBuilder, figure_data_uri, figure_png

__all__ = [
    "greek_by_expiry",
    "greek_surface",
    "greek_vs_strike",
    "actual_vs_synthetic",
    "comparison_scatter",
    "iv_smile",
    "iv_surface",
    "surface_heatmap",
    "contributions",
    "payoff",
    "portfolio_totals",
    "coverage_by_date",
    "iv_status_breakdown",
    "rate_history",
    "repricing_error",
    "surface_coverage",
    "ReportBuilder",
    "figure_data_uri",
    "figure_png",
]
