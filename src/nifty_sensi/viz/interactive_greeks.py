"""Optional ipywidgets Greek explorer (static plots if widgets are absent)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import Settings
from ..models.conventions import GREEKS
from .plots_greeks import greek_surface, greek_vs_strike


def available() -> bool:
    try:
        import ipywidgets  # noqa: F401
        import IPython  # noqa: F401

        return True
    except ImportError:
        return False


def greek_explorer(frame: pd.DataFrame, settings: Settings):
    """Date / Greek / maturity explorer. Returns an ipywidgets container."""
    if not available():
        raise ImportError("ipywidgets is required for the interactive explorer")

    import ipywidgets as widgets
    from IPython.display import clear_output, display

    dates = sorted(frame["date"].unique())
    greeks = [greek for greek in GREEKS if greek in frame.columns]

    date_widget = widgets.Dropdown(options=dates, value=dates[-1], description="Date:")
    greek_widget = widgets.Dropdown(options=greeks, value="delta", description="Greek:")
    tenor_widget = widgets.Dropdown(description="DTE:")
    option_widget = widgets.Dropdown(
        options=[("Calls", ("CE",)), ("Puts", ("PE",)), ("Both", ("CE", "PE"))],
        value=("CE", "PE"),
        description="Options:",
    )
    surface_button = widgets.Button(description="3-D surface")
    output = widgets.Output()

    def refresh_tenors(*_):
        subset = frame[frame["date"] == date_widget.value]
        tenor_widget.options = sorted(subset["days_to_exp"].dropna().unique())

    def refresh(*_):
        with output:
            clear_output(wait=True)
            figure = None
            try:
                figure = greek_vs_strike(
                    frame,
                    date_widget.value,
                    greek_widget.value,
                    days_to_exp=tenor_widget.value,
                    option_types=tuple(option_widget.value),
                )
                display(figure)
            except ValueError as exc:
                print(exc)
            finally:
                if figure is not None:
                    plt.close(figure)

    def show_surface(*_):
        with output:
            clear_output(wait=True)
            figure = None
            try:
                figure = greek_surface(
                    frame,
                    date_widget.value,
                    greek_widget.value,
                    option_type="CE",
                )
                display(figure)
            except ValueError as exc:
                print(exc)
            finally:
                if figure is not None:
                    plt.close(figure)

    date_widget.observe(refresh_tenors, names="value")
    date_widget.observe(refresh, names="value")
    greek_widget.observe(refresh, names="value")
    tenor_widget.observe(refresh, names="value")
    option_widget.observe(refresh, names="value")
    surface_button.on_click(show_surface)

    refresh_tenors()
    refresh()
    return widgets.VBox(
        [widgets.HBox([date_widget, greek_widget, tenor_widget, option_widget, surface_button]), output]
    )
