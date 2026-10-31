"""Optional ipywidgets actual-versus-synthetic history explorer."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from ..analytics.history import (
    SyntheticCache,
    actual_vs_synthetic,
    build_instrument_options,
    get_available_greeks,
    prepare_history,
)
from ..config import Settings
from .plots_history import actual_vs_synthetic as plot_comparison


def available() -> bool:
    try:
        import ipywidgets  # noqa: F401

        return True
    except ImportError:
        return False


def history_explorer(frame: pd.DataFrame, settings: Settings, max_instruments: int = 4):
    if not available():
        raise ImportError("ipywidgets is required for the history explorer")

    import ipywidgets as widgets
    from IPython.display import clear_output, display

    data = prepare_history(frame)
    greeks = get_available_greeks(data)
    instruments = build_instrument_options(data)
    if not instruments:
        raise ValueError("no instruments available")

    date_widget = widgets.Dropdown(
        options=[(pd.Timestamp(d).strftime("%Y-%m-%d"), pd.Timestamp(d)) for d in sorted(data["_date"].unique())],
        value=pd.Timestamp(sorted(data["_date"].unique())[-1]),
        description="As of:",
    )
    instrument_widget = widgets.SelectMultiple(
        options=instruments,
        value=(instruments[0][1],),
        rows=min(10, max(4, len(instruments))),
        description="Instruments:",
        layout=widgets.Layout(width="720px"),
    )
    greek_widget = widgets.Dropdown(options=greeks, value="delta", description="Greek:")
    output = widgets.Output()
    cache = SyntheticCache(data, settings)

    def refresh(*_):
        with output:
            clear_output(wait=True)
            selected = list(instrument_widget.value)[:max_instruments]
            if not selected:
                print("Select at least one instrument.")
                return
            result = actual_vs_synthetic(
                data, selected, greek_widget.value, date_widget.value, settings, cache=cache
            )
            if result.empty:
                print("No valid observations for this selection.")
                return
            figure = None
            try:
                figure = plot_comparison(result, greek_widget.value)
                display(figure)
            finally:
                if figure is not None:
                    plt.close(figure)
            matched = result.dropna(subset=["actual", "synthetic"])
            print(f"Actual observations: {len(result)} | matched synthetic: {len(matched)}")

    for widget in (date_widget, instrument_widget, greek_widget):
        widget.observe(refresh, names="value")
    refresh()
    return widgets.VBox([widgets.HBox([date_widget, greek_widget]), instrument_widget, output])
