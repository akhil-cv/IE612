"""Optional ipywidgets portfolio builder with signed quantities and payoff."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from ..analytics.history import (
    SyntheticCache,
    build_instrument_options,
    get_available_greeks,
    prepare_history,
)
from ..analytics.portfolio import aggregate_portfolio, calculate_contributions, payoff_curve
from ..config import Settings
from .plots_portfolio import contributions as plot_contributions
from .plots_portfolio import payoff as plot_payoff
from .plots_portfolio import portfolio_totals


def available() -> bool:
    try:
        import ipywidgets  # noqa: F401

        return True
    except ImportError:
        return False


def portfolio_builder(frame: pd.DataFrame, settings: Settings, max_instruments: int = 10):
    if not available():
        raise ImportError("ipywidgets is required for the portfolio builder")

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
        rows=min(10, max(5, len(instruments))),
        description="Instruments:",
        layout=widgets.Layout(width="720px"),
    )
    greek_widget = widgets.Dropdown(options=greeks, value="delta", description="Greek:")
    view_widget = widgets.Dropdown(
        options=[("Portfolio total", "portfolio"), ("Contributions", "contributions")],
        value="portfolio",
        description="View:",
    )
    position_box = widgets.VBox()
    output = widgets.Output()
    cache = SyntheticCache(data, settings)
    position_widgets: dict[tuple, dict] = {}
    previous_values: dict[tuple, tuple[int, int]] = {}

    def build_rows(*_):
        selected = list(instrument_widget.value)[:max_instruments]
        for instrument, widgets_for in position_widgets.items():
            previous_values[instrument] = (
                widgets_for["direction"].value,
                widgets_for["quantity"].value,
            )
        position_widgets.clear()
        rows = []
        for instrument in selected:
            label = f"{instrument[0]} K={float(instrument[1]):g} {pd.Timestamp(instrument[2]):%Y-%m-%d}"
            saved_direction, saved_quantity = previous_values.get(instrument, (1, 1))
            direction = widgets.Dropdown(
                options=[("Long (+)", 1), ("Short (-)", -1)],
                value=saved_direction,
                layout=widgets.Layout(width="130px"),
            )
            quantity = widgets.BoundedIntText(
                value=saved_quantity,
                min=1,
                max=1_000_000,
                layout=widgets.Layout(width="110px"),
            )
            direction.observe(refresh, names="value")
            quantity.observe(refresh, names="value")
            position_widgets[instrument] = {"direction": direction, "quantity": quantity}
            rows.append(
                widgets.HBox(
                    [widgets.Label(label, layout=widgets.Layout(width="380px")), direction, quantity]
                )
            )
        position_box.children = tuple(rows) if rows else (widgets.HTML("<i>Select instruments.</i>"),)
        refresh()

    def refresh(*_):
        with output:
            clear_output(wait=True)
            selected = list(instrument_widget.value)[:max_instruments]
            if not selected:
                print("Select at least one instrument.")
                return
            positions = {
                instrument: {
                    "direction": position_widgets[instrument]["direction"].value,
                    "quantity": position_widgets[instrument]["quantity"].value,
                }
                for instrument in selected
                if instrument in position_widgets
            }
            if len(positions) != len(selected):
                return
            greek = greek_widget.value
            table = calculate_contributions(
                data, positions, greek, date_widget.value, settings, cache=cache
            )
            if table.empty:
                print("No historical observations for this selection.")
                return
            figure = None
            try:
                if view_widget.value == "portfolio":
                    portfolio = aggregate_portfolio(table, len(selected))
                    figure = portfolio_totals(portfolio, greek)
                    display_table = portfolio.tail(10).reset_index()
                else:
                    figure = plot_contributions(table, greek)
                    display_table = table.tail(10)
                display(figure)
            finally:
                if figure is not None:
                    plt.close(figure)
            display(display_table)

    def show_payoff(*_):
        with output:
            clear_output(wait=True)
            selected = list(instrument_widget.value)[:max_instruments]
            positions = {
                instrument: {
                    "direction": position_widgets[instrument]["direction"].value,
                    "quantity": position_widgets[instrument]["quantity"].value,
                }
                for instrument in selected
                if instrument in position_widgets
            }
            if not positions:
                print("Select instruments first.")
                return
            figure = None
            try:
                curve = payoff_curve(positions)
                figure = plot_payoff(curve)
                display(figure)
            finally:
                if figure is not None:
                    plt.close(figure)

    payoff_button = widgets.Button(description="Payoff at expiry")
    payoff_button.on_click(show_payoff)

    instrument_widget.observe(build_rows, names="value")
    date_widget.observe(refresh, names="value")
    greek_widget.observe(refresh, names="value")
    view_widget.observe(refresh, names="value")

    build_rows()
    controls = widgets.VBox(
        [
            widgets.HBox([date_widget, greek_widget, view_widget, payoff_button]),
            instrument_widget,
            position_box,
        ]
    )
    return widgets.VBox([controls, output])
