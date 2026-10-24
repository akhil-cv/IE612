"""Compose the full HTML report from a pipeline run."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analytics.history import (
    actual_vs_synthetic,
    build_instrument_options,
    instrument_history,
    prepare_history,
)
from .analytics.portfolio import aggregate_portfolio, calculate_contributions, payoff_curve
from .config import Settings
from .models.conventions import GREEKS
from .pipeline.runner import load_run
from .viz import plots_greeks, plots_history, plots_iv, plots_portfolio, plots_qc
from .viz.report import ReportBuilder


def _read(settings: Settings, name: str, fallback: str) -> pd.DataFrame:
    path = settings.processed_dir / name
    if not path.exists():
        path = settings.interim_dir / fallback
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _sample_dates(greeks: pd.DataFrame, limit: int) -> list[str]:
    dates = sorted(greeks["date"].unique())
    if len(dates) <= limit:
        return dates
    picks = np.linspace(0, len(dates) - 1, limit).astype(int)
    return [dates[i] for i in sorted(set(picks))]


def _liquid_instrument(data: pd.DataFrame) -> tuple | None:
    greek_columns = list(GREEKS)
    usable = (
        data[data[greek_columns].notna().any(axis=1)]
        if all(name in data.columns for name in greek_columns)
        else data
    )
    counts = (
        usable.groupby(["_option", "_strike", "_expiry"]).size().sort_values(ascending=False)
    )
    if counts.empty:
        return None
    option, strike, expiry = counts.index[0]
    return (option, float(strike), pd.Timestamp(expiry))


def _otm_strangle(frame: pd.DataFrame) -> list[tuple]:
    """Nearest OTM call and put at the front expiry (OTM-only data has no straddle)."""
    last_date = frame["date"].max()
    subset = frame[frame["date"] == last_date]
    if "valid_iv" in subset.columns:
        subset = subset[subset["valid_iv"]]
    if subset.empty:
        return []
    spot = float(subset["spot"].median())
    expiries = subset[subset["days_to_exp"] >= 5]["expiry"].unique()
    if len(expiries) == 0:
        expiries = subset["expiry"].unique()
    expiry = sorted(expiries)[0]
    near = subset[subset["expiry"] == expiry]

    legs = []
    calls = near[near["option_type"] == "CE"]
    if not calls.empty:
        strike = min(calls["strike"].unique(), key=lambda value: abs(float(value) - spot))
        legs.append(("CE", float(strike), pd.Timestamp(expiry)))
    puts = near[near["option_type"] == "PE"]
    if not puts.empty:
        strike = min(puts["strike"].unique(), key=lambda value: abs(float(value) - spot))
        legs.append(("PE", float(strike), pd.Timestamp(expiry)))
    return legs


def build_run_report(settings: Settings, run_id: str = "latest", output: str | Path | None = None) -> Path:
    """Build and save a self-contained HTML report for a run."""
    manifest = load_run(settings, run_id)
    run_id = manifest["run_id"]
    run_dir = settings.artifacts_dir / "runs" / run_id

    greeks = _read(settings, "greeks.parquet", "iv.parquet")
    surfaces = _read(settings, "surfaces.parquet", "surface.parquet")
    rates = pd.read_parquet(settings.interim_dir / "rates.parquet") if (settings.interim_dir / "rates.parquet").exists() else pd.DataFrame()
    iv_qc = pd.read_parquet(settings.processed_dir / "iv_qc.parquet") if (settings.processed_dir / "iv_qc.parquet").exists() else pd.DataFrame()
    surface_qc = pd.read_parquet(settings.processed_dir / "surface_qc.parquet") if (settings.processed_dir / "surface_qc.parquet").exists() else pd.DataFrame()

    if greeks.empty:
        raise FileNotFoundError("no processed Greeks found; run the pipeline first")

    report = ReportBuilder(
        settings.report.title,
        subtitle=f"Run {run_id} | {greeks['date'].min()} to {greeks['date'].max()}",
        dpi=settings.report.dpi,
    )

    usable = greeks[greeks["valid_iv"]] if "valid_iv" in greeks else greeks
    report.add_heading("Run overview")
    report.add_metrics(
        {
            "contracts": f"{len(greeks):,}",
            "valid IV": f"{len(usable):,}",
            "dates": f"{greeks['date'].nunique():,}",
            "expiries": f"{greeks['expiry'].nunique():,}",
            "median IV": f"{usable['iv'].median():.2%}" if len(usable) else "n/a",
            "median rate": f"{rates['rate'].median():.2%}" if not rates.empty else "n/a",
        }
    )
    report.add_table(pd.DataFrame(manifest.get("stages", [])), "Pipeline stages")

    report.add_heading("Data quality")
    if "iv_status_label" in greeks:
        status = greeks["iv_status_label"].value_counts().rename_axis("status").reset_index(name="contracts")
        report.add_table(status, "IV solver outcomes")
    if not iv_qc.empty:
        report.add_table(iv_qc.tail(15), "Daily IV coverage (latest 15 dates)")
    if not rates.empty:
        report.add_figure(plots_qc.rate_history(rates), "Put-call-parity rate")
    report.add_figure(plots_qc.coverage_by_date(greeks), "Contract coverage")
    report.add_figure(plots_qc.repricing_error(greeks), "Repricing residuals")
    if not surface_qc.empty:
        report.add_figure(plots_qc.surface_coverage(surface_qc), "IV grid coverage")

    report.add_heading("Implied volatility")
    for date in _sample_dates(greeks, settings.report.max_dates):
        try:
            report.add_figure(plots_iv.iv_smile(greeks, date), f"IV smile — {date}")
        except ValueError:
            continue
    if not surfaces.empty:
        try:
            report.add_figure(
                plots_iv.surface_heatmap(surfaces, surfaces["date"].max()),
                f"IV grid — {surfaces['date'].max()}",
            )
        except ValueError:
            pass

    report.add_heading("Greeks")
    latest = greeks["date"].max()
    for greek in ("delta", "gamma", "vega", "theta"):
        try:
            report.add_figure(plots_greeks.greek_vs_strike(greeks, latest, greek), f"{greek} vs strike — {latest}")
        except ValueError:
            continue
    try:
        report.add_figure(plots_greeks.greek_surface(greeks, latest, "gamma"), f"Gamma surface — {latest}")
    except ValueError:
        pass

    # Actual versus synthetic -------------------------------------------------
    try:
        history = prepare_history(greeks)
        instrument = _liquid_instrument(history)
        if instrument is not None:
            report.add_heading("Actual versus synthetic")
            subset = instrument_history(history, instrument, as_of=history["_date"].max(), greek="delta")
            if len(subset) > 90:
                cutoff = subset["_date"].sort_values().unique()[-90:][0]
                as_of = pd.Timestamp(cutoff)
            else:
                as_of = pd.Timestamp(history["_date"].max())
            comparison = actual_vs_synthetic(history, [instrument], "delta", as_of, settings)
            if not comparison.empty:
                report.add_figure(
                    plots_history.actual_vs_synthetic(comparison, "delta"),
                    f"Delta — actual vs synthetic ({instrument[0]} K={instrument[1]:g})",
                )
                report.add_figure(
                    plots_history.comparison_scatter(comparison, "delta"),
                    "Delta — synthetic vs actual",
                )
    except (ValueError, KeyError) as exc:
        report.add_note(f"Actual vs synthetic section skipped: {exc}")

    # Portfolio --------------------------------------------------------------
    try:
        history = prepare_history(greeks)
        instruments = _otm_strangle(greeks)
        if instruments:
            positions = {instrument: {"direction": 1, "quantity": 1} for instrument in instruments}
            report.add_heading("Portfolio example (long OTM strangle)")
            contributions = calculate_contributions(
                history, positions, "delta", as_of=history["_date"].max(), settings=settings
            )
            portfolio = aggregate_portfolio(contributions, len(instruments))
            if not portfolio.empty:
                report.add_figure(plots_portfolio.portfolio_totals(portfolio, "delta"), "Portfolio delta")
            curve = payoff_curve(positions)
            report.add_figure(plots_portfolio.payoff(curve), "Expiry payoff")
            report.add_table(
                pd.DataFrame(
                    [
                        {"leg": f"{opt} K={strike:g}", "expiry": expiry.date(), "direction": "Long", "quantity": 1}
                        for opt, strike, expiry in instruments
                    ]
                ),
                "Positions",
            )
    except (ValueError, KeyError) as exc:
        report.add_note(f"Portfolio section skipped: {exc}")

    report.add_heading("Configuration")
    report.add_table(
        pd.DataFrame(
            [{"setting": key, "value": str(value)} for key, value in manifest.get("config", {}).items()]
        ),
        "Run configuration",
        max_rows=60,
    )
    report.add_note(
        "Greek units: delta per 1.0 spot, gamma per 1.0 spot^2, vega per vol point, "
        "theta per calendar day, rho per 1% rate, vanna per vol point, vomma per vol point^2, "
        "charm/color/veta per calendar day, zomma per vol point, ultima per vol point^3."
    )

    output = Path(output) if output else run_dir / "report.html"
    figure_count = sum(block.startswith("<figure") for block in report.blocks)
    path = report.save(output)
    plt.close("all")
    print(f"Report written to {path} ({figure_count} figures)")
    return path
