"""Generate ``notebooks/Sensi.ipynb``.

The notebook is intentionally thin: every calculation lives in the package, the
notebook only parameterises a run, renders figures inline and writes the HTML
report. Re-run this script after editing the cell text below::

    python scripts/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "Sensi.ipynb"

MARKDOWN = "markdown"
CODE = "code"

CELLS: list[tuple[str, str]] = [
    (
        MARKDOWN,
        """# NIFTY 50 option sensitivity — end-to-end workflow

This notebook runs the production pipeline in `nifty_sensi` and renders every
figure inline:

1. clean and align the raw option/spot feeds,
2. estimate a put-call-parity risk-free rate per observation date,
3. solve Black-76 implied volatility with explicit solver diagnostics,
4. compute 15 Greeks in documented units,
5. interpolate the IV surface on the fixed moneyness x maturity grid,
6. compare actual Greeks with a synthetic surface and aggregate a portfolio,
7. write a self-contained HTML report with the same figures embedded.

**Setup** — either `pip install -e .` or run from a clone (the next cell adds
`src/` to `sys.path`). Required: `numpy`, `pandas`, `scipy`, `pyarrow`,
`matplotlib`, `IPython`. Interactive widgets are optional (`ipywidgets`).

**Parameters** — edit the next cell directly, or set the environment variables
`NIFTY_SENSI_START`, `NIFTY_SENSI_END` and `NIFTY_SENSI_FORCE` before launch.
""",
    ),
    (
        CODE,
        """import os

# ---------------------------------------------------------------- parameters
START = os.environ.get("NIFTY_SENSI_START") or None  # e.g. "2025-01-01"
END = os.environ.get("NIFTY_SENSI_END") or None      # e.g. "2025-06-30"
FORCE = os.environ.get("NIFTY_SENSI_FORCE", "0") == "1"  # ignore stage caches
REPORT_PATH = "artifacts/report.html"  # relative to the repository root
MAX_DATES = 3      # dates shown in the IV and Greek sections
DPI = 110          # report figure resolution
""",
    ),
    (
        CODE,
        """%matplotlib inline
import sys
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from nifty_sensi.config import load_settings
from nifty_sensi.pipeline.runner import run_pipeline
from nifty_sensi.qc import anomaly_report, dataset_summary
from nifty_sensi.viz import plots_greeks, plots_history, plots_iv, plots_portfolio, plots_qc

plt.rcParams["figure.dpi"] = DPI
plt.rcParams["figure.facecolor"] = "white"


def show(fig):
    # Display once and release, so the inline backend cannot render it again
    # at the end of the cell and later cells cannot flush it.
    display(fig)
    plt.close(fig)


print("repository:", ROOT)
""",
    ),
    (
        MARKDOWN,
        """## 1. Run the pipeline

Stages are cached by content (`config` + input hashes) and a manifest is written
to `artifacts/runs/<run_id>/manifest.json`. Set `FORCE = True` to recompute.
""",
    ),
    (
        CODE,
        """overrides = {"data": {"start": START, "end": END}, "report": {"dpi": DPI}}
settings = load_settings(overrides=overrides)
result = run_pipeline(settings, force=FORCE)

display(result.summary())
greeks = result.greeks
surfaces = result.surfaces
print(f"run {result.run_id} -> {result.run_dir}")
print(f"{len(greeks):,} contracts | {greeks['date'].nunique()} dates | "
      f"{int(greeks['valid_iv'].sum()):,} usable IVs")
""",
    ),
    (
        MARKDOWN,
        """## 2. Dataset overview and quality checks

`valid_iv` marks rows with a converged solver, a price above intrinsic, an IV
inside the configured bounds and (by default) out-of-the-money moneyness.
""",
    ),
    (
        CODE,
        """summary = pd.DataFrame(
    sorted(dataset_summary(greeks).items()), columns=["metric", "value"]
)
display(summary)
display(anomaly_report(greeks))
""",
    ),
    (
        CODE,
        """rates_path = settings.interim_dir / "rates.parquet"
rates = pd.read_parquet(rates_path) if rates_path.exists() else pd.DataFrame()
surface_qc_path = settings.processed_dir / "surface_qc.parquet"
surface_qc = pd.read_parquet(surface_qc_path) if surface_qc_path.exists() else pd.DataFrame()

if not rates.empty:
    show(plots_qc.rate_history(rates))
show(plots_qc.coverage_by_date(greeks))
show(plots_qc.repricing_error(greeks))
show(plots_qc.iv_status_breakdown(greeks))
if not surface_qc.empty:
    show(plots_qc.surface_coverage(surface_qc))
""",
    ),
    (
        MARKDOWN,
        """## 3. Implied-volatility smile and surface

Smiles are shown for a handful of evenly spaced observation dates; the heatmap
is the processed 10 x 8 grid (moneyness `K/S` x maturity in days).
""",
    ),
    (
        CODE,
        """dates = sorted(greeks["date"].unique())
if len(dates) > MAX_DATES:
    picks = np.linspace(0, len(dates) - 1, MAX_DATES).astype(int)
    sample_dates = [dates[i] for i in sorted(set(picks))]
else:
    sample_dates = dates

for date in sample_dates:
    try:
        show(plots_iv.iv_smile(greeks, date))
    except ValueError as exc:
        print(f"{date}: {exc}")

if not surfaces.empty:
    show(plots_iv.surface_heatmap(surfaces, surfaces["date"].max()))
""",
    ),
    (
        MARKDOWN,
        """## 4. Greeks

Units: delta per 1.0 spot, gamma per 1.0 spot², vega per volatility point (1%),
theta per calendar day, rho per 1% of rate, vanna per volatility point, vomma per
volatility point², charm/color/veta per calendar day, zomma per volatility point,
ultima per volatility point³.
""",
    ),
    (
        CODE,
        """latest = greeks["date"].max()
print("latest observation date:", latest)

for greek in ("delta", "gamma", "vega", "theta"):
    try:
        show(plots_greeks.greek_vs_strike(greeks, latest, greek))
    except ValueError as exc:
        print(f"{greek}: {exc}")

try:
    show(plots_greeks.greek_by_expiry(greeks, latest, "vega"))
except ValueError as exc:
    print("vega by expiry:", exc)

try:
    show(plots_greeks.greek_surface(greeks, latest, "gamma"))
except ValueError as exc:
    print("gamma surface:", exc)
""",
    ),
    (
        MARKDOWN,
        """## 5. Actual versus synthetic Greeks

The synthetic surface is rebuilt from observed IVs on a dense strike x maturity
grid (with +/-1% rate scenarios). The comparison below uses the most frequently
observed contract so the history is meaningful.
""",
    ),
    (
        CODE,
        """from nifty_sensi.analytics.history import actual_vs_synthetic, prepare_history

history = prepare_history(greeks)
counts = history[history["delta"].notna()].groupby(["_option", "_strike", "_expiry"]).size()
as_of = pd.Timestamp(history["_date"].max())

if len(counts):
    option, strike, expiry = counts.idxmax()
    instrument = (option, float(strike), pd.Timestamp(expiry))
    comparison = actual_vs_synthetic(history, [instrument], "delta", as_of, settings)
    print(f"instrument: {instrument[0]} K={instrument[1]:g} expiry={instrument[2].date()}")
    print(f"observations: {len(comparison)}")
    if not comparison.empty:
        show(plots_history.actual_vs_synthetic(comparison, "delta"))
        show(plots_history.comparison_scatter(comparison, "delta"))
else:
    print("no contract has enough history for a comparison")
""",
    ),
    (
        MARKDOWN,
        """## 6. Portfolio aggregation

A long out-of-the-money strangle at the front expiry, sized one lot per leg.
Contributions are signed by direction and multiplied by quantity; portfolio
totals are only shown on dates where every leg has an observation.
""",
    ),
    (
        CODE,
        """from nifty_sensi.analytics.portfolio import aggregate_portfolio, calculate_contributions, payoff_curve

today = greeks[(greeks["date"] == latest) & greeks["valid_iv"]]
spot = float(today["spot"].median())
front = today[today["days_to_exp"] >= 5]
if front.empty:
    front = today
expiry = sorted(front["expiry"].unique())[0]

legs = []
if not front.empty:
    for option_type in ("CE", "PE"):
        side = front[(front["expiry"] == expiry) & (front["option_type"] == option_type)]
        if side.empty:
            continue
        row = side.loc[(side["strike"] - spot).abs().idxmin()]
        legs.append((option_type, float(row["strike"]), pd.Timestamp(expiry)))

print("spot:", spot, "| expiry:", expiry)
for leg in legs:
    print(f"  long {leg[0]} K={leg[1]:g}")

if legs:
    positions = {leg: {"direction": 1, "quantity": 1} for leg in legs}
    contributions = calculate_contributions(history, positions, "delta", as_of, settings)
    portfolio = aggregate_portfolio(contributions, len(legs))
    if not portfolio.empty:
        show(plots_portfolio.portfolio_totals(portfolio, "delta"))
    show(plots_portfolio.contributions(contributions, "delta"))
    show(plots_portfolio.payoff(payoff_curve(positions)))
else:
    print("no usable legs for a portfolio example")
""",
    ),
    (
        MARKDOWN,
        """## 7. Optional interactive exploration

The widgets below are available when `ipywidgets` is installed. When it is not,
the static figures above cover the same views; the notebook always runs.
""",
    ),
    (
        CODE,
        """from nifty_sensi.viz import interactive_greeks, interactive_history, interactive_portfolio

if interactive_greeks.available():
    display(interactive_greeks.greek_explorer(greeks, settings))
else:
    print("ipywidgets is not installed: pip install ipywidgets")
    print("Falling back to the static figures rendered above.")

if interactive_history.available():
    display(interactive_history.history_explorer(greeks, settings))

if interactive_portfolio.available():
    display(interactive_portfolio.portfolio_builder(greeks, settings))
""",
    ),
    (
        MARKDOWN,
        """## 8. Self-contained HTML report

The report embeds every figure as base64 PNG, so it is portable: no external
assets, no server, no JavaScript. It is written to `artifacts/report.html`.
""",
    ),
    (
        CODE,
        """from nifty_sensi.reporting import build_run_report

report_file = build_run_report(
    settings, run_id=result.run_id, output=settings.root / REPORT_PATH
)
report_html = Path(report_file).read_text(encoding="utf-8")
print("figures embedded:", report_html.count("<figure>"))
print("size:", round(len(report_html) / 1024), "KB")
print("open:", report_file)
""",
    ),
    (
        MARKDOWN,
        """### Where things are written

| Path | Contents |
| --- | --- |
| `data/interim/prepared.parquet` | cleaned and aligned option rows |
| `data/interim/rates.parquet` | parity rate per observation date + source |
| `data/interim/iv.parquet` | IV, solver status, repricing error, OTM flag |
| `data/processed/greeks.parquet` | all 15 Greeks for usable contracts |
| `data/processed/surfaces.parquet` | 10 x 8 IV grid per date |
| `artifacts/runs/<run_id>/` | manifest, run log and HTML report |
| `artifacts/report.html` | latest portable report |

**Notes on correctness**

* Prices are discounted Black-76 on `F = S exp(rT)`; NIFTY is treated as a price
  index (no dividend yield).
* The risk-free rate comes from put-call parity (median across matched pairs) with
  a rolling-median then constant fallback; the source is recorded per date.
* Deep in-the-money quotes can trade below discounted intrinsic (stale prints);
  they are flagged `below_intrinsic`, excluded from `valid_iv` and never used in
  the surface.
* The calendar-spread diagnostic counts negative moves in total variance
  `w = IV² T`; violations are reported rather than silently smoothed.
""",
    ),
]


def cell(kind: str, source: str) -> dict:
    lines = source.splitlines(keepends=True)
    if kind == MARKDOWN:
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def build() -> dict:
    return {
        "cells": [cell(kind, source) for kind, source in CELLS],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
                "mimetype": "text/x-python",
                "file_extension": ".py",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"notebook: {NOTEBOOK} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
