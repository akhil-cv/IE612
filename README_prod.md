# NIFTY 50 Option Sensitivity — Production Guide

`nifty_sensi` is the installable, tested production version of the IE612 NIFTY
option-sensitivity analysis. It turns the legacy `Sensi.ipynb` workflow into a
five-stage pipeline with a CLI, a fixed output schema, 15 Greeks in documented
units, inline notebook figures, a self-contained HTML report, and a parity
harness against the legacy outputs.

The original analysis files (`Sensi.ipynb`, `Sensi_Rough.ipynb`,
`datacleaning.py`, `README.md`, `log.md`, the committed data feeds) are left
untouched. Everything new lives in `src/`, `config/`, `scripts/`, `tests/`,
`notebooks/` and the generated `data/{raw,interim,processed}` / `artifacts`
directories.

| Item | Value |
| --- | --- |
| Package | `nifty-sensi` 1.0.0 (import name `nifty_sensi`) |
| CLI | `nifty-sensi` (or `python -m nifty_sensi`) |
| Python | 3.10+ |
| Core dependencies | `numpy`, `pandas`, `scipy`, `pyarrow`, `matplotlib` |
| Pricing model | Discounted Black-76 on `F = S·exp(rT)` |
| Greeks | 15 (first, second, third order and dual) |
| Tests | 40 `unittest` tests, 6 modules |
| Report | Single portable HTML file with base64 PNG figures |
| Licence | MIT |

---

## 1. What the package does

For every observation date in the option feed it:

1. **Cleans and aligns** the raw option chain with the NIFTY spot close.
2. **Estimates a risk-free rate** per date from put-call parity, with a
   rolling-median and constant fallback when parity is unavailable.
3. **Solves Black-76 implied volatility** per contract with explicit solver
   statuses and a repricing residual.
4. **Computes 15 Greeks** for every usable contract, in fixed units.
5. **Interpolates an IV surface** on a 10 × 8 moneyness × maturity grid and
   counts calendar-spread violations.

On top of the pipeline it provides:

- historical **actual-versus-synthetic** Greek comparisons,
- **portfolio aggregation** and expiry payoff curves,
- **static and optional interactive** visualisations,
- a **self-contained HTML report** (`artifacts/report.html`),
- a **legacy parity** command that isolates solver differences from the
  deliberate forward/rate fix.

---

## 2. What changed relative to the legacy notebook

| Area | Legacy | `nifty_sensi` |
| --- | --- | --- |
| IV convention | Black-76 with `F = S`, `r = 0` | Correct forward `F = S·exp(rT)` with parity rate |
| Rate | Missing for 10,268 rows; no source tracking | Parity median per date + fallback chain + `rate_source` |
| Solver | Silent `NaN` on failure | Newton + Brent fallback, six status codes, repricing residual |
| Greeks | `opengreeks`, inconsistent units/coverage | Verified engine, 15 Greeks, documented units, FD cross-checks |
| Validation | Ad-hoc notebook cells | Typed settings, input contracts, QC tables, 40 tests |
| Repeatability | Notebook state | Content-addressed stage cache + `manifest.json` per run |
| Output | CSV dumps and notebook plots | Parquet with stable schema, inline notebook figures, portable HTML |
| Scope | All contracts incl. stale ITM quotes | OTM-only `valid_iv` flag; stale quotes flagged, never silently used |

Key correctness fixes found during the rewrite:

- The legacy `iv` is exactly Black-76 implied volatility with **`F = S` and
  `r = 0`**, refined to a relative precision of ~1e-12; the new pipeline
  reproduces it to a median absolute difference of **4.7e-13** when run in that
  convention, so solver and lookup differences are ruled out.
- The new engine's analytic Greeks agree with bumped prices to a maximum
  relative error of ~6e-5 on real rows; repricing residual p95 is ~2.5e-6.
- Deep in-the-money quotes below discounted intrinsic are flagged
  `below_intrinsic`, excluded from `valid_iv`, and never used in the surface.

See [§16](#16-legacy-parity) for measured parity numbers.

---

## 3. Installation

### 3.1 Editable install from this checkout

```powershell
cd C:\GIT\IE612
python -m pip install -e .
```

This installs the runtime dependencies and the `nifty-sensi` entry point. If
`pip` complains that the long-description file referenced by `readme` in
`pyproject.toml` is missing, either create that file or use the `src`-path
bootstrap in [§3.2](#32-run-without-installing-src-layout).

### 3.2 Run without installing (src layout)

Every script and the notebook can run from a clone because they insert `src/`
into `sys.path`. For ad-hoc Python sessions use:

```powershell
# PowerShell
$env:PYTHONPATH = "src"
python -m nifty_sensi --help

# cmd.exe
set PYTHONPATH=src

# bash / zsh
export PYTHONPATH=src
```

### 3.3 Optional extras

```powershell
python -m pip install -e ".[ingest]"     # requests, huggingface_hub, duckdb
python -m pip install -e ".[notebook]"   # ipywidgets, jupyterlab
python -m pip install -e ".[dev]"        # pytest, ruff, nbformat, nbclient
```

- **ingest** is required only for `nifty-sensi ingest` (NSE spot + HuggingFace
  option chain).
- **notebook** is required only for the interactive widget cells.
- Core plotting and the report need only `matplotlib`.

### 3.4 Verify the install

```powershell
python -c "import nifty_sensi; print(nifty_sensi.__version__)"
python -m unittest discover -s tests -t .
```

---

## 4. Quick start

### 4.1 CLI in four commands

```powershell
# 1. Run the pipeline on the committed feeds (cached across runs)
nifty-sensi build

# 2. Inspect quality
nifty-sensi qc

# 3. Build the portable report
nifty-sensi report --out artifacts/report.html

# 4. Compare against the legacy output
nifty-sensi parity --legacy data/df_with_greeks.parquet `
  --convention legacy_zero_rate --details data/interim/parity.parquet
```

Restrict the window and force recomputation:

```powershell
nifty-sensi build --start 2025-01-01 --end 2025-06-30 --force
```

### 4.2 Notebook

```powershell
python scripts/check_notebook.py --start 2025-01-01 --end 2025-01-31 --keep-outputs
```

Open `notebooks/Sensi.ipynb` for the same workflow with every figure inline.
See [§13](#13-notebook).

### 4.3 Python

```python
from nifty_sensi.config import load_settings
from nifty_sensi.pipeline.runner import run_pipeline

settings = load_settings(overrides={"data": {"start": "2025-01-01", "end": "2025-03-31"}})
result = run_pipeline(settings)

print(result.summary())
greeks = result.greeks
surfaces = result.surfaces
print(result.run_id, result.run_dir)
```

---

## 5. Repository layout

```
IE612/
├── README_prod.md              # this production guide
├── README.md                   # original project README (untouched)
├── log.md                      # original project log (untouched)
├── pyproject.toml              # package metadata, extras, CLI entry point
├── datacleaning.py             # legacy module (untouched, still importable)
├── Sensi.ipynb                 # legacy notebook (untouched)
├── Sensi_Rough.ipynb           # legacy notebook (untouched)
├── config/
│   └── default.json            # all pipeline settings
├── src/nifty_sensi/
│   ├── __init__.py             # public API + version
│   ├── __main__.py             # python -m nifty_sensi
│   ├── cli.py                  # argparse CLI
│   ├── config.py               # typed settings, JSON/YAML loader
│   ├── paths.py                # repo discovery, atomic writes, hashing, cache
│   ├── logging_utils.py        # package logger + timers
│   ├── schema.py               # raw feed validation contracts
│   ├── qc.py                   # summaries, anomalies, parity helpers
│   ├── reporting.py            # full HTML report composer
│   ├── datacleaning.py         # legacy-compatible shim
│   ├── ingest_spot.py          # NSE spot downloader
│   ├── ingest_options.py       # HuggingFace option-chain loader
│   ├── models/
│   │   ├── conventions.py      # forward, moneyness, Greek units
│   │   ├── black76.py          # pricing, IV, 15 Greeks
│   │   └── synthetic.py        # dense synthetic Greek surface
│   ├── pipeline/
│   │   ├── prepare.py          # stage 1
│   │   ├── rate.py             # stage 2
│   │   ├── implied_vol.py      # stage 3
│   │   ├── greeks.py           # stage 4
│   │   ├── surface.py          # stage 5
│   │   └── runner.py           # orchestration, caching, manifest
│   ├── analytics/
│   │   ├── history.py          # actual vs synthetic history
│   │   └── portfolio.py        # contributions, aggregation, payoff
│   └── viz/
│       ├── report.py           # HTML builder (base64 figures)
│       ├── plots_qc.py         # coverage, rates, residuals, status
│       ├── plots_iv.py         # smile, surface, heatmap
│       ├── plots_greeks.py     # Greeks vs strike/expiry, surfaces
│       ├── plots_history.py    # actual vs synthetic
│       ├── plots_portfolio.py  # totals, contributions, payoff
│       ├── interactive_greeks.py
│       ├── interactive_history.py
│       └── interactive_portfolio.py
├── scripts/
│   ├── run_pipeline.py         # pipeline + report convenience wrapper
│   ├── fetch_data.py           # safe spot/option download into data/raw
│   ├── build_notebook.py       # regenerates notebooks/Sensi.ipynb
│   ├── check_notebook.py       # Jupyter-free notebook executor/validator
│   └── make_fixtures.py        # deterministic synthetic test fixture
├── notebooks/
│   └── Sensi.ipynb             # thin, production notebook
├── tests/
│   ├── helpers.py
│   ├── test_black76.py
│   ├── test_prepare.py
│   ├── test_rate.py
│   ├── test_iv_greeks.py
│   ├── test_surface.py
│   ├── test_cli_smoke.py
│   └── fixtures/
│       ├── options_sample.csv
│       ├── spot_sample.csv
│       └── truth_sample.csv
├── data/
│   ├── nifty_close.csv         # committed spot feed
│   ├── niftyoptions_df.csv     # committed option feed
│   ├── df_with_greeks.parquet  # legacy output used for parity (untouched)
│   ├── datapath.csv            # legacy per-contract output (untouched)
│   ├── raw/                    # generated downloads (gitignored)
│   ├── interim/                # prepared/rated/iv (gitignored)
│   └── processed/              # greeks/surfaces/QC (gitignored)
├── artifacts/                  # runs, logs, reports, parity (gitignored)
├── archive/                    # legacy VolGAN experiments (untouched)
└── references/                 # VolGAN / arbitrage-free IV papers
```

### 5.1 Untouched legacy files

These are **not modified** by the package and must not be overwritten:
`Sensi.ipynb`, `Sensi_Rough.ipynb`, `datacleaning.py`, `datacleaning.ipynb`,
`data.ipynb`, `streaming_data.ipynb`, `download_nifty_close.py`, `README.md`,
`log.md`, `.gitignore`, `data/niftyoptions_df.csv`, `data/nifty_close.csv`,
`data/df_with_greeks.parquet`, `data/datapath.csv`, `archive/`, `references/`.

---

## 6. Command-line reference

```
nifty-sensi [--config PATH] [--root PATH] [--verbose] COMMAND ...
```

Global options:

| Option | Meaning |
| --- | --- |
| `--config PATH` | JSON or YAML settings file (default `config/default.json` under the root) |
| `--root PATH` | Project root for resolving relative data paths (default: nearest `pyproject.toml`) |
| `--verbose` | DEBUG-level logging |

Exit codes: `0` success, `1` handled error (missing input, bad value, runtime
failure), `2` argparse usage error.

### 6.1 `ingest` — download raw feeds

```powershell
nifty-sensi ingest --source spot --start 2025-01-01 --end 2025-12-31
nifty-sensi ingest --source options --start 2025-01-01 --end 2025-12-31 --revision main
nifty-sensi ingest --source all --start 2025-01-01 --end 2025-12-31
```

| Option | Default | Notes |
| --- | --- | --- |
| `--source` | `all` | `spot`, `options`, or `all` |
| `--start` | `2025-01-01` | first date |
| `--end` | today | last date |
| `--revision` | latest | HuggingFace dataset revision |
| `--out-spot` | `data/raw/nifty_close.parquet` | spot output |
| `--out-options` | `data/raw/options.parquet` | options output |

Requires the `ingest` extra. Outputs are written under `data/raw/`; point
`data.spot_file` / `data.options_file` at them in your config before building.
`scripts/fetch_data.py` is the CSV alternative (see [§14](#14-scripts)).

### 6.2 `build` — run pipeline stages

```powershell
nifty-sensi build
nifty-sensi build --stages prepare,rate
nifty-sensi build --start 2025-01-01 --end 2025-06-30 --force
```

| Option | Meaning |
| --- | --- |
| `--stages` | comma-separated subset of `prepare,rate,iv,greeks,surface`; prerequisites run automatically |
| `--force` | ignore stage caches |
| `--start` / `--end` | override `data.start` / `data.end` |

Running a later stage runs every prerequisite stage first. The run manifest and
log are written to `artifacts/runs/<run_id>/`.

### 6.3 `run-all` — full pipeline alias

```powershell
nifty-sensi run-all --start 2025-01-01 --end 2025-12-31
```

Same implementation as `build` with all stages. It does **not** download data;
run `ingest` (or `scripts/fetch_data.py`) first.

### 6.4 `report` — portable HTML

```powershell
nifty-sensi report
nifty-sensi report --run-id 20261024T061440Z --out artifacts/report.html
```

Defaults to the latest run and writes `artifacts/runs/<run_id>/report.html`
unless `--out` is supplied.

### 6.5 `qc` — quality summary

```powershell
nifty-sensi qc
```

Prints the manifest identity plus row counts, rate coverage, usable-IV share and
the anomaly checklist from the processed Greeks.

### 6.6 `parity` — legacy comparison

```powershell
nifty-sensi parity --legacy data/df_with_greeks.parquet
nifty-sensi parity --legacy data/df_with_greeks.parquet --convention legacy_zero_rate
nifty-sensi parity --legacy data/df_with_greeks.parquet --details data/interim/parity.parquet
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--legacy` | required | legacy parquet containing `iv` (and optionally `r`) |
| `--new` | `data/processed/greeks.parquet` | new output to compare |
| `--tolerance` | `1e-3` | absolute IV tolerance for the “within tolerance” share |
| `--convention` | `pipeline` | `pipeline` compares shipped IVs; `legacy_zero_rate` re-solves with `F=S, r=0` |
| `--out` | `artifacts/parity_report.md` | Markdown report |
| `--details` | none | optional per-contract parquet |

The Markdown report contains the summary, per-maturity breakdown and rate
comparison. Results are discussed in [§16](#16-legacy-parity).

---

## 7. Python API

### 7.1 Top-level compatibility surface

```python
import nifty_sensi as ns

print(ns.__version__)
settings = ns.load_settings()
df, spot_df, spot, dates = ns.optiondata_transform(options, spot_df)
day_df, rate = ns.data_day(dates[0], df, spot_df)
prepared = ns.prepare_options(options, spot_df, settings)
rated, rates = ns.attach_rates(prepared, settings)
```

Exports: `__version__`, `Settings`, `load_settings`, `optiondata_transform`,
`prepare_options`, `data_day`, `estimate_rates`, `attach_rates`.

### 7.2 Pipeline

```python
from nifty_sensi.pipeline.runner import STAGES, load_run, run_pipeline
from nifty_sensi.config import load_settings

settings = load_settings(
    path="config/default.json",
    overrides={"data": {"start": "2025-01-01", "end": "2025-03-31"}},
    root=None,
)
result = run_pipeline(settings, stages=None, force=False)

result.summary()          # DataFrame: stage, status, rows, seconds
result.greeks             # processed contract frame
result.surfaces           # long-format IV grid
result.paths              # stage/file path map
result.manifest           # dict written to manifest.json
manifest = load_run(settings, "latest")
```

Stage functions are individually usable:

```python
from nifty_sensi.pipeline.prepare import prepare_options
from nifty_sensi.pipeline.rate import attach_rates, estimate_rates, parity_rates
from nifty_sensi.pipeline.implied_vol import compute_iv, iv_report
from nifty_sensi.pipeline.greeks import attach_greeks, greeks_report
from nifty_sensi.pipeline.surface import build_surfaces
```

### 7.3 Models

```python
from nifty_sensi.models import black76
from nifty_sensi.models.conventions import GREEKS, forward_price, moneyness
from nifty_sensi.models.synthetic import generate_synthetic_greeks

F = forward_price(spot=23_742.9, rate=0.07, ttm=0.25)
price = black76.price("CE", F, 24_000.0, 0.25, 0.07, 0.18)
iv, status = black76.implied_vol(price, F, 24_000.0, 0.25, 0.07, "CE")
values = black76.greeks("CE", F, 24_000.0, 0.25, 0.07, iv)
```

`price`, `implied_vol` and `greeks` are vectorised and accept scalars or arrays.
Status codes: `IV_OK`, `IV_AT_LOWER`, `IV_AT_UPPER`, `IV_INVALID`,
`IV_NO_CONVERGENCE`, `IV_BELOW_INTRINSIC`, with labels in `STATUS_LABELS`.

### 7.4 Analytics

```python
import pandas as pd

from nifty_sensi.analytics.history import (
    SyntheticCache, actual_vs_synthetic, build_instrument_options,
    instrument_history, prepare_history, select_synthetic_greek,
)
from nifty_sensi.analytics.portfolio import (
    aggregate_portfolio, calculate_contributions, normalise_positions,
    payoff_curve, prepare_portfolio_input,
)

history = prepare_history(result.greeks)
instrument = ("CE", 24_000.0, pd.Timestamp("2025-06-26"))
comparison = actual_vs_synthetic(history, [instrument], "delta", history["_date"].max(), settings)

positions = {instrument: {"direction": 1, "quantity": 50}}
contributions = calculate_contributions(history, positions, "delta", history["_date"].max(), settings)
portfolio = aggregate_portfolio(contributions, len(positions))
curve = payoff_curve(positions)
```

Positions accept either `{instrument: {"direction": ±1, "quantity": n}}`,
`{instrument: n}`, or a list of dicts with `option`, `strike`, `expiry`,
`direction`, `quantity`.

### 7.5 Visualisation

Static figures return Matplotlib `Figure` objects and are used by the notebook
and report:

| Module | Functions |
| --- | --- |
| `plots_qc` | `coverage_by_date`, `rate_history`, `repricing_error`, `surface_coverage`, `iv_status_breakdown` |
| `plots_iv` | `iv_smile`, `iv_surface`, `surface_heatmap` |
| `plots_greeks` | `greek_vs_strike`, `greek_by_expiry`, `greek_surface` |
| `plots_history` | `actual_vs_synthetic`, `comparison_scatter` |
| `plots_portfolio` | `portfolio_totals`, `contributions`, `payoff` |
| `viz.report` | `ReportBuilder`, `figure_png`, `figure_data_uri` |

Interactive modules (`interactive_greeks`, `interactive_history`,
`interactive_portfolio`) expose `available()` and return widgets only when
`ipywidgets` is installed.

### 7.6 Legacy `datacleaning` shim

`nifty_sensi.datacleaning` mirrors the root module so old code can import from
the package: `optiondata_transform`, `data_day`, `kernel`, `kernelVega`,
`Smooth`, `SmoothVega`, `interpolate_m`, `interpolate_tau`,
`interpolate_and_extrapolate`, `interpolate_data`, `interpolate_surface`,
`interpolate_dataBspline`.

The root `datacleaning.py` remains the original implementation.

---

## 8. Pipeline internals

### 8.1 Stages

| # | Stage | Input | Output | Purpose |
| --- | --- | --- | --- | --- |
| 1 | `prepare` | raw option + spot CSV/Parquet | `data/interim/prepared.parquet` | validate, align spot, compute TTM/moneyness, filter, dedupe |
| 2 | `rate` | `prepared.parquet` | `data/interim/rated.parquet`, `rates.parquet` | parity rate per date + fallback, attach `r` |
| 3 | `iv` | `rated.parquet` | `data/interim/iv.parquet`, `iv_qc.parquet` | Black-76 IV, status, repricing, OTM flag, `valid_iv` |
| 4 | `greeks` | `iv.parquet` | `data/processed/greeks.parquet`, `greek_qc.parquet` | 15 Greeks for usable rows |
| 5 | `surface` | `greeks.parquet` | `data/processed/surfaces.parquet`, `surface_qc.parquet` | 10 × 8 IV grid + calendar check |

### 8.2 Stage 1 — prepare

- Parses `date`/`expiry`, computes `date_dt`, `exdate_dt`,
  `time_to_expiry`, `days_to_exp = expiry − date`, `ttm = days/365`.
- Aligns the spot close by observation date; raises `KeyError` listing up to
  five missing dates rather than silently dropping them.
- Applies the optional `data.start`/`data.end` window.
- Drops rows with non-positive price, strike or spot; keeps
  `min_days ≤ days_to_exp ≤ max_days`; deduplicates on
  `(date, expiry, strike, option_type)` (keep last); enforces `price ≥ 0.05`;
  keeps only `CE`/`PE`.
- Sorts deterministically and adds a `row_id`.

Does **not** mutate the caller's DataFrames.

### 8.3 Stage 2 — rate

For each date, matched calls and puts on the same expiry, strike and spot give

```
C − P = S − K·exp(−rT)        ⇒        r = log(K / (S − C + P)) / T
```

- Pairs outside `rate.min`/`rate.max` or with a non-positive denominator are
  discarded.
- The daily estimate is the **median** across valid pairs.
- Fallback chain, recorded per date in `rate_source`:
  1. `parity` — at least one valid pair;
  2. `rolling_median` — centred rolling median over `rolling_window` dates;
  3. `fallback_constant` — configured `rate.fallback`.
- `attach_rates` writes both `rate` (with source metadata) and the documented
  `r` column used by later stages.

### 8.4 Stage 3 — IV

- Builds `forward = S·exp(rT)` and solves discounted Black-76.
- Vectorised Newton with a Brenner-Subrahmanyam seed, step clipping and
  configurable tolerance/maximum iterations; a Brent fallback handles the rare
  non-converged rows with a tightened `xtol`.
- Detects quotes below discounted intrinsic **before** solving and returns
  `NaN` with `below_intrinsic`.
- Records `model_price`, `repricing_error = model − market`, `is_otm`,
  `iv_status`, `iv_status_label`.
- `valid_iv` requires: converged (`ok`), `lower < iv < upper`, and — when
  `options.otm_only` is true — OTM moneyness (`CE: K/S ≥ 1`,
  `PE: K/S < 1`).

### 8.5 Stage 4 — Greeks

Computes all 15 Greeks only for `valid_iv` rows; invalid rows receive `NaN`.
Analytic formulae cover `delta`, `gamma`, `vega`, `theta`, `rho`, `vanna`,
`vomma`, `dual_delta`, `dual_gamma`. Central finite differences of those
analytic quantities cover `speed`, `zomma`, `charm`, `color`, `veta`, `ultima`.
Units are normalised once, as documented in [§10.3](#103-greek-units).

### 8.6 Stage 5 — surface

- Grid: `grid.moneyness` (10 points, 0.60 → 1.40) × `grid.tau_days`
  (7, 14, 30, 60, 90, 180, 270, 365) = **80 points per date**.
- For each observed maturity with at least `grid.min_points` moneyness points,
  average IV by moneyness and interpolate linearly across the moneyness grid
  with flat extrapolation.
- Across maturities, interpolate total IV per moneyness point; a single tenor
  is held flat across the grid.
- `extrapolated` marks moneyness points that required flat extrapolation for at
  least one used tenor.
- Calendar-spread diagnostic: negative changes in total variance
  `w = IV²·T` between adjacent grid maturities, with a `−1e-8` tolerance.
  Violations are **reported, not smoothed** (`surface_qc.parquet`).

### 8.7 Caching

Each stage output has a sidecar `<output>.meta.json` containing a cache key:

```
sha256(stage name + full config fingerprint + sha256 of every input file)
```

- A stage is reused only when the output exists and the key matches.
- The full config participates in the fingerprint, so changing **any** setting
  invalidates all stage caches.
- `--force` (CLI) or `force=True` (API) recomputes everything requested.
- Writes are atomic (temporary file + `os.replace`), so interrupted runs do not
  leave half-written parquet/JSON files.

### 8.8 Run manifest

`artifacts/runs/<run_id>/manifest.json` records:

- `run_id`, `created_at`, `finished_at`;
- full `config` and `config_fingerprint`;
- input paths and SHA-256 hashes;
- per-stage `status` (`computed`/`cached`), rows and seconds;
- QC summaries for prepare, rate sources, IV statuses, Greeks and surface.

`artifacts/latest.json` points at the most recent run. Each run also gets
`run.log`. Stage timing from the reference run is in [§17](#17-reference-run).

---

## 9. Data contracts

### 9.1 Raw option feed

Required columns (extra columns such as `timestamp` are allowed):

| Column | Type | Rule |
| --- | --- | --- |
| `date` | date string | observation date |
| `expiry` | date string | expiry, later than `date` |
| `strike` | numeric | positive |
| `option_type` | string | `CE` or `PE` (case-insensitive) |
| `close` | numeric | option price, column configurable via `options.price_column` |

Committed feed: `data/niftyoptions_df.csv` (columns `date`, `timestamp`,
`expiry`, `strike`, `option_type`, `close`).

### 9.2 Raw spot feed

| Column | Type | Rule |
| --- | --- | --- |
| `date` | date string | must cover every option observation date |
| `close` | numeric | positive |

Committed feed: `data/nifty_close.csv` (OHLC; only `close` is required).

`schema.validate_options` / `validate_spot` return QC counters
(`bad_date`, `bad_expiry`, `bad_strike`, `bad_price`, `bad_type`,
`expiry_not_after_date`, `duplicate_contracts`, `duplicate_dates`) and raise
`DataValidationError` for an empty feed or entirely unparseable dates.

### 9.3 Processed frame columns

`greeks.parquet` (one row per prepared contract):

| Group | Columns |
| --- | --- |
| Identity | `date`, `timestamp`, `expiry`, `strike`, `option_type` |
| Dates | `date_dt`, `exdate_dt`, `time_to_expiry`, `days_to_exp`, `ttm` |
| Market | `spot`, `moneyness`, `price`, `row_id` |
| Rate | `rate`, `rate_pairs`, `rate_source`, `r` |
| IV | `forward`, `iv`, `iv_status`, `iv_status_label`, `model_price`, `repricing_error`, `is_otm`, `valid_iv` |
| Greeks | `delta`, `gamma`, `vega`, `theta`, `rho`, `vanna`, `charm`, `vomma`, `speed`, `zomma`, `color`, `veta`, `ultima`, `dual_delta`, `dual_gamma` |

`surfaces.parquet`: `date`, `date_dt`, `moneyness`, `tau_days`, `tau`, `iv`,
`extrapolated`.

QC tables:

| File | Columns |
| --- | --- |
| `iv_qc.parquet` | `date`, `rows`, `solved`, `usable`, `median_iv`, `p95_abs_repricing`, `usable_pct` |
| `greek_qc.parquet` | `greek`, `non_null`, `coverage_pct`, `median`, `p05`, `p95` |
| `surface_qc.parquet` | `date`, `tenors_used`, `grid_points`, `grid_missing`, `calendar_violations`, `calendar_pairs`, `calendar_violation_pct` |
| `rates.parquet` | `date_dt`, `rate`, `rate_pairs`, `rate_source` |

---

## 10. Model conventions

### 10.1 Pricing

With `F = S·exp((r − q)T)`, `q = 0` for NIFTY (treated as a price index):

```
d1 = [ln(F/K) + ½σ²T] / (σ√T)
d2 = d1 − σ√T

Call = exp(−rT)·[F·N(d1) − K·N(d2)]
Put  = exp(−rT)·[K·N(−d2) − F·N(−d1)]
```

Put-call parity is verified in the tests to ~1e-10. Prices are always
discounted, so the forward is the correct input; use `price_from_spot` or
`forward_price` if you only have spot.

### 10.2 IV solver statuses

| Status | Label | Meaning | `valid_iv` |
| --- | --- | --- | --- |
| 0 | `ok` | converged within tolerance | yes (also needs bounds and OTM) |
| 1 | `at_lower_bound` | pinned to `iv.lower` | no |
| 2 | `at_upper_bound` | pinned to `iv.upper` | no |
| 3 | `invalid_input` | non-finite or non-positive input | no |
| 4 | `no_convergence` | Newton and Brent both failed | no |
| 5 | `below_intrinsic` | quote below discounted intrinsic | no |

`iv` is `NaN` for `below_intrinsic` and `invalid_input`; bound statuses keep the
bound value for diagnostics. Solver diagnostics are persisted, never silently
discarded.

### 10.3 Greek units

| Greek | Unit | Method |
| --- | --- | --- |
| `delta` | per 1.0 spot | analytic |
| `gamma` | per 1.0 spot² | analytic |
| `vega` | per 1 vol point (1%) | analytic ÷ 100 |
| `theta` | per calendar day | analytic ÷ 365 |
| `rho` | per 1 percentage point of rate | analytic ÷ 100 |
| `vanna` | per 1 vol point | analytic |
| `vomma` | per 1 vol point² | analytic |
| `charm` | change in delta per calendar day | central FD of analytic delta |
| `color` | change in gamma per calendar day | central FD of analytic gamma |
| `veta` | change in vega (per vol point) per calendar day | central FD of analytic vega |
| `speed` | ∂gamma/∂S (per 1.0 spot³) | central FD of analytic gamma |
| `zomma` | ∂gamma/∂σ per 1 vol point | central FD of analytic gamma |
| `ultima` | ∂³V/∂σ³ per 1 vol point³ | central FD of analytic vomma |
| `dual_delta` | ∂V/∂K | analytic |
| `dual_gamma` | ∂²V/∂K² | analytic |

`greeks.theta_per_day`, `greeks.vega_per_point` and `greeks.rho_per_point` can
disable the day/point normalisation (used by the unit-scaling test).

Rounding/step choices: spot bump `max(1e-3·S, 0.5)` for speed, vol bump
`1e-3`, time bump `max(1e-3·T, 1e-6)`. Analytic Greeks were validated against
independent bumped prices (max relative error ≈ 5.8e-5 on 300 real rows).

---

## 11. Configuration reference

Configuration is JSON (`config/default.json`) or YAML (requires `PyYAML`).
Precedence: defaults → `--config` file → programmatic/CLI overrides.

### `data`

| Key | Default | Meaning |
| --- | --- | --- |
| `raw_dir` | `data/raw` | downloads |
| `interim_dir` | `data/interim` | prepared/rated/IV outputs |
| `processed_dir` | `data/processed` | Greeks, surfaces, QC |
| `artifacts_dir` | `artifacts` | runs, logs, reports |
| `spot_file` | `data/nifty_close.csv` | spot feed (CSV or Parquet) |
| `options_file` | `data/niftyoptions_df.csv` | option feed (CSV or Parquet) |
| `start` | `null` | inclusive start date |
| `end` | `null` | inclusive end date |

### `options`

| Key | Default | Meaning |
| --- | --- | --- |
| `price_column` | `close` | market price column |
| `otm_only` | `true` | restrict `valid_iv` to OTM contracts |
| `min_price` | `0.05` | NSE tick; cheaper quotes dropped |
| `min_days` | `1` | minimum days to expiry |
| `max_days` | `365` | maximum days to expiry |
| `day_count` | `365.0` | declared day-count basis; this version fixes 365 in code |
| `dedupe` | `true` | drop duplicate contracts (keep last) |

### `iv`

| Key | Default | Meaning |
| --- | --- | --- |
| `lower` | `0.0001` | lower IV bound |
| `upper` | `5.0` | upper IV bound |
| `tolerance` | `1e-8` | price tolerance (relative to one plus the absolute price) |
| `max_iter` | `60` | Newton iterations |

### `rate`

| Key | Default | Meaning |
| --- | --- | --- |
| `min` | `-0.05` | discard parity rates below |
| `max` | `0.30` | discard parity rates above |
| `fallback` | `0.065` | constant when no parity/rolling estimate |
| `rolling_window` | `5` | centred rolling window (dates) |

### `grid`

| Key | Default | Meaning |
| --- | --- | --- |
| `moneyness` | 10 points, 0.6000 … 1.4000 | strike/spot grid |
| `tau_days` | `[7, 14, 30, 60, 90, 180, 270, 365]` | maturity grid (days) |
| `min_points` | `4` | minimum moneyness points to use a tenor |

### `greeks`

| Key | Default | Meaning |
| --- | --- | --- |
| `theta_per_day` | `true` | normalise theta per day |
| `vega_per_point` | `true` | normalise vega/vanna/vomma/etc. per vol point |
| `rho_per_point` | `true` | normalise rho per 1% rate |

### `synthetic`

| Key | Default | Meaning |
| --- | --- | --- |
| `spot_pad` | `6000` | strike grid spans spot ± this |
| `strike_step` | `300` | raw strike spacing |
| `strike_round` | `50` | round grid strikes to this tick |
| `rate_offsets` | `[-0.01, 0.0, 0.01]` | rate scenarios |

### `report`

| Key | Default | Meaning |
| --- | --- | --- |
| `title` | `NIFTY 50 Option Sensitivity Report` | report heading |
| `dpi` | `110` | embedded PNG resolution |
| `max_dates` | `4` | IV smiles shown |

Programmatic overrides are deep-merged:

```python
settings = load_settings(overrides={"data": {"start": "2025-01-01"},
                                    "report": {"dpi": 80}})
print(settings.fingerprint())   # changes when any setting changes
```

---

## 12. Outputs

### 12.1 Files

| Path | Written by | Contents |
| --- | --- | --- |
| `data/interim/prepared.parquet` | prepare | cleaned/aligned contracts |
| `data/interim/rates.parquet` | rate | daily rate + source |
| `data/interim/rated.parquet` | rate | prepared + `r` |
| `data/interim/iv.parquet` | iv | IV diagnostics + `valid_iv` |
| `data/processed/greeks.parquet` | greeks | 15 Greeks per usable contract |
| `data/processed/surfaces.parquet` | surface | 80 points per date |
| `data/processed/iv_qc.parquet` | iv | daily solver/repricing QC |
| `data/processed/greek_qc.parquet` | greeks | per-Greek coverage |
| `data/processed/surface_qc.parquet` | surface | coverage + calendar violations |
| `artifacts/latest.json` | runner | pointer to latest run |
| `artifacts/runs/<id>/manifest.json` | runner | inputs, config, stage metrics, QC |
| `artifacts/runs/<id>/run.log` | runner | timestamped stage log |
| `artifacts/report.html` | report/notebook | portable report |
| `artifacts/parity_report.md` | parity | Markdown parity summary |

All generated paths are gitignored.

### 12.2 HTML report

`build_run_report(settings, run_id, output)` composes:

1. **Run overview** — contracts, usable IV, dates, expiries, median IV/rate,
   pipeline-stage table.
2. **Data quality** — IV solver outcomes, daily coverage, rate history,
   contract coverage, repricing residuals, grid coverage.
3. **Implied volatility** — smiles for up to `report.max_dates` dates and the
   latest grid heatmap.
4. **Greeks** — delta/gamma/vega/theta vs strike and the gamma surface.
5. **Actual versus synthetic** — history and scatter for the most frequently
   observed contract.
6. **Portfolio example** — long OTM strangle at the front expiry (nearest OTM
   call and put), portfolio delta and expiry payoff, positions table.
7. **Configuration** — the full run configuration and a units footnote.

Figures are base64 PNGs inside the single HTML file: no external assets, no
server, no JavaScript. The reference run produces **18 figures** (~1.15 MB).
The builder closes Matplotlib figures afterwards so notebook output stays clean.

### 12.3 Parity report

`artifacts/parity_report.md` contains:

- convention, row counts, matched/unmatched counts;
- median/p95/max absolute IV difference and within-tolerance share;
- per-maturity buckets (`<=7`, `8-30`, `31-90`, `91-180`, `>180` days);
- per-date rate comparison (median legacy/new, median difference, correlation).

---

## 13. Notebook

`notebooks/Sensi.ipynb` is generated by `scripts/build_notebook.py`; edit the
generator, not the JSON, if you need structural changes.

It contains **21 cells (11 code)**:

1. Parameters (env/const): `START`, `END`, `FORCE`, `REPORT_PATH`, `MAX_DATES`,
   `DPI`.
2. Bootstrap: `%matplotlib inline`, locates the repo, inserts `src/`, imports.
3. Run pipeline and display stage summary.
4. Dataset summary and anomaly table.
5. QC figures: rate history, coverage, repricing, status breakdown, surface
   coverage.
6. IV smiles for up to `MAX_DATES` dates + grid heatmap.
7. Greeks vs strike, by expiry, and gamma surface for the latest date.
8. Actual vs synthetic history/scatter.
9. Portfolio: long OTM strangle, contributions, totals, payoff.
10. Optional interactive widgets (skipped with a message if `ipywidgets` is
    absent).
11. Build `artifacts/report.html` and print figure count/size.

Parameters can be set in the cell or via environment variables:

| Variable | Effect |
| --- | --- |
| `NIFTY_SENSI_START` | first date |
| `NIFTY_SENSI_END` | last date |
| `NIFTY_SENSI_FORCE` | `1` ignores stage caches |

### 13.1 Running

Open in JupyterLab and run all cells, or execute headlessly:

```powershell
python scripts/check_notebook.py
python scripts/check_notebook.py --start 2025-01-01 --end 2025-01-31
python scripts/check_notebook.py --keep-outputs   # embed text + PNG outputs
```

The checker executes code cells without Jupyter (strips `%` magics, forces the
Agg backend), reports the number of inline figures, and validates that the
report has at least eight embedded images and no external image references.

---

## 14. Scripts

| Script | Purpose |
| --- | --- |
| `scripts/run_pipeline.py` | run stages and optionally the report without installing the package |
| `scripts/fetch_data.py` | download spot/options into `data/raw` without overwriting existing files |
| `scripts/build_notebook.py` | regenerate `notebooks/Sensi.ipynb` from Python source |
| `scripts/check_notebook.py` | Jupyter-free notebook executor and validator |
| `scripts/make_fixtures.py` | regenerate the deterministic test fixture |

Examples:

```powershell
python scripts/run_pipeline.py --start 2025-01-01 --end 2025-06-30 --report
python scripts/run_pipeline.py --stages prepare,rate --force

python scripts/fetch_data.py --source spot --start 2025-01-01 --end 2025-12-31
python scripts/fetch_data.py --source all --start 2025-01-01 --end 2025-12-31 --force

python scripts/build_notebook.py
python scripts/check_notebook.py --keep-outputs
python scripts/make_fixtures.py
```

`fetch_data.py` refuses to overwrite an existing target unless `--force` is
given, so the committed canonical feeds are never clobbered by accident.

---

## 15. Testing

Run with the standard library (no pytest required):

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -t .
```

With pytest installed (`.[dev]`): `python -m pytest -q`.

40 tests across six modules:

| Module | Coverage |
| --- | --- |
| `test_black76` | reference prices (10.4506 / 5.5735), put-call parity, vectorisation, IV round-trip (scalar and vector), below-intrinsic, bound status, all 15 Greeks vs finite differences, unit scaling, NaN on invalid input |
| `test_prepare` | legacy transform contract, dirty-row removal, no input mutation, spot/moneyness alignment, day window, dedupe, missing spot error |
| `test_rate` | parity rate recovery (6.5%), `r` attachment, fallback chain, rolling-median fill, legacy `data_day`, fixture truth |
| `test_iv_greeks` | IV recovery vs known generator (median < 1e-4, p95 < 6e-3), repricing quality (< 5e-8 relative), OTM-only usability, status labels, Greek coverage/signs, analytic delta vs bumped price |
| `test_surface` | grid shape (dates × 80), finite plausible IVs, no violations on flat IV, no violations for monotone total variance, empty input |
| `test_cli_smoke` | end-to-end build → cached rebuild → HTML report → QC on a temporary root; unknown stage exits 1 |

### Fixtures

`tests/fixtures/` is generated by `scripts/make_fixtures.py` and committed:

- 3 observation dates, 2 tenors (9 and 37 days), 11 strikes per tenor,
  calls and puts — 138 raw rows including 6 deliberately dirty rows;
- prices produced by the package's own Black-76 at a known **6.5% rate** and a
  skewed smile, then rounded to the INR 0.05 tick;
- `truth_sample.csv` carries the generating IV/rate so tests know the answer;
- 132 rows survive preparation; 66 are usable after the OTM filter.

Because prices are tick-rounded, cheap options carry a comparatively large IV
round-trip error by construction; thresholds reflect that.

---

## 16. Legacy parity

Command:

```powershell
nifty-sensi parity --legacy data/df_with_greeks.parquet `
  --convention legacy_zero_rate --details data/interim/parity.parquet
```

### 16.1 What the legacy pipeline actually computed

The legacy `iv` column is **exactly** discounted Black-76 implied volatility
with **`F = S` and `r = 0`**: re-solving with those inputs reproduces all
sampled legacy IVs with a median absolute difference of **4.6e-13** (p95
1.2e-9, 100% within 1e-6). The legacy `r` column was computed but never used in
its IV solve.

The new pipeline deliberately fixes this: it discounts with the parity rate and
uses `F = S·exp(rT)`. `--convention legacy_zero_rate` exists to separate that
intentional change from any solver/lookup difference.

### 16.2 Measured differences (full committed dataset)

| Metric | `legacy_zero_rate` | `pipeline` |
| --- | --- | --- |
| Legacy rows | 481,009 | 481,009 |
| New rows | 512,085 | 512,085 |
| Matched contracts | 415,624 | 415,624 |
| Legacy-only | 65,385 | 65,385 |
| New-only | 96,461 | 96,461 |
| Median absolute IV diff | **4.7e-13** | **1.8 vol pts** |
| p95 absolute IV diff | 1.1e-9 | 8.1 vol pts |
| Max absolute IV diff | 0.1097 | 0.5314 |
| Within 1e-3 | **99.9988%** | 0.43% |

Per-maturity medians under `pipeline` grow with tenor: 0.43 vol pts (`≤7d`),
1.3 pts (`8–30d`), 2.6 pts (`31–90d`), 2.7 pts (`91–180d`),
4.3 pts (`>180d`) — the expected signature of using carry over longer horizons.

Unmatched rows are explained:

- **65,385 legacy-only** rows have `days_to_exp > 365`; the new pipeline caps
  maturity at 365 days by configuration.
- **96,461 new-only** rows are in-the-money contracts the legacy run dropped;
  the new pipeline retains them for audit but marks every one `valid_iv =
  False` under `otm_only`.

Only **5 of 415,624** contracts differ by more than 1e-3 under the legacy
convention. All five are quotes at or below intrinsic where the new solver
pins IV to the lower bound (`0.0001`), while the legacy solver returned a
nominal value (e.g. 0.011–0.11). They are correctly excluded from `valid_iv`.

### 16.3 Rate comparison

Across 357 dates where both outputs carry a rate:

| Statistic | Legacy | New |
| --- | --- | --- |
| Median | 6.78% | 7.16% |
| Median difference | — | +0.23% |
| Standard deviation of difference | — | 2.12% |
| Correlation | — | 0.828 |
| Dates with \|difference\| > 5% | — | 17 |

The legacy rate was itself inconsistent: 10,268 rows have no rate, 159 Greeks
failed, and its `iv` and `implied_volatility` columns disagree by more than
1e-3 on 231,126 rows. The new pipeline has one rate, one IV and one source per
date.

---

## 17. Reference run

Full committed feed (`2025-01-01` → `2026-06-29`), Windows laptop, cold cache:

| Stage | Rows | Seconds |
| --- | --- | --- |
| `prepare` | 512,085 | 1.64 |
| `rate` | 512,085 | 2.61 |
| `iv` | 512,085 | 32.90 |
| `greeks` | 512,085 | 6.52 |
| `surface` | 29,280 | 5.56 |
| **End to end incl. HTML report** | — | **~76** |

Counts:

| Metric | Value |
| --- | --- |
| Raw option rows | 609,845 |
| Prepared contracts | 512,085 |
| Observation dates | 366 |
| Expiries | 94 |
| Strikes | 179 |
| Calls / puts | 255,002 / 257,083 |
| Rate source | `parity` on all 366 dates |
| IV `ok` / `below_intrinsic` / `at_upper_bound` | 436,792 / 75,227 / 66 |
| Usable IVs (`valid_iv`) | 257,423 (50.3%) |
| Repricing residual p95 | 2.5e-6 |
| Surface dates / rows | 366 / 29,280 |
| Grid coverage | 100% |
| Calendar violations | 4,209 of 25,620 pairs (16.4%) |
| Report | 18 embedded figures, ~1.15 MB |

These numbers are diagnostics, not guarantees; they move with the feed and
configuration. The calendar violations are a property of the raw data (16.4% of
adjacent-maturity variance pairs invert), reported rather than repaired.

---

## 18. Caveats and limitations

- **No dividend yield.** NIFTY is treated as a price index (`q = 0`). For a
  dividend-paying underlying, pass `q` to `forward_price`/`price_from_spot`.
- **European exercise.** Black-76 assumes European options; no early-exercise
  premium is modelled.
- **Fixed day count.** `T = calendar days / 365`; the `options.day_count`
  setting is declared but this version fixes the basis at 365.
- **One rate per date.** The parity rate is a single risk-free proxy, not a
  term structure; rate uncertainty propagates into long-dated IVs.
- **OTM-only IV.** By default only OTM contracts are usable, so deep ITM
  information is not represented in the surface.
- **Stale quotes.** Quotes below intrinsic are flagged and excluded; the five
  lower-bound contracts in [§16.2](#162-measured-differences-full-committed-dataset)
  are examples.
- **Calendar violations are reported, not fixed.** No arbitrage-free
  parameterisation (SVI, VolGAN) is applied; the archived VolGAN code is
  reference material only.
- **Synthetic comparison is nearest-neighbour** in maturity, then strike, then
  rate — a diagnostic reconstruction, not a fitted model.
- **Scope.** No Docker image, scheduler, database or dashboard; the pipeline,
  CLI, notebook and static report are the deliverable.
- **Performance.** The IV stage dominates (~33 s for 512k rows). Increase
  `iv.max_iter` only if your data needs it; Newton usually converges in a few
  iterations.

---

## 19. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `ModuleNotFoundError: nifty_sensi` | Install with `pip install -e .`, or set `PYTHONPATH=src` (see [§3.2](#32-run-without-installing-src-layout)) |
| `ModuleNotFoundError: scipy` / `pyarrow` | `pip install -e .` (core dependencies) |
| `option feed not found` / `spot feed not found` | Check `data.options_file` / `data.spot_file`; run `ingest` first if needed |
| `spot price missing for N observation date(s)` | The spot feed does not cover an option date; extend it or restrict `data.start`/`data.end` |
| Report missing | Run `nifty-sensi build` first, then `nifty-sensi report` |
| Stale results | Caches are keyed by config + input hashes; use `--force` after editing inputs in place if needed |
| `PyYAML is required to read YAML configs` | `pip install pyyaml` or use JSON |
| Widgets cell prints “ipywidgets is not installed” | Install `.[notebook]`; static figures still render |
| Matplotlib “non-interactive” warning | Expected headless (Agg); `scripts/check_notebook.py` suppresses it |
| `tabulate` missing | Not required: parity Markdown tables are written directly |
| Parity says “missing new output” | Run `nifty-sensi build` or pass `--new` |
| Permission error writing parquet on Windows | Close the file in Excel/another process; writes are atomic and retry-safe |
| Very long IV stage | Expected for 500k+ rows; use a date window or `--stages` while iterating |

Logs: `artifacts/runs/<run_id>/run.log`; raise verbosity with `--verbose`
(DEBUG) or `python -m nifty_sensi --verbose ...`.

---

## 20. Repository hygiene and generated data

- Generated directories are ignored by **nested** `.gitignore` files
  (`*` plus `!.gitignore`) in `data/raw`, `data/interim`, `data/processed` and
  `artifacts`; the root `.gitignore` is untouched.
- Committed feeds stay in `data/` (`nifty_close.csv`, `niftyoptions_df.csv`);
  the legacy outputs `df_with_greeks.parquet` and `datapath.csv` are reference
  inputs and are not regenerated in place.
- Test fixtures are small and committed so tests run offline.
- New source, config, scripts, tests and the notebook are added; no existing
  file is modified or deleted.
- The working branch for this work is `akhil`; nothing is committed
  automatically by the pipeline.

To clean generated state without touching committed data:

```powershell
Get-ChildItem data/interim, data/processed -File -Exclude .gitignore | Remove-Item -Force
Remove-Item -Recurse -Force artifacts/runs
Remove-Item -Force artifacts/latest.json, artifacts/report.html, artifacts/parity_report.md
```

Nested `.gitignore` files remain.

---

## 21. References, licence and log

- Black, F. (1976), *The pricing of commodity contracts* — Black-76 model.
- Cont, R. & Vuletić, M. (2023), *Simulation of arbitrage-free implied
  volatility surfaces* and *VolGAN* — in `references/`.
- NSE NIFTY 50 historical index data; HuggingFace dataset
  `rissin/nse-options-intraday` (option chain).
- Legacy project notes: `log.md`; original overview: `README.md`.
- Licence: MIT — see `LICENSE`.

---

*This document describes `nifty_sensi` 1.0.0 as implemented in this
repository. The legacy notebooks and modules remain available unchanged for
comparison; use the package and CLI for new work.*
