# Project Log — IE612

Chronological record of project direction changes, major milestones, and key decisions.

---

## 2026-10-18 — Project Pivot: VolGAN → Option Sensitivity Analysis

**Decision**: Repurposed the repository from a VolGAN (GAN-based volatility surface generation) project to an **Option Sensitivity Analysis** project focused on computing and visualizing option Greeks for NIFTY 50 index options.

### Changes Made

| Area | Before | After |
|------|--------|-------|
| **Primary Goal** | Train GAN to generate arbitrage-free IV surfaces (Cont & Vuletic, 2023) | Calculate & visualize 14 Greeks (1st/2nd/3rd order) via Black-76 |
| **Main Notebook** | `VolGAN_example.ipynb`, `VolGAN.ipynb` | `Sensi.ipynb` |
| **Core Library** | Custom PyTorch GAN (`VolGAN.py`) | `opengreeks.black76` (industry-standard) |
| **Output** | Generated IV surfaces, VIX simulation | Interactive Greek visualizations (Plotly) |
| **VolGAN Code** | Root level (`VolGAN.py`, `VolGAN_example.ipynb`) | Archived to `archive/` |

### Files Moved to `archive/`
- `VolGAN.py` — Core VolGAN implementation (Generator, Discriminator, training loops, arbitrage penalties)
- `VolGAN_example.ipynb` — Training & evaluation demo (sampling, reweighting, VIX)
- `VolGAN.ipynb` — Development notebook

### Files Retained/Repurposed
- `datacleaning.py` / `datacleaning.ipynb` — Data preprocessing pipeline (shared foundation)
- `download_nifty_close.py` — NSE spot price downloader (enhanced with logging, pagination)
- `streaming_data.ipynb` / `data.ipynb` — Options data ingestion from HuggingFace
- `data/` — CSV/Parquet data files
- `references/` — Academic papers (kept for context)

### New Capabilities in `Sensi.ipynb`
- Black-76 implied volatility solver (Brent's method)
- 14 Greeks: delta, gamma, vega, theta, rho, vanna, charm, vomma, speed, zomma, color, veta, ultima, dual_delta, dual_gamma
- Interactive Plotly widgets: Greek vs Strike / Expiry / Moneyness / Heatmap
- Per-date, per-option-type filtering

---

## 2026-01 to 2026-06 — Original VolGAN Development Phase

### Data Acquisition
- NIFTY 50 spot prices via NSE API (`download_nifty_close.py`)
- NIFTY options from HuggingFace `rissin/nse-options-intraday` (daily 2025–2026, intraday 2026)
- DuckDB for efficient Parquet querying

### Data Processing (`datacleaning.py`)
- `optiondata_transform()` — Date parsing, TTM, moneyness, spot alignment
- `data_day()` — Put-call parity risk-free rate estimation, OTM filtering
- Kernel smoothing (Nadaraya-Watson) & B-spline interpolation for IV surfaces
- Fixed grid: 10 moneyness × 8 maturities = 80-vector per day

### VolGAN Implementation (`archive/VolGAN.py`)
- **Generator**: Noise + condition (83-dim) → 3-layer MLP (Softplus/Sigmoid) → 81-dim output (Δr, Δlog_IV)
- **Discriminator**: Condition + target (81-dim) → 2-layer MLP (Softplus/Sigmoid) → probability
- **Arbitrage Penalties**: Finite-difference matrices for calendar spread, vertical spread, butterfly (Cont & Vuletic, 2023)
- **Two-Phase Training**:
  1. Gradient Matching (`n_grad` epochs) — balance BCE vs smoothness → learn α, β
  2. Main Training (`n_epochs`) — Generator loss = BCE + α·m_smooth + β·t_smooth
- **Post-Processing**: Importance reweighting by `exp(-β × penalty)`, weighted mean/quantile surfaces, model-implied VIX

### Key Reference
> Cont, R. & Vuletic, M. (2023) — "VolGAN: Generative Adversarial Networks for Volatility Surface Simulation"  
> Cont, R. & Vuletic, M. (2023) — "Simulation of Arbitrage-Free Implied Volatility Surfaces"

---

## Future Direction Ideas (Tracking)

- [ ] **Greek Surface Modeling**: Fit parametric surfaces (SVI, SSVI) to Greeks across moneyness/tenor
- [ ] **Risk Aggregation**: Portfolio-level Greeks, scenario analysis (spot/vol shocks)
- [ ] **Backtesting**: Compare model Greeks vs. realized P&L attribution
- [ ] **Intraday Analysis**: Use 1-min Upstox data for high-frequency Greek dynamics
- [ ] **Alternative Models**: Heston, SABR calibration for cross-validation
- [ ] **Dashboard**: Streamlit/Plotly Dash deployment for interactive exploration

---

## Version History

| Date | Version | Description |
|------|---------|-------------|
| 2026-10-18 | 2.0.0 | Pivot to Option Sensitivity Analysis; VolGAN archived |
| 2026-06-?? | 1.x | VolGAN development (GAN training, arbitrage penalties, VIX) |
| 2026-01-?? | 0.1 | Initial data acquisition & cleaning pipeline |

---

*Log maintained manually. Update on each significant directional change.*