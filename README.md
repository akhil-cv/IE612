# IE612 - NIFTY Option Sensitivity Analysis

Analysis of option Greeks (sensitivities) for NIFTY 50 index options using historical market data. Computes and visualizes first, second, and third-order Greeks using the Black-76 model via the `opengreeks` library.

---

## 📁 Repository Structure

```
IE612/
├── Sensi.ipynb                    # Main analysis: IV & Greeks calculation + interactive visualization
├── datacleaning.py                # Data preprocessing & IV surface construction (reusable module)
├── datacleaning.ipynb             # Data cleaning walkthrough (notebook version)
├── download_nifty_close.py        # NSE historical spot price downloader (robust, logged)
├── streaming_data.ipynb           # Options data acquisition from HuggingFace (daily + intraday)
├── data.ipynb                     # Data exploration & spot price download (alternative)
├── references/
│   ├── R Cont, M Vuletic (2023) VolGAN.pdf
│   └── R Cont, M Vuletic (2023) Simulation of arbitrage free IV.pdf
├── data/
│   ├── nifty_close.csv            # NIFTY 50 spot prices (OHLC)
│   ├── niftyoptions_df.csv        # Raw options data (from HuggingFace)
│   ├── datapath.csv               # Per-contract implied volatilities (output)
│   └── surfacepath.csv            # Daily IV surfaces on fixed grid (output)
├── archive/                       # Legacy VolGAN implementation (GAN-based IV surface generation)
│   ├── VolGAN.py                  # Core VolGAN implementation
│   ├── VolGAN_example.ipynb       # Training & evaluation demo
│   └── VolGAN.ipynb               # VolGAN development notebook
└── LICENSE
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch numpy pandas scipy matplotlib tqdm yfinance statsmodels scikit-learn requests duckdb opengreeks plotly ipywidgets
```

### 2. Download Spot Prices (NSE)

```bash
python download_nifty_close.py
```
Produces `data/nifty_close.csv` with daily OHLC for NIFTY 50 (configurable date range).

### 3. Fetch Options Data

Run `streaming_data.ipynb` — pulls NIFTY options from HuggingFace dataset `rissin/nse-options-intraday`:
- `historical_daily/NIFTY/NIFTY_2025.parquet`
- `historical_daily/NIFTY/NIFTY_2026.parquet`
- `upstox_intraday/NIFTY/NIFTY_2026.parquet`

Outputs `data/niftyoptions_df.csv`.

### 4. Build IV Surfaces & Calculate Greeks

Run `Sensi.ipynb` (or use `datacleaning.py` functions):

```python
from datacleaning import optiondata_transform, data_day
import datacleaning as dc

# Transform raw data → add TTM, moneyness, spot, risk-free rate
niftyoptions_df, niftyspot_df, spot, obs_dates = optiondata_transform(opts_df, spot_df)

# Per-day: filter OTM, estimate r via put-call parity, compute BS-IV
# Interpolate onto fixed (moneyness × τ) grid → flatten to 80-vector
# Saves: data/datapath.csv, data/surfacepath.csv

# Then in Sensi.ipynb:
# - Calculate implied volatility via Black-76
# - Compute all Greeks: delta, gamma, vega, theta, rho, vanna, charm, vomma, speed, zomma, color, veta, ultima, dual_delta, dual_gamma
# - Interactive Plotly visualizations (strike, expiry, moneyness, heatmaps)
```

**Grid:** 10 moneyness points (0.6–1.4) × 8 maturities (7–365 days) = 80 points per surface

---

## 🔬 Greeks Computed

| Category | Greeks |
|----------|--------|
| **First Order** | Delta, Vega, Theta, Rho |
| **Second Order** | Gamma, Vanna, Charm, Vomma, Veta |
| **Third Order** | Speed, Zomma, Color, Ultima |
| **Dual** | Dual Delta, Dual Gamma |

All computed via `opengreeks.black76` using Black-76 implied volatility backed out from market prices.

---

## 📊 Data Pipeline

```
NSE API + HuggingFace
        │
        ▼
nifty_close.csv + niftyoptions_df.csv
        │
        ▼  [datacleaning.py: optiondata_transform, data_day]
        ├── spot price alignment
        ├── TTM (days/365)
        ├── moneyness (K/spot)
        ├── put-call parity → risk-free rate (r)
        └── OTM filtering (CE: m≥1, PE: m<1)
        │
        ▼  Black-76 IV solver (brentq)
datapath.csv (per-contract IV) + surfacepath.csv (daily 80-vector)
        │
        ▼  [Sensi.ipynb: opengreeks.black76]
implied_volatility + 14 Greeks per contract
        │
        ▼  [Interactive Plotly Widgets]
Greek vs Strike | Greek vs Expiry | Greek vs Moneyness | Strike-Expiry Heatmap
```

---

## 🎯 Key Features

- **Robust Data Download**: `download_nifty_close.py` handles NSE API pagination, rate limiting, and session management
- **Put-Call Parity Rate Estimation**: Implied risk-free rate per observation date
- **Black-76 IV Calculation**: Numerically stable Brent's method solver
- **Comprehensive Greeks**: 14 sensitivities via `opengreeks` (industry-standard)
- **Interactive Exploration**: Plotly + ipywidgets for dynamic Greek visualization
- **Legacy Preserved**: VolGAN GAN-based approach archived for reference

---

## 📚 References

- `references/R Cont, M Vuletic (2023) VolGAN.pdf` — Original VolGAN paper (archived approach)
- `opengreeks` library — Black-76 Greeks implementation

---

## 📝 License

MIT License — see `LICENSE` file.

---

## 📋 Project Log

See `log.md` for a chronological record of project direction changes and major milestones.