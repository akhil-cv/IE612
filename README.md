# IE612 - VolGAN: Volatility Surface Generation with GANs

Implementation of **VolGAN** (Volatility Generative Adversarial Network) for simulating arbitrage-free implied volatility surfaces on NIFTY 50 index options, based on:

> **Cont, R. & Vuletic, M. (2023)** — "VolGAN: Generative Adversarial Networks for Volatility Surface Simulation"  
> **Cont, R. & Vuletic, M. (2023)** — "Simulation of Arbitrage-Free Implied Volatility Surfaces"

---

## 📁 Repository Structure

```
IE612/
├── VolGAN.py                 # Core VolGAN implementation
├── datacleaning.py           # Data preprocessing & IV surface construction
├── download_nifty_close.py   # NSE historical spot price downloader
├── VolGAN_example.ipynb      # Training & evaluation demo
├── datacleaning.ipynb        # Data cleaning walkthrough
├── streaming_data.ipynb      # Data acquisition from HuggingFace
├── references/               # Academic papers
├── data/
│   ├── nifty_close.csv       # NIFTY 50 spot prices
│   ├── niftyoptions_df.csv   # Raw options data
│   ├── datapath.csv          # Per-contract implied volatilities
│   └── surfacepath.csv       # Daily IV surfaces (80-vector grid)
└── WORKFLOW.md               # Detailed pipeline documentation
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch numpy pandas scipy matplotlib tqdm yfinance statsmodels scikit-learn requests duckdb
```

### 2. Download Spot Prices (NSE)

```bash
python download_nifty_close.py
```
Produces `data/nifty_close.csv` with daily OHLC.

### 3. Fetch Options Data

Run `streaming_data.ipynb` — pulls NIFTY options from HuggingFace dataset `rissin/nse-options-intraday`:
- `historical_daily/NIFTY/NIFTY_2025.parquet`
- `historical_daily/NIFTY/NIFTY_2026.parquet`
- `upstox_intraday/NIFTY/NIFTY_2026.parquet`

Outputs `data/niftyoptions_df.csv`.

### 4. Build IV Surfaces

Run `datacleaning.ipynb` (or use `datacleaning.py` functions):
```python
from datacleaning import optiondata_transform, data_day, interpolate_data

# Transform raw data → add TTM, moneyness, spot
niftyoptions_df, niftyspot_df, spot, obs_dates = optiondata_transform(opts_df, spot_df)

# Per-day: filter OTM, estimate r via put-call parity, compute BS-IV
# Interpolate onto fixed (moneyness × τ) grid → flatten to 80-vector
# Saves: data/datapath.csv, data/surfacepath.csv
```

**Grid:** 10 moneyness points × 8 maturities = 80 points per surface

### 5. Train VolGAN

```python
import VolGAN

gen, gen_opt, disc, disc_opt, *rest = VolGAN.VolGAN(
    datapath="data/datapath.csv",
    surfacepath="data/surfacepath.csv",
    tr=0.85,              # train split
    noise_dim=32,
    hidden_dim=16,
    n_epochs=10000,
    n_grad=25,            # gradient matching epochs
    lrg=0.0001,
    lrd=0.0001,
    batch_size=100,
    device='cuda'         # or 'cpu'
)
```

### 6. Evaluate & Generate

See `VolGAN_example.ipynb` for:
- Sampling 10,000 paths from trained generator
- Computing arbitrage penalties (calendar, butterfly, vertical spread)
- Importance reweighting by `exp(-β × penalty)`
- Weighted mean/quantile surfaces per day
- Model-implied VIX calculation

---

## 🔬 Key Components

### Arbitrage Penalties (Cont & Vuletic, 2023)
Finite-difference matrices enforce:
- **Calendar spread**: Call price increasing in maturity (∂C/∂τ ≥ 0)
- **Vertical spread**: Call price decreasing in strike (∂C/∂K ≤ 0)
- **Butterfly**: Call price convex in strike (∂²C/∂K² ≥ 0)

Applied to Black-76 prices computed from generated IV surfaces.

### Generator Architecture
```
Input: noise (dim) + condition (83) → Linear(hidden) → Softplus
                                          → Linear(2×hidden) → Softplus
                                          → Linear(81) → Sigmoid
Output: [Δr_t, Δlog_IV_t]  (81-dim)
```

### Discriminator Architecture
```
Input: condition (83) + target (81) → Linear(hidden) → Softplus
                                         → Linear(1) → Sigmoid
Output: Probability real vs fake
```

### Two-Phase Training
1. **Gradient Matching** (`n_grad` epochs) — balance BCE vs smoothness penalties → learn `α, β`
2. **Main Training** (`n_epochs`) — Generator loss = `BCE + α·m_smooth + β·t_smooth`

---

## 📊 Data Pipeline

```
NSE API + HuggingFace
        │
        ▼
nifty_close.csv + niftyoptions_df.csv
        │
        ▼  [datacleaning.py]
datapath.csv (per-contract IV) + surfacepath.csv (daily 80-vector)
        │
        ▼  [VolGAN.DataPreprocesssing]
Condition (83) + Target (81) tensors
        │
        ▼  [VolGAN.VolGAN()]
Trained Generator + Discriminator
        │
        ▼  [VolGAN_example.ipynb]
Generated surfaces → Arbitrage penalties → Reweighting → Statistics/VIX
```

---

## 📚 References

- `references/R Cont, M Vuletic (2023) VolGAN.pdf`
- `references/R Cont, M Vuletic (2023) Simulation of arbitrage free IV.pdf`

---

## 📝 License

MIT License — see `LICENSE` file.