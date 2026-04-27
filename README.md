# Hierarchical Forecasting of U.S. Class I Freight Rail Carloads

**Northern Arizona University — Department of Mathematics and Statistics**
*Sakina Lord · James Hope-Meek · Megan Ruza Dsouza*
*Mentor: Dr. Robert Buscaglia*

---

![Capstone Poster](Capstone_Poster.png)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Background & Motivation](#background--motivation)
3. [Data](#data)
4. [Repository Structure](#repository-structure)
5. [Methodology](#methodology)
   - [Data Pipeline](#data-pipeline)
   - [GAM: Total Volume Forecasting](#gam-total-volume-forecasting)
   - [RNN-LSTM: Market Share Forecasting](#rnn-lstm-market-share-forecasting)
   - [Hierarchical Combination](#hierarchical-combination)
   - [Extended Benchmark Pipeline](#extended-benchmark-pipeline)
6. [Model Validation](#model-validation)
7. [Results](#results)
8. [Dependencies](#dependencies)
9. [Getting Started](#getting-started)
10. [Limitations & Future Work](#limitations--future-work)
11. [References](#references)

---

## Project Overview

This project develops a **two-stage hierarchical forecasting framework** for predicting commodity flows through the U.S. Class I freight rail network. The framework decomposes the forecasting problem into two interpretable sub-problems:

1. **Total Market Volume** — predicted by a Generalized Additive Model (GAM) with penalized splines.
2. **Market Share Allocation** — predicted by a Recurrent Neural Network with Long Short-Term Memory (RNN-LSTM), outputting proportions for each Company × Commodity × Reception Type pair.

Final predictions are obtained by multiplying the GAM-derived volume estimate by the RNN-derived proportions:

```
Â_Prediction = Ĉ_volume × P̂_Market
```

This design prioritizes **interpretability at the aggregate level** (via the GAM) while leveraging the **sequence modeling power of deep learning** at the disaggregate, high-dimensional level (via the RNN-LSTM).

---

## Background & Motivation

The North American freight rail network spans over **140,000 track-miles** and is a critical backbone of the national economy, supporting more than 1.1 million jobs and representing the largest private infrastructure investment of any industry. Accurate demand forecasting enables rail operators to:

- Optimize locomotive and crew scheduling
- Improve terminal staffing and equipment allocation
- Inform long-range capital investment decisions

This analysis covers **8 Class I freight rail companies** (those with annual operating revenue above $490 million, adjusted for inflation), tracking weekly commodity movements across the STB reporting period from March 2017 through February 2026. The dataset captures major structural disruptions including:

- The **COVID-19 pandemic** (2020–2021)
- **Post-pandemic demand recovery**
- The **CP–KCS merger** into CPKC
- A **reporting regulation change** effective June 20, 2020, which added chemicals and plastics as distinct commodity categories

---

## Data

**Source:** [Surface Transportation Board (STB) — Rail Service Data](https://www.stb.gov/reports-data/rail-service-data/)

Each Class I railroad is required to submit weekly reports detailing the number of carloads moved for each commodity group.

| Field | Description |
|---|---|
| `week` | Week ending date (Wednesday) |
| `year` | Calendar year |
| `company` | Class I railroad identifier |
| `commodity_group` | Commodity name |
| `commodity_group_code` | STB commodity code |
| `originated` | Carloads that began on this railroad |
| `received` | Carloads that arrived from another railroad |
| `originated_received` | Sum per row |
| `total_originated_received` | Cumulative total |

**Coverage:**
- **Period:** March 2017 – February 2026 (weekly frequency)
- **Companies:** 8 Class I railroads
- **Commodity groups:** 22 per company
- **Observations:** 66,154 rows × 7 fields

> **Note:** The STB data is scraped from the STB website using the Python scraping script in `Scraping Files/` and cleaned via the R pipeline in the project root before being written to `Data/`.

---

## Repository Structure

```
Transportation/
│
├── Data/                        # Cleaned weekly carload CSVs
│   └── Weekly_Cargo_Data_2017_2026.csv
│
├── Docs/                        # Capstone poster, write-ups, references
│
├── Meeting Notes/               # Weekly meeting agendas and notes
│
├── Modeling/                    # Core model scripts
│   ├── freight_pipeline.py      # GAM benchmark pipeline (4 models + CV)
│   └── ...                      # RNN-LSTM training scripts
│
├── Scraping Files/              # Python STB web scraper
│
├── Time Series/                 # Exploratory time series analysis
│
├── Visuals/                     # Generated figures and diagnostics
│   └── freight_pipeline/
│       └── 2017_onwards/        # Pipeline output PNGs
│
├── sandbox/                     # Experimental / scratch notebooks
│
├── CAPPY.Rproj                  # RStudio project file
├── .gitignore
└── README.md
```

---

## Methodology

### Data Pipeline

```
STB Website → Python scraper → Raw CSVs → R cleaning pipeline → Cleaned Data
```

The R cleaning pipeline standardizes column names, resolves company codes across the CP/KCS/CPKC merger, handles missing weeks, and flags the June 2020 reporting change. All cleaned data lands in `Data/`.

---

### GAM: Total Volume Forecasting

A **Generalized Additive Model** is fit separately for each of three target series — Originated carloads, Received carloads, and Total carloads:

```
Ĉ(t) = β₀ + f_trend(t) + Σ f_c(t) + Σ f_k(t) + f_seasonal(t) + f_cal(t)
```

| Component | Description |
|---|---|
| `f_trend(t)` | Overall market trend — natural cubic spline, knots selected by CV |
| `f_c(t)` | Company-level splines — 8 knots each, evenly spaced (bi-quarterly) |
| `f_k(t)` | Commodity-level splines — 8 knots each, evenly spaced |
| `f_seasonal(t)` | Fourier basis — 12 harmonics capturing monthly seasonal cycles |
| `f_cal(t)` | Calendar indicators — quarter dummies + holiday-adjacent weeks |

**Regularization:** All components are estimated jointly via Ridge (L2) regression (`sklearn.linear_model.Ridge`), which shrinks low-impact basis coefficients toward zero. The final Total volume GAM has **309 effective coefficients** after penalization.

**Hyperparameter selection:** Number of overall trend knots and Ridge penalty strength `α` were chosen by **5×2 repeated time-series cross-validation**, minimizing held-out RMSE.

**Implementation:** `sklearn.preprocessing.SplineTransformer` + `sklearn.linear_model.Ridge`
**Dependencies:** Python 3.10.12; scikit-learn 1.7.2

---

### RNN-LSTM: Market Share Forecasting

The market share allocation problem involves predicting **352 time series simultaneously** — one proportion per Company × Commodity × Reception Type combination — where each proportion represents that pair's share of the week's total carload volume (bounded [0, 1], summing to 1 across all pairs).

**Architecture:**
- Input: 352-dimensional proportion vectors over a rolling lookback window
- Core: Recurrent Neural Network with **Long Short-Term Memory (LSTM)** hidden state, which selectively retains long-range dependencies while avoiding the vanishing gradient problem
- Final: Dense output layer (352 units, softmax normalization)

**Hyperparameter search:** ~1,000 models were trained across a grid of:

| Hyperparameter | Search Range |
|---|---|
| Lookback (weeks) | varied |
| Hidden layer size | 64 – 1024 nodes |
| Learning rate | varied |
| Dropout | 0.0 – 0.4 |
| Number of layers | 1 – 3 |

The top 40 configurations were re-trained with **5-seed cross-validation**. The final selected model uses:
- **1 hidden layer** (128 nodes)
- **Dropout: 0.0 – 0.1**
- **4 total layers** (including input/output)
- **767,712 parameters**

**Loss function:** Huber Loss (robust to outliers in proportion space)

**Dependencies:** Python 3.12.13; PyTorch 2.10.0; Pandas 2.2.2; NumPy 2.0.2

---

### Hierarchical Combination

```
Final Carload Forecast = GAM Volume Estimate × RNN Proportion Estimate
```

This hierarchical structure ensures that disaggregate (company-commodity) forecasts are **coherent** with the aggregate total — a constraint naturally enforced by the multiplicative combination.

---

#### Extended Benchmark Pipeline (`freight_pipeline.py`) - Not Used For Final Evaluation

In addition to the primary GAM + RNN framework, `freight_pipeline.py` implements a **four-model benchmark suite** for aggregate volume forecasting with a complete cross-validation and visualization framework. This is suitable for rapid experimentation and comparison.

**Models included:**

| Model | Class | Key Design |
|---|---|---|
| **GAM** | `GAMSpline` | Penalized cubic spline trend + Fourier seasonality, Ridge-estimated |
| **ProphetLite** | `ProphetLite` | Piecewise-linear trend + Fourier seasonality, Ridge-estimated |
| **SARIMALite** | `SARIMALite` | SARIMA(2,1,2)×(1,1,1)[52] via conditional MLE (L-BFGS-B) |
| **OLS Fixed-Effects** | `FixedEffectsOLS` | Two-way FE (company + commodity) on log-carloads + Fourier + calendar |

**Validation:** 5×2 repeated time-series cross-validation with randomized jitter on split points (preserving temporal order, no data leakage). Produces 10 non-overlapping train/test folds per model.
   
**Forecasting:** 26-week (6-month) ahead forecasts are generated for all four models after final full-sample fitting. The first 13 of these weeks are used in the hierarchical model.

---

### Model Validation

All models are evaluated using **5×2 repeated time-series cross-validation** — a forward-chaining scheme where training always precedes testing in time to prevent lookahead bias.

**Metrics reported:**

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error (carloads) |
| RMSE | Root Mean Squared Error (carloads) |
| MAPE | Mean Absolute Percentage Error (%) |
| R² | Coefficient of determination |

CV results are saved to `Visuals/freight_pipeline/2017_onwards/cv_results.csv` and `cv_summary.csv`.

---

## Results: 13 Week Forecast

### GAM vs. Naive Benchmark (Volume Forecasting)

| Target | GAM × RNN RMSE | Naive (mean) RMSE |
|---|---|---|
| Originated | **8,896** | 20,067 |
| Received | **1,258** | 169,181 |

The GAM reduces RMSE by up to **10× vs. a naive mean baseline**, demonstrating strong aggregate predictive performance.

### RNN-LSTM Performance (Market Share)

- **Test RMSE:** 0.02865 (proportion scale)
- **Per-category error:** ~2.865%

### Key Findings

- Freight carload volumes exhibit **high stability** outside macro shocks (COVID, recessions), which means naive mean models perform deceptively well on average.
- The RNN-LSTM successfully identifies trends beyond the mean in proportion space, though the high number of series (352) means small changes can be absorbed as noise.
- **BNSF carloads are systematically underpredicted** at the commodity level, implying compensatory overprediction for other railroads — a distributional precision problem inherent to the hierarchical constraint.
- GAM **outperformed Facebook's Prophet** in aggregate volume forecasting across all three carload categories (Originated, Received, Total).
- Each model showed better individual performance when trained on single companies versus the combined dataset, suggesting that **company-stratified models** may improve disaggregate accuracy.

---

## Dependencies

### Python (Core Pipeline & Modeling)

```
# Benchmark pipeline (freight_pipeline.py)
python>=3.10
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn>=1.7.2

# RNN-LSTM model
python>=3.12
pytorch>=2.10.0
pandas>=2.2.2
numpy>=2.0.2
```

Install Python dependencies:
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
pip install torch pandas numpy   # for RNN scripts
```

### R (Data Cleaning Pipeline)

```r
# Core packages used in cleaning pipeline
tidyverse
lubridate
readr
```

Install R dependencies:
```r
install.packages(c("tidyverse", "lubridate", "readr"))
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/littlleHawk/Transportation.git
cd Transportation
```

### 2. Scrape fresh data (optional)

If you want to update the dataset beyond February 2026, run the STB scraper:

```bash
cd "Scraping Files"
python stb_scraper.py   # writes raw CSVs to Data/raw/
```

Then run the R cleaning pipeline from within RStudio (open `CAPPY.Rproj`) or from the terminal:

```bash
Rscript cleaning_pipeline.R   # writes cleaned data to Data/
```

### 3. Run the benchmark pipeline

```bash
cd Modeling
python freight_pipeline.py
```

This will:
- Load `Data/Weekly_Cargo_Data_2017_2026.csv`
- Fit and cross-validate all four models (GAM, ProphetLite, SARIMALite, OLS-FE)
- Generate a 26-week forecast
- Save 15 figures to `Visuals/freight_pipeline/2017_onwards/`
- Save CV tables to the same directory

### 4. Use the pipeline as a module (downstream scripts)

```python
import sys
sys.path.append("Modeling")
import freight_pipeline as fp

artefacts = fp.build_fitted_models(
    data_path="Data/Weekly_Cargo_Data_2017_2026.csv",
    run_cv=True,      # set False for fast iteration
    verbose=True,
)

gam   = artefacts["gam_model"]
ts    = artefacts["ts"]
df_cv = artefacts["df_cv"]

# Forecast 26 weeks ahead with the GAM
import numpy as np
t_future = np.arange(ts["week_num"].iloc[-1] + 1,
                     ts["week_num"].iloc[-1] + 27)
forecast = gam.predict(t_future)
```

---

## Limitations & Future Work

- **Distributional precision:** The hierarchical constraint ensures aggregate coherence, but systematic biases at the company-commodity level (especially BNSF) suggest that bottom-up or MinT reconciliation approaches could improve disaggregate accuracy.
- **Manual bias corrections** (+30,000 originated / +15,000 received for BNSF) were applied in the final model; a principled company-stratified intercept or separate per-company GAM would be more robust.
- **RNN complexity vs. data stability:** The relatively stable nature of freight proportions over time means the RNN's added complexity offers modest gains over a naive mean for many pairs. A simpler hierarchical time series model (e.g., ETS or ARIMA per series with MinT reconciliation) may achieve comparable accuracy with greater interpretability.
- **Reporting break (June 2020):** The chemical/plastics commodity split introduces a structural break that is currently handled by truncating the pre-break series. A unified treatment with an indicator variable or separate models by regime would be more rigorous.
- **External regressors** (fuel prices, industrial production indices, GDP) are not currently included; incorporating macroeconomic covariates could improve forecasting performance during disruption periods.

---

## References

1. "The Public Benefits of Freight Railroads." GORAIL, May 2023. https://gorail.org/wp-content/uploads/GoRail_Public-Benefits_update-050123.pdf
2. "An Introduction to Class I Freight Railroads." RailInc, Mar 2023. https://public.railinc.com/about-railinc/blog/introduction-class-i-freight-railroads
3. "Rail Service Data." Surface Transportation Board. https://www.stb.gov/reports-data/rail-service-data/
4. "Report: Weekly Carloads by Railroad." RSI Logistics. https://www.rsilogistics.com/resources/railroadperformance/weekly-cars-by-railroad/
5. Petition for Rulemaking to Amend 49 CFR §§ 1250. Surface Transportation Board, 2020.

---

*Northern Arizona University · Department of Mathematics and Statistics · Capstone Project 2025–2026*
