# Store Sales - Time Series Forecasting: Design Spec

**Date:** 2026-03-23
**Status:** Approved
**Competition:** [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
**Goal:** Competitive RMSLE score + production-quality portfolio project

---

## 1. Problem Definition

Predict daily unit sales for ~1782 time series (date x store_nbr x family) at Corporacion Favorita grocery stores in Ecuador.

- **Metric:** RMSLE (Root Mean Squared Logarithmic Error)
- **RMSLE behavior:** Penalizes underestimation more than overestimation. Sensitive to low-magnitude values.
- **Target:** `sales` column (float >= 0)
- **Prediction horizon:** 16 days (test set)

## 2. Data Inventory

| File | Description | Key Columns |
|------|-------------|-------------|
| `train.csv` | Historical sales | id, date, store_nbr, family, sales, onpromotion |
| `test.csv` | Prediction target dates | id, date, store_nbr, family, onpromotion |
| `stores.csv` | Store metadata | store_nbr, city, state, type, cluster |
| `oil.csv` | Daily oil prices | date, dcoilwtico |
| `holidays_events.csv` | Holidays and events | date, type, locale, locale_name, description, transferred |
| `transactions.csv` | Daily store transactions | date, store_nbr, transactions |
| `sample_submission.csv` | Submission format | id, sales |

## 3. Architecture

### Pipeline Overview

```
Data Ingestion -> EDA & Profiling -> Feature Engineering -> Modeling -> Ensemble -> Post-Processing -> Submission
```

### Module Decomposition

1. **Data Layer** (`src/data/`): Load, merge, clean, handle missing values, type casting.
2. **Feature Layer** (`src/features/`): Generate all feature families from raw data.
3. **Model Layer** (`src/models/`): Train individual models with consistent interface.
4. **Evaluation Layer** (`src/evaluation/`): RMSLE metric, TimeSeriesSplit validation.
5. **Tuning Layer** (`src/tuning/`): Optuna-based hyperparameter optimization.
6. **Submission Layer** (`src/submission/`): Post-processing, clipping, CSV generation.

## 4. Feature Engineering Strategy

### 4.1 Temporal Features
- day, day_of_week, month, year, week_of_year, quarter, day_of_year
- is_weekend, is_month_start, is_month_end
- sin/cos cyclical encoding for day_of_week and month

### 4.2 Lag Features
- Sales lags: 1, 7, 14, 28 days
- Rolling statistics (mean, std, min, max): windows of 7, 14, 28, 90 days
- Expanding mean per store x family

### 4.3 External Features
- Oil: price, 7-day lag, 28-day rolling mean, daily difference
- Holidays: is_holiday, holiday_type one-hot, days_to_next_holiday, days_since_last_holiday, is_transferred
- Transactions: lag 7, rolling mean 7/14/28

### 4.4 Categorical / Store Features
- store_type, cluster, city, state (label or target encoded)
- family (target encoded)

### 4.5 Promotion Features
- onpromotion (binary), promo_lag_7, promo_rolling_14
- promo_duration (consecutive promo days)

### 4.6 Cross Features
- family x store_type interaction
- family x cluster interaction
- day_of_week x family interaction

### 4.7 Statistical Aggregations
- Mean/median/std sales per store, family, store x family over 30/90/365 day windows
- Coefficient of variation per series

## 5. Modeling Strategy

### 5.1 Individual Models

| Model | Objective/Loss | Rationale |
|-------|---------------|-----------|
| LightGBM | Tweedie (log link) | Top-1 solution used this; handles zero-inflated sales well |
| XGBoost | reg:squaredlogerror | Direct RMSLE optimization |
| CatBoost | RMSE on log1p(sales) | Native categorical handling; diversity for ensemble |

### 5.2 Ensemble Strategy
- **Level 1:** Weighted average of 3 GBM models (weights optimized via scipy.optimize on hold-out)
- **Level 2 (if time permits):** Ridge regression stacking with OOF predictions as features

### 5.3 Grouping Strategy
Train separate models per product family cluster (high-volume vs low-volume families) to capture different dynamics.

## 6. Validation Strategy

- **TimeSeriesSplit:** 5 folds, expanding window, gap of 16 days between train/val to avoid leakage
- **Hold-out:** Last 16 days of training data (mirrors test set structure)
- **Metric:** RMSLE on validation sets; track per-family and per-store breakdown
- **Leakage prevention:** No future information in lag/rolling features; gap between train/val splits

## 7. Post-Processing

- Clip predictions to >= 0 (RMSLE undefined for negative)
- Round small predictions (<0.01) to 0
- Sanity check: distribution of predictions vs training data distribution

## 8. Technology Stack

```
Python 3.11+
- pandas >= 2.0 (data manipulation)
- numpy >= 1.24 (numerical operations)
- scikit-learn >= 1.3 (preprocessing, metrics, splits)
- lightgbm >= 4.0 (primary model)
- xgboost >= 2.0 (secondary model)
- catboost >= 1.2 (tertiary model)
- optuna >= 3.0 (hyperparameter tuning)
- plotly >= 5.0 (interactive visualizations)
- matplotlib >= 3.7 (static plots)
- seaborn >= 0.12 (statistical plots)
- statsforecast >= 1.0 (statistical baseline)
- pytest >= 7.0 (testing)
- ruff (linting)
```

## 9. Repository Structure

```
store-sales-forecasting/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── preprocessor.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── temporal.py
│   │   ├── lag_features.py
│   │   ├── external.py
│   │   └── builder.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── lgbm_model.py
│   │   ├── xgb_model.py
│   │   ├── catboost_model.py
│   │   └── ensemble.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── validation.py
│   ├── tuning/
│   │   ├── __init__.py
│   │   └── optimizer.py
│   └── submission/
│       ├── __init__.py
│       └── generator.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_ensemble_and_submission.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_loader.py
│   ├── test_preprocessor.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_metrics.py
├── docs/
│   └── superpowers/specs/
├── .github/workflows/
│   └── ci.yml
├── pyproject.toml
├── Makefile
├── README.md
└── .gitignore
```

## 10. Testing Strategy

- **Unit tests** for each feature module: correct shapes, no unexpected NaNs, correct dtypes
- **Unit tests** for metrics: RMSLE matches sklearn implementation
- **Unit tests** for models: fit/predict interface, output shape consistency
- **Integration test**: full pipeline from raw CSV to submission CSV
- **Coverage target:** >80%

## 11. Git Workflow

- `main`: stable, deployable
- `develop`: integration branch
- `feature/*`: one branch per pipeline phase
- Commit messages: conventional commits (feat:, fix:, docs:, test:, refactor:)

## 12. Success Criteria

| Criteria | Target |
|----------|--------|
| RMSLE on public leaderboard | < 0.40 |
| CV-LB correlation | > 0.90 |
| Test coverage | > 80% |
| Code quality | ruff clean, type hints, docstrings |
| Documentation | README with methodology, results, visuals |
| Portfolio readiness | Reproducible end-to-end with Makefile |

## 13. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Data leakage in features | Gap in TimeSeriesSplit; validate feature timestamps |
| Overfitting to CV | Track LB vs CV gap; use multiple validation windows |
| Zero-inflated sales | Tweedie loss; separate zero/non-zero models if needed |
| Missing oil prices | Forward-fill then linear interpolation |
| Holiday complexity | Careful handling of transferred holidays |

## 14. Phases

1. **Phase 1 - Foundation:** Project setup, data loading, EDA notebook
2. **Phase 2 - Features:** Full feature engineering pipeline with tests
3. **Phase 3 - Modeling:** Individual models, validation, tuning
4. **Phase 4 - Ensemble:** Weighted average, stacking, final submission
5. **Phase 5 - Polish:** README, CI/CD, documentation, portfolio packaging
