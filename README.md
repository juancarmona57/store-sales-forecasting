# Store Sales - Time Series Forecasting

Predict grocery sales for Corporacion Favorita stores in Ecuador using an ensemble of gradient boosting models.

**Kaggle Competition:** [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

## Results

| Model | CV RMSLE | LB RMSLE |
|-------|----------|----------|
| LightGBM (Tweedie) | TBD | TBD |
| XGBoost (SquaredLogError) | TBD | TBD |
| CatBoost (Log1p RMSE) | TBD | TBD |
| **Ensemble (Weighted Avg)** | **TBD** | **TBD** |

## Approach

### Feature Engineering
- **Temporal:** Calendar features with cyclical encoding (sin/cos)
- **Lag Features:** Sales lags (1, 7, 14, 28 days) with rolling statistics
- **External:** Oil prices, holidays/events, transaction counts
- **Categorical:** Store type, cluster, city, state, product family
- **Interactions:** Family x store_type, family x cluster, dow x family
- **Promotions:** Lag, rolling frequency, consecutive duration
- **Aggregations:** Rolling mean/std per store x family

### Modeling
- **LightGBM** with Tweedie objective (handles zero-inflated sales)
- **XGBoost** with squared log error (direct RMSLE optimization)
- **CatBoost** with RMSE on log1p-transformed target
- **Ensemble:** Weighted average with scipy-optimized weights

### Validation
- TimeSeriesSplit with 16-day gap (prevents feature leakage)
- 5-fold expanding window cross-validation
- Hold-out: last 16 days of training data

## Quick Start

```bash
# Install
make install

# Download data (requires Kaggle API credentials)
make download-data

# Run tests
make test

# Lint
make lint
```

## Project Structure

```
src/
├── config.py           # Paths, constants, hyperparameters
├── data/               # Data loading and preprocessing
├── features/           # Feature engineering modules
│   ├── temporal.py     # Calendar and cyclical features
│   ├── lag_features.py # Lag and rolling window features
│   ├── external.py     # Oil, holiday, transaction features
│   ├── promotion.py    # Promotion duration and lag features
│   ├── cross_features.py # Interaction features
│   ├── aggregations.py # Statistical aggregations
│   └── builder.py      # Feature pipeline orchestrator
├── models/             # Model implementations
│   ├── base.py         # Abstract base class
│   ├── lgbm_model.py   # LightGBM Tweedie
│   ├── xgb_model.py    # XGBoost SquaredLogError
│   ├── catboost_model.py # CatBoost Log1p
│   └── ensemble.py     # Weighted average ensemble
├── evaluation/         # Metrics and validation
├── tuning/             # Optuna hyperparameter optimization
└── submission/         # Submission generation
```

## Tech Stack

Python 3.11+ | pandas | scikit-learn | LightGBM | XGBoost | CatBoost | Optuna | Plotly

## License

MIT
