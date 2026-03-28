# Store Sales Forecasting - Knowledge Base

## Competition
- **Kaggle**: Store Sales - Time Series Forecasting
- **Metric**: RMSLE (Root Mean Squared Logarithmic Error)
- **Goal**: RMSLE < 0.36
- **Test period**: Aug 16-31, 2017 (16 days after training ends Aug 15)
- **Data**: Corporacion Favorita grocery stores in Ecuador

## Score History (Kaggle Leaderboard)

| Version | Score | Description | Status |
|---------|-------|-------------|--------|
| v1 (submission.csv) | 3.20150 | Initial naive attempt | BAD |
| v7 | 0.70038 | First working pipeline with basic features | OK |
| v9 | 1.04156 | Iterative prediction attempt (error accumulation) | WORSE |
| v10 | 1.22707 | log1p+RMSE broke predictions | WORSE |
| v11 | 0.44756 | Tweedie + safe lags + imputed transactions + direct pred | Previous best |
| v15 | 0.43979 | 2-LGBM ensemble (Tweedie 1.5+1.2), val=0.4026 | Improvement |
| v18 | 0.41205 | 3-model TW ensemble (1.1+1.2+1.5) + earthquake excl + tx forecast, val=0.4024 | Superseded |
| v19 | 0.41246 | Segment-optimized (HIGH/MED/LOW), stat means for LOW, val=0.3997 | Close to v18 |
| v20 | 0.40305 | 5-TW × 3-seed (15 models), optimized weights (31.5% TW1.0 + 51.3% TW1.5), val=0.4023 | Previous best |
| v20_equal | 0.40513 | Same as v20 but equal weights | Worse than opt |
| **v21** | **0.40106** | 5-TW × 3-seed + 12 new P1 features + structural zero filter, weights (43.6% TW1.0 + 56.4% TW1.5), val=0.4072 | **BEST KAGGLE** |
| v21_equal | 0.40506 | Same as v21 but equal weights | Worse than opt |
| v12-v14 | not submitted | Various intermediate versions | Skipped |
| v16 | 0.41662 | Two-Stage Hurdle, val=0.4017 | Not better |
| v17 | abandoned | Per-day DART — too slow, 0.43+ | Abandoned |
| v17b | 0.41444 | Family-grouped (HIGH/MED/LOW), val=0.4032 | Not better |

## What Works (PROVEN)

1. **LightGBM Tweedie (variance_power=1.5)**: Predicts on original scale, handles zero-inflated data
2. **XGBoost reg:squaredlogerror**: Optimizes RMSLE directly on original scale
3. **Direct prediction (NOT iterative)**: Predict all 16 days at once using safe features
4. **Safe lags >= 16**: All lag features use shift >= 16 to guarantee availability for all test days
5. **Transaction imputation**: Per-store mean from last 28 training days (test has no transaction data)
6. **Target statistics (static)**: store x family mean/median/std, DOW means — always available at inference
7. **Combined train+test for feature computation**: Ensures lags from training data flow into test rows
8. **Training on last 2 years**: More recent data = more relevant patterns
9. **Multi-seed averaging (3 seeds per model)**: Reduces variance, more robust predictions
10. **Structural zero filtering**: Remove rows before first sale per store×family — removes 464k noise rows, improves signal
11. **Same-DOW lag features**: lag_dow_3w(21), lag_dow_4w(28), lag_dow_8w(56), lag_dow_52w(364) + DOW rolling mean — captures weekly seasonality better than raw lags
12. **Trend ratio features**: dow_trend_3w_4w, dow_yoy_ratio, recent_vs_overall_ratio — detect momentum/trend changes
13. **Holiday type ordinal + days_to_christmas/new_year**: Strong seasonal signals for grocery retail
14. **Earthquake period exclusion**: Remove Apr 16 - May 15, 2016 from training (53k rows of anomalous data)
15. **Optimized Tweedie diversity**: TW1.0 (Poisson-like) + TW1.5 (Gamma-like) complement each other — v21 uses 43.6%/56.4% split

## What DOES NOT Work (PROVEN FAILURES)

1. **log1p + RMSE**: Training on log1p(sales) with RMSE produces wrong-scale predictions at inference. Score went from 0.70 to 1.23. NEVER use this approach.
2. **Iterative prediction**: Predict day-by-day and plug predictions as next day's lag. Error accumulates exponentially. Mean prediction decayed from 56 (day 1) to 6 (day 16). Score: 1.04.
3. **Unsafe lags (< 16)**: Lags [1, 7, 14] are NaN for most test days, get filled with 0, predictions collapse.
4. **transactions=0 for test**: Model learned transactions as important feature, predicted near-zero sales.
5. **fillna(0) on category columns**: Causes TypeError. Must filter to numeric columns only.
6. **TimeSeriesSplitWithGap without reset_index**: After dropna, index is sparse, causes IndexError.
7. **Inconsistent category encodings**: astype("category") creates different codes on subsets. Must save mappings from training.

## V16 Results: Two-Stage Hurdle Model

- Stage 1 classifier: AUC=0.9874, F1=0.9748 (threshold=0.30)
- Stage 2 regression (non-zero only): RMSLE=0.378093
- **Hurdle Hard**: 0.4229 — hard threshold hurts, too aggressive on zeros
- **Hurdle Soft**: 0.4017 — P(nonzero) × amount, marginal improvement
- **Pure Regression**: 0.4033 — single Tweedie 1.2 on all data
- **Blend (50/50)**: 0.4077 — hard+baseline blend
- **Conclusion**: Hurdle model provides minimal benefit (~0.002 improvement). Tweedie already handles zeros well. The classifier is excellent but the multiplication dampens non-zero predictions.

## Winning Solution Research (Deep Dive)

### What Top Solutions (< 0.36 RMSLE) Actually Do:
1. **Per-day models**: 16 separate LGBMs, one per forecast day. Each model specializes in a specific horizon.
2. **DART boosting**: Dropout regularization outperforms gbdt for this dataset.
3. **Simpler trees**: num_leaves=16-32, max_depth=4-5 (NOT 256-512). Top solutions use much less complex trees.
4. **Tweedie variance_power=1.1**: Closer to Poisson, better for count-like data.
5. **Per-store models**: 54 separate models for 54 stores (M5 winner pattern).
6. **Earthquake exclusion**: Remove April 16 - May 15, 2016 earthquake recovery period from training.
7. **CNN+DNN / WaveNet**: Deep learning models capture long-range dependencies. 5th place used dilated causal convolution.
8. **3+ model ensemble**: Per-day + global + deep learning, each with different random seeds.
9. **Bayesian group means**: Instead of simple means, use Bayesian priors from broader groups for low-volume combos.

### Score Benchmarks from Research:
- Simple LGBM: ~0.5-0.6
- Well-tuned LGBM + good features: ~0.38-0.42 ← WE ARE HERE
- Top solutions (ensemble + advanced FE): < 0.36
- Leaderboard leaders: ~0.30

## What Needs Testing

1. ~~**Zero sales handling**: Separate classifier for zero/non-zero + regression for non-zero~~ → TESTED v16, minimal benefit
2. ~~**Temporal embeddings**: Fourier features for seasonality~~ → ADDED v14, modest improvement
3. **Per-day models**: 16 separate LGBMs (v17 implementing)
4. **DART boosting**: Dropout regularization (v17 implementing)
5. **Per-store models**: Separate LGBM per store_nbr
6. **Stacking**: Train meta-model on out-of-fold predictions from diverse base models
7. **Hyperparameter tuning**: Optuna/Bayesian optimization
8. **Feature selection**: Remove low-importance features to reduce noise
9. **Deep learning**: WaveNet / dilated CNN for time series (requires PyTorch)

## Current Architecture

```
src/
  config.py          - Hyperparameters, paths, constants
  pipeline.py        - Main orchestrator (load, train, predict, submit)
  data/
    loader.py        - Load raw CSV files
    preprocessor.py  - Clean and merge datasets
  features/
    builder.py       - Orchestrates all feature modules
    temporal.py      - Calendar features (dow, month, cyclical)
    lag_features.py  - Lag and rolling features (safe shift >=16)
    external.py      - Oil, holiday, transaction features
    promotion.py     - Promotion-related features
    cross_features.py - Categorical interactions
    aggregations.py  - Rolling aggregation per store x family
    target_stats.py  - Static lookup stats from training
  models/
    base.py          - Abstract base class
    lgbm_model.py    - LightGBM Tweedie
    xgb_model.py     - XGBoost squaredlogerror
    catboost_model.py - CatBoost RMSE
    ensemble.py      - Weighted averaging + weight optimization
  evaluation/
    metrics.py       - RMSLE metric
    validation.py    - TimeSeriesSplitWithGap
  submission/
    generator.py     - Generate submission CSV
```

## Feature List (v13, 88 cols)

### Temporal
- day, day_of_week, month, year, week_of_year, quarter, day_of_year
- is_weekend, is_month_start, is_month_end
- dow_sin, dow_cos, month_sin, month_cos

### Lag Features (safe, >=16)
- sales_lag_{16, 17, 21, 28, 35, 42, 56, 91, 182, 364}
- sales_rolling_mean_{7, 14, 28}, sales_rolling_std_{7, 14, 28}

### External
- dcoilwtico, oil_lag_7, oil_lag_14, oil_rolling_mean_28, oil_rolling_std_28, oil_diff
- is_holiday, is_national_holiday, days_to_next_holiday, days_since_last_holiday
- transactions (imputed), transactions_lag_{16, 21, 28}, transactions_rolling_mean_{7, 14, 28}

### Promotion
- onpromotion, promo_lag_7, promo_lag_14, promo_rolling_14, promo_duration

### Cross Features
- family_x_store_type, family_x_cluster, dow_x_family

### Aggregations
- sales_store_family_mean_{30, 90}, sales_store_family_std_{30, 90}, sales_expanding_mean

### Target Statistics (static)
- sf_mean, sf_median, sf_std, sf_q25, sf_q75, sf_zero_ratio
- sf_dow_mean, family_mean, family_std, family_dow_mean
- store_mean, store_std, store_dow_mean

### Interaction & Derived
- is_payday, days_to_payday, is_earthquake_period
- sales_velocity_16_28, sales_velocity_16_42, sales_yoy_change
- promo_x_sf_mean, promo_x_sf_dow_mean, lag16_to_mean_ratio

### Categorical
- family, store_type, cluster, city, state

## Hyperparameters (Current)

### LightGBM
- objective: tweedie (variance_power=1.5)
- learning_rate: 0.03, num_leaves: 255, min_child_samples: 50
- feature_fraction: 0.8, bagging_fraction: 0.8
- n_estimators: 3000, early_stopping: 100 rounds

### XGBoost
- objective: reg:squaredlogerror
- learning_rate: 0.03, max_depth: 8
- subsample: 0.8, colsample_bytree: 0.8
- n_estimators: 3000, early_stopping: 100 rounds

### CatBoost
- loss_function: RMSE
- learning_rate: 0.03, depth: 8
- iterations: 3000, early_stopping: 100 rounds

## Winning Solution Patterns (Research — Updated)

1. **Tweedie objective (variance_power=1.1)**: Closer to Poisson, top solutions prefer 1.1 over 1.5
2. **Per-day direct models**: 16 separate models, one per forecast day (not one global model)
3. **DART boosting**: Dropout regularization prevents overfitting, outperforms gbdt
4. **Simpler trees**: num_leaves=16-32, max_depth=4-5 (less overfitting)
5. **Transaction forecasting**: Separate LGBM to forecast transactions for test period
6. **Earthquake exclusion**: Remove April-May 2016 earthquake recovery period from training
7. **Per-store models**: 54 separate models capture store-specific patterns (M5 winner)
8. **Ensemble diversity**: Per-day + global + deep learning, multi-seed averaging
9. **Bayesian group means**: Hierarchical priors for low-volume store×family combos
10. **Feature engineering depth**: 100+ features from domain knowledge

## Business Logic: Family Segmentation Analysis

### Key Insight
33 product families have vastly different characteristics. A single global model
wastes capacity trying to predict noise in low-volume families (BOOKS has 97% zeros!).

### Segments:
| Segment | Families | Rows% | Mean Sales | Zero% | Strategy |
|---------|----------|-------|------------|-------|----------|
| HIGH    | 13       | 39%   | 883.6      | 12%   | Full-power LGBM, Tweedie 1.1, num_leaves=256 |
| MEDIUM  | 6        | 18%   | 43.8       | 21%   | Moderate LGBM, Tweedie 1.3, num_leaves=128 |
| LOW     | 14       | 42%   | 4.0        | 53%   | Simple model, Tweedie 1.5, num_leaves=32, high regularization |

### HIGH volume families (mean > 100):
GROCERY I, BEVERAGES, PRODUCE, CLEANING, DAIRY, BREAD/BAKERY, POULTRY, MEATS,
PERSONAL CARE, DELI, HOME CARE, EGGS, FROZEN FOODS

### MEDIUM volume families (10 < mean <= 100):
PREPARED FOODS, LIQUOR/WINE/BEER, SEAFOOD, GROCERY II, HOME AND KITCHEN I/II

### LOW volume families (mean <= 10):
CELEBRATION, LINGERIE, LADIESWEAR, PLAYERS AND ELECTRONICS, AUTOMOTIVE,
LAWN AND GARDEN, PET SUPPLIES, BEAUTY, SCHOOL AND OFFICE SUPPLIES, MAGAZINES,
HARDWARE, HOME APPLIANCES, BABY CARE, BOOKS

### Day-of-Week Patterns:
- Top families peak on **Sunday**, valley on **Thursday** (1.6-1.8x ratio)
- PRODUCE peaks on **Wednesday** (different pattern!)
- Low-volume families: noisy, no clear pattern → model should rely on means

### Implications:
- RMSLE penalizes proportionally: prediction error of 1 on true=0.1 is huge in log space
- LOW volume families contribute disproportionate RMSLE when model overpredicts
- Better to use simple means/medians for BOOKS, BABY CARE than complex models
- HIGH volume families benefit from aggressive feature engineering and capacity

## V17 Results: Per-Day DART Models (ABANDONED)

- DART boosting extremely slow: ~4 min per model × 16 = 64+ min total
- Day 1 RMSLE: 0.4375 (WORSE than global 0.40)
- Per-day models with DART and simpler trees did NOT help
- **Conclusion**: Per-day models train on ALL data but only validate on 1 day. Not enough validation signal for early stopping. DART overhead not justified.

## V17b Results: Family-Grouped Models

- HIGH volume (13 families): RMSLE = **0.1924** ← incredible!
- MEDIUM volume (6 families): RMSLE = 0.4563
- LOW volume (14 families): RMSLE = 0.5178
- Overall segmented: 0.4076
- Best blend (5% seg / 95% global): 0.4032
- **Conclusion**: Segmented HIGH model is spectacular, but global model is better for MEDIUM+LOW because it leverages cross-family patterns. The segmented approach only helps if we find a way to improve MEDIUM+LOW models.

## V18 Results: Hybrid Segmented + Multi-Tweedie Ensemble

- HIGH dedicated (TW 1.1): 0.1924
- Global TW 1.1: 0.4038
- Global TW 1.5: 0.4060
- Global TW 1.2: **0.4029** ← best single model
- **Best 3-model blend**: **0.4024** (30% TW1.1 + 30% TW1.5 + 40% TW1.2)
- Hybrid segmentation did NOT help on top of ensemble
- **Conclusion**: Multi-Tweedie ensemble with earthquake exclusion + tx forecasting is the current best approach. Marginal improvement over v15 (0.4026→0.4024).

## V19 Results: Segment-Optimized Models

- HIGH (TW 1.1 dedicated): 0.1924 — unchanged
- MEDIUM (TW 1.3 dedicated): 0.4554 — slight improvement from tuned params
- LOW stat means: 0.5147 — BETTER than GBM (0.5178)!
- LOW blend (55% stat + 45% GBM): 0.5063 — best LOW approach
- Combined Segmented: **0.4013** — new best pure-segmented
- **Best per-seg blend** (seg 65% + global 35% for MED+LOW): **0.3997** — best val ever!
- Kaggle: 0.41246 — slightly worse than v18 despite better val
- **Conclusion**: Segmentation helps val but doesn't generalize as well as pure ensemble. Statistical means for LOW families confirmed better than GBM for noise-dominated data.

## Data Characteristics (Business Logic)

### Sales Data:
- **Not electronic-only**: Corporación Favorita is a physical grocery chain in Ecuador
- All transaction types (cash, card, etc.) included
- Granularity: daily totals per store×family (no hourly data)
- 54 stores, 33 product families

### Structural Zeros:
- Many zeros are **structural** — certain stores don't carry certain families
- BOOKS (97% zeros): most stores simply don't sell books
- BABY CARE (94% zeros): niche category, few stores carry it
- For structural zeros: predict 0 directly (no model needed)
- For intermittent zeros: use store×family×dow means

## Score Progression Analysis

| Version | Val RMSLE | Kaggle | Val-to-LB Gap | Key Change |
|---------|-----------|--------|---------------|------------|
| v11     | ~0.41     | 0.4476 | ~0.04         | Basic pipeline |
| v15     | 0.4026    | 0.4398 | 0.037         | 2-LGBM ensemble |
| v18     | 0.4024    | **0.4121** | **0.010**  | +Tx forecast +earthquake excl |
| v19     | **0.3997**| 0.4125 | 0.013         | +Segmentation |

**Key insight**: The biggest Kaggle improvement (v11→v18) came from:
1. **Transaction forecasting** (replacing static imputation)
2. **Earthquake period exclusion** (removing noisy data)
3. **Multi-Tweedie ensemble** (TW 1.1 + 1.2 + 1.5)

The val-to-LB gap shrank dramatically (0.04→0.01), suggesting the model generalizes much better now.

## Next Steps to Break 0.36

Current: Kaggle 0.412. Need to drop 0.05 more.

### High-Impact (likely to help):
1. **Train on full data** (no val holdout) for final submission — gains ~0.005-0.01
2. **More Tweedie variants** in ensemble (1.0, 1.3, 1.4) — more diversity
3. **Multi-seed averaging** — train same model 3-5x with different seeds
4. **Feature selection** via Optuna — remove noisy features
5. **Per-store models** for top stores — capture store-specific patterns

### Medium-Impact (experimental):
6. **Deep learning** (WaveNet/dilated CNN) — different architecture adds diversity
7. **Stacking** — meta-model on out-of-fold predictions
8. **Target encoding** with Bayesian priors for low-volume combos

### Low-Impact (already tested/diminishing returns):
9. Hurdle model — minimal benefit (v16)
10. Per-day models — worse results (v17)
11. Family segmentation — helps val but not LB (v19)

## Feature Importance (v13 LGBM Tweedie)

Top 15 features by importance (split count):
1. family_x_cluster: 149228 ← Cross feature dominates!
2. dow_x_family: 141636
3. transactions: 64357
4. day_of_year: 55260
5. oil_rolling_std_28: 48236
6. city: 46598
7. family_x_store_type: 46479
8. dcoilwtico: 41103
9. oil_rolling_mean_28: 37373
10. day: 35606
11. oil_lag_14: 33893
12. transactions_lag_28: 33813
13. oil_diff: 33738
14. oil_lag_7: 33347
15. sales_store_family_std_90: 32059

**Key insight**: Sales lag features are NOT in top 15. Cross features (family x cluster, dow x family) and external features (transactions, oil) dominate. This means:
- The model is learning store-family-day patterns from categorical crosses, not from time series lags
- Oil prices have outsized importance (5 oil features in top 15)
- Transaction features matter (but are imputed for test)
- Need more informative lag features or reduce noise from unimportant features

## Performance Observations

- **CatBoost too slow**: 1.2M rows x 88 features takes 30+ minutes. Not worth the diversity benefit.
- **XGBoost worse than LGBM**: 0.48 vs 0.41. Drop from ensemble.
- **LGBM Tweedie is the clear winner**: Most consistent, fastest, lowest error
- **Validation vs Kaggle gap**: Val RMSLE ~0.40, Kaggle ~0.45. ~0.05 gap due to distribution shift.
- **Learning rate 0.01 helps**: More iterations (3613 best vs 835 with LR=0.03) = better convergence

## Pipeline Files

| File | Description | Best Score |
|------|-------------|------------|
| pipeline.py | Single LGBM Tweedie baseline | v14 output |
| pipeline_v15.py | 2-LGBM ensemble (TW 1.5+1.2) | val 0.4026 |
| pipeline_v16.py | Two-Stage Hurdle + Tx forecast | val 0.4017 |
| pipeline_v17.py | Per-day DART (abandoned) | 0.43+ |
| pipeline_v17b.py | Family-grouped 3 models | val 0.4013 |
| **pipeline_v18.py** | **3-model TW ensemble + earthquake + tx** | **Kaggle 0.4121** |
| pipeline_v19.py | Segment-optimized + stat means | val 0.3997 |
