"""Overview page — problem statement, solution, KPIs."""
import streamlit as st
from utils.data_loader import get_model_scores, load_stores, load_train

# ── KPI row ──────────────────────────────────────────────────────────────────
scores = get_model_scores()
train = load_train()
stores = load_stores()

best_rmsle = min(v["rmsle"] for v in scores.values())

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    ":material/trophy: Best RMSLE",
    f"{best_rmsle:.4f}",
    "Ensemble",
    border=True,
)
col2.metric(
    ":material/dataset: Training rows",
    f"{len(train):,}",
    "2013–2017",
    border=True,
)
col3.metric(
    ":material/store: Stores",
    str(train.store_nbr.nunique()),
    f"{stores.type.nunique()} types",
    border=True,
)
col4.metric(
    ":material/category: Product families",
    str(train.family.nunique()),
    "33 categories",
    border=True,
)

st.space("medium")

# ── Problem statement ─────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader(":material/quiz: Problem statement")
    st.markdown("""
**Corporación Favorita** is a large Ecuadorian grocery retailer operating **54 stores**
across the country. The business challenge: forecast daily sales for **33 product families**
at each store for a **16-day horizon** (August 16–31, 2017).

Accurate forecasts enable better:
- Inventory management and waste reduction
- Promotional campaign planning
- Supplier order optimization

**Competition metric:** Root Mean Squared Logarithmic Error (**RMSLE**)

$$\\text{RMSLE} = \\sqrt{\\frac{1}{n}\\sum_{i=1}^{n}(\\log(1+\\hat{y}_i) - \\log(1+y_i))^2}$$

Lower is better. The log transformation makes the metric scale-invariant and penalizes
under-prediction and over-prediction equally.
""")

st.space("medium")

# ── Solution architecture ─────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader(":material/architecture: Solution architecture")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Data sources**")
        st.markdown("""
| File | Rows | Description |
|------|------|-------------|
| `train.csv` | 3,000,888 | Daily sales per store × family |
| `stores.csv` | 54 | Store metadata (type, cluster, city) |
| `oil.csv` | 1,218 | Daily Ecuador oil price |
| `holidays_events.csv` | 350 | National/local holidays & events |
| `transactions.csv` | 83,488 | Daily store transaction counts |
""")

    with col_b:
        st.markdown("**Modeling strategy**")
        st.markdown("""
```
Raw data (5 CSVs)
        │
        ▼
Feature Engineering (61 features)
  ├── Temporal (cyclical encoding)
  ├── Lag features (1, 7, 14, 28 days)
  ├── Rolling stats (7, 14, 28, 90 days)
  ├── External (oil, holidays, transactions)
  ├── Promotion features
  ├── Cross features (family × store type)
  └── Aggregation features
        │
        ▼
 ┌──────┬──────────┬──────────┐
 │ LGBM │ XGBoost  │ CatBoost │
 │Tweedi│sqloglerr │   RMSE   │
 └──────┴──────────┴──────────┘
        │ Scipy weight optimization
        ▼
    Ensemble → Submission
```
""")

st.space("medium")

# ── Tech stack ────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader(":material/build: Technology stack")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Data**")
        st.badge("pandas", color="blue")
        st.badge("numpy", color="blue")
    with c2:
        st.markdown("**Models**")
        st.badge("LightGBM", color="green")
        st.badge("XGBoost", color="orange")
        st.badge("CatBoost", color="violet")
    with c3:
        st.markdown("**Tuning**")
        st.badge("Optuna", color="red")
        st.badge("TimeSeriesSplit", color="orange")
    with c4:
        st.markdown("**Quality**")
        st.badge("pytest · 66 tests", color="green")
        st.badge("ruff · CI/CD", color="gray")
