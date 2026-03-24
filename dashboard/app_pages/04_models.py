"""Models page — training results, parameters, and comparison."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils.data_loader import get_model_scores, load_weights

scores = get_model_scores()
weights = load_weights()

# ── Model comparison KPIs ─────────────────────────────────────────────────────
with st.container(horizontal=True):
    for name, meta in scores.items():
        delta = f"weight {meta['weight']*100:.1f}%" if name != "Ensemble" else "final result"
        st.metric(
            f"{name}",
            f"{meta['rmsle']:.4f}",
            delta,
            delta_color="off" if name != "Ensemble" else "normal",
            border=True,
        )

st.space("medium")

# ── Bar chart comparison ──────────────────────────────────────────────────────
col_l, col_r = st.columns([2, 1])

with col_l:
    with st.container(border=True):
        st.markdown("**RMSLE comparison (lower is better)**")
        df_scores = [{"Model": k, "RMSLE": v["rmsle"], "Color": v["color"]}
                     for k, v in scores.items()]
        import pandas as pd
        df_scores = pd.DataFrame(df_scores)
        fig = px.bar(
            df_scores, x="Model", y="RMSLE",
            color="Model",
            color_discrete_map={k: v["color"] for k, v in scores.items()},
            text="RMSLE",
        )
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        # Add Kaggle top-1 reference line
        fig.add_hline(y=0.37538, line_dash="dash", line_color="gold",
                      annotation_text="Kaggle #1 (0.37538)", annotation_position="top right")
        fig.update_layout(showlegend=False, yaxis_range=[0.3, 0.75])
        st.plotly_chart(fig, use_container_width=True)

with col_r:
    with st.container(border=True):
        st.markdown("**Ensemble weights (optimized)**")
        weight_data = pd.DataFrame([
            {"Model": "LightGBM", "Weight": weights.get("lgbm", 0.368)},
            {"Model": "XGBoost", "Weight": weights.get("xgb", 0.003)},
            {"Model": "CatBoost", "Weight": weights.get("catboost", 0.629)},
        ])
        fig2 = px.pie(
            weight_data, values="Weight", names="Model",
            color="Model",
            color_discrete_map={
                "LightGBM": "#4C8BF5",
                "XGBoost": "#FF7043",
                "CatBoost": "#66BB6A",
            },
        )
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig2, use_container_width=True)

st.space("medium")

# ── Model config cards ────────────────────────────────────────────────────────
st.subheader(":material/settings: Model configurations")

tab1, tab2, tab3 = st.tabs(["LightGBM", "XGBoost", "CatBoost"])

with tab1:
    col_l, col_r = st.columns(2)
    with col_l:
        with st.container(border=True):
            st.markdown("**Key parameters**")
            st.markdown("""
| Parameter | Value |
|-----------|-------|
| Objective | `tweedie` |
| Tweedie variance power | 1.1 |
| Learning rate | 0.05 |
| Num leaves | 127 |
| Min child samples | 20 |
| Feature fraction | 0.8 |
| Bagging fraction | 0.8 |
| Bagging freq | 1 |
| Reg alpha | 0.1 |
| Reg lambda | 1.0 |
""")
    with col_r:
        with st.container(border=True):
            st.markdown("**Why Tweedie?**")
            st.markdown("""
The **Tweedie distribution** is ideal for sales data because:

- :material/check_circle: Handles **zero-inflated** distributions (31% zero sales)
- :material/check_circle: Models data with **non-negative** support naturally
- :material/check_circle: The log link function aligns with **RMSLE** metric
- :material/check_circle: Works well when variance grows with the mean (compound Poisson–Gamma)

The `variance_power=1.1` setting is close to Poisson (1.0), appropriate for
count-like sales data.
""")

with tab2:
    col_l, col_r = st.columns(2)
    with col_l:
        with st.container(border=True):
            st.markdown("**Key parameters**")
            st.markdown("""
| Parameter | Value |
|-----------|-------|
| Objective | `reg:squaredlogerror` |
| Learning rate | 0.05 |
| Max depth | 6 |
| Num estimators | 1000 |
| Subsample | 0.8 |
| Colsample by tree | 0.8 |
| Min child weight | 20 |
| Tree method | `hist` |
| Reg alpha | 0.1 |
| Reg lambda | 1.0 |
""")
    with col_r:
        with st.container(border=True):
            st.markdown("**Notes**")
            st.markdown("""
`reg:squaredlogerror` directly minimizes log-scale squared error — mathematically
equivalent to RMSLE.

XGBoost received **very low ensemble weight (0.3%)** because:
- Pre-trained model was not aligned with this dataset's preprocessing
- CatBoost and LightGBM captured the variance more effectively
- scipy weight optimizer assigned near-zero weight during optimization
""")

with tab3:
    col_l, col_r = st.columns(2)
    with col_l:
        with st.container(border=True):
            st.markdown("**Key parameters**")
            st.markdown("""
| Parameter | Value |
|-----------|-------|
| Loss function | `RMSE` |
| Target on log1p | ✓ (manual) |
| Learning rate | 0.05 |
| Iterations | 1000 |
| Depth | 6 |
| L2 leaf reg | 3.0 |
| Bagging temperature | 0.2 |
| Random strength | 0.5 |
| Border count | 128 |
""")
    with col_r:
        with st.container(border=True):
            st.markdown("**Why CatBoost dominates (62.8% weight)?**")
            st.markdown("""
CatBoost excels at this task because:

- :material/check_circle: **Native categorical handling** — no label encoding needed for family, store type, city
- :material/check_circle: **Ordered boosting** — reduces target leakage automatically
- :material/check_circle: **Symmetric trees** — faster inference on tabular data
- :material/check_circle: Lower overfitting with default regularization

The manual `log1p` target transform with `RMSE` objective is mathematically equivalent
to minimizing RMSLE when predictions are clipped to non-negative values.
""")

st.space("medium")

# ── Validation strategy ───────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader(":material/analytics: Validation strategy — TimeSeriesSplitWithGap")
    st.markdown("""
A custom expanding-window cross-validator with a **16-day gap** was used to simulate
the exact test horizon structure:

```
Train fold 1: ──────────────────────────┐ gap │ Val 1
Train fold 2: ─────────────────────────────── │ gap │ Val 2
Train fold 3: ──────────────────────────────────────── │ gap │ Val 3
              2013                            2017     └─16d─┘
```

- **Gap = 16 days**: matches the test set horizon (Aug 16–31, 2017)
- **Expanding window**: each fold adds more historical data
- **Leakage prevention**: rolling features computed on shifted series

Train split: **2,945,646 rows** | Validation split: **28,512 rows**
""")
