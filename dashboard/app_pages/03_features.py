"""Feature engineering page."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils.data_loader import get_feature_importance

fi = get_feature_importance()

# ── Feature categories summary ────────────────────────────────────────────────
CATEGORIES = {
    "Temporal": {
        "icon": ":material/calendar_today:",
        "count": 14,
        "color": "#4C8BF5",
        "features": "day, day_of_week, month, year, week_of_year, quarter, day_of_year, is_weekend, is_month_start, is_month_end, dow_sin, dow_cos, month_sin, month_cos",
        "description": "Calendar attributes with cyclical sin/cos encoding to capture periodicity without ordinal bias.",
    },
    "Lag features": {
        "icon": ":material/history:",
        "count": 4,
        "color": "#FF7043",
        "features": "sales_lag_1, sales_lag_7, sales_lag_14, sales_lag_28",
        "description": "Past sales at 1, 7, 14, and 28-day lags. Grouped by store×family with leakage-free shift.",
    },
    "Rolling statistics": {
        "icon": ":material/show_chart:",
        "count": 10,
        "color": "#66BB6A",
        "features": "rolling_mean/std at 7, 14, 28, 90 days + expanding mean",
        "description": "Rolling window statistics capturing short and long-term trends. Applied after shift(1) to prevent leakage.",
    },
    "External": {
        "icon": ":material/water_drop:",
        "count": 12,
        "color": "#AB47BC",
        "features": "oil_lag_7, oil_lag_14, oil_rolling_mean/std_28, oil_diff, is_holiday, is_national_holiday, days_to/since_holiday, transactions_lag_7, transactions_rolling_mean_7/14",
        "description": "Oil price indicators (Ecuador macro), holiday flags with proximity, and transaction volume.",
    },
    "Promotion": {
        "icon": ":material/sell:",
        "count": 5,
        "color": "#FFA726",
        "features": "onpromotion, promo_lag_7, promo_rolling_mean_7/14, promo_rolling_sum_7",
        "description": "Current and historical promotion activity per store×family.",
    },
    "Cross features": {
        "icon": ":material/join_inner:",
        "count": 3,
        "color": "#26C6DA",
        "features": "family_x_store_type, family_x_cluster, dow_x_family",
        "description": "Interaction terms: product family combined with store type, cluster, and day-of-week.",
    },
    "Aggregation": {
        "icon": ":material/functions:",
        "count": 13,
        "color": "#EF5350",
        "features": "store_mean_sales, family_mean_sales, store_family_mean_sales, city/state/type/cluster aggs",
        "description": "Historical averages grouped at store, family, store×family, city, state, type, and cluster levels.",
    },
}

# KPI row
total_features = sum(v["count"] for v in CATEGORIES.values())
c1, c2, c3 = st.columns(3)
c1.metric(":material/tune: Total features", str(total_features), border=True)
c2.metric(":material/category: Feature categories", str(len(CATEGORIES)), border=True)
c3.metric(":material/trending_up: Top feature", "sales_rolling_mean_7", "highest gain", border=True)

st.space("medium")

# Category cards
st.subheader("Feature categories")
cols = st.columns(2)
for i, (cat, meta) in enumerate(CATEGORIES.items()):
    with cols[i % 2]:
        with st.container(border=True):
            st.markdown(f"{meta['icon']} **{cat}** — {meta['count']} features")
            st.caption(meta["description"])
            st.code(meta["features"], language=None)

st.space("medium")

# ── Feature importance chart ──────────────────────────────────────────────────
st.subheader(":material/bar_chart: LightGBM feature importance (gain)")

if fi.empty:
    st.warning("LightGBM model not found. Run the training pipeline first.", icon=":material/warning:")
else:
    top_n = st.slider("Show top N features", 10, min(60, len(fi)), 25)
    top_fi = fi.head(top_n)

    fig = px.bar(
        top_fi[::-1],
        x="importance_pct",
        y="feature",
        orientation="h",
        title=f"Top {top_n} features by gain (% of total)",
        labels={"importance_pct": "Importance (%)", "feature": "Feature"},
        color="importance_pct",
        color_continuous_scale="Blues",
    )
    fig.update_coloraxes(showscale=False)
    fig.update_layout(height=max(400, top_n * 22))
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        # Category breakdown of importance
        def classify(name):
            if any(x in name for x in ["rolling", "expanding"]):
                return "Rolling stats"
            if "lag" in name and "promo" not in name and "oil" not in name and "transact" not in name:
                return "Lag features"
            if any(x in name for x in ["day", "month", "year", "week", "quarter", "dow", "weekend", "sin", "cos"]):
                return "Temporal"
            if any(x in name for x in ["oil", "holiday", "transact"]):
                return "External"
            if "promo" in name:
                return "Promotion"
            if "x_" in name:
                return "Cross"
            return "Aggregation"

        fi["category"] = fi["feature"].apply(classify)
        cat_imp = fi.groupby("category")["importance_pct"].sum().reset_index().sort_values("importance_pct", ascending=False)
        fig2 = px.pie(cat_imp, values="importance_pct", names="category",
                      title="Importance share by category",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        st.markdown("**Full feature importance table**")
        st.dataframe(
            fi[["feature", "importance_pct", "category"]].head(40).rename(columns={
                "feature": "Feature",
                "importance_pct": "Importance (%)",
                "category": "Category",
            }),
            hide_index=True,
            use_container_width=True,
        )
