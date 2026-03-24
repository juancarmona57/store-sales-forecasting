"""EDA page — exploratory data analysis with real data."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils.data_loader import load_holidays, load_oil, load_stores, load_train, load_transactions

train = load_train()
stores = load_stores()
oil = load_oil()
holidays = load_holidays()
transactions = load_transactions()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Filters**")
    year_options = sorted(train.date.dt.year.unique())
    selected_years = st.multiselect("Year", year_options, default=year_options)
    store_types = sorted(stores.type.unique())
    selected_types = st.multiselect("Store type", store_types, default=store_types)

# Filter data
filtered_stores = stores[stores.type.isin(selected_types)]["store_nbr"].tolist()
mask = train.date.dt.year.isin(selected_years) & train.store_nbr.isin(filtered_stores)
df = train[mask].copy()

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total sales", f"{df.sales.sum()/1e6:.1f}M", border=True)
c2.metric("Avg daily sales", f"{df.sales.mean():.1f}", border=True)
c3.metric("Zero-sales rows", f"{(df.sales==0).mean()*100:.1f}%", border=True)
c4.metric("Promoted rows", f"{(df.onpromotion>0).mean()*100:.1f}%", border=True)

st.space("medium")

# ── Monthly sales trend ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    ":material/trending_up: Sales trend",
    ":material/store: Stores",
    ":material/category: Families",
    ":material/water_drop: Oil & macro",
    ":material/celebration: Holidays",
])

with tab1:
    monthly = (
        df.assign(month=df.date.dt.to_period("M").dt.to_timestamp())
        .groupby("month")["sales"]
        .sum()
        .reset_index()
    )
    fig = px.area(
        monthly, x="month", y="sales",
        title="Total monthly sales over time",
        labels={"sales": "Total sales", "month": "Month"},
        color_discrete_sequence=["#4C8BF5"],
    )
    fig.update_layout(showlegend=False, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        dow_sales = df.assign(dow=df.date.dt.day_name()).groupby("dow")["sales"].mean().reset_index()
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow_sales["dow"] = pd.Categorical(dow_sales["dow"], categories=order, ordered=True)
        dow_sales = dow_sales.sort_values("dow")
        fig2 = px.bar(dow_sales, x="dow", y="sales", title="Average sales by day of week",
                      color="sales", color_continuous_scale="Blues")
        fig2.update_coloraxes(showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        month_sales = df.assign(m=df.date.dt.month).groupby("m")["sales"].mean().reset_index()
        month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                       7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        month_sales["month"] = month_sales["m"].map(month_names)
        fig3 = px.bar(month_sales, x="month", y="sales", title="Average sales by month",
                      color="sales", color_continuous_scale="Greens")
        fig3.update_coloraxes(showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

with tab2:
    df_stores = df.merge(stores, on="store_nbr")

    col_l, col_r = st.columns(2)
    with col_l:
        type_sales = df_stores.groupby("type")["sales"].sum().reset_index()
        fig = px.pie(type_sales, values="sales", names="type",
                     title="Sales share by store type",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        cluster_sales = df_stores.groupby("cluster")["sales"].mean().reset_index()
        fig = px.bar(cluster_sales, x="cluster", y="sales",
                     title="Average sales by cluster",
                     color="sales", color_continuous_scale="Viridis")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # City heatmap
    city_type = df_stores.groupby(["city","type"])["sales"].mean().reset_index()
    fig = px.bar(
        city_type.sort_values("sales", ascending=False).head(40),
        x="city", y="sales", color="type",
        title="Average daily sales by city and store type",
        barmode="group",
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    family_sales = df.groupby("family")["sales"].mean().sort_values(ascending=True).reset_index()
    fig = px.bar(
        family_sales, x="sales", y="family", orientation="h",
        title="Average daily sales by product family",
        color="sales", color_continuous_scale="Teal",
    )
    fig.update_coloraxes(showscale=False)
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        promo_impact = df.groupby("family").apply(
            lambda x: pd.Series({
                "promoted": x[x.onpromotion > 0]["sales"].mean(),
                "not_promoted": x[x.onpromotion == 0]["sales"].mean(),
            })
        ).reset_index()
        promo_long = promo_impact.melt(id_vars="family", var_name="status", value_name="avg_sales")
        top_fam = df.groupby("family")["sales"].mean().nlargest(10).index
        promo_long_top = promo_long[promo_long.family.isin(top_fam)]
        fig = px.bar(promo_long_top, x="family", y="avg_sales", color="status",
                     title="Promotion impact on top 10 families",
                     barmode="group", color_discrete_map={
                         "promoted": "#4C8BF5", "not_promoted": "#FF7043"
                     })
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        zero_by_family = df.groupby("family").apply(
            lambda x: (x.sales == 0).mean() * 100
        ).reset_index(name="zero_pct").sort_values("zero_pct", ascending=False)
        fig = px.bar(zero_by_family, x="family", y="zero_pct",
                     title="Zero-sales rate by family (%)",
                     color="zero_pct", color_continuous_scale="Reds")
        fig.update_coloraxes(showscale=False)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    # Oil price timeline
    fig = px.line(oil.dropna(), x="date", y="dcoilwtico",
                  title="Ecuador oil price (WTI) over time",
                  labels={"dcoilwtico": "USD/barrel", "date": "Date"},
                  color_discrete_sequence=["#FF7043"])
    fig.add_vrect(x0="2016-04-01", x1="2016-05-01",
                  fillcolor="red", opacity=0.15,
                  annotation_text="Earthquake", annotation_position="top left")
    fig.add_vrect(x0="2014-07-01", x1="2016-01-01",
                  fillcolor="orange", opacity=0.08,
                  annotation_text="Oil price crash", annotation_position="top left")
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        # Oil vs sales correlation
        train_monthly = (
            train.assign(month=train.date.dt.to_period("M").dt.to_timestamp())
            .groupby("month")["sales"].sum().reset_index()
        )
        oil_monthly = (
            oil.assign(month=oil.date.dt.to_period("M").dt.to_timestamp())
            .groupby("month")["dcoilwtico"].mean().reset_index()
        )
        merged = train_monthly.merge(oil_monthly, on="month").dropna()
        corr = merged["sales"].corr(merged["dcoilwtico"])
        fig = px.scatter(merged, x="dcoilwtico", y="sales",
                         title=f"Oil price vs total sales (corr={corr:.2f})",
                         trendline="ols",
                         labels={"dcoilwtico": "Oil price (USD)", "sales": "Total sales"},
                         color_discrete_sequence=["#4C8BF5"])
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Transactions trend
        txn_monthly = (
            transactions.assign(month=transactions.date.dt.to_period("M").dt.to_timestamp())
            .groupby("month")["transactions"].sum().reset_index()
        )
        fig = px.area(txn_monthly, x="month", y="transactions",
                      title="Monthly transaction count",
                      color_discrete_sequence=["#66BB6A"])
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    national = holidays[holidays.locale == "National"]
    national_holiday_dates = national["date"].values

    df_hol = train.copy()
    df_hol["is_holiday"] = df_hol.date.isin(national_holiday_dates)
    hol_sales = df_hol.groupby("is_holiday")["sales"].mean().reset_index()
    hol_sales["label"] = hol_sales["is_holiday"].map({True: "Holiday", False: "Regular day"})

    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.bar(hol_sales, x="label", y="sales",
                     title="Average sales: holidays vs regular days",
                     color="label",
                     color_discrete_map={"Holiday": "#AB47BC", "Regular day": "#78909C"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        holiday_types = holidays.groupby("type").size().reset_index(name="count")
        fig = px.pie(holiday_types, values="count", names="type",
                     title="Holiday types distribution",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)

    # Holiday events by month
    holidays_by_month = (
        national.assign(month=national.date.dt.month)
        .groupby("month").size().reset_index(name="count")
    )
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    holidays_by_month["month_name"] = holidays_by_month["month"].map(month_names)
    fig = px.bar(holidays_by_month, x="month_name", y="count",
                 title="National holidays per month",
                 color="count", color_continuous_scale="Purples")
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)
