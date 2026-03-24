"""Store Sales Forecasting — Project Dashboard.

Multi-page Streamlit app covering EDA, feature engineering,
modeling, Kaggle comparison, and project history.

Run with:
    streamlit run dashboard/streamlit_app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Store Sales Forecasting",
    page_icon=":material/storefront:",
    layout="wide",
)

page = st.navigation(
    {
        "": [
            st.Page("app_pages/01_overview.py", title="Overview", icon=":material/home:"),
        ],
        "Analysis": [
            st.Page("app_pages/02_eda.py", title="Exploratory data analysis", icon=":material/bar_chart:"),
            st.Page("app_pages/03_features.py", title="Feature engineering", icon=":material/tune:"),
        ],
        "Results": [
            st.Page("app_pages/04_models.py", title="Models & validation", icon=":material/model_training:"),
            st.Page("app_pages/05_kaggle.py", title="Kaggle leaderboard", icon=":material/emoji_events:"),
        ],
        "Project": [
            st.Page("app_pages/06_timeline.py", title="Development timeline", icon=":material/timeline:"),
        ],
    },
    position="sidebar",
)

with st.sidebar:
    st.markdown("## :material/storefront: Store Sales Forecasting")
    st.caption("Corporación Favorita · Ecuador · 2013–2017")
    st.markdown(":material/code: [GitHub](https://github.com/pablocarmona0311/store-sales-forecasting)")
    st.markdown(":material/leaderboard: [Kaggle competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)")
    st.divider()

# Page title
ICONS = {
    "Overview": ":material/home:",
    "Exploratory data analysis": ":material/bar_chart:",
    "Feature engineering": ":material/tune:",
    "Models & validation": ":material/model_training:",
    "Kaggle leaderboard": ":material/emoji_events:",
    "Development timeline": ":material/timeline:",
}
icon = ICONS.get(page.title, ":material/analytics:")
st.title(f"{icon} {page.title}")

page.run()
