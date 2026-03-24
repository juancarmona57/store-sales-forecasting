"""Project timeline — full development history."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Git commit history ────────────────────────────────────────────────────────
COMMITS = [
    {"hash": "b7293f2", "phase": 1, "type": "feat", "title": "Initialize project structure",
     "detail": "pyproject.toml (deps: lightgbm, xgboost, catboost, optuna, pandas, plotly), src/config.py with all constants, LAG_DAYS, ROLLING_WINDOWS, model hyperparams. Makefile with Windows-compatible commands."},
    {"hash": "e7ce586", "phase": 1, "type": "feat", "title": "Data loader module",
     "detail": "7 loader functions: load_train, load_test, load_stores, load_oil, load_holidays, load_transactions, load_raw_data. Merges stores (renaming type→store_type), oil, transactions. 5 tests passing."},
    {"hash": "96aac07", "phase": 1, "type": "feat", "title": "Data preprocessor",
     "detail": "preprocess_train (clip negatives, fill onpromotion NaN, sort by date+store+family), preprocess_oil (ffill/bfill/interpolate), preprocess_holidays (transferred flag), preprocess_transactions. 5 tests."},
    {"hash": "25f6137", "phase": 2, "type": "feat", "title": "RMSLE metric + validation",
     "detail": "rmsle() using np.log1p with negative clipping. TimeSeriesSplitWithGap with expanding window and configurable gap. split() generator + get_holdout_split(). 4 tests."},
    {"hash": "0c19303", "phase": 2, "type": "feat", "title": "Temporal features",
     "detail": "add_temporal_features(): 14 features including cyclical sin/cos encoding for day_of_week and month to capture periodicity without ordinal bias."},
    {"hash": "0fae109", "phase": 2, "type": "feat", "title": "Lag + rolling features",
     "detail": "add_lag_features() with grouped shift per store×family. add_rolling_features() with shift(1)+grouped rolling to prevent cross-group contamination and target leakage. 12 tests."},
    {"hash": "9fd4740", "phase": 2, "type": "feat", "title": "External features (oil, holidays, transactions)",
     "detail": "oil_lag_7/14, oil_rolling_mean/std_28, oil_diff. Holiday proximity (days_to_next, days_since_last). Transactions rolling per-store with leakage-safe grouped rolling. 5 tests."},
    {"hash": "f9ac2db", "phase": 2, "type": "feat", "title": "Feature builder orchestrator",
     "detail": "build_features() applies all 7 feature categories in sequence. get_feature_columns() excludes id, date, target. Final: 61 features total."},
    {"hash": "72ea225", "phase": 2, "type": "feat", "title": "Promotion + cross + aggregation features",
     "detail": "promo_lag_7, promo_rolling_mean/sum. Cross: family×store_type, family×cluster, dow×family. Aggregations: store/family/store_family/city/state/type/cluster mean sales."},
    {"hash": "1d2b37d", "phase": 3, "type": "feat", "title": "Base model + LightGBM Tweedie",
     "detail": "Abstract BaseModel with fit/predict/save/load interface. LGBMModel: Tweedie objective (variance_power=1.1), early stopping, non-negative clipping. Native booster serialization."},
    {"hash": "d8f8bfe", "phase": 3, "type": "feat", "title": "XGBoost + CatBoost models",
     "detail": "XGBModel: reg:squaredlogerror, hist tree method. CatBoostModel: RMSE on log1p target, ordered boosting, native categorical handling. Both implement BaseModel interface."},
    {"hash": "16fa103", "phase": 3, "type": "feat", "title": "Ensemble + weight optimization",
     "detail": "weighted_average_ensemble() with scipy.optimize. Optimizes weights to minimize RMSLE on validation set. Final: lgbm=36.8%, xgb=0.3%, catboost=62.8%."},
    {"hash": "bc79dbe", "phase": 3, "type": "feat", "title": "Submission generator",
     "detail": "generate_submission(): loads test, builds features, predicts with ensemble. postprocess_predictions(): clips to 0, rounds zeros. Output: 28,512-row CSV in submissions/."},
    {"hash": "e9d3b12", "phase": 3, "type": "feat", "title": "Optuna hyperparameter optimizer",
     "detail": "Model-specific search spaces for LightGBM, XGBoost, CatBoost. TimeSeriesSplitWithGap cross-validation. Pruning with MedianPruner. Configurable n_trials."},
    {"hash": "08eb237", "phase": 4, "type": "test", "title": "Validation tests",
     "detail": "TimeSeriesSplitWithGap leakage prevention tests. Ensures no future data leaks into training windows. Gap correctness assertions."},
    {"hash": "e1eb8c5", "phase": 4, "type": "test", "title": "Feature builder integration tests",
     "detail": "End-to-end tests for build_features(). Verifies all 61 features present, no NaN in key columns, correct dtypes, no leakage."},
    {"hash": "f7f62a2", "phase": 4, "type": "ci", "title": "CI/CD + README",
     "detail": "GitHub Actions workflow: ruff lint + pytest on push/PR. README as portfolio case study: business problem, dataset, architecture, results, installation guide."},
    {"hash": "6c49bbc", "phase": 4, "type": "docs", "title": "Jupyter notebooks",
     "detail": "4 notebooks: 01_eda (sales trends, zero-sales, oil), 02_features (cyclical encoding demo, lag/rolling), 03_modeling (validation splits, model params), 04_ensemble (weight optimization, submission)."},
]

PHASES = {
    1: {"name": "Foundation", "color": "#4C8BF5", "icon": ":material/foundation:"},
    2: {"name": "Feature engineering", "color": "#66BB6A", "icon": ":material/tune:"},
    3: {"name": "Modeling & ensemble", "color": "#FF7043", "icon": ":material/model_training:"},
    4: {"name": "Quality & delivery", "color": "#AB47BC", "icon": ":material/verified:"},
}

TYPE_ICONS = {
    "feat": ":material/add_circle:",
    "test": ":material/bug_report:",
    "ci": ":material/sync:",
    "docs": ":material/description:",
}

# ── Summary KPIs ──────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric(":material/commit: Total commits", str(len(COMMITS)), border=True)
col2.metric(":material/layers: Phases", "4", "Foundation → Delivery", border=True)
col3.metric(":material/bug_report: Tests", "66", "all passing", border=True)
col4.metric(":material/schedule: Built in", "1 day", "2026-03-23", border=True)

st.space("medium")

# ── Phase progress ────────────────────────────────────────────────────────────
st.subheader(":material/flag: Phase summary")
cols = st.columns(4)
for i, (phase_num, phase) in enumerate(PHASES.items()):
    phase_commits = [c for c in COMMITS if c["phase"] == phase_num]
    with cols[i]:
        with st.container(border=True):
            st.markdown(f"{phase['icon']} **Phase {phase_num}**")
            st.markdown(f"**{phase['name']}**")
            st.badge(f"{len(phase_commits)} commits", color="green")
            st.caption(", ".join(c["title"] for c in phase_commits[:3]) + ("..." if len(phase_commits) > 3 else ""))

st.space("medium")

# ── Timeline chart ────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader(":material/timeline: Development timeline")
    import pandas as pd
    timeline_df = pd.DataFrame([{
        "Commit": c["hash"],
        "Phase": f"Phase {c['phase']}: {PHASES[c['phase']]['name']}",
        "Title": c["title"],
        "Type": c["type"],
        "Order": i,
    } for i, c in enumerate(COMMITS)])

    fig = px.scatter(
        timeline_df,
        x="Order",
        y="Phase",
        color="Phase",
        hover_name="Title",
        hover_data={"Order": False, "Commit": True, "Type": True},
        color_discrete_map={
            f"Phase {k}: {v['name']}": v["color"] for k, v in PHASES.items()
        },
        size_max=15,
    )
    fig.update_traces(marker=dict(size=14, symbol="circle"))
    fig.update_layout(
        xaxis_title="Commit sequence",
        yaxis_title="",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

st.space("medium")

# ── Detailed commit log ───────────────────────────────────────────────────────
st.subheader(":material/list: Full commit history")

selected_phase = st.segmented_control(
    "Filter by phase",
    options=["All"] + [f"Phase {k}" for k in PHASES],
    default="All",
)

for commit in reversed(COMMITS):
    phase_label = f"Phase {commit['phase']}"
    if selected_phase != "All" and phase_label != selected_phase:
        continue

    phase_color = PHASES[commit["phase"]]["color"]
    type_icon = TYPE_ICONS.get(commit["type"], ":material/code:")

    with st.container(border=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown(f"{type_icon} **{commit['title']}**")
            st.caption(commit["detail"])
        with c2:
            st.code(commit["hash"], language=None)
            st.badge(phase_label, color="blue")
