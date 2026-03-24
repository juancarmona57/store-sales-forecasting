"""Kaggle leaderboard comparison page."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils.data_loader import get_kaggle_leaderboard, get_model_scores, load_submission

scores = get_model_scores()
leaderboard = get_kaggle_leaderboard()
submission = load_submission()

KAGGLE_LB_SCORE = 0.70038   # actual public leaderboard score (submitted 2026-03-23)
VAL_SCORE = scores.get("Ensemble (val)", {}).get("rmsle", 0.3680)
OUR_SCORE = KAGGLE_LB_SCORE  # use real LB score for leaderboard positioning

# ── Positioning banner ────────────────────────────────────────────────────────
# Interpolate estimated rank from the reference leaderboard
ranks = leaderboard["rank"].values
lb_scores = leaderboard["score"].values
# lb_scores is decreasing (lower=better); reverse for np.interp (needs increasing x)
our_rank_est = int(np.interp(OUR_SCORE, lb_scores[::-1], ranks[::-1]))
our_rank_est = max(1, our_rank_est)
total_teams = 5000  # approximate

col1, col2, col3, col4 = st.columns(4)
col1.metric(":material/emoji_events: Validation RMSLE", f"{VAL_SCORE:.4f}", "holdout split", border=True)
col2.metric(":material/leaderboard: Kaggle public LB", f"{KAGGLE_LB_SCORE:.5f}", "actual submission", border=True)
col3.metric(":material/group: Approx. total teams", f"{total_teams:,}", border=True)
pct = our_rank_est / total_teams * 100
col4.metric(":material/military_tech: Estimated percentile", f"Top {pct:.1f}%", border=True)

with st.container(border=True):
    st.markdown(":material/warning: **Gap analysis: val 0.3680 → LB 0.7004**")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**Root cause: same-day transactions leakage**
- The model was trained using `transactions` (same day) as a feature
- In production, same-day transactions are **not available** at forecast time
- We imputed test transactions from lag-7 (same day last week) — good approximation but not exact
- Error propagation in the 16-step iterative forecast compounds this
""")
    with c2:
        st.markdown("""
**How to fix for a real score ≤ 0.40:**
1. Remove `transactions` from training features (or use only lag versions)
2. Use lags ≥ 16 to avoid iterative forecasting entirely
3. Re-train all 3 models with corrected features
4. Expected improvement: 0.70 → 0.38–0.42

:green-badge[In progress] Retrain pipeline already being planned
""")

st.space("medium")

# ── Leaderboard chart ─────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader(":material/bar_chart: Score vs public leaderboard")

    # Build combined dataframe
    lb_df = leaderboard.copy()
    our_row = pd.DataFrame([{"rank": our_rank_est, "score": OUR_SCORE, "label": "⭐ Our score"}])
    combined = pd.concat([lb_df, our_row], ignore_index=True).sort_values("rank")

    fig = go.Figure()

    # Reference line scores
    fig.add_trace(go.Scatter(
        x=lb_df["rank"], y=lb_df["score"],
        mode="lines+markers",
        name="Leaderboard reference",
        line=dict(color="#78909C", width=2),
        marker=dict(size=6),
        hovertemplate="Rank #%{x}<br>RMSLE: %{y:.5f}<extra></extra>",
    ))

    # Our score line
    fig.add_hline(
        y=OUR_SCORE, line_dash="dot", line_color="#4C8BF5", line_width=2,
        annotation_text=f"Our ensemble: {OUR_SCORE:.4f}",
        annotation_position="top left",
        annotation_font=dict(color="#4C8BF5", size=13),
    )

    # Our score marker
    fig.add_trace(go.Scatter(
        x=[our_rank_est], y=[OUR_SCORE],
        mode="markers",
        name="Our score",
        marker=dict(color="#4C8BF5", size=16, symbol="star"),
        hovertemplate=f"Our rank: ~#{our_rank_est}<br>RMSLE: {OUR_SCORE:.4f}<extra></extra>",
    ))

    fig.update_layout(
        xaxis_title="Rank",
        yaxis_title="RMSLE",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

st.space("medium")

# ── Reference table ───────────────────────────────────────────────────────────
col_l, col_r = st.columns([1, 1])

with col_l:
    with st.container(border=True):
        st.markdown("**Leaderboard reference points**")
        display_lb = leaderboard[["rank", "score", "label"]].copy()
        display_lb["vs ours"] = (display_lb["score"] - OUR_SCORE).map(lambda x: f"{x:+.5f}")
        display_lb["better?"] = display_lb["score"].apply(
            lambda s: ":material/arrow_upward:" if s < OUR_SCORE else ":material/arrow_downward:"
        )
        st.dataframe(
            display_lb.rename(columns={"rank": "Rank", "score": "RMSLE", "label": "Position"}),
            hide_index=True,
            use_container_width=True,
        )

with col_r:
    with st.container(border=True):
        st.markdown("**Score distribution context**")
        # Score distribution reference
        fig = go.Figure()
        scores_ref = [0.37538, 0.37602, 0.37671, 0.37720, 0.37775,
                      0.37901, 0.38142, 0.38483, 0.39214, 0.41095,
                      0.44321, 0.49872, 0.58341, 0.70000]
        fig.add_trace(go.Box(
            y=scores_ref,
            name="Leaderboard range",
            marker_color="#78909C",
            boxpoints=False,
        ))
        fig.add_hline(y=OUR_SCORE, line_dash="dot", line_color="#4C8BF5",
                      annotation_text=f"Ours: {OUR_SCORE:.4f}", annotation_position="right")
        fig.update_layout(yaxis_title="RMSLE", showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

st.space("medium")

# ── Submission preview ────────────────────────────────────────────────────────
if submission is not None:
    with st.container(border=True):
        st.subheader(":material/upload_file: Submission preview")
        col_l, col_r = st.columns([1, 2])
        with col_l:
            st.metric("Total predictions", f"{len(submission):,}", border=True)
            st.metric("Zero predictions", f"{(submission.sales == 0).sum():,}", border=True)
            st.metric("Max prediction", f"{submission.sales.max():.1f}", border=True)
            st.metric("Mean prediction", f"{submission.sales.mean():.2f}", border=True)
        with col_r:
            fig = px.histogram(
                submission, x="sales",
                nbins=100,
                title="Distribution of predicted sales",
                labels={"sales": "Predicted sales"},
                color_discrete_sequence=["#4C8BF5"],
            )
            fig.update_layout(yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(submission.head(20), hide_index=True, use_container_width=True)

# ── What would improve the score ─────────────────────────────────────────────
with st.container(border=True):
    st.subheader(":material/lightbulb: Paths to improvement")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Feature engineering**")
        st.markdown("""
- Fourier series for seasonality
- Store-level sales momentum
- Holiday bridge days detection
- Pay cycle features (quincenal)
- Family-store affinity scores
""")
    with c2:
        st.markdown("**Modeling**")
        st.markdown("""
- N-BEATS / TFT deep learning
- Prophet for trend + seasonality
- Linear model for residuals
- Stacking with meta-learner
- Per-family specialized models
""")
    with c3:
        st.markdown("**Validation & tuning**")
        st.markdown("""
- More Optuna trials (50→500)
- Walk-forward validation
- RMSLE-aware custom loss
- Threshold tuning per family
- Post-process by store type
""")
