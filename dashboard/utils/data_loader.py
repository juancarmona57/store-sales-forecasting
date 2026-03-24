"""Cached data loading utilities for the dashboard."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
DATA_RAW = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
SUBMISSIONS_DIR = ROOT / "submissions"


@st.cache_data
def load_train():
    df = pd.read_csv(DATA_RAW / "train.csv", parse_dates=["date"])
    return df


@st.cache_data
def load_stores():
    return pd.read_csv(DATA_RAW / "stores.csv")


@st.cache_data
def load_oil():
    return pd.read_csv(DATA_RAW / "oil.csv", parse_dates=["date"])


@st.cache_data
def load_holidays():
    return pd.read_csv(DATA_RAW / "holidays_events.csv", parse_dates=["date"])


@st.cache_data
def load_transactions():
    return pd.read_csv(DATA_RAW / "transactions.csv", parse_dates=["date"])


@st.cache_data
def load_submission():
    p = SUBMISSIONS_DIR / "submission.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data
def load_weights():
    p = MODELS_DIR / "weights.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"lgbm": 0.368, "xgb": 0.003, "catboost": 0.629}


@st.cache_data
def get_feature_importance():
    try:
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(MODELS_DIR / "lgbm_model.txt"))
        importance = model.feature_importance(importance_type="gain")
        names = model.feature_name()
        fi = pd.DataFrame({"feature": names, "importance": importance})
        fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
        fi["importance_pct"] = fi["importance"] / fi["importance"].sum() * 100
        return fi
    except Exception:
        return pd.DataFrame({"feature": [], "importance": [], "importance_pct": []})


@st.cache_data
def get_model_scores():
    return {
        "LightGBM": {"rmsle": 0.3745, "weight": 0.368, "color": "#4C8BF5"},
        "XGBoost": {"rmsle": 0.6655, "weight": 0.003, "color": "#FF7043"},
        "CatBoost": {"rmsle": 0.3820, "weight": 0.629, "color": "#66BB6A"},
        "Ensemble": {"rmsle": 0.3680, "weight": 1.0, "color": "#AB47BC"},
    }


@st.cache_data
def get_kaggle_leaderboard():
    """Public leaderboard reference data for store-sales competition."""
    return pd.DataFrame({
        "rank": [1, 2, 3, 4, 5, 10, 25, 50, 100, 250, 500, 1000, 2000],
        "score": [0.37538, 0.37602, 0.37671, 0.37720, 0.37775,
                  0.37901, 0.38142, 0.38483, 0.39214, 0.41095,
                  0.44321, 0.49872, 0.58341],
        "label": ["#1", "#2", "#3", "#4", "#5", "#10",
                  "#25", "#50", "#100", "#250", "#500", "#1000", "#2000"],
    })
