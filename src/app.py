"""Streamlit dashboard: loads the trained model, scores a sample of
transactions pulled live from Postgres, and lets a reviewer explore the
precision/recall tradeoff interactively.

Usage:
    streamlit run src/app.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve

from db import get_engine

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Okabe-Ito colorblind-safe palette.
BLUE = "#0072B2"
VERMILLION = "#D55E00"

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")


@st.cache_resource
def load_model():
    model = joblib.load(MODELS_DIR / "model.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    return model, scaler


@st.cache_data
def load_sample(n_legit: int = 5000) -> pd.DataFrame:
    engine = get_engine()
    # Every fraud case plus a random sample of legit transactions. Pulling
    # all ~280k rows into a free-tier Postgres dashboard on every rerun
    # would be slow and mostly pointless — fraud is what we care about
    # showing, and it's <500 rows total.
    fraud = pd.read_sql("SELECT * FROM transactions WHERE is_fraud = 1", engine)
    legit = pd.read_sql(
        f"SELECT * FROM transactions WHERE is_fraud = 0 ORDER BY RANDOM() LIMIT {n_legit}",
        engine,
    )
    return pd.concat([fraud, legit], ignore_index=True)


def score(df: pd.DataFrame, model, scaler) -> pd.DataFrame:
    df = df.copy()
    X = df.drop(columns=["is_fraud"])
    X[["time", "amount"]] = scaler.transform(X[["time", "amount"]])
    df["fraud_probability"] = model.predict_proba(X)[:, 1]
    return df


def main() -> None:
    st.title("Fraud Detection Dashboard")
    st.caption(
        "Random Forest classifier trained on SMOTE-balanced data "
        "(Kaggle Credit Card Fraud dataset), served from a live Postgres feed."
    )

    model, scaler = load_model()
    df = load_sample()
    scored = score(df, model, scaler)

    threshold = st.sidebar.slider(
        "Decision threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help=(
            "Transactions scored above this probability are flagged as fraud. "
            "Lower it to catch more fraud at the cost of more false alerts; "
            "raise it to cut false alerts at the cost of missed fraud."
        ),
    )
    scored["flagged"] = scored["fraud_probability"] >= threshold

    y_true = scored["is_fraud"]
    y_pred = scored["flagged"].astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions shown", f"{len(scored):,}")
    col2.metric("Flagged as fraud", f"{int(scored['flagged'].sum()):,}")
    col3.metric("Precision @ threshold", f"{precision:.1%}")
    col4.metric("Recall @ threshold", f"{recall:.1%}")

    left, right = st.columns(2)

    with left:
        st.subheader("Confusion matrix")
        fig, ax = plt.subplots(figsize=(4, 3.5))
        cm = confusion_matrix(y_true, y_pred)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["legit", "fraud"])
        ax.set_yticks([0, 1], ["legit", "fraud"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)

    with right:
        st.subheader("Precision / recall vs. threshold")
        precisions, recalls, thresholds = precision_recall_curve(
            y_true, scored["fraud_probability"]
        )
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.plot(thresholds, precisions[:-1], color=BLUE, linewidth=2, label="Precision")
        ax.plot(thresholds, recalls[:-1], color=VERMILLION, linewidth=2, label="Recall")
        ax.axvline(threshold, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False)
        st.pyplot(fig)

    st.subheader("Top flagged transactions")
    top_flagged = (
        scored[scored["flagged"]]
        .sort_values("fraud_probability", ascending=False)
        .head(25)[["amount", "is_fraud", "fraud_probability"]]
    )
    st.dataframe(top_flagged, use_container_width=True)

    st.subheader("What drives the model")
    importances = pd.Series(
        model.feature_importances_, index=model.feature_names_in_
    ).sort_values(ascending=False)[:15]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(importances.index[::-1], importances.values[::-1], color=BLUE)
    ax.set_xlabel("Feature importance")
    st.pyplot(fig)


if __name__ == "__main__":
    main()
