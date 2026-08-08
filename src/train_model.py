"""Train a fraud classifier on data pulled from Postgres.

Fraud is ~0.17% of transactions in this dataset, so the two things that
matter most here are (1) never letting resampling leak into the test set,
and (2) evaluating with metrics that don't lie to you about a 99.8%-accurate
model that never predicts fraud at all.

Usage:
    python src/train_model.py
"""

from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from db import get_engine

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_data() -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM transactions", engine)
    print(f"Loaded {len(df)} rows from Postgres")
    return df


def split_and_scale(df: pd.DataFrame):
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    # Stratify on y so both splits keep the same ~0.17% fraud rate — with a
    # class this rare, a plain random split could easily starve the test
    # set of fraud examples and make the evaluation meaningless.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # V1-V28 are already PCA components (roughly zero-mean, unit-variance
    # by construction). Time and Amount are raw and on very different
    # scales, so only they need explicit scaling. Fit the scaler on the
    # training set only — fitting on the full dataset would leak test-set
    # statistics into training.
    scaler = StandardScaler()
    X_train[["time", "amount"]] = scaler.fit_transform(X_train[["time", "amount"]])
    X_test[["time", "amount"]] = scaler.transform(X_test[["time", "amount"]])

    return X_train, X_test, y_train, y_test, scaler


def resample_train_only(X_train, y_train):
    # SMOTE must run AFTER the train/test split, on the training data only.
    # SMOTE synthesizes new minority-class rows by interpolating between
    # real fraud cases and their nearest neighbors. If you SMOTE before
    # splitting, synthetic points derived from a test-set fraud case can
    # end up in the training set (or vice versa) — the model then gets
    # evaluated on near-duplicates of what it trained on, and your metrics
    # are inflated and won't hold up on real, unseen transactions.
    print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
    X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
    print(f"After SMOTE:  {y_res.value_counts().to_dict()}")
    return X_res, y_res


def evaluate(name: str, model, X_test, y_test) -> None:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["legit", "fraud"]))
    print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_test, y_pred))

    # ROC-AUC looks great on imbalanced data almost regardless of model
    # quality, because true negatives (the 99.8% legit class) dominate the
    # curve. PR-AUC (average precision) only cares about the minority
    # class's precision/recall tradeoff, which is what actually matters
    # for "did we catch fraud without drowning analysts in false alerts."
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"PR-AUC:  {average_precision_score(y_test, y_proba):.4f}")


def main() -> None:
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
    X_train_res, y_train_res = resample_train_only(X_train, y_train)

    # Baseline: fast, linear, interpretable — a reasonable "why do we need
    # anything fancier" reference point.
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train_res, y_train_res)
    evaluate("Logistic Regression", logreg, X_test, y_test)

    # Main model: captures non-linear interactions between the PCA
    # features, generally scores higher PR-AUC on this dataset.
    forest = RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1, random_state=42
    )
    forest.fit(X_train_res, y_train_res)
    evaluate("Random Forest", forest, X_test, y_test)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(forest, MODELS_DIR / "model.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    print(f"\nSaved model + scaler to {MODELS_DIR}")


if __name__ == "__main__":
    main()
