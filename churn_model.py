"""
Churn Prediction Model - Internee.pk Internship Task
Objective: Predict which users are likely to leave the platform.
Models: Random Forest and Gradient Boosting (scikit-learn)
Data: LMS engagement logs / user activity.

Usage:
    python churn_model.py                  # uses lms_data.csv in this folder
    python churn_model.py --data other.csv # any CSV with a 'churn' column (1 = left, 0 = stayed)
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             classification_report, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
OUT_DIR = "outputs"


# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------
def make_synthetic_lms_data(n=3000, seed=RANDOM_STATE):
    """Simulate LMS engagement logs. Replace with real data via --data."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "logins_per_week": rng.poisson(4, n),
        "avg_session_minutes": rng.gamma(4, 9, n),
        "days_since_last_login": rng.exponential(8, n).round(),
        "courses_enrolled": rng.integers(1, 8, n),
        "course_completion_pct": rng.beta(2, 2, n) * 100,
        "lessons_completed": rng.poisson(25, n),
        "quiz_avg_score": np.clip(rng.normal(65, 18, n), 0, 100),
        "assignments_submitted": rng.poisson(6, n),
        "forum_posts": rng.poisson(2, n),
        "support_tickets": rng.poisson(0.7, n),
        "video_watch_pct": rng.beta(2.5, 2, n) * 100,
        "tenure_days": rng.integers(7, 720, n),
    })
    # Churn is driven by low engagement (+ noise)
    z = (
        -0.45 * df["logins_per_week"]
        + 0.11 * df["days_since_last_login"]
        - 0.025 * df["course_completion_pct"]
        - 0.012 * df["avg_session_minutes"]
        - 0.015 * df["quiz_avg_score"]
        - 0.04 * df["lessons_completed"]
        + 0.25 * df["support_tickets"]
        - 0.1 * df["forum_posts"]
        + 2.8
        + rng.normal(0, 1, n)
    )
    df["churn"] = (1 / (1 + np.exp(-z)) > 0.5).astype(int)
    # simulate a few missing values, like real logs
    for col in ["quiz_avg_score", "avg_session_minutes"]:
        df.loc[rng.choice(n, 40, replace=False), col] = np.nan
    return df


def load_data(path):
    path = path or "lms_data.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Loaded {path}: {df.shape}")
    else:
        df = make_synthetic_lms_data()
        print(f"{path} not found -> using synthetic LMS data: {df.shape}")
    return df


def engineer_features(df):
    """Behavioural features built from the raw LMS logs."""
    df = df.copy()
    if {"courses_completed", "courses_enrolled"} <= set(df.columns):
        df["completion_rate"] = df["courses_completed"] / df["courses_enrolled"].replace(0, np.nan)
    if {"logins_per_week", "avg_session_minutes"} <= set(df.columns):
        df["weekly_minutes"] = df["logins_per_week"] * df["avg_session_minutes"]
    if "days_since_last_login" in df.columns:
        df["inactive_14d"] = (df["days_since_last_login"] > 14).astype(int)
    return df


def prepare(df, target="churn"):
    """Clean the target to clean 0/1 and keep numeric features only."""
    y = df[target]
    if y.dtype == object:
        y = y.astype(str).str.strip().str.lower().map(
            {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0})
    mask = y.notna()                      # drop rows with unknown target
    # drop identifier columns (they carry no behaviour and cause leakage/noise)
    id_cols = [c for c in df.columns if c.lower() in ("user_id", "id", "userid")]
    X = engineer_features(df.loc[mask].drop(columns=[target] + id_cols))
    X = X.select_dtypes(include=[np.number])
    y = y[mask].astype(int)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, y


# ----------------------------------------------------------------------------
# 2. MODELS
# ----------------------------------------------------------------------------
def build_models():
    return {
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
    }


def main(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data(args.data)
    X, y = prepare(df)
    print(f"\nFeatures: {X.shape[1]} numeric | Rows: {len(X)}")
    print("Churn distribution:\n", y.value_counts(normalize=True).round(3).to_string())

    # --- EDA chart: engagement vs churn
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col in zip(axes, ["logins_per_week", "completion_rate", "days_since_last_login"]):
        if col in X.columns:
            X.assign(churn=y).boxplot(column=col, by="churn", ax=ax)
            ax.set_title(col); ax.set_xlabel("churn (1 = left)")
    plt.suptitle("Engagement vs churn"); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/eda_engagement_vs_churn.png", dpi=150); plt.close()

    # --- Train / test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # --- Cross-validation (safe: folds never exceed minority class size)
    n_splits = int(max(2, min(5, y_tr.value_counts().min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    print(f"\nCross-validation ({n_splits}-fold, stratified):")
    models, rows = build_models(), []
    for name, model in models.items():
        s = cross_validate(model, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1)
        row = {"model": name}
        row.update({m: round(s[f"test_{m}"].mean(), 4) for m in scoring})
        rows.append(row)
    cv_df = pd.DataFrame(rows).set_index("model")
    print(cv_df.to_string())
    cv_df.to_csv(f"{OUT_DIR}/cv_results.csv")

    # --- Hyper-parameter tuning of the best model (by CV ROC-AUC)
    best_name = cv_df["roc_auc"].idxmax()
    print(f"\nBest model by CV ROC-AUC: {best_name} -> tuning...")
    grids = {
        "Random Forest": {"clf__max_depth": [None, 8, 14],
                          "clf__min_samples_leaf": [1, 3, 5]},
        "Gradient Boosting": {"clf__n_estimators": [100, 200],
                              "clf__learning_rate": [0.05, 0.1],
                              "clf__max_depth": [2, 3]},
    }
    gs = GridSearchCV(models[best_name], grids[best_name], cv=cv,
                      scoring="roc_auc", n_jobs=1)
    gs.fit(X_tr, y_tr)
    print("Best params:", gs.best_params_)
    best = gs.best_estimator_

    # --- Final evaluation on held-out test set
    proba = best.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    print("\nTest-set results:")
    print(classification_report(y_te, pred, target_names=["Stayed", "Churned"]))
    print("ROC-AUC:", round(roc_auc_score(y_te, proba), 4))

    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["Stayed", "Churned"])
    plt.title(f"Confusion matrix - {best_name}"); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150); plt.close()

    RocCurveDisplay.from_predictions(y_te, proba)
    plt.title(f"ROC curve - {best_name}"); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/roc_curve.png", dpi=150); plt.close()

    # --- Feature importance (what drives churn?)
    imp = pd.Series(best.named_steps["clf"].feature_importances_, index=X.columns)
    imp = imp.sort_values()
    imp.plot(kind="barh", figsize=(7, 5), color="#3f9d1f")
    plt.title("Feature importance"); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150); plt.close()
    print("\nTop churn drivers:\n", imp.sort_values(ascending=False).head(5).round(3).to_string())

    # --- Actionable output: at-risk users for personalized support
    risk = X_te.copy()
    if "user_id" in df.columns:
        risk.insert(0, "user_id", df.loc[X_te.index, "user_id"].values)
    risk["churn_probability"] = proba.round(3)
    risk["risk_level"] = pd.cut(proba, [-0.01, 0.4, 0.7, 1.0],
                                labels=["Low", "Medium", "High"])
    risk.sort_values("churn_probability", ascending=False).to_csv(
        f"{OUT_DIR}/at_risk_users.csv", index=False)
    print("\nRisk levels:\n", risk["risk_level"].value_counts().to_string())
    print(f"\nSaved charts and CSVs in ./{OUT_DIR}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="path to your LMS CSV (needs a 'churn' column)")
    main(ap.parse_args())   # protected by the __main__ guard (needed on Windows)
