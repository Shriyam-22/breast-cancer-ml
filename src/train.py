"""
Train and evaluate multiple ML models for breast cancer diagnosis.
Dataset: Wisconsin Diagnostic Breast Cancer (built into scikit-learn, 569 samples, 30 features).

Run:  python src/train.py
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             RocCurveDisplay)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
MODELS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42


def load_data():
    data = load_breast_cancer(as_frame=True)
    X = data.data
    # Relabel so 1 = malignant (the class we care about detecting)
    y = (data.target == 0).astype(int).rename("malignant")
    return X, y


def eda(X, y):
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Class balance -> benign: {(y == 0).sum()}, malignant: {(y == 1).sum()}")
    print(f"Missing values: {int(X.isna().sum().sum())}")

    plt.figure(figsize=(5, 4))
    sns.countplot(x=y.map({0: "Benign", 1: "Malignant"}))
    plt.title("Class Distribution"); plt.xlabel(""); plt.tight_layout()
    plt.savefig(FIG_DIR / "class_distribution.png", dpi=120); plt.close()

    mean_cols = [c for c in X.columns if c.startswith("mean")]
    plt.figure(figsize=(9, 7))
    sns.heatmap(X[mean_cols].corr(), cmap="coolwarm", center=0, annot=True, fmt=".1f")
    plt.title("Correlation of 'mean' Features"); plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=120); plt.close()


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def main():
    X, y = load_data()
    eda(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results, fitted = {}, {}
    print("\nModel comparison (5-fold CV on training set):")
    for name, clf in get_models().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        cv_recall = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="recall")
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        results[name] = {"cv_recall": float(cv_recall.mean()), "cv_f1": float(cv_f1.mean())}
        print(f"  {name:<20} recall={cv_recall.mean():.3f}  f1={cv_f1.mean():.3f}")

    # Pick best by CV F1, tie-break on recall (missing a cancer is the costly error)
    best_name = max(results, key=lambda n: (results[n]["cv_f1"], results[n]["cv_recall"]))
    best = fitted[best_name]
    print(f"\nBest model: {best_name}")

    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]
    test_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    print("\nHeld-out test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k:<10} {v:.3f}")
    print("\n" + classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout(); plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=120); plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, pipe in fitted.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_title("ROC Curves (Test Set)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "roc_curves.png", dpi=120); plt.close()

    rf = fitted["Random Forest"].named_steps["clf"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
    plt.figure(figsize=(7, 5))
    sns.barplot(x=imp.values, y=imp.index, color="#4C72B0")
    plt.title("Top 10 Features (Random Forest Importance)"); plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance.png", dpi=120); plt.close()

    joblib.dump(best, MODELS_DIR / "model.joblib")
    meta = {
        "best_model": best_name,
        "feature_names": list(X.columns),
        "feature_means": X.mean().round(4).to_dict(),
        "feature_min": X.min().round(4).to_dict(),
        "feature_max": X.max().round(4).to_dict(),
        "cv_results": results,
        "test_metrics": {k: round(float(v), 4) for k, v in test_metrics.items()},
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved model to {MODELS_DIR / 'model.joblib'} and figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
