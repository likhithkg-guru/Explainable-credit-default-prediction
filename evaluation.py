import json
import joblib
import matplotlib
matplotlib.use("Agg")  # save images without opening windows
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (roc_curve, roc_auc_score,
                             precision_recall_curve, ConfusionMatrixDisplay)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from preprocess import get_train_test
from train_model import build_models

NAMES = ["decision_tree", "random_forest", "xgboost"]
LABELS = {"decision_tree": "Decision Tree",
          "random_forest": "Random Forest",
          "xgboost": "XGBoost"}
COLORS = {"decision_tree": "#E4572E",
          "random_forest": "#2E86AB",
          "xgboost": "#3FA34D"}


def load_models():
    return {n: joblib.load(f"models/{n}.joblib") for n in NAMES}


def plot_metric_comparison():
    with open("models/metrics.json") as f:
        results = json.load(f)
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, n in enumerate(NAMES):
        vals = [results[n][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=LABELS[n], color=COLORS[n])
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison on held-out test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig("assets/metric_comparison.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrices(models, X_test, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, n in zip(axes, NAMES):
        ConfusionMatrixDisplay.from_estimator(
            models[n], X_test, y_test, ax=ax, cmap="Blues",
            display_labels=["No default", "Default"], colorbar=False)
        ax.set_title(LABELS[n])
    fig.tight_layout()
    fig.savefig("assets/confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_roc_curves(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(6, 5))
    for n in NAMES:
        proba = models[n].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, color=COLORS[n], label=f"{LABELS[n]} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random guessing")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig("assets/roc_curves.png", dpi=150)
    plt.close(fig)


def plot_threshold_tradeoff(models, X_test, y_test):
    """Shows how precision and recall change as the decision threshold moves."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, n in zip(axes, NAMES):
        proba = models[n].predict_proba(X_test)[:, 1]
        precision, recall, thr = precision_recall_curve(y_test, proba)
        ax.plot(thr, precision[:-1], label="Precision")
        ax.plot(thr, recall[:-1], label="Recall")
        ax.axvline(0.5, color="grey", linestyle=":")
        ax.set_title(LABELS[n])
        ax.set_xlabel("Decision threshold")
    axes[0].legend()
    fig.suptitle("Precision vs recall as the threshold changes")
    fig.tight_layout()
    fig.savefig("assets/threshold_tradeoff.png", dpi=150)
    plt.close(fig)


def cross_validate(X_train, y_train):
    """5-fold CV on the training set: checks that results are stable."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    out = {}
    for n, model in build_models(y_train).items():
        print(f"Cross-validating {n}...")
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc")
        out[n] = {"mean_auc": float(scores.mean()), "std_auc": float(scores.std())}
        print(f"  ROC-AUC: {scores.mean():.3f} (+/- {scores.std():.3f})")
    with open("models/cv_results.json", "w") as f:
        json.dump(out, f, indent=2)


def main():
    X_train, X_test, y_train, y_test = get_train_test()
    models = load_models()

    print("Majority-class baseline accuracy:", round(1 - y_test.mean(), 4))
    plot_metric_comparison()
    plot_confusion_matrices(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)
    plot_threshold_tradeoff(models, X_test, y_test)
    print("Charts saved in assets/")

    cross_validate(X_train, y_train)


if __name__ == "__main__":
    main()