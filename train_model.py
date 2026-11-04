import json
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier

from preprocess import get_train_test

MODELS_DIR = "models"


def build_models(y_train):
    # Defaults are the minority class, so we tell each model to pay
    # extra attention to them.
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    return {
        "decision_tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=50,
            class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=42),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=ratio, eval_metric="logloss",
            random_state=42),
    }


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def main():
    X_train, X_test, y_train, y_test = get_train_test()
    results = {}

    for name, model in build_models(y_train).items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_test, y_test)
        joblib.dump(model, f"{MODELS_DIR}/{name}.joblib")

    # Save everything later modules need
    with open(f"{MODELS_DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"{MODELS_DIR}/feature_columns.json", "w") as f:
        json.dump(list(X_train.columns), f)
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    table = pd.DataFrame(results).T.drop(columns="confusion_matrix")
    print("\nTest-set results (computed on held-out data):")
    print(table.round(3))


if __name__ == "__main__":
    main()