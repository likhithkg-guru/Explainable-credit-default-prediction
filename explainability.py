import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer

MODELS_DIR = Path("models")
CLASS_NAMES = ["No default", "Default"]
RED, GREEN = "#D64545", "#2E9E5B"

DISCLAIMER = (
    "These charts show which inputs the model relied on for this estimate. "
    "They describe the model's behaviour, not the real-world cause of default. "
    "This is an educational prototype, not a credit decision."
)


def _positive_class(sv):
    """SHAP returns different shapes depending on model and version.
    This always gives an array of shape (rows, features) for the 'Default' class."""
    if isinstance(sv, list):
        return np.array(sv[1])
    sv = np.array(sv)
    if sv.ndim == 3:
        return sv[:, :, 1]
    return sv


def _base_value(expected_value):
    return float(np.atleast_1d(np.array(expected_value))[-1])


class Explainer:
    def __init__(self, model_name="xgboost"):
        self.model_name = model_name
        self.model = joblib.load(MODELS_DIR / f"{model_name}.joblib")
        with open(MODELS_DIR / "feature_columns.json") as f:
            self.columns = json.load(f)

        # SHAP: fast exact method for tree models
        self.shap_explainer = shap.TreeExplainer(self.model)
        self.base_value = _base_value(self.shap_explainer.expected_value)
        # Decision Tree / Random Forest explain in probability units,
        # XGBoost explains in log-odds units. Show this on the chart axis.
        self.units = "log-odds" if model_name == "xgboost" else "probability"

        # LIME: needs the training data to learn feature ranges
        X_train = pd.read_csv("data/X_train.csv")[self.columns]
        self.lime_explainer = LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=self.columns,
            class_names=CLASS_NAMES,
            mode="classification",
            random_state=42,
        )

    def _prepare(self, row):
        """Make sure the row is a one-row DataFrame in the right column order."""
        if isinstance(row, pd.Series):
            row = row.to_frame().T
        return row[self.columns].astype(float).reset_index(drop=True)

    def predict_proba(self, row):
        return float(self.model.predict_proba(self._prepare(row))[0, 1])

    # ---------- SHAP ----------
    def shap_local(self, row):
        row = self._prepare(row)
        sv = _positive_class(self.shap_explainer.shap_values(row))[0]
        df = pd.DataFrame({
            "feature": self.columns,
            "feature_value": row.iloc[0].values,
            "shap_value": sv,
        })
        df["label"] = df["feature"] + " = " + df["feature_value"].round(2).astype(str)
        order = df["shap_value"].abs().sort_values(ascending=False).index
        return df.loc[order].reset_index(drop=True)

    def shap_global(self, X, n=500):
        sample = X[self.columns].astype(float).sample(min(n, len(X)), random_state=42)
        sv = _positive_class(self.shap_explainer.shap_values(sample))
        df = pd.DataFrame({
            "feature": self.columns,
            "mean_abs_shap": np.abs(sv).mean(axis=0),
        })
        return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # ---------- LIME ----------
    def lime_local(self, row, num_features=10):
        row = self._prepare(row)

        def predict_fn(arr):
            # LIME sends plain numbers; put the column names back
            return self.model.predict_proba(pd.DataFrame(arr, columns=self.columns))

        exp = self.lime_explainer.explain_instance(
            row.iloc[0].values, predict_fn, num_features=num_features, labels=(1,)
        )
        df = pd.DataFrame(exp.as_list(label=1), columns=["rule", "weight"])
        order = df["weight"].abs().sort_values(ascending=False).index
        return df.loc[order].reset_index(drop=True)


# ---------- Charts ----------
def plot_contributions(df, label_col, value_col, title, xlabel, top_n=10):
    d = df.head(top_n).iloc[::-1]
    colors = [RED if v > 0 else GREEN for v in d[value_col]]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(d) + 1.8))
    ax.barh(d[label_col], d[value_col], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(xlabel + "   (red = raises estimated risk, green = lowers it)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_global_importance(df, title="Overall feature importance (mean |SHAP|)", top_n=15):
    d = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(d) + 1.5))
    ax.barh(d["feature"], d["mean_abs_shap"], color="#2E86AB")
    ax.set_xlabel("Average absolute SHAP value")
    ax.set_title(title)
    fig.tight_layout()
    return fig