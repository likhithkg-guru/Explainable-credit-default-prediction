import json
from functools import lru_cache
from validation import validate_row
import joblib
import pandas as pd

MODEL_LABELS = {
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


@lru_cache(maxsize=1)
def feature_columns():
    with open("models/feature_columns.json") as f:
        return json.load(f)


@lru_cache(maxsize=3)
def load_model(model_name):
    return joblib.load(f"models/{model_name}.joblib")


def build_row_dict(limit_bal, education, marriage, age, months):
    """months = list of 6 dicts, most recent month first:
    {"status": ..., "bill": ..., "paid": ...}"""
    row = {"LIMIT_BAL": limit_bal, "EDUCATION": education,
           "MARRIAGE": marriage, "AGE": age}
    for i, m in enumerate(months, start=1):
        row[f"PAY_{i}"] = m["status"]
        row[f"BILL_AMT{i}"] = m["bill"]
        row[f"PAY_AMT{i}"] = m["paid"]
    return row


def row_to_frame(row):
    return pd.DataFrame([row])[feature_columns()].astype(float)


def predict_scores(model_name, df):
    """Risk scores (not calibrated probabilities) for every row of df."""
    model = load_model(model_name)
    return model.predict_proba(df[feature_columns()].astype(float))[:, 1]


def risk_band(score):
    """Simple demonstration bands. They are NOT official credit categories."""
    if score < 0.30:
        return "Lower", "#2E9E5B"
    if score < 0.60:
        return "Moderate", "#E0A030"
    return "Higher", "#D64545"
MAX_ROWS = 10000


def band_array(scores):
    """Same cut-offs as risk_band(): Lower < 0.30 <= Moderate < 0.60 <= Higher."""
    return pd.cut(scores, bins=[-0.001, 0.30, 0.60, 1.001],
                  labels=["Lower", "Moderate", "Higher"], right=False)


def check_batch(df):
    """Validate every row of an uploaded CSV.
    Returns (valid_rows, problems_table, rows_with_warnings, missing_columns).
    Row numbers start at 1 for the first data row."""
    cols = feature_columns()
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        return None, pd.DataFrame(columns=["row", "problem"]), 0, missing_cols

    numeric = df[cols].apply(pd.to_numeric, errors="coerce")  # text -> missing
    numeric.index = numeric.index + 1

    # Same grouping of unknown category codes as preprocess.py
    for col, valid_codes, other in (("EDUCATION", [1, 2, 3, 4], 4),
                                    ("MARRIAGE", [1, 2, 3], 3)):
        bad = numeric[col].notna() & ~numeric[col].isin(valid_codes)
        numeric.loc[bad, col] = other

    problems, valid_rows, warning_rows = [], [], 0
    for row_no, record in zip(numeric.index, numeric.to_dict("records")):
        errors, warnings = validate_row(record)
        if errors:
            problems.extend({"row": row_no, "problem": e} for e in errors)
        else:
            valid_rows.append(row_no)
            if warnings:
                warning_rows += 1
    return (numeric.loc[valid_rows],
            pd.DataFrame(problems, columns=["row", "problem"]),
            warning_rows, [])