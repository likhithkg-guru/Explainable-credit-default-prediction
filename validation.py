from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=1)
def training_ranges():
    X = pd.read_csv("data/X_train.csv")
    return {c: (float(X[c].min()), float(X[c].max())) for c in X.columns}


def validate_row(row):
    """Returns (errors, warnings). Errors block prediction; warnings don't.
    Passing these checks means the format is plausible, NOT that it is truthful."""
    errors, warnings = [], []

    # 1. Missing or non-numeric values
    for name, value in row.items():
        if pd.isna(value):
            errors.append(f"{name} is missing.")
            continue
        try:
            float(value)
        except (TypeError, ValueError):
            errors.append(f"{name} is not a valid number.")
    if errors:
        return errors, warnings

    # 2. Valid ranges
    if row["LIMIT_BAL"] <= 0:
        errors.append("Credit limit must be greater than 0.")
    if not (18 <= row["AGE"] <= 100):
        errors.append("Age must be between 18 and 100.")
    for i in range(1, 7):
        s = row[f"PAY_{i}"]
        if float(s) != int(s) or not (-2 <= s <= 8):
            errors.append(f"Month {i} repayment status must be a whole number from -2 to 8.")
        if row[f"PAY_AMT{i}"] < 0:
            errors.append(f"Month {i} amount paid cannot be negative.")
    if errors:
        return errors, warnings

    # 3. Unusual-value warnings: outside what the model saw in training
    for name, (lo, hi) in training_ranges().items():
        if not (lo <= row[name] <= hi):
            warnings.append(
                f"{name} = {row[name]:,.0f} is outside the range the model was "
                f"trained on ({lo:,.0f} to {hi:,.0f}). The estimate may be unreliable."
            )
    return errors, warnings