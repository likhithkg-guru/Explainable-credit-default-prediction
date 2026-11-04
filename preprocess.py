from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/default_of_credit_card_clients.xls")
TARGET = "default"

# ID is just a row number. SEX is dropped on purpose: using gender to
# estimate credit risk is a fairness problem, and it is not needed here.
DROP_COLUMNS = ["ID", "SEX"]


def load_raw(path=DATA_PATH):
    """Read the Excel file. The real header is on the second row."""
    df = pd.read_excel(path, header=1, engine="calamine") 
    df.columns = [str(c).strip() for c in df.columns]
    # The dataset names the first repayment column PAY_0; rename for consistency.
    df = df.rename(columns={
        "PAY_0": "PAY_1",
        "default payment next month": TARGET,
    })
    return df


def clean(df):
    """Fix undocumented category codes and handle missing values."""
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns]).copy()

    # EDUCATION is documented as 1-4. Other codes are grouped into 4 ("others").
    df["EDUCATION"] = df["EDUCATION"].where(df["EDUCATION"].isin([1, 2, 3, 4]), 4)
    # MARRIAGE is documented as 1-3. Other codes are grouped into 3 ("others").
    df["MARRIAGE"] = df["MARRIAGE"].where(df["MARRIAGE"].isin([1, 2, 3]), 3)

    # Defensive: fill any missing numeric values with the median.
    df = df.fillna(df.median(numeric_only=True))
    return df


def get_features_and_target(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    return X, y


def get_train_test(test_size=0.2, random_state=42):
    df = clean(load_raw())
    X, y = get_features_and_target(df)
    # stratify keeps the default/non-default ratio the same in both sets
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test()
    print("Train:", X_train.shape, "Test:", X_test.shape)
    print("Features:", list(X_train.columns))
    print("Default rate (train):", round(y_train.mean(), 3))