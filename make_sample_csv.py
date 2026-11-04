import pandas as pd

X = pd.read_csv("data/X_test.csv")
sample = X.sample(200, random_state=1).reset_index(drop=True)
sample.to_csv("data/sample_customers.csv", index=False)

# A smaller file with deliberate problems, to test validation
bad = sample.head(30).copy()
bad.loc[2, "LIMIT_BAL"] = None      # data row 3: missing value
bad.loc[4, "AGE"] = 15              # data row 5: age out of range
bad.loc[6, "PAY_1"] = 12            # data row 7: invalid repayment status
bad.loc[8, "LIMIT_BAL"] = 5000000   # data row 9: warning only
bad.to_csv("data/sample_customers_with_errors.csv", index=False)
print("Saved data/sample_customers.csv and data/sample_customers_with_errors.csv")