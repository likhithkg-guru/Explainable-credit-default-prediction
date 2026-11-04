import pandas as pd

df = pd.read_excel(path, header=1, engine="calamine") 

print("Shape:", df.shape)
print("\nColumns:\n", list(df.columns))
print("\nMissing values per column:\n", df.isnull().sum().sum(), "total")
print("\nTarget distribution:")
print(df.iloc[:, -1].value_counts(normalize=True))
print("\nEDUCATION values:", sorted(df["EDUCATION"].unique()))
print("MARRIAGE values:", sorted(df["MARRIAGE"].unique()))
print("\nFirst rows:")
print(df.head())