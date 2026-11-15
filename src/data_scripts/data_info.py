import pandas as pd

path = "data/raw/kaggle/transactions_data.csv"
df = pd.read_csv(path, nrows=5)
print("Colonnes :", list(df.columns))
print(df.head())
print("\nTypes :")
print(df.dtypes)