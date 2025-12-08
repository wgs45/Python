from sklearn.datasets import load_wine
import pandas as pd

wine = load_wine()
df2 = pd.DataFrame(wine.data, columns=wine.feature_names)
df2['target'] = wine.target

print("=== First 20 rows ===")
print(df2.head(20))

print("\n=== Shape of the dataset ===")
print(df2.shape)

print("\n=== Descriptive statistics ===")
print(df2.describe())

print("\n=== Unique classes ===")
print(df2['target'].unique())

