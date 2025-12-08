from sklearn.datasets import load_wine
import pandas as pd

wine = load_wine()
df = pd.DataFrame(wine.data)
df['target'] = wine.target

print(df.head())
print(df.isnull().sum())
print(df.shape[1])
