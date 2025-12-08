from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 載入資料
wine = load_wine()
df3 = pd.DataFrame(wine.data, columns=wine.feature_names)
df3['target'] = wine.target

# X = 特徵欄位 (13 個)
X = df3.iloc[:, 0:13].values

# y = 目標欄位
y = df3['target'].values

# 切分訓練集 / 測試集 (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=== Shapes ===")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("\n=== y 分布 ===")
print("y_train:", pd.Series(y_train).value_counts().to_dict())
print("y_test :", pd.Series(y_test).value_counts().to_dict())

# （可選）特徵縮放
scale = True
if scale:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

