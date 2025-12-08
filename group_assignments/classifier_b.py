from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import pandas as pd

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

X = df.iloc[:, 0:13].values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf_svc_poly = SVC(kernel='poly', degree=2, C=0.1)
clf_svc_poly.fit(X_train, y_train)

print("[SVC Polynomial (degree=2, C=0.1)]")
print(f"    Train Accuracy:    {clf_svc_poly.score(X_train, y_train):.4f}")
print(f"    Test  Accuracy:    {clf_svc_poly.score(X_test, y_test):.4f}")
print("-" * 30)


clf_rf = RandomForestClassifier(random_state=42)
clf_rf.fit(X_train, y_train)
print("[RandomForestClassifier]")
print(f"    Train Accuracy:    {clf_rf.score(X_train, y_train):.4f}")
print(f"    Test  Accuracy:    {clf_rf.score(X_test, y_test):.4f}")
print("-" * 30)

clf_mlp = MLPClassifier(max_iter=10000, random_state=42)
clf_mlp.fit(X_train, y_train)
print("[MLPClassifier]")
print(f"    Train Accuracy:    {clf_mlp.score(X_train, y_train):.4f}")
print(f"    Test  Accuracy:    {clf_mlp.score(X_test, y_test):.4f}")
print("-" * 30)
