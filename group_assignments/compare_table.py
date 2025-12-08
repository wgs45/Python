import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Reload dataset
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target

X = df.iloc[:, :13].values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === All 9 models ===
models = {
    "KNN (k=1)": KNeighborsClassifier(n_neighbors=1),
    "Decision Tree (max_depth=2)": DecisionTreeClassifier(max_depth=2),
    "Logistic Regression (C=1e10)": LogisticRegression(C=10**10, max_iter=500, solver='liblinear'),
    "LDA": LinearDiscriminantAnalysis(),
    "GaussianNB": GaussianNB(),
    "SVC (RBF)": SVC(),
    "SVC Polynomial (deg=2, C=0.1)": SVC(kernel='poly', degree=2, C=0.1),
    "Random Forest": RandomForestClassifier(random_state=42),
    "MLPClassifier": MLPClassifier(max_iter=10000, random_state=42)
}

rows = []

for name, model in models.items():
    model.fit(X_train, y_train)
    rows.append({
        "Model": name,
        "Train Accuracy": model.score(X_train, y_train),
        "Test Accuracy": model.score(X_test, y_test)
    })

comparison = pd.DataFrame(rows)

print(comparison.sort_values(by="Test Accuracy", ascending=False))
