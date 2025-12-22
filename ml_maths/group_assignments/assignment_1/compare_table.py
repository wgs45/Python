import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import all necessary model classes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# --- Data Preparation Setup (Standardized process) ---
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


# --- Automated Model Comparison ---

# === All 9 models defined in a dictionary ===
# This dictionary maps a human-readable name to the initialized model object.
models = {
    # Using k=1 makes it prone to noise, but often high accuracy on small datasets
    "KNN (k=1)": KNeighborsClassifier(n_neighbors=1), 
    # Constraining the depth makes the tree simpler and less likely to overfit
    "Decision Tree (max_depth=2)": DecisionTreeClassifier(max_depth=2, random_state=42), 
    # High C implies less regularization
    "Logistic Regression (C=1e10)": LogisticRegression(C=10**10, max_iter=500, solver='liblinear'),
    "LDA": LinearDiscriminantAnalysis(),
    "GaussianNB": GaussianNB(),
    # Default SVC uses RBF kernel, C=1
    "SVC (RBF)": SVC(),
    "SVC Polynomial (deg=2, C=0.1)": SVC(kernel='poly', degree=2, C=0.1),
    "Random Forest": RandomForestClassifier(random_state=42),
    "MLPClassifier": MLPClassifier(max_iter=10000, random_state=42)
}

# A list to store results as dictionaries, which Pandas can easily convert to a DataFrame
rows = []

# Loop through the dictionary of models
for name, model in models.items():
    # Fit the current model on the training data
    model.fit(X_train, y_train)
    
    # Append the results (Model Name, Train Accuracy, Test Accuracy) to the rows list
    rows.append({
        "Model": name,
        "Train Accuracy": model.score(X_train, y_train),
        "Test Accuracy": model.score(X_test, y_test)
    })

# Convert the list of results into a Pandas DataFrame for easy analysis
comparison = pd.DataFrame(rows)

# Print the comparison table, sorted by the 'Test Accuracy' column in descending order
print(comparison.sort_values(by="Test Accuracy", ascending=False))

