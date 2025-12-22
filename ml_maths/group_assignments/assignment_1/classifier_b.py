# Import necessary libraries and models
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier  # Ensemble model
from sklearn.neural_network import MLPClassifier    # Neural Network model
from sklearn.svm import SVC                         # Support Vector Classifier
import pandas as pd

# --- Data Preparation Setup (Standardized process) ---

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

X = df.iloc[:, 0:13].values
y = df['target'].values

# Split data into training and testing sets (80/20 split, stratified, reproducible)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features (crucial for SVM and MLP)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --- Advanced Model Building and Evaluation ---

# Initialize, fit, and score Support Vector Classifier with a Polynomial Kernel
# Kernel tricks help classify data that isn't linearly separable.
# C=0.1 is a regularization parameter.
clf_svc_poly = SVC(kernel='poly', degree=2, C=0.1)
clf_svc_poly.fit(X_train, y_train)

print("[SVC Polynomial (degree=2, C=0.1)]")
# Evaluate training and test accuracy using the built-in .score() method
print(f"    Train Accuracy:    {clf_svc_poly.score(X_train, y_train):.4f}")
print(f"    Test  Accuracy:    {clf_svc_poly.score(X_test, y_test):.4f}")
print("-" * 30)


# Initialize, fit, and score a Random Forest Classifier
# Random Forests are powerful ensemble methods based on decision trees.
clf_rf = RandomForestClassifier(random_state=42)
clf_rf.fit(X_train, y_train)
print("[RandomForestClassifier]")
print(f"    Train Accuracy:    {clf_rf.score(X_train, y_train):.4f}")
print(f"    Test  Accuracy:    {clf_rf.score(X_test, y_test):.4f}")
print("-" * 30)

# Initialize, fit, and score a Multi-Layer Perceptron (Neural Network)
# max_iter is increased to 10000 to ensure the network has enough steps to converge.
clf_mlp = MLPClassifier(max_iter=10000, random_state=42)
clf_mlp.fit(X_train, y_train)
print("[MLPClassifier]")
print(f"    Train Accuracy:    {clf_mlp.score(X_train, y_train):.4f}")
print(f"    Test  Accuracy:    {clf_mlp.score(X_test, y_test):.4f}")
print("-" * 30)

