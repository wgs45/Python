from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# --- Data Preparation Setup (Same as previous snippets) ---

# Declare variables manually (to ensure the code runs standalone)
wine = load_wine()
df4 = pd.DataFrame(wine.data, columns=wine.feature_names)
df4['target'] = wine.target

X = df4.iloc[:, 0:13].values
y = df4['target'].values

# Split data into training and testing sets (80/20 split, stratified, reproducible)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --- Model Building and Evaluation ---

# Import necessary classification models and evaluation metrics from scikit-learn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=== KNN Classifier ===")
# Initialize the K-Nearest Neighbors model with K=5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)
# Train the model using the training data
knn.fit(X_train, y_train)
# Make predictions on the unseen test data
y_pred_knn = knn.predict(X_test)
# Evaluate and print performance metrics
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("\n")

print("=== Decision Tree Classifier ===")
# Initialize the Decision Tree model (random_state ensures reproducibility)
dt = DecisionTreeClassifier(random_state=42)
# Train the model
dt.fit(X_train, y_train)
# Make predictions
y_pred_dt = dt.predict(X_test)
# Evaluate and print performance metrics
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("\n")

print("=== Logistic Regression ===")
# Initialize Logistic Regression. max_iter increased to ensure convergence if needed.
# solver='liblinear' is often efficient for smaller datasets like this one.
lr = LogisticRegression(max_iter=500, solver='liblinear')
# Train the model
lr.fit(X_train, y_train)
# Make predictions
y_pred_lr = lr.predict(X_test)
# Evaluate and print performance metrics
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("\n")

# Comparison Table
results = {
    "Model": ["KNN", "Decision Tree", "Logistic Regression"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_knn),
        accuracy_score(y_test, y_pred_dt),
        accuracy_score(y_test, y_pred_lr)
    ]
}

print("=== Model Accuracy Comparison ===")
print(pd.DataFrame(results))

# --- Additional Models Added in the Snippet ---

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Initialize, fit, and score Linear Discriminant Analysis (LDA)
clf_lda = LinearDiscriminantAnalysis()
clf_lda.fit(X_train, y_train)
print("[LDA]")
# Using .score() gives a quick accuracy metric
print(f"  Train Accuracy: {clf_lda.score(X_train, y_train):.4f}")
print(f"  Test Accuracy:  {clf_lda.score(X_test, y_test):.4f}")
print("-" * 30)

# Initialize, fit, and score Gaussian Naive Bayes (GNB)
clf_gnb = GaussianNB()
clf_gnb.fit(X_train, y_train)
print("[GaussianNB]")
print(f"  Train Accuracy: {clf_gnb.score(X_train, y_train):.4f}")
print(f"  Test Accuracy:  {clf_gnb.score(X_test, y_test):.4f}")
print("-" * 30)

# Initialize, fit, and score Support Vector Classifier (SVC)
clf_svc = SVC() # Uses default Radial Basis Function (RBF) kernel
clf_svc.fit(X_train, y_train)
print("[SVC (RBF)]")
print(f"  Train Accuracy: {clf_svc.score(X_train, y_train):.4f}")
print(f"  Test Accuracy:  {clf_svc.score(X_test, y_test):.4f}")
print("-" * 30)

