# Import the necessary libraries
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load Data
wine = load_wine()
# Create a DataFrame using meaningful feature names
df3 = pd.DataFrame(wine.data, columns=wine.feature_names)
df3['target'] = wine.target

# Define Features (X) and Target (y)

# X = Feature columns (13 features). .iloc selects all rows (:) and columns 0 through 12.
X = df3.iloc[:, 0:13].values
# Note: wine.data would also work directly if you don't need the DataFrame structure for X

# y = Target column (the class label). .values extracts the underlying NumPy array.
y = df3['target'].values

# Split Training / Testing sets (80/20 split)
# This is crucial for evaluating model performance on unseen data.
X_train, X_test, y_train, y_test = train_test_split(
    X,           # The features to split
    y,           # The target variable to split
    test_size=0.2, # Allocate 20% of data to the test set
    random_state=42, # Ensures the split is the same every time the code runs (reproducibility)
    stratify=y   # Ensures the class distribution in y is maintained in both train and test sets
)

print("=== Shapes ===")
print("X_train:", X_train.shape) # Should be (142 samples, 13 features)
print("X_test :", X_test.shape)  # Should be (36 samples, 13 features)

print("\n=== y Distribution ===")
# Verify that the 'stratify' parameter worked correctly by printing counts for each class (0, 1, 2)
print("y_train:", pd.Series(y_train).value_counts().to_dict())
print("y_test :", pd.Series(y_test).value_counts().to_dict())

# (Optional but recommended) Feature Scaling
scale = True
if scale:
    # Initialize the StandardScaler
    scaler = StandardScaler()
    
    # Fit the scaler ONLY on the training data and then transform the training data.
    X_train = scaler.fit_transform(X_train)
    
    # Transform the test data using the *same* scaling parameters learned from the training data.
    # We do NOT call .fit() on X_test to prevent data leakage.
    X_test = scaler.transform(X_test)

