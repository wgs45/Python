# Import the necessary library to load the wine dataset from scikit-learn
from sklearn.datasets import load_wine

# Import the pandas library for data manipulation and analysis
import pandas as pd

# Load the 'wine' dataset.
wine = load_wine()

# Convert the raw feature data into a Pandas DataFrame.
# Crucially, this time we use the 'columns' parameter to assign
# human-readable feature names (like 'alcohol', 'malic_acid') directly from the dataset object,
# making the DataFrame much easier to interpret.
df2 = pd.DataFrame(wine.data, columns=wine.feature_names)

# Add the target variable (the type of wine: class 0, 1, or 2) as a new column named 'target'.
df2['target'] = wine.target

# Display the first 20 rows of the DataFrame (instead of the default 5).
# This provides a more extensive visual check of the named columns and data types.
print("=== First 20 rows ===")
print(df2.head(20))

# Print the total number of rows and columns (e.g., (178 rows, 14 columns)).
print("\n=== Shape of the dataset ===")
print(df2.shape)

# Generate and print descriptive (summary) statistics for all numerical columns.
# This includes count, mean, standard deviation, min/max values, and quartiles,
# which gives a quick snapshot of the data distribution.
print("\n=== Descriptive statistics ===")
print(df2.describe())

# Print the unique values present in the 'target' column.
# This confirms the number of distinct classes we are trying to predict (0, 1, and 2).
print("\n=== Unique classes ===")
print(df2['target'].unique())

