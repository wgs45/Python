# Import the necessary library to load the wine dataset from scikit-learn
from sklearn.datasets import load_wine

# Import the pandas library, which is essential for data manipulation and analysis in Python
import pandas as pd

# Load the 'wine' dataset. This function returns a Bunch object, which acts like a dictionary
# and contains the data (features), target labels, feature names, and description.
wine = load_wine()

# Convert the raw feature data (which is a NumPy array within the 'wine' object) into a
# Pandas DataFrame for easier manipulation and viewing.
df = pd.DataFrame(wine.data)

# Add the target variable (the type of wine: class 0, 1, or 2) as a new column named 'target'
# to the DataFrame.
df['target'] = wine.target

# Display the first 5 rows of the DataFrame to get a quick visual check of the data structure,
# the features, and the newly added 'target' column.
print(df.head())

# Check for missing values (NaN, null, etc.) in each column of the DataFrame.
# .isnull() creates a boolean mask, and .sum() counts the 'True' values (missing entries)
# per column. This helps identify data cleaning needs.
print(df.isnull().sum())

# Print the total number of columns in the DataFrame. The .shape attribute returns a tuple
# of (rows, columns), so accessing index 1 gives the column count.
print(df.shape[1])

