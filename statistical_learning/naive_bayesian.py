from sklearn.naive_bayes import GaussianNB
import numpy as np

# Initialize the model
model = GaussianNB()

# User input
# Example: [[1, 2], [2, 3]]
X = np.array(eval(input("Enter training data (X) as a 2D list: ")))
# Example: [0, 1]
y = np.array(eval(input("Enter target labels (y) as a list: ")))

# Fit the model
model.fit(X, y)

# Test input
# Example: [[2, 3]]
test_data = np.array(eval(input("Enter test data as a 2D list: ")))
predictions = model.predict(test_data)

print("Predictions:", predictions)
