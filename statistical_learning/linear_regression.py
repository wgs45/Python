from sklearn.linear_model import LinearRegression
import numpy as np

# Initialize the model
model = LinearRegression()

# User input
# Example: [[1], [2]]
X = np.array(eval(input("Enter training data (X) as a 2D list: ")))
# Example: [3, 5]
y = np.array(eval(input("Enter target values (y) as a list: ")))

# Fit the model
model.fit(X, y)

# Test input
test_data = np.array(
    eval(input("Enter test data as a 2D list: ")))  # Example: [[3]]
predictions = model.predict(test_data)

print("Predictions:", predictions)
