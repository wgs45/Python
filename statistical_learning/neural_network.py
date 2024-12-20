from sklearn.neural_network import MLPClassifier
import numpy as np

# Initialize the model
model = MLPClassifier(max_iter=1000)

# User input
# Example: [[1, 2], [3, 4]]
X = np.array(eval(input("Enter training data (X) as a 2D list: ")))
# Example: [0, 1]
y = np.array(eval(input("Enter target labels (y) as a list: ")))

# Fit the model
model.fit(X, y)

# Test input
# Example: [[3, 4]]
test_data = np.array(eval(input("Enter test data as a 2D list: ")))
predictions = model.predict(test_data)

print("Predictions:", predictions)
