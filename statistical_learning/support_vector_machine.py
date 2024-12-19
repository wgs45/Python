from sklearn.svm import SVC
import numpy as np

# Function to input data with error handling


def input_data():
    n = int(input("Enter the number of training samples: "))
    X = []
    y = []
    print("Enter the features and labels for training data:")
    for i in range(n):
        while True:
            try:
                # Features input
                features = list(map(float, input(f"Enter features for sample {
                                i+1} (space-separated): ").split()))
                if len(features) == 0:
                    raise ValueError("Features cannot be empty.")
                break
            except ValueError as e:
                print(f"Invalid input. {e}. Please try again.")

        while True:
            try:
                # Label input
                label = int(input(f"Enter label for sample {i+1}: "))
                break
            except ValueError:
                print("Invalid label. Please enter an integer value.")

        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)


# Get training data
X_train, y_train = input_data()

# Train the SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# User input for prediction
while True:
    try:
        test_features = list(
            map(float, input("Enter features for prediction (space-separated): ").split()))
        if len(test_features) == 0:
            raise ValueError("Test features cannot be empty.")
        break
    except ValueError as e:
        print(f"Invalid input. {e}. Please try again.")

# Make prediction
prediction = model.predict([test_features])
print(f"Prediction: {prediction[0]}")
