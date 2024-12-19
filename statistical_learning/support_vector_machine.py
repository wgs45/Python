from sklearn.svm import SVC
import numpy as np


def input_data():
    n = int(input("Enter the number of training samples: "))
    X = []
    y = []
    print("Enter the features and labels for training data:")
    for i in range(n):
        features = list(map(float, input(f"Enter features for sample {
                        i+1} (space-separated): ").split()))
        label = int(input(f"Enter label for sample {i+1}: "))
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)


# Get training data
X_train, y_train = input_data()

# Train the SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# User input for prediction
test_features = list(
    map(float, input("Enter features for prediction (space-separated): ").split()))
prediction = model.predict([test_features])
print(f"Prediction: {prediction[0]}")
