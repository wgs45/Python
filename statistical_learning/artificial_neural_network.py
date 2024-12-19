from keras.models import Sequential
from keras.layers import Dense
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

# Build the ANN model
model = Sequential()
# First layer
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Output layer

# Compile and train the model
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, verbose=0)

# User input for prediction
test_features = list(
    map(float, input("Enter features for prediction (space-separated): ").split()))
prediction = model.predict([test_features])
print(f"Prediction: {prediction[0][0]}")
