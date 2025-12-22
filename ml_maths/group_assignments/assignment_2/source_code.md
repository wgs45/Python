# Data Generation

```py
import numpy as np

# Initializes the random number generator to ensure the same
# "random" values are generated every time the script runs.
np.random.seed(42)

def generate_xor(n_samples=1000, noise=0.1):
    # Generates a (n_samples, 2) array of 0s and 1s
    X = np.random.randint(0, 2, (n_samples, 2))

    # Applies XOR logic: 1 if inputs are different, 0 if they are the same
    y = (X[:, 0] ^ X[:, 1]).reshape(-1, 1)

    # Adds small random variations (noise) so the points aren't exactly on 0 or 1
    X = X + noise * np.random.randn(n_samples, 2)
    return X, y

def generate_quadrant(train_size=4000, test_size=1000):
    def create_data(n):
        # Creates random points with coordinates between -1 and 1
        X = np.random.uniform(-1, 1, (n, 2))
        y = np.zeros((n, 1))

        # Assigns class 1 if points are in the top-right or bottom-left quadrants
        # (where both x and y have the same sign), otherwise class 0
        y[(X[:, 0] * X[:, 1]) > 0] = 1
        return X, y

    # Generates separate datasets for training and testing
    X_train, y_train = create_data(train_size)
    X_test, y_test = create_data(test_size)

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    # Create the XOR-pattern data
    X_xor, y_xor = generate_xor()

    # Create the quadrant-pattern data split into training and testing sets
    X_train, y_train, X_test, y_test = generate_quadrant()

    # Output the dimensions of the generated arrays to verify counts
    print("XOR:", X_xor.shape, y_xor.shape)
    print("Train:", X_train.shape, y_train.shape)
    print("Test:", X_test.shape, y_test.shape)
```

## Result

```txt
XOR: (1000, 2) (1000, 1)
Train: (4000, 2) (4000, 1)
Test: (1000, 2) (1000, 1)
```

---

# Visualization and Validation

```py
import numpy as np
import matplotlib.pyplot as plt

def visualize_xor(X, y):
    plt.figure()
    # Plot points where y is 0 as circles ('o')
    plt.scatter(X[y[:, 0] == 0, 0], X[y[:, 0] == 0, 1], marker='o', label='Class 0')
    # Plot points where y is 1 as crosses ('x')
    plt.scatter(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], marker='x', label='Class 1')
    plt.title("XOR Dataset")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend() # Show the labels for Class 0 and Class 1
    plt.show()

def visualize_quadrant(X, y, title="Quadrant Dataset"):
    plt.figure()
    # Separate and plot the two classes based on their labels (0 or 1)
    plt.scatter(X[y[:, 0] == 0, 0], X[y[:, 0] == 0, 1], marker='o', label='Class 0')
    plt.scatter(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], marker='x', label='Class 1')
    # Draw horizontal and vertical lines at 0 to clearly show the four quadrants
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.show()

def validate_data(X, y, name="Dataset"):
    # Print summary statistics to ensure the data was generated correctly
    print(f"{name} validation:")
    print("X shape:", X.shape) # Number of samples and features
    print("y shape:", y.shape) # Number of labels
    # Counts how many samples belong to each class (ideally ~50/50 split)
    print("Class distribution:", np.unique(y, return_counts=True))
    print("-" * 30)

if __name__ == "__main__":
    # Generate the data using the functions from your previous script
    X_xor, y_xor = generate_xor()
    X_train, y_train, X_test, y_test = generate_quadrant()

    # Print the shapes and class counts for each dataset
    validate_data(X_xor, y_xor, "XOR")
    validate_data(X_train, y_train, "Quadrant Train")
    validate_data(X_test, y_test, "Quadrant Test")

    # Display the plots
    visualize_xor(X_xor, y_xor)
    visualize_quadrant(X_train, y_train, "Quadrant Training Data")
```

## Result

```txt
XOR validation:
X shape: (1000, 2)
y shape: (1000, 1)
Class distribution: (array([0, 1]), array([473, 527]))
------------------------------
Quadrant Train validation:
X shape: (4000, 2)
y shape: (4000, 1)
Class distribution: (array([0., 1.]), array([1971, 2029]))
------------------------------
Quadrant Test validation:
X shape: (1000, 2)
y shape: (1000, 1)
Class distribution: (array([0., 1.]), array([489, 511]))
------------------------------
```

---

# Neural Network Structure

```py
# Activation function that squashes values between 0 and 1
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# The mathematical derivative of sigmoid; used during backpropagation to calculate gradients
def sigmoid_deriv(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Converts raw scores into probabilities that sum to 1 (used for multi-class classification)
def softmax(z):
    # Subtracting np.max(z) is a numerical stability trick to prevent overflow errors
    exp = np.exp(z - np.max(z))
    return exp / np.sum(exp, axis=1, keepdims=True)

# Randomly initializes the weights and biases for a 2-layer neural network
def init_weights(input_dim, hidden_dim, output_dim):
    # Small random weights (multiplied by 0.01) prevent gradients from exploding
    W1 = np.random.randn(input_dim, hidden_dim) * 0.01
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, output_dim) * 0.01
    b2 = np.zeros((1, output_dim))
    return W1, b1, W2, b2

# Passes input data through the network layers to get a prediction
def forward(X, W1, b1, W2, b2, use_softmax=False):
    # Layer 1: Linear transformation (X @ W1 + b1) followed by sigmoid activation
    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)

    # Layer 2: Second linear transformation
    Z2 = A1 @ W2 + b2
    # Choose Softmax for multi-class or Sigmoid for binary classification/regression
    if use_softmax:
        A2 = softmax(Z2)
    else:
        A2 = sigmoid(Z2)

    return Z1, A1, Z2, A2

# Initialize a network with 2 inputs, 6 hidden neurons, and 1 output neuron
W1, b1, W2, b2 = init_weights(2, 6, 1)

# Run the XOR data through the network once
Z1, A1, Z2, A2 = forward(X_xor, W1, b1, W2, b2)

# Print results: A1 is the hidden layer output, A2 is the final prediction
print("A1 shape:", A1.shape) # Should be (n_samples, 6)
print("A2 shape:", A2.shape) # Should be (n_samples, 1)
```

## Result

```txt
A1 shape: (1000, 6)
A2 shape: (1000, 1)
```

---

# Shape verification test

```py
import numpy as np

# Define the architecture dimensions
input_size = 4    # Number of features per input sample
hidden_size = 5   # Number of neurons in the hidden layer
output_size = 3   # Number of output nodes (e.g., for 3 classes)
batch_size = 10   # Number of samples processed at once

# Initialize weights and biases with random values
W1 = np.random.randn(input_size, hidden_size)
b1 = np.random.randn(hidden_size)
W2 = np.random.randn(hidden_size, output_size)
b2 = np.random.randn(output_size)

# Create a dummy batch of input data
X = np.random.randn(batch_size, input_size)

# ReLU activation: turns negative values to 0, keeps positive values as is
def relu(z):
    return np.maximum(0, z)

# --- Forward propagation ---
# Layer 1: Matrix multiplication of input and weights, then add bias
Z1 = X @ W1 + b1
# Apply ReLU activation to the first layer's output
A1 = relu(Z1)
# Layer 2: Matrix multiplication of hidden layer output and second weights, then add bias
Z2 = A1 @ W2 + b2
# Final output (linear activation in this case)
A2 = Z2

def shape_verification():
    # 'assert' checks if a condition is true; if not, it stops the program with an error message.
    # These checks ensure that matrix dimensions align correctly for dot products.
    assert X.shape == (batch_size, input_size), f"X shape incorrect: {X.shape}"
    assert W1.shape == (input_size, hidden_size), f"W1 shape incorrect: {W1.shape}"
    assert b1.shape == (hidden_size,), f"b1 shape incorrect: {b1.shape}"

    # After X(10,4) @ W1(4,5), the result should be (10,5)
    assert Z1.shape == (batch_size, hidden_size), f"Z1 shape incorrect: {Z1.shape}"
    assert A1.shape == (batch_size, hidden_size), f"A1 shape incorrect: {A1.shape}"

    assert W2.shape == (hidden_size, output_size), f"W2 shape incorrect: {W2.shape}"
    assert b2.shape == (output_size,), f"b2 shape incorrect: {b2.shape}"

    # After A1(10,5) @ W2(5,3), the final result should be (10,3)
    assert Z2.shape == (batch_size, output_size), f"Z2 shape incorrect: {Z2.shape}"
    assert A2.shape == (batch_size, output_size), f"A2 shape incorrect: {A2.shape}"

    print("✅ Shape verification passed for all layers")

# Execute the verification function
shape_verification()
```

---

# Backpropagation (Gradient checking and validation)

```py
# Calculates Binary Cross-Entropy loss (measures how far predictions A2 are from labels Y)
def compute_loss(Y, A2):
    m = Y.shape[0]
    # 1e-8 is added to prevent log(0) which would result in NaN (Not a Number)
    cost = -np.mean(Y * np.log(A2 + 1e-8) + (1 - Y) * np.log(1 - A2 + 1e-8))
    return cost

# Backpropagation: Calculates gradients (slopes) to determine how to change weights to reduce loss
def backward(X, Y, Z1, A1, Z2, A2, W2):
    m = X.shape[0] # Number of training samples

    # Calculate error at the output layer
    dZ2 = A2 - Y
    # Gradient for second layer weights and biases (average over samples)
    dW2 = (1 / m) * (A1.T @ dZ2)
    db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

    # Calculate error at the hidden layer by "chaining" back from dZ2
    # Multiplied by the sigmoid derivative to account for the activation function
    dZ1 = (dZ2 @ W2.T) * (A1 * (1 - A1))
    # Gradient for first layer weights and biases
    dW1 = (1 / m) * (X.T @ dZ1)
    db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2

# Optimizer: Adjusts weights and biases by subtracting a portion (learning rate) of the gradient
def update(W1, b1, W2, b2, dW1, db1, dW2, db2, lr=0.1):
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    return W1, b1, W2, b2

# --- Execution Flow ---
# 1. Initialize network
W1, b1, W2, b2 = init_weights(2, 6, 1)

# 2. Forward pass: get predictions
Z1, A1, Z2, A2 = forward(X_xor, W1, b1, W2, b2)

# 3. Compute error (loss)
loss = compute_loss(y_xor, A2)

# 4. Backward pass: find gradients
dW1, db1, dW2, db2 = backward(X_xor, y_xor, Z1, A1, Z2, A2, W2)

# 5. Update weights to improve the model
W1, b1, W2, b2 = update(W1, b1, W2, b2, dW1, db1, dW2, db2)

print("Loss:", loss)
print("Updated W1:", W1.shape)
print("Updated b1:", b1.shape)
print("Updated W2:", W2.shape)
print("Updated b2:", b2.shape)

# Ensures that the calculated gradients match the dimensions of the weights they are meant to update
def verify_grads(W1, b1, W2, b2, dW1, db1, dW2, db2):
    assert dW1.shape == W1.shape, "dW1 shape error"
    assert db1.shape == b1.shape, "db1 shape error"
    assert dW2.shape == W2.shape, "dW2 shape error"
    assert db2.shape == b2.shape, "db2 shape error"

    print("Gradient check passed")

verify_grads(W1, b1, W2, b2, dW1, db1, dW2, db2)
```

## Result

```txt
Loss: 0.6936362139679347
Updated W1: (2, 6)
Updated b1: (1, 6)
Updated W2: (6, 1)
Updated b2: (1, 1)
Gradient check passed
```

---

# Training Loop

```py
import numpy as np

# ReLU Activation: Outputs input if positive, otherwise zero
def relu(z):
    return np.maximum(0, z)

# Forward pass specifically using ReLU for the hidden layer and Sigmoid for output
def forward_relu_hidden(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = 1 / (1 + np.exp(-Z2)) # Sigmoid output
    return Z1, A1, Z2, A2

# Initialize hyperparameters and data dimensions
X_train, y_train, X_test, y_test = generate_quadrant(train_size=4000, test_size=1000)
hidden_dim  = 6
epochs      = 2200
lr          = 0.01
batch_size  = 128
eps         = 1e-8 # Constant to prevent log(0) errors

input_dim  = X_train.shape[1]
output_dim = y_train.shape[1]

# He Initialization: Optimized weight setup for ReLU layers to prevent vanishing gradients
W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim) * 0.01
b2 = np.zeros((1, output_dim))

train_loss_hist = []
test_loss_hist  = []
N = X_train.shape[0]

# --- Main Training Loop ---
for epoch in range(epochs):
    # (a) Shuffle indices every epoch to ensure the model doesn't learn the order of data
    idx = np.random.permutation(N)
    Xs = X_train[idx]
    ys = y_train[idx]

    # (b) Mini-batch Gradient Descent: process small chunks of data at a time
    for start in range(0, N, batch_size):
        end = start + batch_size
        Xb = Xs[start:end]
        yb = ys[start:end]
        m  = Xb.shape[0]

        # 1. Forward Pass
        Z1, A1, Z2, A2 = forward_relu_hidden(Xb, W1, b1, W2, b2)

        # 2. Loss Calculation with clipping to keep values between [eps, 1-eps]
        A2_clip = np.clip(A2, eps, 1 - eps)
        loss = -np.mean(yb * np.log(A2_clip) + (1 - yb) * np.log(1 - A2_clip))

        # 3. Backward Pass (Output Layer)
        dZ2 = (A2 - yb)
        dW2 = (1 / m) * (A1.T @ dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        # 4. Backward Pass (Hidden Layer)
        # Note: (Z1 > 0) is the derivative of ReLU
        dZ1 = (dZ2 @ W2.T) * (Z1 > 0)
        dW1 = (1 / m) * (Xb.T @ dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        # 5. Parameter Update
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    # Periodically evaluate performance on the full training and testing sets
    if epoch % 200 == 0:
        _, _, _, A2_tr = forward_relu_hidden(X_train, W1, b1, W2, b2)
        tr_loss = -np.mean(y_train * np.log(np.clip(A2_tr, eps, 1-eps)) + (1-y_train) * np.log(1-np.clip(A2_tr, eps, 1-eps)))

        _, _, _, A2_te = forward_relu_hidden(X_test, W1, b1, W2, b2)
        te_loss = -np.mean(y_test * np.log(np.clip(A2_te, eps, 1-eps)) + (1-y_test) * np.log(1-np.clip(A2_te, eps, 1-eps)))

        train_loss_hist.append(tr_loss)
        test_loss_hist.append(te_loss)

        print(f"Epoch {epoch:4d} | train_loss={tr_loss:.4f} | test_loss={te_loss:.4f}")
```

## Result

```txt
Epoch    0 | train_loss=0.6865 | test_loss=0.6857
Epoch  200 | train_loss=0.2691 | test_loss=0.2617
Epoch  400 | train_loss=0.1838 | test_loss=0.1787
Epoch  600 | train_loss=0.1460 | test_loss=0.1414
Epoch  800 | train_loss=0.1215 | test_loss=0.1176
Epoch 1000 | train_loss=0.0992 | test_loss=0.0963
Epoch 1200 | train_loss=0.0819 | test_loss=0.0801
Epoch 1400 | train_loss=0.0697 | test_loss=0.0690
Epoch 1600 | train_loss=0.0611 | test_loss=0.0612
Epoch 1800 | train_loss=0.0550 | test_loss=0.0557
Epoch 2000 | train_loss=0.0504 | test_loss=0.0517
```

---

# Training visualization

```py
import matplotlib.pyplot as plt
import numpy as np

epochs_recorded = np.arange(len(train_loss_hist)) * 200

plt.figure(figsize=(6, 4))
plt.plot(epochs_recorded, train_loss_hist, label="Train Loss")
plt.plot(epochs_recorded, test_loss_hist, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Training / Testing Loss Curve")
plt.legend()
plt.grid(True)
plt.show()
```

## Result

```txt
Graph
```

---

# Testing and Visualization (Accuracy, prediction and Drawing Decision Boundaries)

```py
import numpy as np

# Network Architecture
input_size = 4
hidden_size = 5
output_size = 3

# Randomly initialize weights and biases for a two-layer network
W1 = np.random.randn(input_size, hidden_size)
b1 = np.random.randn(hidden_size)
W2 = np.random.randn(hidden_size, output_size)
b2 = np.random.randn(output_size)

def predict(X):
    # Layer 1: Linear transformation followed by ReLU activation
    Z1 = X @ W1 + b1
    A1 = np.maximum(0, Z1)

    # Layer 2: Final linear transformation
    Z2 = A1 @ W2 + b2

    # Binary classification threshold: values > 0.5 become 1, others become 0
    return (Z2 > 0.5).astype(int)

def compute_accuracy(y_true, y_pred):
    # Calculates the percentage of correct predictions (0.0 to 1.0)
    return np.mean(y_true == y_pred)

# Generate synthetic test data (100 samples) and random binary ground-truth labels
X_test = np.random.randn(100, input_size)
y_test = np.random.randint(0, 2, (100, output_size))

# Run the test data through the network to get predictions
y_pred = predict(X_test)
# Compare predictions against the actual labels
acc = compute_accuracy(y_test, y_pred)

print("Test Accuracy:", acc)
# Display the first 5 prediction results
print(y_pred[:5])
```

## Result

```txt
Test Accuracy: 0.5066666666666667
[[1 0 0]
 [1 1 1]
 [0 0 0]
 [1 0 0]
 [0 0 0]]
```
