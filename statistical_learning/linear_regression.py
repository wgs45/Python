import numpy as np


def linear_regression(X, y, learning_rate=0.01, iterations=1000):
    # Initialize parameters
    m = 0  # Slope
    b = 0  # Intercept
    n = len(X)  # Number of data points

    # Gradient Descent
    for _ in range(iterations):
        # Predictions
        y_pred = m * X + b

        # Calculate Gradients
        dm = (-2 / n) * np.sum(X * (y - y_pred))
        db = (-2 / n) * np.sum(y - y_pred)

        # Update Parameters
        m -= learning_rate * dm
        b -= learning_rate * db

    return m, b


def get_user_input():
    print("Enter the number of data points:")
    n = int(input())

    print(f"Enter {n} values for X (space-separated):")
    X = np.array(list(map(float, input().split())))

    print(f"Enter {n} values for y (space-separated):")
    y = np.array(list(map(float, input().split())))

    return X, y


# Main programs
if __name__ == "__main__":
    # Get user data
    X, y = get_user_input()

    # Ask for learning rate and iterations
    print("Enter the learning rate (e.g., 0.01):")
    learning_rate = float(input())

    print("Enter the number of iterations:")
    iterations = int(input())

    # Train the model
    m, b = linear_regression(X, y, learning_rate, iterations)

    # Predict and display results
    y_pred = m * X + b

    print("\nResults:")
    print(f"Slope (m): {m}")
    print(f"Intercept (b): {b}")
    print(f"Predicted y values: {y_pred}")
