import numpy as np

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def k_nearest_neighbors(X_train, y_train, X_test, k):
    predictions = []
    for test_instance in X_test:
        distances = [manhattan_distance(test_instance, train_instance) for train_instance in X_train]
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = y_train[nearest_indices]
        predicted_class = np.bincount(nearest_labels).argmax()
        predictions.append(predicted_class)
    return predictions

# Training data
X_train = np.array([
    [0.376, 0.488],
    [0.312, 0.544],
    [0.298, 0.624],
    [0.394, 0.6],
    [0.506, 0.512],
    [0.488, 0.334],
    [0.478, 0.398],
    [0.606, 0.366],
    [0.428, 0.294],
    [0.542, 0.252]
])
y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Test data
X_test = np.array([
    [0.55, 0.364],
    [0.558, 0.47],
    [0.456, 0.45],
    [0.45, 0.57]
])

# Number of neighbors to consider (k value)
k_neighbors = 3

# Find k-nearest neighbors
predictions = k_nearest_neighbors(X_train, y_train, X_test, k_neighbors)

# Print the predicted class for each test instance
for i, pred_class in enumerate(predictions):
    print(f"Test instance {i + 1}: Predicted Class = {pred_class}")
