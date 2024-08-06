import numpy as np
from collections import Counter


def load_dataset(filename):
    data = np.genfromtxt(filename, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y


def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))


def k_nearest_neighbors(X_train, y_train, X_test, k):
    predictions = []
    for test_instance in X_test:
        distances = [manhattan_distance(test_instance, train_instance) for train_instance in X_train]
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = y_train[nearest_indices]
        predicted_class = Counter(nearest_labels).most_common(1)[0][0]
        predictions.append(predicted_class)
    return predictions


def accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total


def confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label, pred_label] += 1
    return matrix


def evaluate_knn(train_filename, test_filename, k_values):
    X_train, y_train = load_dataset(train_filename)
    X_test, y_test = load_dataset(test_filename)
    num_classes = len(np.unique(y_train))

    for k in k_values:
        predictions = k_nearest_neighbors(X_train, y_train, X_test, k)
        acc = accuracy(y_test, predictions)
        cm = confusion_matrix(y_test, predictions, num_classes)

        print(f"\nResults for k = {k}:")
        print("Accuracy:", acc)
        print("Confusion Matrix:")
        print(cm)


# Specify the filenames and k values for each dataset with full paths
datasets = [
    ("iris/iris.trn", "iris/iris.tst"),
    ("optics/opt.trn", "optics/opt.tst"),
    ("letter/let.trn", "letter/let.tst"),
    ("faces/data.trn", "faces/data.tst"),
    ("fp/fp.trn", "fp/fp.tst")
]

k_values = [1, 3, 5]  # Add more k values if needed

# Evaluate kNN on each dataset
for train_filename, test_filename in datasets:
    print(f"\nDataset: {train_filename}")
    evaluate_knn(train_filename, test_filename, k_values)
