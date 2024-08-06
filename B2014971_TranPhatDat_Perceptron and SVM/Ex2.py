import numpy as np
import os

def load_data(filename):
    data = np.loadtxt(filename, delimiter=",")
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def perceptron_train(X_train, y_train, learning_rate, max_iterations):
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

    num_features = X_train.shape[1]
    weights = np.zeros(num_features)

    for _ in range(max_iterations):
        for i in range(X_train.shape[0]):
            if np.dot(X_train[i], weights) * y_train[i] <= 0:
                weights += learning_rate * y_train[i] * X_train[i]

    return weights

def perceptron_test(X_test, y_test, weights):
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

    y_pred = np.sign(np.dot(X_test, weights))

    accuracy = np.mean(y_pred == y_test)

    confusion_matrix = np.zeros((2, 2))
    for i in range(len(y_test)):
        if y_test[i] == 1:
            if y_pred[i] == 1:
                confusion_matrix[0, 0] += 1
            else:
                confusion_matrix[0, 1] += 1
        else:
            if y_pred[i] == -1:
                confusion_matrix[1, 1] += 1
            else:
                confusion_matrix[1, 0] += 1

    return accuracy, confusion_matrix

def main(train_filename, test_filename, learning_rate, max_iterations):
    print(f"Dataset: {train_filename.split('/')[0]}")

    X_train, y_train = load_data(train_filename)
    X_test, y_test = load_data(test_filename)

    weights = perceptron_train(X_train, y_train, learning_rate, max_iterations)

    accuracy, confusion_matrix = perceptron_test(X_test, y_test, weights)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(confusion_matrix)

if __name__ == "__main__":
    data_directories = {
        "spam": ["spam/spam.data", "spam/spam.data"],
        "ovarian": ["ovarian/ovarian.data", "ovarian/ovarian.data"],
        "leukemia": ["leukemia/ALLAML.trn", "leukemia/ALLAML.tst"]
    }

    learning_rate = 0.1
    max_iterations = 1000

    for dataset, filenames in data_directories.items():
        train_filename, test_filename = filenames
        main(train_filename, test_filename, learning_rate, max_iterations)
