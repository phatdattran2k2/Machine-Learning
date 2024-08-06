# Name: Tran Phat Dat
# ID Student: B2014971

import numpy as np
from sklearn.svm import SVC

def load_data(filename):
    data = np.loadtxt(filename, delimiter=",")
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def svm_train(X_train, y_train, C_value, kernel_type):
    clf = SVC(C=C_value, kernel=kernel_type)
    clf.fit(X_train, y_train)
    return clf

def svm_test(X_test, y_test, model):
    y_pred = model.predict(X_test)
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

def main(train_filename, test_filename, C_value, kernel_type):
    print(f"Dataset: {train_filename.split('/')[0]}")

    X_train, y_train = load_data(train_filename)
    X_test, y_test = load_data(test_filename)

    model = svm_train(X_train, y_train, C_value, kernel_type)

    accuracy, confusion_matrix = svm_test(X_test, y_test, model)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(confusion_matrix)

if __name__ == "__main__":
    data_directories = {
        "spam": ["spam/spam.data", "spam/spam.data"],
        "ovarian": ["ovarian/ovarian.data", "ovarian/ovarian.data"],
        "leukemia": ["leukemia/ALLAML.trn", "leukemia/ALLAML.tst"]
    }

    C_value = 1.0  # Tham số điều chuẩn
    kernel_type = 'linear'  # Loại kernel (linear, poly, rbf, sigmoid, etc.)

    for dataset, filenames in data_directories.items():
        train_filename, test_filename = filenames
        main(train_filename, test_filename, C_value, kernel_type)
