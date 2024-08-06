import numpy as np
import csv
from csv import reader
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def NaiveBayes(filename1, filename2):
    print("Implement",filename1,": ")

    trainset = load_csv(filename1)
    testset = load_csv(filename2)
    temp1 = [i[0:-1] for i in trainset]
    yTrain = np.array(list(map(int, [i[-1] for i in trainset])))
    temp2 = [i[0:-1] for i in testset]
    yTest = np.array(list(map(int, [i[-1] for i in testset])))
    xTrain = np.array([list(map(float,i)) for i in temp1])
    xTest = np.array([list(map(float,i)) for i in temp2])

    model = GaussianNB()
    model.fit(xTrain,yTrain)
    y_predicted = model.predict(xTest)
    score = accuracy_score(yTest, y_predicted)*100

    print("Accuracy: ", score, "%")
    print("Confusion Matrix: \n", metrics.confusion_matrix(yTest,y_predicted))
    print("Classification Report: \n", metrics.classification_report(yTest,y_predicted))

NaiveBayes('iris/iris.trn', 'iris/iris.tst')
NaiveBayes('letter/let.trn', 'letter/let.tst')        
NaiveBayes('optics/opt.trn', 'optics/opt.tst')
NaiveBayes('fp/fp.trn', 'fp/fp.tst')
NaiveBayes('leukemia/ALLAML.trn', 'leukemia/ALLAML.tst')