from matplotlib.lines import Line2D
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

iris = load_iris()

classifiers = [
    LogisticRegression(max_iter=1000),
    KNeighborsClassifier(),
    LinearSVC(max_iter=10000)]


class TestResult:
    def __init__(self, p, r, f, c):
        self.precision = p
        self.recall = r
        self.f1score = f
        self.classifier = c

    def __str__(self):
        return self.classifier + " precision:" + str(self.precision) + " recall:" + str(self.recall) + " f1:" + str(self.f1score)


def test_classifier(classifier_param):
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, )#random_state=42)
    classifier_param.fit(x_train, y_train)
    predictions = classifier_param.predict(x_test)

    return TestResult(
                precision_score(y_test, predictions, average='weighted'),
                recall_score(y_test, predictions, average='weighted'),
                f1_score(y_test, predictions, average='weighted'),
                classifier_param.__class__.__name__
            )


classifier_collumn = 'Classifier'

collumns = [classifier_collumn, 'Precision', 'Recall', 'f1score']
dataframe = pd.DataFrame(columns=collumns)
dataframe[classifier_collumn] = dataframe.Classifier.astype('str')
iterations_per_classifier = 100
count = 0

for classifier in classifiers:
    while count < iterations_per_classifier:
        testResult = test_classifier(classifier)
        df2 = pd.DataFrame([[classifier.__class__.__name__, testResult.precision, testResult.recall, testResult.f1score]], columns=collumns)
        df2[classifier_collumn] = df2.Classifier.astype('str')
        dataframe = dataframe.append(df2)
        count += 1
    count = 0

classifier_names = []

for classifier in classifiers:
    classifier_names.append(classifier.__class__.__name__)

for name in classifier_names:
    print('')
    print(name)
    test = dataframe[dataframe[classifier_collumn] == name]
    print(test.describe())












