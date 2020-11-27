from matplotlib.lines import Line2D
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


def format_float(num):
    return np.format_float_positional(num, trim='-')

iris = load_iris()  # load iris sample dataset
#print(iris)

X = iris.data[:,2:] # petal length and width, so 2D information
#X = iris.data # 4D
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, )#random_state=42) # , random_state=42

print("train:")
# print(X_train)

print("test:")
# print(X_test)


# check how many samples we have
print("Number of samples: " + str(len(y)))
print(iris.target_names)


Lin_SVC = LinearSVC(max_iter=10000)
Lin_SVC.fit(X_train, y_train)


print("Model is trained")
predictions = Lin_SVC.predict(X_test)

# looking at how good our predictor is.
print("Evaluating performance: Cross validation ")

scores = cross_val_score(Lin_SVC, X_train, y_train, cv=3, scoring="accuracy")
print(scores)



print("Evaluating performance: Confusion matrix")
matrix = confusion_matrix(y_test, predictions)
print(matrix)
#output is the confusion matrix


print("precision: " + str(precision_score(y_test, predictions, average='weighted')))
print("recall: " + str(recall_score(y_test, predictions, average='weighted')))
print("F1 score: " + str(f1_score(y_test, predictions, average='weighted')))
print(classification_report(y_test, predictions))

