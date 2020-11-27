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

def format_float(num):
    return np.format_float_positional(num, trim='-')

iris = load_iris()  # load iris sample dataset
#print(iris)

# X = iris.data[:,2:] # petal length and width, so 2D information
X = iris.data # 4D
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # , random_state=42

print("train:")
# print(X_train)

print("test:")
# print(X_test)


# check how many samples we have
print("Number of samples: " + str(len(y)))
print(iris.target_names)


log_reg = LogisticRegression(max_iter=1000)
#so our y will be 1, if it is verginica (label is 2 of that flower), or 0 otherwise
y = (y_train == 2).astype(np.int)
log_reg.fit(X_train, y)


print("Model is trained")
predictions = log_reg.predict(X_test)

# looking at how good our predictor is.
print("Evaluating performance: Cross validation ")

scores = cross_val_score(log_reg, X_train, y_train, cv=3, scoring="accuracy")
print(scores)

ytest = (y_test == 2).astype(np.int)
print("Evaluating performance: Confusion matrix")
matrix = confusion_matrix(ytest, predictions)
print(matrix)
#output is the confusion matrix


print("precision: " + str(precision_score(ytest, predictions)))
print("recall: " + str(recall_score(ytest, predictions)))
print("F1 score: " + str(f1_score(ytest, predictions)))
print(classification_report(ytest, predictions))

