# example of using mnist set for classification
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier

print("MNIST example")
print("fetching data.....can take some time....")

mnist = fetch_openml("mnist_784", version=1, cache=True)

print("have data now")

X, y = mnist["data"], mnist["target"]
print(X.shape)
print(y.shape)

# convert the string array to int array
y = y.astype(np.uint8)

train_size = 69000
X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

some_digit = X_train[0]
print("label : "+str(y_train[0]))

# visualize the distributions
plt.hist(y_train, bins=10)

knn = KNeighborsClassifier(n_neighbors=3)
print("Training the model... please wait..")
knn.fit(X_train, y_train)

print("Model is trained")
predictions = knn.predict(X_test)


print("Evaluating performance: Confusion matrix")
matrix = confusion_matrix(y_test, predictions)
print(matrix)


print("precision: " + str(precision_score(y_test, predictions, average='weighted')))
print("recall: " + str(recall_score(y_test, predictions, average='weighted')))
print("F1 score: " + str(f1_score(y_test, predictions, average='weighted')))
print(classification_report(y_test, predictions))

plt.show()
print("end of program")










