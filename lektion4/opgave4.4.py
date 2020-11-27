from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import numpy as np

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


tree_clf = RandomForestClassifier(max_depth=10) ## indicate we do not want the tree to be deeper than 2 levels
tree_clf.fit(X_train, y_train) # training the classifier


print("Model is trained")
predictions = tree_clf.predict(X_test)

print("Evaluating performance: Confusion matrix")
matrix = confusion_matrix(y_test, predictions)
print(matrix)

print("precision: " + str(precision_score(y_test, predictions, average='weighted')))
print("recall: " + str(recall_score(y_test, predictions, average='weighted')))
print("F1 score: " + str(f1_score(y_test, predictions, average='weighted')))












