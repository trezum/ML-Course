from sklearn import tree
from sklearn.datasets import make_moons
from matplotlib import pyplot
from pandas import DataFrame
# generate 2d classification dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=1000, noise=0.1)
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
colors = {0: 'red', 1: 'blue'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
#pyplot.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, )#random_state=42) # , random_state=42

tree_clf = RandomForestClassifier(max_depth=5) ## indicate we do not want the tree to be deeper than 2 levels
tree_clf.fit(X_train, y_train) # training the classifier


print("Model is trained")
predictions = tree_clf.predict(X_test)

# looking at how good our predictor is.
print("Evaluating performance: Cross validation ")

scores = cross_val_score(tree_clf, X_train, y_train, cv=3, scoring="accuracy")
print(scores)

print("Evaluating performance: Confusion matrix")
matrix = confusion_matrix(y_test, predictions)
print(matrix)

print("precision: " + str(precision_score(y_test, predictions, average='weighted')))
print("recall: " + str(recall_score(y_test, predictions, average='weighted')))
print("F1 score: " + str(f1_score(y_test, predictions, average='weighted')))


accuracy = []
error = []
# training - with different number of trees - from 1 til 70
for i in range(1, 50):
    clf = RandomForestClassifier(n_estimators=i)
    clf.fit(X, y)
    acc = clf.score(X, y)
    accuracy.append(acc)

plt.figure(figsize=(8,8))
plt.plot(accuracy, label='Accuracy')
plt.legend()
plt.title("RandomForest training - different number of trees")
plt.xlabel("Number of Trees used")
plt.show()













