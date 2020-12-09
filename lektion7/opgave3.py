from sklearn import datasets
from sklearn.cluster import KMeans  # This will be used for the algorithm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier


def doKnn(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    print("Model is trained")
    predictions = knn.predict(X_test)

    print("Evaluating performance: Confusion matrix")
    matrix = confusion_matrix(y_test, predictions)
    print(matrix)
    # output is the confusion matrix

    print("precision: " + str(precision_score(y_test, predictions, average='weighted')))
    print("recall: " + str(recall_score(y_test, predictions, average='weighted')))
    print("F1 score: " + str(f1_score(y_test, predictions, average='weighted')))
    print(classification_report(y_test, predictions))



iris_df = datasets.load_iris()
X, y = iris_df.data, iris_df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("-----KNN with full featureset-----")
doKnn(X_train, X_test, y_train, y_test)

pca = PCA(n_components=2)
print(X.shape)
pca.fit(X_train)
X_train_reduced_two = pca.transform(X_train)
X_test_reduced_two = pca.transform(X_test)

print("-----X_train_reduced to two-----")
print(X_train_reduced_two)

print("-----KNN with the two features-----")
doKnn(X_train_reduced_two, X_test_reduced_two, y_train, y_test)

pca = PCA(n_components=1)
print(X.shape)
pca.fit(X_train)
X_train_reduced_one = pca.transform(X_train)
X_test_reduced_one = pca.transform(X_test)

print("-----X_train_reduced to two-----")
print(X_train_reduced_one)

print("-----KNN with one features-----")
doKnn(X_train_reduced_one, X_test_reduced_one, y_train, y_test)


