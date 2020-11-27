print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest

titanic_dataframe = pd.read_csv('titanic_train_500_age_passengerclass.csv', sep=',', header=0)



def prepare_data_mean():
    data = titanic_dataframe.copy()
    yvalues = pd.DataFrame(dict(Survived=[]), dtype=int)
    yvalues["Survived"] = data["Survived"].copy()

    data.drop('Survived', axis=1, inplace=True)
    data.drop('PassengerId', axis=1, inplace=True)

    x_train = data.head(400)
    x_train = x_train.fillna(x_train.mean())

    x_test = data.tail(100)
    x_test = x_test.fillna(x_test.mean())

    y_train = yvalues.head(400)
    y_test = yvalues.tail(100)

    return x_train, x_test, y_train, y_test


X_train, X_test, X_train, X_test = prepare_data_mean()

rng = np.random.RandomState(42)

# Generate train data
X = 0.3 * rng.randn(100, 2)
#X_train = np.r_[X + 2, X - 2]
# Generate some regular novel observations
X = 0.3 * rng.randn(20, 2)
#X_test = np.r_[X + 2, X - 2]
# Generate some abnormal novel observations
# = rng.uniform(low=-4, high=4, size=(20, 2))

# fit the model
clf = IsolationForest(max_samples=100, random_state=rng)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
#y_pred_outliers = clf.predict(X_outliers)

# plot the line, the samples, and the nearest vectors to the plane
#xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
#Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = clf.decision_function(X_train)
# Z = Z.reshape(xx.shape)
#
# plt.title("IsolationForest")
# plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=20, edgecolor='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green', s=20, edgecolor='k')
#c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red', s=20, edgecolor='k')

plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([b1, b2],#, c],
           ["training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left")
plt.show()