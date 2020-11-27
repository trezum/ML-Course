#this program explores the data set of the housing.csv.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

data = pd.read_csv("iris.csv")

data.hist(bins=20, figsize=(20, 15))
plt.show()


pd.set_option('display.max_columns', None)
print(data.head())
print(data.info())
print(data.describe())
print(data.corr())

attributes = ["sepal.length", "sepal.width", "petal.length", "petal.width"]

scatter_matrix(data[attributes], figsize=(12, 8))
plt.show()

data.plot(kind="scatter", x="petal.length", y="petal.width", alpha=0.1)
plt.show()

data.plot(kind="scatter", x="petal.length", y="sepal.width", alpha=0.1)
plt.show()

data.plot(kind="scatter", x="sepal.length", y="petal.width", alpha=0.1)
plt.show()