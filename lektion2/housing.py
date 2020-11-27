#this program explores the data set of the housing.csv.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

data = pd.read_csv("housing.csv")

data.hist(bins=50, figsize=(20,15))
plt.show()



data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.2,
s=data["population"]/100, label="population", figsize=(10,7),
c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
plt.legend()
plt.show()


pd.set_option('display.max_columns', None)
print(data.head())
print(data.info())
print(data["ocean_proximity"].value_counts())
print(data.describe())

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(data[attributes], figsize=(12, 8))
plt.show()

data.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
plt.show()