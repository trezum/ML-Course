import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from datetime import datetime
import os
import glob

path = r'C:\Datasets\Stocks' # use your path
all_files = glob.glob(path + "/*.txt")

#print(all_files)
li = []
for filename in all_files:
    print(filename)
    if os.stat(filename).st_size != 0:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)
    else:
        print(filename + "Empty")


df = pd.concat(li, axis=0, ignore_index=True)

df.drop('OpenInt', axis=1, inplace=True)
df.drop('High', axis=1, inplace=True)
df.drop('Low', axis=1, inplace=True)
df.drop('Volume', axis=1, inplace=True)
df = df.dropna(axis='rows')

df.drop(df[df['Open'] == 0].index, inplace=True)
df.drop(df[df['Close'] == 0].index, inplace=True)

df['ChangePercentage'] = ((df.Close - df.Open) / df.Open)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')


# printing extreme data
dftest_negative = df.loc[df['ChangePercentage'] < -100]
print(dftest_negative)
dftest_positive = df.loc[df['ChangePercentage'] > 1000]
print(dftest_positive)


# Removing extreme data.
df.drop(df[df['ChangePercentage'] <= -100].index, inplace=True)
df.drop(df[df['ChangePercentage'] >= 1000].index, inplace=True)



print(df.describe())
print(df.info)




for y in range(1, 13):
    df_year = df[df['Date'].dt.month == y]
    print(str(y) + ": " + str(df_year['ChangePercentage'].mean(axis=0)))



# day, month, week, year
# https://stackoverflow.com/a/59604826/2067677











