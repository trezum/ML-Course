from pandas import DataFrame
from sklearn.datasets import make_circles
import pandas as pd
from matplotlib import pyplot as plt

# generate 2d classification dataset
X, y = make_circles(n_samples=1000, noise=0.05)
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
   group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])

plt.show()
