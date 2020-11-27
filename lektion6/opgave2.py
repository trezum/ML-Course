from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
from pandas import DataFrame
from matplotlib.colors import ListedColormap


# generate 2d classification dataset
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])


k = 2
#running kmeans clustering into two
kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
# this will contain the labels for our predicted clusters (either 0 or 1)
labels = kmeans.labels_
# the centers of the calculated clusters
clusters = kmeans.cluster_centers_
# printing our cluster centers - there will be 2 of them.
print(clusters)


# the color values are simply RGB values, so the colormap for k = 2, will give red ($FF0000) and green ($00FF00) colors
cmap_bold = [ListedColormap(['#FF0000', '#00FF00']),
             ListedColormap(['#FF0000', '#00FF00', '#0000FF']),
             ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFF00']),
             ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF'])]

# now plot the same points, but this time assigning the colors to indicate the clusters
plt.figure() # creating a new figure
plt.title('Clustering with Kmeans - k = 2')

plt.scatter(X[:, 0], X[:, 1], c=labels, edgecolor='black', cmap=cmap_bold[1], s=20)



plt.plot(clusters[0],clusters[1],'ys',markersize=15)

plt.show()

