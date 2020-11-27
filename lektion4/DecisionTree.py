
# coding: utf-8

# ## Example / Tutorial for the use of Decision Trees

# In this example we will look at how to use decision trees for classification.
# This technique works for labeled data and is therefore in the category of <b> supervised learning </b>.
# First we must have some imports to get up and running:

# In[31]:
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# Okay, so now we can start to look at some data:

# In[32]:

iris = load_iris()  # load iris sample dataset
X = iris.data[:,2:] # petal length and width, so 2D information
y = iris.target
# check how many samples we have
print("Number of samples: " +str(len(y)))
#visulize the dataset
plt.figure()
#define colors - red, green, blue
colormap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
# pplot labxel
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=colormap,edgecolor='black', s=500)


# Above you can see a visualiation of the data. In this case we are just looking at 2D, but it is important to note that decision trees work on n-dimensional data without any problems.
# As you can see from the data, we have 3 groups of data, each with a different color. So this data will be our training set. We now want to train a decision tree on this data, and then this decision tree classifier should of course be able to predict the classification of future data points



# In[34]:
tree_clf = DecisionTreeClassifier(max_depth=2) ## indicate we do not want the tree to be deeper than 2 levels
tree_clf.fit(X,y) # training the classifier
print("seed : "+ str(tree_clf.random_state))

#prediction
print("probability of point = (2,1) = "+str(tree_clf.predict_proba([[2,1]])))
print("probability of point = (4,1) = "+str(tree_clf.predict_proba([[4,1]])))
print("probability of point = (5,2) =  "+str(tree_clf.predict_proba([[5,2]])))


tree.plot_tree(tree_clf)
plt.show()













