# (0 = died, 1 = survived)
# (1 = first class, 2 = second class, 3 = third class)


# import panda library and a few others we will need
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

# skipping the header
data=pd.read_csv('titanic_train_500_age_passengerclass.csv', sep=',', header=0)

# show the data
print(data.describe(include='all'))
# the describe is a great way to get an overview of the data
print(data.values)

yvalues = pd.DataFrame(dict(Survived=[]), dtype=int)
yvalues["Survived"] = data["Survived"].copy()

# now the yvalues should contain just the survived column - our labels
x = data["Age"]
y = data["Pclass"]

plt.figure()
plt.scatter(x.values, y.values, color='black', s=20)
plt.show()

# now the yvalues should contain just the survived column - our labels
x = data["Age"]
y = data["Survived"]

plt.figure()
plt.scatter(x.values, y.values, color='black', s=20)
plt.show()